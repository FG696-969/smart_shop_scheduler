"""Scheduling algorithms: FIFO, fixed-parameter GA, SLGA, and CP-AOL-SLGA.

The CP-AOL-SLGA here is a course-project version: it extends SLGA by allowing
an RL-style action to select different search operators and a simplified
bottleneck-machine local search.
"""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scheduler_core import Jobs, ScheduleRecord, build_fifo_chromosome, decode_chromosome


@dataclass
class AlgorithmResult:
    name: str
    chromosome: List[int]
    schedule: List[ScheduleRecord]
    makespan: int
    history: List[int] = field(default_factory=list)
    pc_history: List[float] = field(default_factory=list)
    pm_history: List[float] = field(default_factory=list)
    state_history: List[int] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    runtime: float = 0.0


def chromosome_base(jobs: Jobs, completed_counts: Optional[List[int]] = None) -> List[int]:
    completed_counts = completed_counts or [0] * len(jobs)
    genes: List[int] = []
    for job_id, job in enumerate(jobs):
        remaining = len(job) - completed_counts[job_id]
        genes.extend([job_id] * max(0, remaining))
    return genes


def create_population(jobs: Jobs, population_size: int, completed_counts: Optional[List[int]] = None) -> List[List[int]]:
    base = chromosome_base(jobs, completed_counts)
    population: List[List[int]] = []
    for _ in range(population_size):
        chrom = base[:]
        random.shuffle(chrom)
        population.append(chrom)
    if population:
        # FIFO-like sequence over remaining operations.
        fifo: List[int] = []
        remaining = [len(job) - (completed_counts or [0] * len(jobs))[i] for i, job in enumerate(jobs)]
        for round_id in range(max(remaining) if remaining else 0):
            for job_id, count in enumerate(remaining):
                if round_id < count:
                    fifo.append(job_id)
        population[0] = fifo
    return population


def evaluate(
    chromosome: Sequence[int],
    jobs: Jobs,
    num_machines: int,
    start_after: int = 0,
    initial_job_ready: Optional[List[int]] = None,
    initial_machine_ready: Optional[List[int]] = None,
    initial_next_operation: Optional[List[int]] = None,
    machine_breakdowns: Optional[Dict[int, Tuple[int, int]]] = None,
    emergency_job_start: Optional[int] = None,
    emergency_job_id: Optional[int] = None,
) -> int:
    _schedule, makespan = decode_chromosome(
        chromosome,
        jobs,
        num_machines,
        start_after=start_after,
        initial_job_ready=initial_job_ready,
        initial_machine_ready=initial_machine_ready,
        initial_next_operation=initial_next_operation,
        machine_breakdowns=machine_breakdowns,
        emergency_job_start=emergency_job_start,
        emergency_job_id=emergency_job_id,
    )
    return makespan


def tournament_selection(population: Sequence[List[int]], fitness_values: Sequence[int], tournament_size: int) -> List[int]:
    idxs = random.sample(range(len(population)), min(tournament_size, len(population)))
    best = min(idxs, key=lambda idx: fitness_values[idx])
    return population[best][:]


def count_preserving_crossover(parent1: Sequence[int], parent2: Sequence[int], required_counts: Dict[int, int]) -> List[int]:
    if not parent1:
        return []
    child: List[Optional[int]] = [None] * len(parent1)
    counts: Counter[int] = Counter()
    for i, gene in enumerate(parent1):
        if random.random() < 0.5 and counts[gene] < required_counts[gene]:
            child[i] = gene
            counts[gene] += 1
    open_positions = [i for i, gene in enumerate(child) if gene is None]
    cursor = 0
    for gene in parent2:
        if counts[gene] < required_counts[gene]:
            child[open_positions[cursor]] = gene
            counts[gene] += 1
            cursor += 1
            if cursor >= len(open_positions):
                break
    return [int(gene) for gene in child]


def swap_mutation(chromosome: List[int]) -> List[int]:
    if len(chromosome) >= 2:
        i, j = random.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome


def insertion_mutation(chromosome: List[int]) -> List[int]:
    if len(chromosome) >= 2:
        i, j = random.sample(range(len(chromosome)), 2)
        gene = chromosome.pop(i)
        chromosome.insert(j, gene)
    return chromosome


def diversity(population: Sequence[Sequence[int]]) -> float:
    if not population:
        return 0.0
    return len({tuple(ch) for ch in population}) / len(population)


def state_index(fitness_values: Sequence[int], population: Sequence[Sequence[int]], first_best: int, first_avg: float, num_states: int = 10) -> int:
    best_ratio = min(fitness_values) / first_best if first_best else 1.0
    avg_ratio = (sum(fitness_values) / len(fitness_values)) / first_avg if first_avg else 1.0
    div = diversity(population)
    score = 0.45 * best_ratio + 0.35 * avg_ratio + 0.20 * div
    score = max(0.0, min(0.999999, score / 1.5))
    return int(score * num_states)


def required_counts_for_population(jobs: Jobs, completed_counts: Optional[List[int]] = None) -> Dict[int, int]:
    completed_counts = completed_counts or [0] * len(jobs)
    return {job_id: max(0, len(job) - completed_counts[job_id]) for job_id, job in enumerate(jobs)}


def bottleneck_local_search(
    chromosome: List[int],
    jobs: Jobs,
    num_machines: int,
    current_makespan: int,
    **decode_kwargs,
) -> Tuple[List[int], int]:
    """Simplified critical-path/bottleneck-machine local search.

    Finds the machine where the last operation completes and tries a few adjacent
    swaps in the chromosome. Accepted only if makespan improves.
    """
    if len(chromosome) < 2:
        return chromosome, current_makespan
    schedule, _ = decode_chromosome(chromosome, jobs, num_machines, **decode_kwargs)
    last_record = max(schedule, key=lambda r: r["end"], default=None)
    if not last_record:
        return chromosome, current_makespan
    bottleneck_machine = last_record["machine_id"]
    machine_jobs = [r["job_id"] for r in sorted(schedule, key=lambda r: r["start"]) if r["machine_id"] == bottleneck_machine]
    candidate_positions = []
    for j1, j2 in zip(machine_jobs, machine_jobs[1:]):
        try:
            p1 = chromosome.index(j1)
            p2 = chromosome.index(j2, p1 + 1)
            candidate_positions.append((p1, p2))
        except ValueError:
            continue
    random.shuffle(candidate_positions)
    best = chromosome[:]
    best_makespan = current_makespan
    for p1, p2 in candidate_positions[:8]:
        trial = best[:]
        trial[p1], trial[p2] = trial[p2], trial[p1]
        try:
            value = evaluate(trial, jobs, num_machines, **decode_kwargs)
        except Exception:
            continue
        if value < best_makespan:
            best, best_makespan = trial, value
    return best, best_makespan


def run_fifo(jobs: Jobs, num_machines: int, random_seed: int = 42, **decode_kwargs) -> AlgorithmResult:
    started = time.time()
    chromosome = build_fifo_chromosome(jobs) if not decode_kwargs.get("initial_next_operation") else chromosome_base(jobs, decode_kwargs.get("initial_next_operation"))
    schedule, makespan = decode_chromosome(chromosome, jobs, num_machines, **decode_kwargs)
    return AlgorithmResult("FIFO", chromosome, schedule, makespan, history=[makespan], runtime=time.time() - started)


def run_ga(
    jobs: Jobs,
    num_machines: int,
    population_size: int = 80,
    generations: int = 140,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    random_seed: int = 42,
    local_search: bool = False,
    **decode_kwargs,
) -> AlgorithmResult:
    started = time.time()
    random.seed(random_seed)
    completed_counts = decode_kwargs.get("initial_next_operation")
    population = create_population(jobs, population_size, completed_counts)
    required_counts = required_counts_for_population(jobs, completed_counts)
    if not population or not population[0]:
        schedule = []
        makespan = decode_kwargs.get("start_after", 0)
        return AlgorithmResult("GA", [], schedule, int(makespan), history=[int(makespan)], runtime=time.time() - started)

    best_chrom = population[0][:]
    best_make = evaluate(best_chrom, jobs, num_machines, **decode_kwargs)
    history: List[int] = []
    for _ in range(generations):
        fitness = [evaluate(ch, jobs, num_machines, **decode_kwargs) for ch in population]
        elite_idx = min(range(len(population)), key=lambda i: fitness[i])
        if fitness[elite_idx] < best_make:
            best_chrom = population[elite_idx][:]
            best_make = fitness[elite_idx]
        if local_search:
            best_chrom, best_make = bottleneck_local_search(best_chrom, jobs, num_machines, best_make, **decode_kwargs)
        history.append(int(best_make))
        new_pop = [best_chrom[:]]
        while len(new_pop) < population_size:
            p1 = tournament_selection(population, fitness, tournament_size)
            p2 = tournament_selection(population, fitness, tournament_size)
            child = count_preserving_crossover(p1, p2, required_counts) if random.random() < crossover_rate else p1[:]
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_pop.append(child)
        population = new_pop
    schedule, final_make = decode_chromosome(best_chrom, jobs, num_machines, **decode_kwargs)
    return AlgorithmResult("GA", best_chrom, schedule, final_make, history=history, runtime=time.time() - started)


def run_slga(
    jobs: Jobs,
    num_machines: int,
    population_size: int = 80,
    generations: int = 140,
    tournament_size: int = 3,
    random_seed: int = 42,
    cp_aol: bool = False,
    **decode_kwargs,
) -> AlgorithmResult:
    """RL-tuned GA. If cp_aol=True, also selects mutation/local-search actions."""
    started = time.time()
    random.seed(random_seed)
    completed_counts = decode_kwargs.get("initial_next_operation")
    population = create_population(jobs, population_size, completed_counts)
    required_counts = required_counts_for_population(jobs, completed_counts)
    if not population or not population[0]:
        schedule = []
        makespan = decode_kwargs.get("start_after", 0)
        return AlgorithmResult("CP-AOL-SLGA" if cp_aol else "SLGA", [], schedule, int(makespan), history=[int(makespan)], runtime=time.time() - started)

    actions = [
        {"name": "High crossover / low mutation", "pc": 0.90, "pm": 0.08, "op": "swap", "local": False},
        {"name": "Balanced GA", "pc": 0.80, "pm": 0.18, "op": "swap", "local": False},
        {"name": "Exploration mutation", "pc": 0.65, "pm": 0.28, "op": "swap", "local": False},
        {"name": "Insertion mutation", "pc": 0.75, "pm": 0.22, "op": "insert", "local": False},
        {"name": "Bottleneck local search", "pc": 0.80, "pm": 0.14, "op": "swap", "local": True},
    ]
    if not cp_aol:
        actions = actions[:4]

    num_states = 10
    q = [[0.0 for _ in actions] for _ in range(num_states)]
    alpha, gamma, epsilon = 0.45, 0.25, 0.25
    fitness = [evaluate(ch, jobs, num_machines, **decode_kwargs) for ch in population]
    first_best = min(fitness)
    first_avg = sum(fitness) / len(fitness)
    state = state_index(fitness, population, first_best, first_avg, num_states)
    best_chrom = population[fitness.index(first_best)][:]
    best_make = first_best

    history: List[int] = []
    pc_history: List[float] = []
    pm_history: List[float] = []
    state_history: List[int] = []
    action_history: List[str] = []
    reward_history: List[float] = []

    for generation in range(generations):
        if random.random() < epsilon:
            action_idx = random.randrange(len(actions))
        else:
            max_q = max(q[state])
            best_actions = [i for i, value in enumerate(q[state]) if value == max_q]
            action_idx = random.choice(best_actions)
        action = actions[action_idx]
        pc, pm = action["pc"], action["pm"]
        prev_best = min(fitness)

        elite_idx = min(range(len(population)), key=lambda i: fitness[i])
        elite = population[elite_idx][:]
        new_pop = [elite]
        while len(new_pop) < population_size:
            p1 = tournament_selection(population, fitness, tournament_size)
            p2 = tournament_selection(population, fitness, tournament_size)
            child = count_preserving_crossover(p1, p2, required_counts) if random.random() < pc else p1[:]
            if random.random() < pm:
                if action.get("op") == "insert":
                    child = insertion_mutation(child)
                else:
                    child = swap_mutation(child)
            new_pop.append(child)
        population = new_pop
        fitness = [evaluate(ch, jobs, num_machines, **decode_kwargs) for ch in population]
        elite_idx = min(range(len(population)), key=lambda i: fitness[i])
        if fitness[elite_idx] < best_make:
            best_chrom = population[elite_idx][:]
            best_make = fitness[elite_idx]
        if action.get("local"):
            best_chrom, best_make = bottleneck_local_search(best_chrom, jobs, num_machines, best_make, **decode_kwargs)

        next_state = state_index(fitness, population, first_best, first_avg, num_states)
        reward = (prev_best - min(fitness)) / max(prev_best, 1)
        if best_make < prev_best:
            reward += 0.05
        q[state][action_idx] += alpha * (reward + gamma * max(q[next_state]) - q[state][action_idx])

        history.append(int(best_make))
        pc_history.append(float(pc))
        pm_history.append(float(pm))
        state_history.append(int(state))
        action_history.append(str(action["name"]))
        reward_history.append(float(reward))
        state = next_state

    schedule, final_make = decode_chromosome(best_chrom, jobs, num_machines, **decode_kwargs)
    return AlgorithmResult(
        "CP-AOL-SLGA" if cp_aol else "SLGA",
        best_chrom,
        schedule,
        final_make,
        history=history,
        pc_history=pc_history,
        pm_history=pm_history,
        state_history=state_history,
        action_history=action_history,
        reward_history=reward_history,
        runtime=time.time() - started,
    )


def run_algorithm(name: str, jobs: Jobs, num_machines: int, random_seed: int = 42, fast_mode: bool = True, **decode_kwargs) -> AlgorithmResult:
    if name == "FIFO":
        return run_fifo(jobs, num_machines, random_seed=random_seed, **decode_kwargs)
    if name == "GA":
        return run_ga(jobs, num_machines, population_size=70 if fast_mode else 120, generations=100 if fast_mode else 250, random_seed=random_seed, **decode_kwargs)
    if name == "SLGA":
        return run_slga(jobs, num_machines, population_size=70 if fast_mode else 120, generations=110 if fast_mode else 250, random_seed=random_seed, **decode_kwargs)
    if name == "CP-AOL-SLGA":
        return run_slga(jobs, num_machines, population_size=70 if fast_mode else 120, generations=120 if fast_mode else 260, random_seed=random_seed, cp_aol=True, **decode_kwargs)
    raise ValueError(f"Unknown algorithm: {name}")

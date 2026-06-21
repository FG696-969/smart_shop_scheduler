from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from scheduler_core import Jobs, decode_chromosome

from .common import evaluate


def count_preserving_crossover(
    parent1: Sequence[int],
    parent2: Sequence[int],
    required_counts: Dict[int, int],
) -> List[int]:
    if not parent1:
        return []
    child: List[Optional[int]] = [None] * len(parent1)
    counts: Counter[int] = Counter()
    for index, gene in enumerate(parent1):
        if random.random() < 0.5 and counts[gene] < required_counts[gene]:
            child[index] = gene
            counts[gene] += 1
    open_positions = [index for index, gene in enumerate(child) if gene is None]
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
        first, second = random.sample(range(len(chromosome)), 2)
        chromosome[first], chromosome[second] = chromosome[second], chromosome[first]
    return chromosome


def insertion_mutation(chromosome: List[int]) -> List[int]:
    if len(chromosome) >= 2:
        first, second = random.sample(range(len(chromosome)), 2)
        gene = chromosome.pop(first)
        chromosome.insert(second, gene)
    return chromosome


def diversity(population: Sequence[Sequence[int]]) -> float:
    if not population:
        return 0.0
    return len({tuple(chromosome) for chromosome in population}) / len(population)


def bottleneck_local_search(
    chromosome: List[int],
    jobs: Jobs,
    num_machines: int,
    current_makespan: int,
    **decode_kwargs,
) -> Tuple[List[int], int]:
    """Try a small set of swaps associated with the last-finishing machine."""
    if len(chromosome) < 2:
        return chromosome, current_makespan
    schedule, _ = decode_chromosome(
        chromosome, jobs, num_machines, **decode_kwargs
    )
    last_record = max(schedule, key=lambda record: record["end"], default=None)
    if not last_record:
        return chromosome, current_makespan
    bottleneck_machine = last_record["machine_id"]
    machine_jobs = [
        record["job_id"]
        for record in sorted(schedule, key=lambda record: record["start"])
        if record["machine_id"] == bottleneck_machine
    ]
    candidate_positions = []
    for first_job, second_job in zip(machine_jobs, machine_jobs[1:]):
        try:
            first_position = chromosome.index(first_job)
            second_position = chromosome.index(second_job, first_position + 1)
            candidate_positions.append((first_position, second_position))
        except ValueError:
            continue
    random.shuffle(candidate_positions)
    best = chromosome[:]
    best_makespan = current_makespan
    for first_position, second_position in candidate_positions[:8]:
        trial = best[:]
        trial[first_position], trial[second_position] = (
            trial[second_position],
            trial[first_position],
        )
        try:
            value = evaluate(trial, jobs, num_machines, **decode_kwargs)
        except Exception:
            continue
        if value < best_makespan:
            best, best_makespan = trial, value
    return best, best_makespan

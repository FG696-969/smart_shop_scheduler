from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scheduler_core import Jobs, ScheduleRecord, decode_chromosome


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


def chromosome_base(
    jobs: Jobs, completed_counts: Optional[List[int]] = None
) -> List[int]:
    completed_counts = completed_counts or [0] * len(jobs)
    genes: List[int] = []
    for job_id, job in enumerate(jobs):
        remaining = len(job) - completed_counts[job_id]
        genes.extend([job_id] * max(0, remaining))
    return genes


def create_population(
    jobs: Jobs,
    population_size: int,
    completed_counts: Optional[List[int]] = None,
) -> List[List[int]]:
    base = chromosome_base(jobs, completed_counts)
    population: List[List[int]] = []
    for _ in range(population_size):
        chromosome = base[:]
        random.shuffle(chromosome)
        population.append(chromosome)
    if population:
        fifo: List[int] = []
        completed = completed_counts or [0] * len(jobs)
        remaining = [len(job) - completed[index] for index, job in enumerate(jobs)]
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


def tournament_selection(
    population: Sequence[List[int]],
    fitness_values: Sequence[int],
    tournament_size: int,
) -> List[int]:
    indexes = random.sample(
        range(len(population)), min(tournament_size, len(population))
    )
    best = min(indexes, key=lambda index: fitness_values[index])
    return population[best][:]


def required_counts_for_population(
    jobs: Jobs, completed_counts: Optional[List[int]] = None
) -> Dict[int, int]:
    completed_counts = completed_counts or [0] * len(jobs)
    return {
        job_id: max(0, len(job) - completed_counts[job_id])
        for job_id, job in enumerate(jobs)
    }

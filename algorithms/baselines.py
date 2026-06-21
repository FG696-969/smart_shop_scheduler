from __future__ import annotations

import random
import time
from typing import List

from scheduler_core import Jobs, build_fifo_chromosome, decode_chromosome

from .common import (
    AlgorithmResult,
    chromosome_base,
    create_population,
    evaluate,
    required_counts_for_population,
    tournament_selection,
)
from .operators import (
    bottleneck_local_search,
    count_preserving_crossover,
    swap_mutation,
)


def run_fifo(
    jobs: Jobs,
    num_machines: int,
    random_seed: int = 42,
    **decode_kwargs,
) -> AlgorithmResult:
    started = time.time()
    completed_counts = decode_kwargs.get("initial_next_operation")
    chromosome = (
        chromosome_base(jobs, completed_counts)
        if completed_counts
        else build_fifo_chromosome(jobs)
    )
    schedule, makespan = decode_chromosome(
        chromosome, jobs, num_machines, **decode_kwargs
    )
    return AlgorithmResult(
        "FIFO",
        chromosome,
        schedule,
        makespan,
        history=[makespan],
        runtime=time.time() - started,
    )


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
        makespan = decode_kwargs.get("start_after", 0)
        return AlgorithmResult(
            "GA",
            [],
            [],
            int(makespan),
            history=[int(makespan)],
            runtime=time.time() - started,
        )

    best_chromosome = population[0][:]
    best_makespan = evaluate(
        best_chromosome, jobs, num_machines, **decode_kwargs
    )
    history: List[int] = []
    for _ in range(generations):
        fitness = [
            evaluate(chromosome, jobs, num_machines, **decode_kwargs)
            for chromosome in population
        ]
        elite_index = min(range(len(population)), key=lambda index: fitness[index])
        if fitness[elite_index] < best_makespan:
            best_chromosome = population[elite_index][:]
            best_makespan = fitness[elite_index]
        if local_search:
            best_chromosome, best_makespan = bottleneck_local_search(
                best_chromosome,
                jobs,
                num_machines,
                best_makespan,
                **decode_kwargs,
            )
        history.append(int(best_makespan))
        new_population = [best_chromosome[:]]
        while len(new_population) < population_size:
            first_parent = tournament_selection(
                population, fitness, tournament_size
            )
            second_parent = tournament_selection(
                population, fitness, tournament_size
            )
            child = (
                count_preserving_crossover(
                    first_parent, second_parent, required_counts
                )
                if random.random() < crossover_rate
                else first_parent[:]
            )
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_population.append(child)
        population = new_population
    schedule, final_makespan = decode_chromosome(
        best_chromosome, jobs, num_machines, **decode_kwargs
    )
    return AlgorithmResult(
        "GA",
        best_chromosome,
        schedule,
        final_makespan,
        history=history,
        runtime=time.time() - started,
    )

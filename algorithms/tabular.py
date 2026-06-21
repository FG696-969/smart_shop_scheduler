from __future__ import annotations

import random
import time
from typing import List, Sequence

from rl.actions import SEARCH_ACTIONS
from scheduler_core import Jobs, decode_chromosome

from .common import (
    AlgorithmResult,
    create_population,
    evaluate,
    required_counts_for_population,
    tournament_selection,
)
from .operators import (
    bottleneck_local_search,
    count_preserving_crossover,
    diversity,
    insertion_mutation,
    swap_mutation,
)


def state_index(
    fitness_values: Sequence[int],
    population: Sequence[Sequence[int]],
    first_best: int,
    first_average: float,
    num_states: int = 10,
) -> int:
    best_ratio = min(fitness_values) / first_best if first_best else 1.0
    average_ratio = (
        (sum(fitness_values) / len(fitness_values)) / first_average
        if first_average
        else 1.0
    )
    score = 0.45 * best_ratio + 0.35 * average_ratio + 0.20 * diversity(population)
    score = max(0.0, min(0.999999, score / 1.5))
    return int(score * num_states)


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
    started = time.time()
    random.seed(random_seed)
    completed_counts = decode_kwargs.get("initial_next_operation")
    population = create_population(jobs, population_size, completed_counts)
    required_counts = required_counts_for_population(jobs, completed_counts)
    result_name = "CP-AOL-SLGA" if cp_aol else "SLGA"
    if not population or not population[0]:
        makespan = decode_kwargs.get("start_after", 0)
        return AlgorithmResult(
            result_name,
            [],
            [],
            int(makespan),
            history=[int(makespan)],
            runtime=time.time() - started,
        )

    actions = SEARCH_ACTIONS if cp_aol else SEARCH_ACTIONS[:4]
    num_states = 10
    q_table = [[0.0 for _ in actions] for _ in range(num_states)]
    alpha, gamma, epsilon = 0.45, 0.25, 0.25
    fitness = [
        evaluate(chromosome, jobs, num_machines, **decode_kwargs)
        for chromosome in population
    ]
    first_best = min(fitness)
    first_average = sum(fitness) / len(fitness)
    state = state_index(
        fitness, population, first_best, first_average, num_states
    )
    best_chromosome = population[fitness.index(first_best)][:]
    best_makespan = first_best

    history: List[int] = []
    pc_history: List[float] = []
    pm_history: List[float] = []
    state_history: List[int] = []
    action_history: List[str] = []
    reward_history: List[float] = []

    for _generation in range(generations):
        if random.random() < epsilon:
            action_index = random.randrange(len(actions))
        else:
            max_q = max(q_table[state])
            best_actions = [
                index for index, value in enumerate(q_table[state]) if value == max_q
            ]
            action_index = random.choice(best_actions)
        action = actions[action_index]
        previous_best = min(fitness)

        elite_index = min(range(len(population)), key=lambda index: fitness[index])
        elite = population[elite_index][:]
        new_population = [elite]
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
                if random.random() < action.crossover_rate
                else first_parent[:]
            )
            if random.random() < action.mutation_rate:
                child = (
                    insertion_mutation(child)
                    if action.mutation == "insert"
                    else swap_mutation(child)
                )
            new_population.append(child)
        population = new_population
        fitness = [
            evaluate(chromosome, jobs, num_machines, **decode_kwargs)
            for chromosome in population
        ]
        elite_index = min(range(len(population)), key=lambda index: fitness[index])
        if fitness[elite_index] < best_makespan:
            best_chromosome = population[elite_index][:]
            best_makespan = fitness[elite_index]
        if action.local_search:
            best_chromosome, best_makespan = bottleneck_local_search(
                best_chromosome,
                jobs,
                num_machines,
                best_makespan,
                **decode_kwargs,
            )

        next_state = state_index(
            fitness, population, first_best, first_average, num_states
        )
        reward = (previous_best - min(fitness)) / max(previous_best, 1)
        if best_makespan < previous_best:
            reward += 0.05
        q_table[state][action_index] += alpha * (
            reward
            + gamma * max(q_table[next_state])
            - q_table[state][action_index]
        )

        history.append(int(best_makespan))
        pc_history.append(float(action.crossover_rate))
        pm_history.append(float(action.mutation_rate))
        state_history.append(int(state))
        action_history.append(action.label)
        reward_history.append(float(reward))
        state = next_state

    schedule, final_makespan = decode_chromosome(
        best_chromosome, jobs, num_machines, **decode_kwargs
    )
    return AlgorithmResult(
        result_name,
        best_chromosome,
        schedule,
        final_makespan,
        history=history,
        pc_history=pc_history,
        pm_history=pm_history,
        state_history=state_history,
        action_history=action_history,
        reward_history=reward_history,
        runtime=time.time() - started,
    )

from __future__ import annotations

import random
import time
from typing import List, Optional

import numpy as np

from rl.actions import SEARCH_ACTIONS, SearchAction
from rl.dqn_agent import DQNAgent
from rl.state_encoder import StateContext, encode_state
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


def _remaining_operation_context(
    jobs: Jobs,
    num_machines: int,
    completed_counts: Optional[List[int]],
) -> tuple[float, tuple[float, ...]]:
    completed = completed_counts or [0] * len(jobs)
    total_operations = sum(len(job) for job in jobs)
    remaining_operations = 0
    machine_loads = [0.0] * num_machines
    for job_id, job in enumerate(jobs):
        next_operation = completed[job_id] if job_id < len(completed) else 0
        remaining_operations += max(0, len(job) - next_operation)
        for machine_id, processing_time in job[next_operation:]:
            machine_loads[machine_id] += processing_time
    remaining_ratio = remaining_operations / max(total_operations, 1)
    return float(remaining_ratio), tuple(machine_loads)


def _breakdown_pressure(decode_kwargs: dict, current_makespan: float) -> float:
    breakdowns = decode_kwargs.get("machine_breakdowns") or {}
    duration = sum(max(0, end - start) for start, end in breakdowns.values())
    return float(duration / max(abs(current_makespan), 1.0))


def _state_context(
    jobs: Jobs,
    num_machines: int,
    completed_counts: Optional[List[int]],
    initial_best: float,
    initial_average: float,
    generation: int,
    generations: int,
    stagnation: int,
    current_makespan: float,
    decode_kwargs: dict,
) -> StateContext:
    remaining_ratio, machine_loads = _remaining_operation_context(
        jobs, num_machines, completed_counts
    )
    return StateContext(
        initial_best=initial_best,
        initial_average=initial_average,
        generation=generation,
        generation_budget=generations,
        stagnation=stagnation,
        remaining_ratio=remaining_ratio,
        machine_loads=machine_loads,
        breakdown_pressure=_breakdown_pressure(decode_kwargs, current_makespan),
        emergency_job=decode_kwargs.get("emergency_job_id") is not None,
    )


def _evolve_population(
    population: List[List[int]],
    fitness: List[int],
    population_size: int,
    tournament_size: int,
    required_counts: dict[int, int],
    action: SearchAction,
) -> List[List[int]]:
    elite_index = min(range(len(population)), key=lambda index: fitness[index])
    new_population = [population[elite_index][:]]
    while len(new_population) < population_size:
        first_parent = tournament_selection(population, fitness, tournament_size)
        second_parent = tournament_selection(population, fitness, tournament_size)
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
    return new_population


def run_dqn_ga(
    jobs: Jobs,
    num_machines: int,
    agent: DQNAgent,
    population_size: int = 80,
    generations: int = 140,
    random_seed: int = 42,
    training: bool = False,
    **decode_kwargs,
) -> AlgorithmResult:
    started = time.time()
    random.seed(random_seed)
    inference_exploration = bool(
        decode_kwargs.pop("inference_exploration", False)
    )
    completed_counts = decode_kwargs.get("initial_next_operation")
    population = create_population(jobs, population_size, completed_counts)
    required_counts = required_counts_for_population(jobs, completed_counts)
    if not population or not population[0]:
        makespan = int(decode_kwargs.get("start_after", 0))
        return AlgorithmResult(
            "DQN-AOL-GA",
            [],
            [],
            makespan,
            history=[makespan],
            runtime=time.time() - started,
        )

    fitness = [
        evaluate(chromosome, jobs, num_machines, **decode_kwargs)
        for chromosome in population
    ]
    initial_best = min(fitness)
    initial_average = sum(fitness) / len(fitness)
    best_index = fitness.index(initial_best)
    best_chromosome = population[best_index][:]
    best_makespan = initial_best
    stagnation = 0

    history: List[int] = []
    pc_history: List[float] = []
    pm_history: List[float] = []
    action_history: List[str] = []
    reward_history: List[float] = []
    loss_history: List[float] = []
    epsilon_history: List[float] = []
    q_value_history: List[List[float]] = []

    for generation in range(generations):
        previous_best = min(fitness)
        previous_average = sum(fitness) / len(fitness)
        previous_diversity = diversity(population)
        context = _state_context(
            jobs,
            num_machines,
            completed_counts,
            initial_best,
            initial_average,
            generation,
            generations,
            stagnation,
            previous_best,
            decode_kwargs,
        )
        state = encode_state(fitness, population, context)
        q_values = agent.q_values(state)
        action_index = agent.select_action(
            state, explore=training or inference_exploration
        )
        action = SEARCH_ACTIONS[action_index]

        population = _evolve_population(
            population,
            fitness,
            population_size,
            3,
            required_counts,
            action,
        )
        fitness = [
            evaluate(chromosome, jobs, num_machines, **decode_kwargs)
            for chromosome in population
        ]
        generation_best_index = min(
            range(len(population)), key=lambda index: fitness[index]
        )
        generation_best_chromosome = population[generation_best_index][:]
        generation_best_makespan = fitness[generation_best_index]
        if action.local_search:
            refined_chromosome, refined_makespan = bottleneck_local_search(
                generation_best_chromosome,
                jobs,
                num_machines,
                generation_best_makespan,
                **decode_kwargs,
            )
            if refined_makespan < generation_best_makespan:
                population[generation_best_index] = refined_chromosome
                fitness[generation_best_index] = refined_makespan
                generation_best_chromosome = refined_chromosome
                generation_best_makespan = refined_makespan

        improved_global_best = generation_best_makespan < best_makespan
        if improved_global_best:
            best_chromosome = generation_best_chromosome[:]
            best_makespan = generation_best_makespan

        current_average = sum(fitness) / len(fitness)
        current_diversity = diversity(population)
        best_improvement = (previous_best - generation_best_makespan) / max(
            abs(previous_best), 1.0
        )
        average_improvement = (previous_average - current_average) / max(
            abs(previous_average), 1.0
        )
        diversity_recovery = (
            max(0.0, current_diversity - previous_diversity)
            if best_improvement <= 0.0
            else 0.0
        )
        reward = (
            0.65 * best_improvement
            + 0.20 * average_improvement
            + 0.10 * diversity_recovery
            - 0.05 * action.cost
        )
        if improved_global_best:
            reward += 0.03
        done = generation == generations - 1
        if done and best_makespan < initial_best:
            reward += 0.02
        reward = float(np.clip(reward, -1.0, 1.0))

        stagnation = 0 if generation_best_makespan < previous_best else stagnation + 1
        next_context = _state_context(
            jobs,
            num_machines,
            completed_counts,
            initial_best,
            initial_average,
            generation + 1,
            generations,
            stagnation,
            generation_best_makespan,
            decode_kwargs,
        )
        next_state = encode_state(fitness, population, next_context)
        if training:
            agent.remember(state, action_index, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                loss_history.append(loss)

        history.append(int(best_makespan))
        pc_history.append(float(action.crossover_rate))
        pm_history.append(float(action.mutation_rate))
        action_history.append(action.label)
        reward_history.append(reward)
        epsilon_history.append(float(agent.epsilon))
        q_value_history.append([float(value) for value in q_values])

    schedule, final_makespan = decode_chromosome(
        best_chromosome, jobs, num_machines, **decode_kwargs
    )
    return AlgorithmResult(
        "DQN-AOL-GA",
        best_chromosome,
        schedule,
        final_makespan,
        history=history,
        pc_history=pc_history,
        pm_history=pm_history,
        action_history=action_history,
        reward_history=reward_history,
        runtime=time.time() - started,
        loss_history=loss_history,
        epsilon_history=epsilon_history,
        q_value_history=q_value_history,
    )

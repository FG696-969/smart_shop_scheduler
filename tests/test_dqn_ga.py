import copy

import numpy as np
import pytest
import torch

from algorithms import AlgorithmResult, run_algorithm
import algorithms.dqn_ga as dqn_ga
from algorithms.dqn_ga import _state_context, run_dqn_ga
from data_loader import load_dataset
from rl.dqn_agent import DQNAgent, DQNConfig
from scheduler_core import validate_chromosome


def test_algorithm_result_preserves_old_constructor_compatibility():
    result = AlgorithmResult("legacy", [], [], 0)

    assert result.loss_history == []
    assert result.epsilon_history == []
    assert result.q_value_history == []


def test_dqn_ga_produces_feasible_schedule_and_learning_traces():
    jobs, metadata = load_dataset("FT06")
    generations = 8
    agent = DQNAgent(DQNConfig(seed=42, batch_size=4, target_sync_interval=2))

    result = run_dqn_ga(
        jobs,
        int(metadata["machines"]),
        agent=agent,
        population_size=12,
        generations=generations,
        random_seed=42,
        training=True,
    )

    validate_chromosome(result.chromosome, jobs)
    assert result.name == "DQN-AOL-GA"
    assert result.makespan == max(record["end"] for record in result.schedule)
    assert len(result.history) == generations
    assert len(result.action_history) == generations
    assert len(result.reward_history) == generations
    assert len(result.epsilon_history) == generations
    assert len(result.q_value_history) == generations
    assert all(len(values) == 5 for values in result.q_value_history)
    assert len(result.loss_history) == generations - agent.config.batch_size + 1
    assert all(np.isfinite(value) for value in result.loss_history)
    assert all(-1.0 <= reward <= 1.0 for reward in result.reward_history)


def test_dqn_ga_seed_reproduces_inference_trace_without_mutating_agent():
    jobs, metadata = load_dataset("FT06")
    config = DQNConfig(seed=17, epsilon_start=0.7)
    first_agent = DQNAgent(config)
    second_agent = DQNAgent(config)
    before_online = copy.deepcopy(first_agent.online.state_dict())
    before_target = copy.deepcopy(first_agent.target.state_dict())
    kwargs = {
        "population_size": 10,
        "generations": 6,
        "random_seed": 17,
        "training": False,
    }

    first = run_dqn_ga(
        jobs, int(metadata["machines"]), agent=first_agent, **kwargs
    )
    second = run_dqn_ga(
        jobs, int(metadata["machines"]), agent=second_agent, **kwargs
    )

    assert first.action_history == second.action_history
    assert first.q_value_history == second.q_value_history
    assert first.makespan == second.makespan
    assert first_agent.epsilon == config.epsilon_start
    assert first_agent.update_count == 0
    assert len(first_agent.replay_buffer) == 0
    for name, tensor in first_agent.online.state_dict().items():
        torch.testing.assert_close(tensor, before_online[name])
    for name, tensor in first_agent.target.state_dict().items():
        torch.testing.assert_close(tensor, before_target[name])


def test_registry_requires_agent_for_dqn_ga():
    jobs, metadata = load_dataset("Custom 5x4")

    with pytest.raises(ValueError, match="^DQN-AOL-GA requires a DQNAgent$"):
        run_algorithm(
            "DQN-AOL-GA",
            jobs,
            int(metadata["machines"]),
            fast_mode=True,
        )


def test_dqn_state_context_uses_remaining_work_and_disturbance_data():
    jobs = [[(0, 4), (1, 3)], [(1, 5), (0, 2)]]

    context = _state_context(
        jobs=jobs,
        num_machines=2,
        completed_counts=[1, 0],
        initial_best=30,
        initial_average=35,
        generation=2,
        generations=8,
        stagnation=1,
        current_makespan=20,
        decode_kwargs={
            "machine_breakdowns": {0: (5, 10)},
            "emergency_job_id": 2,
        },
    )

    assert context.remaining_ratio == 0.75
    assert context.machine_loads == (2.0, 8.0)
    assert context.breakdown_pressure == 0.25
    assert context.emergency_job is True


def test_reward_uses_only_weighted_terms_and_clips_exactly():
    assert hasattr(dqn_ga, "_calculate_reward")
    reward = dqn_ga._calculate_reward(
        best_improvement=0.2,
        average_improvement=-0.1,
        diversity_recovery=0.4,
        action_cost=0.3,
    )

    assert reward == pytest.approx(0.135)
    assert dqn_ga._calculate_reward(4.0, 0.0, 0.0, 0.0) == 1.0
    assert dqn_ga._calculate_reward(-4.0, 0.0, 0.0, 0.0) == -1.0


def test_reward_adds_terminal_bonus_once_only_for_final_improvement():
    assert hasattr(dqn_ga, "_calculate_reward")
    nonterminal = dqn_ga._calculate_reward(0.1, 0.2, 0.3, 0.4)
    terminal = dqn_ga._calculate_reward(
        0.1,
        0.2,
        0.3,
        0.4,
        terminal_improved=True,
    )

    assert nonterminal == pytest.approx(0.115)
    assert terminal == pytest.approx(0.135)

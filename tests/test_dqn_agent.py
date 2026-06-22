import copy

import numpy as np
import torch

from rl.dqn_agent import DQNAgent, DQNConfig


def _fill_replay(agent: DQNAgent, count: int = 8) -> None:
    for index in range(count):
        state = np.full(10, index / 10, dtype=np.float32)
        agent.remember(state, index % 5, 0.1 * index, state + 0.01, index == count - 1)


def test_agent_greedy_action_and_q_values_are_seeded_and_deterministic():
    state = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    first = DQNAgent(DQNConfig(seed=11))
    second = DQNAgent(DQNConfig(seed=11))

    np.testing.assert_array_equal(first.q_values(state), second.q_values(state))
    assert first.select_action(state, explore=False) == second.select_action(
        state, explore=False
    )


def test_training_step_updates_online_network_and_epsilon():
    agent = DQNAgent(DQNConfig(seed=3, batch_size=4, epsilon_decay=0.5))
    _fill_replay(agent)
    before = [parameter.detach().clone() for parameter in agent.online.parameters()]

    loss = agent.train_step()

    assert loss is not None and loss >= 0.0
    assert agent.update_count == 1
    assert agent.epsilon == 0.5
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online.parameters())
    )


def test_target_network_synchronizes_at_configured_interval():
    agent = DQNAgent(
        DQNConfig(seed=5, batch_size=4, target_sync_interval=1, epsilon_start=0.2)
    )
    _fill_replay(agent)

    assert agent.train_step() is not None

    for online_parameter, target_parameter in zip(
        agent.online.parameters(), agent.target.parameters()
    ):
        torch.testing.assert_close(online_parameter, target_parameter)


def test_inference_and_insufficient_training_do_not_mutate_agent():
    agent = DQNAgent(DQNConfig(seed=7, batch_size=4))
    state = np.zeros(10, dtype=np.float32)
    before_online = copy.deepcopy(agent.online.state_dict())
    before_target = copy.deepcopy(agent.target.state_dict())
    before_epsilon = agent.epsilon

    agent.q_values(state)
    agent.select_action(state, explore=False)
    assert agent.train_step() is None

    assert agent.update_count == 0
    assert agent.epsilon == before_epsilon
    for name, tensor in agent.online.state_dict().items():
        torch.testing.assert_close(tensor, before_online[name])
    for name, tensor in agent.target.state_dict().items():
        torch.testing.assert_close(tensor, before_target[name])


def test_double_dqn_uses_online_argmax_and_target_value():
    agent = DQNAgent(
        DQNConfig(
            state_size=2,
            action_size=3,
            gamma=1.0,
            lr=0.0,
            batch_size=1,
            target_sync_interval=100,
        )
    )
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([0.0, 5.0, 0.0]))
        agent.target.layers[-1].bias.copy_(torch.tensor([0.0, 2.0, 9.0]))
    agent.remember(
        np.zeros(2, dtype=np.float32),
        0,
        0.0,
        np.ones(2, dtype=np.float32),
        False,
    )

    loss = agent.train_step()

    assert loss == 1.5

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.checkpoint import (
    DQN_STATE_VERSION,
    GA_ACTION_VERSION,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from rl.dqn_agent import DQNAgent, DQNConfig


def test_checkpoint_round_trip_restores_training_state(tmp_path: Path):
    config = DQNConfig(
        seed=9,
        batch_size=2,
        replay_capacity=7,
        target_sync_interval=3,
        epsilon_decay=0.8,
    )
    source = DQNAgent(config)
    for index in range(5):
        state = np.full(10, index, dtype=np.float32)
        source.remember(state, index % 5, 0.5 + index, state + 1, index == 4)
    assert source.train_step() is not None
    source.epsilon = 1.0
    path = tmp_path / "agent.pt"
    metadata = CheckpointMetadata(
        DQN_STATE_VERSION, GA_ACTION_VERSION, 9, ("FT06",), 2
    )

    save_checkpoint(path, source, metadata)
    expected_replay_sample = source.replay_buffer.sample(3)
    exploration_state = np.linspace(0, 1, 10, dtype=np.float32)
    expected_actions = [source.select_action(exploration_state) for _ in range(6)]
    expected_numpy = source._np_random.random_sample(4)
    expected_torch = torch.rand(4)

    loaded, loaded_metadata = load_checkpoint(path)

    state = np.linspace(0, 1, 10, dtype=np.float32)
    np.testing.assert_allclose(loaded.q_values(state), source.q_values(state))
    assert loaded_metadata == metadata
    assert loaded.epsilon == source.epsilon
    assert loaded.update_count == source.update_count
    assert loaded.optimizer.state_dict()["state"]
    assert len(loaded.replay_buffer) == len(source.replay_buffer) == 5
    assert (
        loaded.replay_buffer.state_dict()["transitions"]
        == source.replay_buffer.state_dict()["transitions"]
    )

    actual_replay_sample = loaded.replay_buffer.sample(3)
    np.testing.assert_array_equal(
        actual_replay_sample.states, expected_replay_sample.states
    )
    np.testing.assert_array_equal(
        actual_replay_sample.actions, expected_replay_sample.actions
    )
    np.testing.assert_array_equal(
        actual_replay_sample.rewards, expected_replay_sample.rewards
    )
    np.testing.assert_array_equal(
        actual_replay_sample.next_states, expected_replay_sample.next_states
    )
    np.testing.assert_array_equal(
        actual_replay_sample.dones, expected_replay_sample.dones
    )
    assert [loaded.select_action(exploration_state) for _ in range(6)] == expected_actions
    np.testing.assert_array_equal(loaded._np_random.random_sample(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch)

    source_loss = source.train_step()
    loaded_loss = loaded.train_step()
    assert loaded_loss == pytest.approx(source_loss, rel=0.0, abs=0.0)
    assert loaded.epsilon == source.epsilon
    assert loaded.update_count == source.update_count
    for source_parameter, loaded_parameter in zip(
        source.online.parameters(), loaded.online.parameters()
    ):
        torch.testing.assert_close(loaded_parameter, source_parameter, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            CheckpointMetadata("old-state", GA_ACTION_VERSION, 9, (), 0),
            "state schema",
        ),
        (
            CheckpointMetadata(DQN_STATE_VERSION, "old-actions", 9, (), 0),
            "action schema",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_schema(
    tmp_path: Path, metadata: CheckpointMetadata, message: str
):
    path = tmp_path / "agent.pt"
    save_checkpoint(path, DQNAgent(DQNConfig(seed=9)), metadata)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_size", 11),
        ("action_size", 4),
        ("lr", 0.002),
        ("gamma", 0.8),
        ("batch_size", 8),
        ("replay_capacity", 99),
        ("target_sync_interval", 3),
        ("epsilon_start", 0.7),
        ("epsilon_end", 0.2),
        ("epsilon_decay", 0.9),
        ("seed", 13),
    ],
)
def test_checkpoint_rejects_training_config_mismatch(
    tmp_path: Path, field: str, value: object
):
    saved_config = DQNConfig(seed=9)
    path = tmp_path / "agent.pt"
    save_checkpoint(
        path,
        DQNAgent(saved_config),
        CheckpointMetadata(DQN_STATE_VERSION, GA_ACTION_VERSION, 9, (), 0),
    )

    with pytest.raises(ValueError, match=rf"DQN config.*{field}.*checkpoint.*caller"):
        load_checkpoint(path, replace(saved_config, **{field: value}))


def test_failed_checkpoint_save_preserves_existing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "agent.pt"
    path.write_bytes(b"previous-good-checkpoint")

    def fail_save(_payload, temporary_path):
        Path(temporary_path).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)

    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(
            path,
            DQNAgent(DQNConfig()),
            CheckpointMetadata(
                DQN_STATE_VERSION, GA_ACTION_VERSION, 42, ("FT06",), 1
            ),
        )

    assert path.read_bytes() == b"previous-good-checkpoint"
    assert not path.with_suffix(".pt.tmp").exists()

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
    config = DQNConfig(seed=9, batch_size=2, target_sync_interval=1)
    source = DQNAgent(config)
    for index in range(2):
        state = np.full(10, index, dtype=np.float32)
        source.remember(state, index, 0.5, state + 1, index == 1)
    assert source.train_step() is not None
    path = tmp_path / "agent.pt"
    metadata = CheckpointMetadata(
        DQN_STATE_VERSION, GA_ACTION_VERSION, 9, ("FT06",), 2
    )

    save_checkpoint(path, source, metadata)
    loaded, loaded_metadata = load_checkpoint(path, config)

    state = np.linspace(0, 1, 10, dtype=np.float32)
    np.testing.assert_allclose(loaded.q_values(state), source.q_values(state))
    assert loaded_metadata == metadata
    assert loaded.epsilon == source.epsilon
    assert loaded.update_count == source.update_count
    assert loaded.optimizer.state_dict()["state"]


@pytest.mark.parametrize(
    ("metadata", "config", "message"),
    [
        (
            CheckpointMetadata("old-state", GA_ACTION_VERSION, 9, (), 0),
            DQNConfig(),
            "state schema",
        ),
        (
            CheckpointMetadata(DQN_STATE_VERSION, "old-actions", 9, (), 0),
            DQNConfig(),
            "action schema",
        ),
        (
            CheckpointMetadata(DQN_STATE_VERSION, GA_ACTION_VERSION, 9, (), 0),
            DQNConfig(state_size=11),
            "state size",
        ),
        (
            CheckpointMetadata(DQN_STATE_VERSION, GA_ACTION_VERSION, 9, (), 0),
            DQNConfig(action_size=4),
            "action size",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_schema_or_network_size(
    tmp_path: Path, metadata: CheckpointMetadata, config: DQNConfig, message: str
):
    path = tmp_path / "agent.pt"
    save_checkpoint(path, DQNAgent(DQNConfig(seed=9)), metadata)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, config)


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

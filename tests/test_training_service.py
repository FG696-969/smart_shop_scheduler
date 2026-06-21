from pathlib import Path

import pytest

from rl.checkpoint import DQN_STATE_VERSION, GA_ACTION_VERSION, load_checkpoint
from rl.dqn_agent import DQNConfig
from services.training import TrainingConfig, train_dqn


def test_training_service_writes_checkpoint_and_complete_report(tmp_path: Path):
    checkpoint = tmp_path / "dqn.pt"

    report = train_dqn(
        TrainingConfig(
            datasets=("Custom 5x4",),
            episodes=2,
            population_size=10,
            generations=5,
            base_seed=5,
            checkpoint_path=checkpoint,
        )
    )

    assert checkpoint.exists()
    assert len(report.episode_rewards) == 2
    assert len(report.final_makespans) == 2
    assert len(report.runtimes) == 2
    assert report.losses
    assert sum(report.action_counts.values()) == 10
    assert report.checkpoint_path == checkpoint
    _agent, metadata = load_checkpoint(checkpoint, DQNConfig(seed=5))
    assert metadata.state_version == DQN_STATE_VERSION
    assert metadata.action_version == GA_ACTION_VERSION
    assert metadata.datasets == ("Custom 5x4",)
    assert metadata.episodes == 2


def test_training_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "dqn.pt"
    checkpoint.write_bytes(b"last-known-good")

    def fail_episode(*_args, **_kwargs):
        raise RuntimeError("episode failed")

    monkeypatch.setattr("services.training.run_dqn_ga", fail_episode)

    with pytest.raises(RuntimeError, match="episode failed"):
        train_dqn(
            TrainingConfig(
                datasets=("Custom 5x4",),
                episodes=1,
                population_size=10,
                generations=4,
                base_seed=5,
                checkpoint_path=checkpoint,
            )
        )

    assert checkpoint.read_bytes() == b"last-known-good"


def test_training_runs_each_episode_for_every_requested_dataset(tmp_path: Path):
    checkpoint = tmp_path / "multi.pt"

    report = train_dqn(
        TrainingConfig(
            datasets=("Custom 5x4", "FT06"),
            episodes=1,
            population_size=10,
            generations=4,
            base_seed=11,
            checkpoint_path=checkpoint,
        )
    )

    assert len(report.episode_rewards) == 2
    assert len(report.final_makespans) == 2
    _agent, metadata = load_checkpoint(checkpoint, DQNConfig(seed=11))
    assert metadata.datasets == ("Custom 5x4", "FT06")
    assert metadata.episodes == 2

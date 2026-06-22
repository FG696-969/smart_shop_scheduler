from pathlib import Path

import pytest

from rl.checkpoint import (
    GA_ACTION_VERSION,
    CheckpointMetadata,
    save_checkpoint,
)
from rl.dqn_agent import DQNAgent, DQNConfig
from services.scheduling import ScheduleRequest, run_schedule


def test_baseline_schedule_does_not_need_or_load_a_model(tmp_path: Path):
    request = ScheduleRequest(
        dataset="Custom 5x4",
        algorithm="FIFO",
        seed=7,
        fast_mode=True,
        checkpoint_path=tmp_path / "missing.pt",
    )

    result = run_schedule(request)

    assert result.name == "FIFO"
    assert result.makespan > 0


def test_dqn_schedule_reports_actionable_missing_checkpoint(tmp_path: Path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="checkpoint.*not found.*Train"):
        run_schedule(
            ScheduleRequest(
                dataset="Custom 5x4",
                algorithm="DQN-AOL-GA",
                seed=7,
                fast_mode=True,
                checkpoint_path=missing,
            )
        )


def test_dqn_schedule_reports_actionable_incompatible_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "old.pt"
    save_checkpoint(
        checkpoint,
        DQNAgent(DQNConfig(seed=7)),
        CheckpointMetadata("old-state", GA_ACTION_VERSION, 7, (), 0),
    )

    with pytest.raises(ValueError, match="Incompatible.*Retrain"):
        run_schedule(
            ScheduleRequest(
                dataset="Custom 5x4",
                algorithm="DQN-AOL-GA",
                seed=7,
                fast_mode=True,
                checkpoint_path=checkpoint,
            )
        )


def test_dqn_schedule_loads_compatible_checkpoint(tmp_path: Path):
    from rl.checkpoint import DQN_STATE_VERSION

    checkpoint = tmp_path / "agent.pt"
    save_checkpoint(
        checkpoint,
        DQNAgent(DQNConfig(seed=7)),
        CheckpointMetadata(
            DQN_STATE_VERSION, GA_ACTION_VERSION, 7, ("Custom 5x4",), 1
        ),
    )

    result = run_schedule(
        ScheduleRequest(
            dataset="Custom 5x4",
            algorithm="DQN-AOL-GA",
            seed=7,
            fast_mode=True,
            checkpoint_path=checkpoint,
        )
    )

    assert result.name == "DQN-AOL-GA"
    assert result.chromosome


def test_dqn_schedule_uses_saved_training_config(tmp_path: Path):
    from rl.checkpoint import DQN_STATE_VERSION

    checkpoint = tmp_path / "trained-agent.pt"
    save_checkpoint(
        checkpoint,
        DQNAgent(
            DQNConfig(
                seed=7,
                batch_size=4,
                replay_capacity=50,
                target_sync_interval=12,
            )
        ),
        CheckpointMetadata(
            DQN_STATE_VERSION, GA_ACTION_VERSION, 7, ("Custom 5x4",), 1
        ),
    )

    result = run_schedule(
        ScheduleRequest(
            dataset="Custom 5x4",
            algorithm="DQN-AOL-GA",
            seed=99,
            fast_mode=True,
            checkpoint_path=checkpoint,
        )
    )

    assert result.name == "DQN-AOL-GA"

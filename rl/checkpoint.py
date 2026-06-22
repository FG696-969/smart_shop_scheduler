from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from .dqn_agent import DQNAgent, DQNConfig

DQN_STATE_VERSION = "dqn-state-v2"
GA_ACTION_VERSION = "ga-actions-v1"
STATE_VERSION = DQN_STATE_VERSION
ACTION_VERSION = GA_ACTION_VERSION


@dataclass(frozen=True)
class CheckpointMetadata:
    state_version: str
    action_version: str
    seed: int
    datasets: tuple[str, ...]
    episodes: int


def save_checkpoint(
    path: str | os.PathLike[str],
    agent: DQNAgent,
    metadata: CheckpointMetadata,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "agent": agent.state_dict(),
        "config": asdict(agent.config),
        "metadata": asdict(metadata),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: str | os.PathLike[str],
    config: DQNConfig | None = None,
    expected_state_version: str = DQN_STATE_VERSION,
    expected_action_version: str = GA_ACTION_VERSION,
) -> tuple[DQNAgent, CheckpointMetadata]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = CheckpointMetadata(
        state_version=str(payload["metadata"]["state_version"]),
        action_version=str(payload["metadata"]["action_version"]),
        seed=int(payload["metadata"]["seed"]),
        datasets=tuple(payload["metadata"]["datasets"]),
        episodes=int(payload["metadata"]["episodes"]),
    )
    if metadata.state_version != expected_state_version:
        raise ValueError(
            "Incompatible state schema: "
            f"expected {expected_state_version}, got {metadata.state_version}"
        )
    if metadata.action_version != expected_action_version:
        raise ValueError(
            "Incompatible action schema: "
            f"expected {expected_action_version}, got {metadata.action_version}"
        )

    saved_config = DQNConfig(**payload["config"])
    if config is not None:
        training_fields = [
            field.name for field in fields(DQNConfig) if field.name != "device"
        ]
        mismatches = [
            field_name
            for field_name in training_fields
            if getattr(saved_config, field_name) != getattr(config, field_name)
        ]
    else:
        mismatches = []
    if mismatches:
        mismatch_details = ", ".join(
            f"{field_name} (checkpoint={getattr(saved_config, field_name)!r}, "
            f"caller={getattr(config, field_name)!r})"
            for field_name in mismatches
        )
        raise ValueError(
            "DQN config mismatch for training-relevant field(s): "
            f"{mismatch_details}. Omit config to use the checkpoint configuration."
        )

    resolved_config = saved_config if config is None else config
    agent = DQNAgent(resolved_config)
    agent.load_state_dict(payload["agent"])
    return agent, metadata

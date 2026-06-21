from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .dqn_agent import DQNAgent, DQNConfig

DQN_STATE_VERSION = "dqn-state-v1"
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
        "online": agent.online.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "update_count": agent.update_count,
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
    config: DQNConfig,
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

    saved_config = payload["config"]
    if int(saved_config["state_size"]) != config.state_size:
        raise ValueError(
            "Incompatible state size: "
            f"checkpoint has {saved_config['state_size']}, config has {config.state_size}"
        )
    if int(saved_config["action_size"]) != config.action_size:
        raise ValueError(
            "Incompatible action size: "
            f"checkpoint has {saved_config['action_size']}, config has {config.action_size}"
        )

    agent = DQNAgent(config)
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload["target"])
    agent.optimizer.load_state_dict(payload["optimizer"])
    agent.epsilon = float(payload["epsilon"])
    agent.update_count = int(payload["update_count"])
    return agent, metadata

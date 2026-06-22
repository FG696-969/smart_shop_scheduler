from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from algorithms.dqn_ga import run_dqn_ga
from data_loader import load_dataset
from rl.checkpoint import (
    DQN_STATE_VERSION,
    GA_ACTION_VERSION,
    CheckpointMetadata,
    save_checkpoint,
)
from rl.dqn_agent import DQNAgent, DQNConfig


@dataclass(frozen=True)
class TrainingConfig:
    datasets: tuple[str, ...]
    episodes: int
    population_size: int
    generations: int
    base_seed: int
    checkpoint_path: Path | str


@dataclass
class TrainingReport:
    episode_rewards: list[float] = field(default_factory=list)
    final_makespans: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    runtimes: list[float] = field(default_factory=list)
    checkpoint_path: Path = Path()


def _validate_config(config: TrainingConfig) -> None:
    if not config.datasets:
        raise ValueError("Training requires at least one dataset")
    if config.episodes <= 0:
        raise ValueError("Training episodes must be positive")
    if config.population_size <= 0 or config.generations <= 0:
        raise ValueError("Population size and generations must be positive")


def train_dqn(config: TrainingConfig) -> TrainingReport:
    _validate_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    batch_size = min(32, max(1, config.generations))
    agent = DQNAgent(
        DQNConfig(
            seed=config.base_seed,
            batch_size=batch_size,
            target_sync_interval=max(1, config.generations * 2),
        )
    )
    episode_rewards: list[float] = []
    final_makespans: list[int] = []
    losses: list[float] = []
    runtimes: list[float] = []
    action_counts: Counter[str] = Counter()

    for episode_index in range(config.episodes):
        dataset = config.datasets[episode_index % len(config.datasets)]
        jobs, metadata = load_dataset(dataset)
        result = run_dqn_ga(
            jobs,
            int(metadata["machines"]),
            agent=agent,
            population_size=config.population_size,
            generations=config.generations,
            random_seed=config.base_seed + episode_index,
            training=True,
        )
        episode_rewards.append(float(sum(result.reward_history)))
        final_makespans.append(int(result.makespan))
        losses.extend(result.loss_history)
        action_counts.update(result.action_history)
        runtimes.append(float(result.runtime))

    metadata = CheckpointMetadata(
        state_version=DQN_STATE_VERSION,
        action_version=GA_ACTION_VERSION,
        seed=config.base_seed,
        datasets=tuple(config.datasets),
        episodes=len(episode_rewards),
    )
    save_checkpoint(checkpoint_path, agent, metadata)
    return TrainingReport(
        episode_rewards=episode_rewards,
        final_makespans=final_makespans,
        losses=losses,
        action_counts=dict(action_counts),
        runtimes=runtimes,
        checkpoint_path=checkpoint_path,
    )

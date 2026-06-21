from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from services.training import TrainingConfig, train_dqn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DQN-AOL-GA controller")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/dqn_aol_ga.pt")
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    population_size = 12 if args.fast else 80
    generations = 6 if args.fast else 140
    try:
        report = train_dqn(
            TrainingConfig(
                datasets=tuple(args.datasets),
                episodes=args.episodes,
                population_size=population_size,
                generations=generations,
                base_seed=args.base_seed,
                checkpoint_path=args.checkpoint,
            )
        )
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    for index, (reward, makespan, runtime) in enumerate(
        zip(
            report.episode_rewards,
            report.final_makespans,
            report.runtimes,
        ),
        start=1,
    ):
        print(
            f"Episode {index}/{len(report.episode_rewards)}: "
            f"reward={reward:.4f} makespan={makespan} runtime={runtime:.3f}s"
        )
    print(
        f"Checkpoint saved: {report.checkpoint_path} "
        f"(episodes={len(report.episode_rewards)}, losses={len(report.losses)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

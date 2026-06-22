from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from algorithms import AlgorithmResult, run_algorithm
from data_loader import load_dataset
from rl.checkpoint import load_checkpoint


@dataclass(frozen=True)
class ScheduleRequest:
    dataset: str
    algorithm: str
    seed: int = 42
    fast_mode: bool = True
    checkpoint_path: Path | str | None = None
    inference_exploration: bool = False


def run_schedule(request: ScheduleRequest) -> AlgorithmResult:
    jobs, metadata = load_dataset(request.dataset)
    num_machines = int(metadata["machines"])
    if request.algorithm != "DQN-AOL-GA":
        return run_algorithm(
            request.algorithm,
            jobs,
            num_machines,
            random_seed=request.seed,
            fast_mode=request.fast_mode,
        )

    if request.checkpoint_path is None:
        raise FileNotFoundError(
            "DQN checkpoint not specified. Train a model and provide checkpoint_path."
        )
    checkpoint_path = Path(request.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DQN checkpoint not found at {checkpoint_path}. "
            "Train a model before running DQN-AOL-GA."
        )
    try:
        agent, _metadata = load_checkpoint(checkpoint_path)
    except Exception as exc:
        raise ValueError(
            f"Incompatible DQN checkpoint at {checkpoint_path}: {exc}. "
            "Retrain the model with the current scheduler version."
        ) from exc
    return run_algorithm(
        request.algorithm,
        jobs,
        num_machines,
        random_seed=request.seed,
        fast_mode=request.fast_mode,
        agent=agent,
        training=False,
        inference_exploration=request.inference_exploration,
    )

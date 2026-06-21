from __future__ import annotations

from scheduler_core import Jobs

from .baselines import run_fifo, run_ga
from .common import AlgorithmResult
from .tabular import run_slga


def run_algorithm(
    name: str,
    jobs: Jobs,
    num_machines: int,
    random_seed: int = 42,
    fast_mode: bool = True,
    **decode_kwargs,
) -> AlgorithmResult:
    if name == "FIFO":
        return run_fifo(
            jobs, num_machines, random_seed=random_seed, **decode_kwargs
        )
    if name == "GA":
        return run_ga(
            jobs,
            num_machines,
            population_size=70 if fast_mode else 120,
            generations=100 if fast_mode else 250,
            random_seed=random_seed,
            **decode_kwargs,
        )
    if name == "SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=70 if fast_mode else 120,
            generations=110 if fast_mode else 250,
            random_seed=random_seed,
            **decode_kwargs,
        )
    if name == "CP-AOL-SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=70 if fast_mode else 120,
            generations=120 if fast_mode else 260,
            random_seed=random_seed,
            cp_aol=True,
            **decode_kwargs,
        )
    raise ValueError(f"Unknown algorithm: {name}")

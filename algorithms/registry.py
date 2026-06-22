from __future__ import annotations

from scheduler_core import Jobs
from rl.dqn_agent import DQNAgent

from .baselines import run_fifo, run_ga
from .common import AlgorithmResult
from .dqn_ga import run_dqn_ga
from .tabular import run_slga


def run_algorithm(
    name: str,
    jobs: Jobs,
    num_machines: int,
    random_seed: int = 42,
    fast_mode: bool = True,
    agent: DQNAgent | None = None,
    training: bool = False,
    **decode_kwargs,
) -> AlgorithmResult:
    population_size = 60 if fast_mode else 120
    generations = 100 if fast_mode else 250
    if name == "FIFO":
        return run_fifo(
            jobs, num_machines, random_seed=random_seed, **decode_kwargs
        )
    if name == "GA":
        return run_ga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=random_seed,
            **decode_kwargs,
        )
    if name == "SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=random_seed,
            **decode_kwargs,
        )
    if name == "CP-AOL-SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=random_seed,
            cp_aol=True,
            **decode_kwargs,
        )
    if name == "DQN-AOL-GA":
        if not isinstance(agent, DQNAgent):
            raise ValueError("DQN-AOL-GA requires a DQNAgent")
        return run_dqn_ga(
            jobs,
            num_machines,
            agent=agent,
            population_size=population_size,
            generations=generations,
            random_seed=random_seed,
            training=training,
            **decode_kwargs,
        )
    raise ValueError(f"Unknown algorithm: {name}")

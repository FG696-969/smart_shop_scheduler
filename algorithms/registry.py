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
    if name == "DQN-AOL-GA":
        if not isinstance(agent, DQNAgent):
            raise ValueError("DQN-AOL-GA requires a DQNAgent")
        dqn_kwargs = {}
        if fast_mode:
            dqn_kwargs = {"population_size": 20, "generations": 12}
        return run_dqn_ga(
            jobs,
            num_machines,
            agent=agent,
            random_seed=random_seed,
            training=training,
            **dqn_kwargs,
            **decode_kwargs,
        )
    raise ValueError(f"Unknown algorithm: {name}")

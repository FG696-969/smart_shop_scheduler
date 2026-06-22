import pytest

import algorithms.registry as registry
from algorithms import run_algorithm
from data_loader import load_dataset
from rl.dqn_agent import DQNAgent, DQNConfig
from scheduler_core import validate_chromosome


@pytest.mark.parametrize("name", ["FIFO", "GA", "SLGA", "CP-AOL-SLGA"])
def test_seeded_ft06_baselines_remain_feasible(name: str):
    jobs, metadata = load_dataset("FT06")

    result = run_algorithm(
        name,
        jobs,
        int(metadata["machines"]),
        random_seed=42,
        fast_mode=True,
    )

    validate_chromosome(result.chromosome, jobs)
    assert result.name == name
    assert result.makespan == max(record["end"] for record in result.schedule)
    assert result.makespan > 0


def test_fast_mode_uses_the_same_search_budget_for_all_ga_variants(monkeypatch):
    calls: list[tuple[str, int, int]] = []

    def fake_ga(_jobs, _machines, *, population_size, generations, **_kwargs):
        calls.append(("GA", population_size, generations))

    def fake_slga(
        _jobs,
        _machines,
        *,
        population_size,
        generations,
        cp_aol=False,
        **_kwargs,
    ):
        calls.append(
            ("CP-AOL-SLGA" if cp_aol else "SLGA", population_size, generations)
        )

    def fake_dqn(
        _jobs,
        _machines,
        *,
        population_size,
        generations,
        **_kwargs,
    ):
        calls.append(("DQN-AOL-GA", population_size, generations))

    monkeypatch.setattr(registry, "run_ga", fake_ga)
    monkeypatch.setattr(registry, "run_slga", fake_slga)
    monkeypatch.setattr(registry, "run_dqn_ga", fake_dqn)
    jobs = [[(0, 1)]]
    agent = DQNAgent(DQNConfig(seed=7))

    for name in ("GA", "SLGA", "CP-AOL-SLGA", "DQN-AOL-GA"):
        registry.run_algorithm(name, jobs, 1, fast_mode=True, agent=agent)

    assert calls == [
        ("GA", 60, 100),
        ("SLGA", 60, 100),
        ("CP-AOL-SLGA", 60, 100),
        ("DQN-AOL-GA", 60, 100),
    ]

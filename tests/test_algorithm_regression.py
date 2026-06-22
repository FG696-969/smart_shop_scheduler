import pytest

import algorithms.registry as registry
from algorithms import AlgorithmResult, run_algorithm
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


def test_fast_mode_uses_the_same_total_search_budget_for_all_ga_variants(monkeypatch):
    calls: list[tuple[str, int, int]] = []

    def fake_ga(_jobs, _machines, *, population_size, generations, **_kwargs):
        calls.append(("GA", population_size, generations))
        return AlgorithmResult("GA", [], [], 10)

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
        return AlgorithmResult("CP-AOL-SLGA" if cp_aol else "SLGA", [], [], 10)

    def fake_dqn(
        _jobs,
        _machines,
        *,
        population_size,
        generations,
        **_kwargs,
    ):
        calls.append(("DQN-AOL-GA", population_size, generations))
        return AlgorithmResult("DQN-AOL-GA", [], [], 10)

    monkeypatch.setattr(registry, "run_ga", fake_ga)
    monkeypatch.setattr(registry, "run_slga", fake_slga)
    monkeypatch.setattr(registry, "run_dqn_ga", fake_dqn)
    jobs = [[(0, 1)]]
    agent = DQNAgent(DQNConfig(seed=7))

    for name in ("GA", "SLGA", "CP-AOL-SLGA", "DQN-AOL-GA"):
        registry.run_algorithm(name, jobs, 1, fast_mode=True, agent=agent)

    total_budgets: dict[str, int] = {}
    for name, population_size, generations in calls:
        total_budgets[name] = total_budgets.get(name, 0) + population_size * generations

    assert total_budgets == {
        "GA": 6000,
        "SLGA": 6000,
        "CP-AOL-SLGA": 6000,
        "DQN-AOL-GA": 6000,
    }


def test_dqn_inference_uses_equal_budget_restarts_and_returns_the_best(monkeypatch):
    calls: list[tuple[int, int, int]] = []
    makespans = {42: 61, 43: 55, 44: 58}

    def fake_dqn(
        _jobs,
        _machines,
        *,
        population_size,
        generations,
        random_seed,
        **_kwargs,
    ):
        calls.append((population_size, generations, random_seed))
        return AlgorithmResult(
            "DQN-AOL-GA",
            [],
            [],
            makespans[random_seed],
            runtime=0.25,
        )

    monkeypatch.setattr(registry, "run_dqn_ga", fake_dqn)
    agent = DQNAgent(DQNConfig(seed=7))

    result = registry.run_algorithm(
        "DQN-AOL-GA", [[(0, 1)]], 1, random_seed=42, fast_mode=True, agent=agent
    )

    assert calls == [(20, 100, 42), (20, 100, 43), (20, 100, 44)]
    assert result.makespan == 55
    assert result.runtime == pytest.approx(0.75)

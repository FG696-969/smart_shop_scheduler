import pytest

from algorithms import run_algorithm
from data_loader import load_dataset
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

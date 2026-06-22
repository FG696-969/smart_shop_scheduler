from experiments.benchmark import summarize_results


def test_summarize_results_groups_runs_and_computes_bks_gap():
    rows = [
        {"dataset": "FT06", "algorithm": "GA", "seed": 1, "makespan": 60, "runtime": 1.0},
        {"dataset": "FT06", "algorithm": "GA", "seed": 2, "makespan": 56, "runtime": 3.0},
        {"dataset": "FT06", "algorithm": "DQN-AOL-GA", "seed": 1, "makespan": 55, "runtime": 2.0},
    ]

    summary = summarize_results(rows, {"FT06": 55})

    ga = next(row for row in summary if row["algorithm"] == "GA")
    assert ga["runs"] == 2
    assert ga["best_makespan"] == 56
    assert ga["mean_makespan"] == 58.0
    assert ga["std_makespan"] == 2.0
    assert ga["mean_runtime"] == 2.0
    assert ga["gap_to_bks_pct"] == 5.45


def test_summarize_results_uses_none_when_bks_is_unavailable():
    rows = [
        {"dataset": "Custom 5x4", "algorithm": "FIFO", "seed": 1, "makespan": 30, "runtime": 0.1}
    ]

    summary = summarize_results(rows, {"Custom 5x4": None})

    assert summary[0]["gap_to_bks_pct"] is None

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from algorithms.baselines import run_fifo, run_ga
from algorithms.dqn_ga import run_dqn_ga
from algorithms.tabular import run_slga
from data_loader import load_dataset
from rl.checkpoint import load_checkpoint
from rl.dqn_agent import DQNAgent

ALGORITHMS = ("FIFO", "GA", "SLGA", "CP-AOL-SLGA", "DQN-AOL-GA")


def summarize_results(
    rows: Sequence[dict[str, object]],
    optimums: dict[str, int | None],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["algorithm"]))].append(row)

    summary: list[dict[str, object]] = []
    for (dataset, algorithm), group in sorted(grouped.items()):
        makespans = [int(row["makespan"]) for row in group]
        runtimes = [float(row["runtime"]) for row in group]
        mean_makespan = statistics.fmean(makespans)
        optimum = optimums.get(dataset)
        gap = (
            round((mean_makespan - optimum) / optimum * 100, 2)
            if optimum
            else None
        )
        summary.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "runs": len(group),
                "best_makespan": min(makespans),
                "mean_makespan": round(mean_makespan, 2),
                "std_makespan": round(statistics.pstdev(makespans), 2),
                "mean_runtime": round(statistics.fmean(runtimes), 4),
                "gap_to_bks_pct": gap,
            }
        )
    return summary


def _run_algorithm(
    name: str,
    jobs,
    num_machines: int,
    seed: int,
    population_size: int,
    generations: int,
    agent: DQNAgent,
):
    if name == "FIFO":
        return run_fifo(jobs, num_machines, random_seed=seed)
    if name == "GA":
        return run_ga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=seed,
        )
    if name == "SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=seed,
        )
    if name == "CP-AOL-SLGA":
        return run_slga(
            jobs,
            num_machines,
            population_size=population_size,
            generations=generations,
            random_seed=seed,
            cp_aol=True,
        )
    if name == "DQN-AOL-GA":
        restart_count = 3
        restart_population = population_size // restart_count
        results = [
            run_dqn_ga(
                jobs,
                num_machines,
                agent=agent,
                population_size=restart_population,
                generations=generations,
                random_seed=seed + restart,
                training=False,
            )
            for restart in range(restart_count)
        ]
        best_result = min(results, key=lambda result: result.makespan)
        best_result.runtime = sum(result.runtime for result in results)
        return best_result
    raise ValueError(f"Unknown algorithm: {name}")


def run_static_benchmark(
    checkpoint_path: Path | str,
    datasets: Sequence[str],
    seeds: Sequence[int],
    population_size: int,
    generations: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    agent, _metadata = load_checkpoint(checkpoint_path)
    rows: list[dict[str, object]] = []
    optimums: dict[str, int | None] = {}
    for dataset in datasets:
        jobs, metadata = load_dataset(dataset)
        num_machines = int(metadata["machines"])
        raw_optimum = metadata.get("optimum")
        optimums[dataset] = int(raw_optimum) if raw_optimum is not None else None
        for algorithm in ALGORITHMS:
            for seed in seeds:
                result = _run_algorithm(
                    algorithm,
                    jobs,
                    num_machines,
                    int(seed),
                    population_size,
                    generations,
                    agent,
                )
                row = {
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "makespan": int(result.makespan),
                    "runtime": round(float(result.runtime), 6),
                }
                rows.append(row)
                print(
                    f"{dataset:>4} | {algorithm:<12} | seed={seed:<3} "
                    f"| Cmax={result.makespan:<4} | {result.runtime:.3f}s"
                )
    return rows, summarize_results(rows, optimums)


def write_csv(path: Path | str, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot write an empty benchmark result")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Scheduler 2.0 benchmarks")
    parser.add_argument("--checkpoint", default="models/dqn_aol_ga.pt")
    parser.add_argument("--datasets", nargs="+", default=["FT06", "LA01", "LA02"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/experiments")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw, summary = run_static_benchmark(
        args.checkpoint,
        args.datasets,
        args.seeds,
        args.population,
        args.generations,
    )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "static_raw.csv", raw)
    write_csv(output_dir / "static_summary.csv", summary)
    print(f"Saved benchmark results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

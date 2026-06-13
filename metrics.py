"""Metrics for schedule monitoring and dynamic rescheduling."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

ScheduleRecord = Dict[str, int]


def calculate_metrics(schedule: Sequence[ScheduleRecord], num_machines: int) -> Dict[str, object]:
    makespan = max([r["end"] for r in schedule], default=0)
    machine_processing = [0] * num_machines
    for record in schedule:
        machine_processing[record["machine_id"]] += record["processing_time"]
    utilization = [p / makespan if makespan else 0 for p in machine_processing]
    avg_utilization = sum(utilization) / num_machines if num_machines else 0
    total_processing = sum(machine_processing)
    total_idle = num_machines * makespan - total_processing
    return {
        "makespan": int(makespan),
        "machine_processing": machine_processing,
        "machine_utilization": utilization,
        "average_utilization": avg_utilization,
        "total_idle_time": int(total_idle),
        "finished_operations": len(schedule),
    }


def completed_operations(schedule: Sequence[ScheduleRecord], current_time: int) -> int:
    return sum(1 for r in schedule if r["end"] <= current_time)


def schedule_deviation(original: Sequence[ScheduleRecord], new: Sequence[ScheduleRecord]) -> int:
    """Sum of start-time changes for operations appearing in both schedules."""
    original_start = {(r["job_id"], r["operation_id"]): r["start"] for r in original}
    deviation = 0
    for record in new:
        key = (record["job_id"], record["operation_id"])
        if key in original_start:
            deviation += abs(record["start"] - original_start[key])
    return int(deviation)


def total_tardiness(schedule: Sequence[ScheduleRecord], due_dates: Dict[int, int] | None = None) -> int:
    if not due_dates:
        return 0
    job_completion: Dict[int, int] = {}
    for r in schedule:
        job_completion[r["job_id"]] = max(job_completion.get(r["job_id"], 0), r["end"])
    return int(sum(max(0, completion - due_dates.get(job_id, completion)) for job_id, completion in job_completion.items()))


def metrics_for_display(metrics: Dict[str, object]) -> Dict[str, str]:
    return {
        "Cmax": str(metrics.get("makespan", 0)),
        "Average Utilization": f"{float(metrics.get('average_utilization', 0)) * 100:.2f}%",
        "Total Idle Time": str(metrics.get("total_idle_time", 0)),
        "Finished Operations": str(metrics.get("finished_operations", 0)),
    }

"""Core JSSP decoding logic.

The chromosome is a repeated job-id sequence. The k-th occurrence of a job ID
means the k-th operation of that job. Decoding enforces:
1. Job precedence constraints.
2. Machine capacity constraints.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

Jobs = List[List[Tuple[int, int]]]
ScheduleRecord = Dict[str, int]


def job_label(job_id: int) -> str:
    return f"J{job_id + 1}"


def machine_label(machine_id: int) -> str:
    return f"M{machine_id + 1}"


def op_label(op_id: int) -> str:
    return f"O{op_id + 1}"


def num_machines_from_jobs(jobs: Jobs) -> int:
    return max(machine for job in jobs for machine, _ in job) + 1


def validate_chromosome(chromosome: Sequence[int], jobs: Jobs) -> None:
    expected = {job_id: len(job) for job_id, job in enumerate(jobs)}
    counts = Counter(chromosome)
    if len(chromosome) != sum(expected.values()):
        raise ValueError("Invalid chromosome length.")
    if dict(counts) != expected:
        raise ValueError("Invalid chromosome counts; every job must appear once per operation.")


def build_fifo_chromosome(jobs: Jobs) -> List[int]:
    chromosome: List[int] = []
    max_ops = max(len(job) for job in jobs)
    for op_round in range(max_ops):
        for job_id, job in enumerate(jobs):
            if op_round < len(job):
                chromosome.append(job_id)
    return chromosome


def decode_chromosome(
    chromosome: Sequence[int],
    jobs: Jobs,
    num_machines: Optional[int] = None,
    start_after: int = 0,
    initial_job_ready: Optional[List[int]] = None,
    initial_machine_ready: Optional[List[int]] = None,
    initial_next_operation: Optional[List[int]] = None,
    machine_breakdowns: Optional[Dict[int, Tuple[int, int]]] = None,
    emergency_job_start: Optional[int] = None,
    emergency_job_id: Optional[int] = None,
) -> Tuple[List[ScheduleRecord], int]:
    """Decode a full or partial chromosome into a feasible schedule.

    If initial_* arrays are provided, the function decodes only remaining
    operations after a disturbance event. machine_breakdowns stores unavailable
    intervals as {machine_id: (start, end)}.
    """
    if num_machines is None:
        num_machines = num_machines_from_jobs(jobs)

    if initial_next_operation is None:
        validate_chromosome(chromosome, jobs)
        next_operation = [0] * len(jobs)
    else:
        next_operation = list(initial_next_operation)

    job_ready = list(initial_job_ready) if initial_job_ready is not None else [0] * len(jobs)
    machine_ready = list(initial_machine_ready) if initial_machine_ready is not None else [0] * num_machines
    machine_breakdowns = machine_breakdowns or {}
    schedule: List[ScheduleRecord] = []

    for job_id in chromosome:
        op_id = next_operation[job_id]
        if op_id >= len(jobs[job_id]):
            continue
        machine_id, processing_time = jobs[job_id][op_id]
        release_time = start_after
        if emergency_job_id is not None and job_id == emergency_job_id and emergency_job_start is not None:
            release_time = max(release_time, emergency_job_start)

        start = max(job_ready[job_id], machine_ready[machine_id], release_time)
        end = start + processing_time

        # Simplified machine breakdown handling: operations cannot overlap the
        # down interval; if overlap exists, delay to breakdown end.
        if machine_id in machine_breakdowns:
            down_start, down_end = machine_breakdowns[machine_id]
            if start < down_end and end > down_start:
                start = max(start, down_end)
                end = start + processing_time

        schedule.append(
            {
                "job_id": job_id,
                "operation_id": op_id,
                "machine_id": machine_id,
                "start": int(start),
                "end": int(end),
                "processing_time": int(processing_time),
            }
        )
        next_operation[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end

    makespan = max([r["end"] for r in schedule], default=start_after)
    return schedule, makespan


def schedule_to_operation_key(record: ScheduleRecord) -> Tuple[int, int]:
    return record["job_id"], record["operation_id"]


def sort_schedule(schedule: Sequence[ScheduleRecord]) -> List[ScheduleRecord]:
    return sorted(schedule, key=lambda r: (r["machine_id"], r["start"], r["job_id"], r["operation_id"]))

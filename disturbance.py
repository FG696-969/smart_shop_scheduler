"""Disturbance and dynamic rescheduling logic.

This module implements a practical course-project version of dynamic
rescheduling:
- operations completed before the event time are frozen;
- unfinished operations are rescheduled with updated machine/job ready times;
- optional machine breakdown blocks a machine during a time interval;
- optional emergency job is inserted at its release time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from algorithms import AlgorithmResult, run_algorithm
from rl.dqn_agent import DQNAgent
from scheduler_core import Jobs, ScheduleRecord, decode_chromosome


def parse_emergency_route(route_text: str) -> List[Tuple[int, int]]:
    """Parse route like 'M1:4, M3:5, M2:3' into [(0,4),(2,5),(1,3)]."""
    route: List[Tuple[int, int]] = []
    if not route_text.strip():
        return route
    for part in route_text.split(","):
        item = part.strip().upper().replace(" ", "")
        if not item:
            continue
        if not item.startswith("M") or ":" not in item:
            raise ValueError("Emergency route must use format M1:4, M3:5, M2:3")
        machine_str, time_str = item.split(":", 1)
        machine_id = int(machine_str[1:]) - 1
        processing_time = int(time_str)
        if machine_id < 0 or processing_time <= 0:
            raise ValueError("Machine ID must be >= M1 and processing time must be positive.")
        route.append((machine_id, processing_time))
    return route


def event_time(
    breakdown_enabled: bool,
    breakdown_start: int,
    emergency_enabled: bool,
    emergency_time: int,
) -> int:
    candidates: List[int] = []
    if breakdown_enabled:
        candidates.append(int(breakdown_start))
    if emergency_enabled:
        candidates.append(int(emergency_time))
    return min(candidates) if candidates else 0


def frozen_state(
    original_schedule: Sequence[ScheduleRecord],
    jobs: Jobs,
    num_machines: int,
    current_time: int,
) -> Tuple[List[ScheduleRecord], List[int], List[int], List[int]]:
    """Freeze operations completed before or at current_time."""
    frozen = [dict(r) for r in original_schedule if r["end"] <= current_time]
    completed_counts = [0] * len(jobs)
    job_ready = [current_time] * len(jobs)
    machine_ready = [current_time] * num_machines

    for record in sorted(frozen, key=lambda r: r["end"]):
        job_id = record["job_id"]
        op_id = record["operation_id"]
        completed_counts[job_id] = max(completed_counts[job_id], op_id + 1)
        job_ready[job_id] = max(job_ready[job_id], record["end"])
        machine_ready[record["machine_id"]] = max(machine_ready[record["machine_id"]], record["end"])

    return frozen, completed_counts, job_ready, machine_ready


def build_remaining_fifo_chromosome(jobs: Jobs, completed_counts: List[int]) -> List[int]:
    remaining = [len(job) - completed_counts[job_id] for job_id, job in enumerate(jobs)]
    chromosome: List[int] = []
    for round_id in range(max(remaining) if remaining else 0):
        for job_id, count in enumerate(remaining):
            if round_id < count:
                chromosome.append(job_id)
    return chromosome


def combine_schedules(frozen: Sequence[ScheduleRecord], new_part: Sequence[ScheduleRecord]) -> List[ScheduleRecord]:
    return sorted([dict(r) for r in frozen] + [dict(r) for r in new_part], key=lambda r: (r["start"], r["machine_id"], r["job_id"], r["operation_id"]))


def apply_disturbance_no_optimization(
    original_schedule: Sequence[ScheduleRecord],
    jobs: Jobs,
    num_machines: int,
    current_time: int,
    breakdown: Optional[Tuple[int, int, int]] = None,
    emergency_route: Optional[List[Tuple[int, int]]] = None,
    emergency_time: Optional[int] = None,
) -> Tuple[Jobs, List[ScheduleRecord], List[str]]:
    """Create a disturbed baseline using FIFO ordering for remaining tasks."""
    logs: List[str] = []
    working_jobs: Jobs = [list(job) for job in jobs]
    emergency_job_id: Optional[int] = None
    if emergency_route:
        emergency_job_id = len(working_jobs)
        working_jobs.append(emergency_route)
        logs.append(f"Emergency job J{emergency_job_id + 1} inserted at t={emergency_time} with {len(emergency_route)} operations.")

    frozen, completed_counts, job_ready, machine_ready = frozen_state(original_schedule, working_jobs, num_machines, current_time)
    if emergency_job_id is not None:
        completed_counts.append(0) if len(completed_counts) < len(working_jobs) else None
        job_ready.append(int(emergency_time or current_time)) if len(job_ready) < len(working_jobs) else None

    machine_breakdowns: Dict[int, Tuple[int, int]] = {}
    if breakdown:
        machine_id, start, duration = breakdown
        machine_breakdowns[machine_id] = (start, start + duration)
        logs.append(f"Machine M{machine_id + 1} breakdown interval: {start}-{start + duration}.")

    chrom = build_remaining_fifo_chromosome(working_jobs, completed_counts)
    rescheduled_part, _ = decode_chromosome(
        chrom,
        working_jobs,
        num_machines,
        start_after=current_time,
        initial_job_ready=job_ready,
        initial_machine_ready=machine_ready,
        initial_next_operation=completed_counts,
        machine_breakdowns=machine_breakdowns,
        emergency_job_start=emergency_time,
        emergency_job_id=emergency_job_id,
    )
    return working_jobs, combine_schedules(frozen, rescheduled_part), logs


def dynamic_reschedule(
    original_schedule: Sequence[ScheduleRecord],
    jobs: Jobs,
    num_machines: int,
    algorithm: str,
    current_time: int,
    breakdown: Optional[Tuple[int, int, int]] = None,
    emergency_route: Optional[List[Tuple[int, int]]] = None,
    emergency_time: Optional[int] = None,
    random_seed: int = 42,
    fast_mode: bool = True,
    agent: Optional[DQNAgent] = None,
    training: bool = False,
) -> Tuple[Jobs, AlgorithmResult, List[str]]:
    """Freeze completed operations and reschedule the remaining part."""
    logs: List[str] = []
    working_jobs: Jobs = [list(job) for job in jobs]
    emergency_job_id: Optional[int] = None
    if emergency_route:
        emergency_job_id = len(working_jobs)
        working_jobs.append(emergency_route)
        logs.append(f"[{current_time:04d}] Emergency job J{emergency_job_id + 1} inserted; release time = {emergency_time}.")

    frozen, completed_counts, job_ready, machine_ready = frozen_state(original_schedule, working_jobs, num_machines, current_time)
    if emergency_job_id is not None and len(completed_counts) < len(working_jobs):
        completed_counts.append(0)
        job_ready.append(int(emergency_time or current_time))

    logs.append(f"[{current_time:04d}] Frozen completed operations: {len(frozen)}.")
    machine_breakdowns: Dict[int, Tuple[int, int]] = {}
    if breakdown:
        machine_id, start, duration = breakdown
        machine_breakdowns[machine_id] = (start, start + duration)
        logs.append(f"[{current_time:04d}] Machine M{machine_id + 1} unavailable from {start} to {start + duration}.")

    dqn_kwargs = (
        {"agent": agent, "training": training}
        if algorithm == "DQN-AOL-GA"
        else {}
    )
    result = run_algorithm(
        algorithm,
        working_jobs,
        num_machines,
        random_seed=random_seed,
        fast_mode=fast_mode,
        **dqn_kwargs,
        start_after=current_time,
        initial_job_ready=job_ready,
        initial_machine_ready=machine_ready,
        initial_next_operation=completed_counts,
        machine_breakdowns=machine_breakdowns,
        emergency_job_start=emergency_time,
        emergency_job_id=emergency_job_id,
    )
    result.schedule = combine_schedules(frozen, result.schedule)
    result.makespan = max([r["end"] for r in result.schedule], default=current_time)
    logs.append(f"[{current_time:04d}] Dynamic rescheduling completed by {algorithm}; new Cmax = {result.makespan}.")
    return working_jobs, result, logs

from algorithms.dqn_ga import run_dqn_ga
from data_loader import load_dataset
from disturbance import dynamic_reschedule
from rl.dqn_agent import DQNAgent, DQNConfig


def test_disturbance_accepts_dqn_agent_and_preserves_frozen_records():
    jobs, metadata = load_dataset("FT06")
    machines = int(metadata["machines"])
    agent = DQNAgent(DQNConfig(seed=23))
    initial = run_dqn_ga(
        jobs,
        machines,
        agent,
        population_size=10,
        generations=5,
        random_seed=23,
        training=False,
    )
    frozen_before_event = {
        (record["job_id"], record["operation_id"]): dict(record)
        for record in initial.schedule
        if record["end"] <= 20
    }

    updated_jobs, result, logs = dynamic_reschedule(
        initial.schedule,
        jobs,
        machines,
        algorithm="DQN-AOL-GA",
        current_time=20,
        breakdown=(1, 20, 15),
        emergency_route=[(0, 4), (2, 5), (1, 3)],
        emergency_time=20,
        random_seed=23,
        fast_mode=True,
        agent=agent,
        training=False,
    )

    assert len(updated_jobs) == len(jobs) + 1
    assert result.makespan >= 20
    assert any("DQN-AOL-GA" in line for line in logs)
    result_by_key = {
        (record["job_id"], record["operation_id"]): record
        for record in result.schedule
    }
    expected_keys = {
        (job_id, operation_id)
        for job_id, job in enumerate(updated_jobs)
        for operation_id in range(len(job))
    }
    assert set(result_by_key) == expected_keys

    for job_id, job in enumerate(updated_jobs):
        records = [
            result_by_key[(job_id, operation_id)]
            for operation_id in range(len(job))
        ]
        for operation_id, record in enumerate(records):
            machine_id, processing_time = job[operation_id]
            assert record["machine_id"] == machine_id
            assert record["processing_time"] == processing_time
            assert record["end"] - record["start"] == processing_time
        for previous, current in zip(records, records[1:]):
            assert previous["end"] <= current["start"]

    for machine_id in range(machines):
        machine_records = sorted(
            (
                record
                for record in result.schedule
                if record["machine_id"] == machine_id
            ),
            key=lambda record: record["start"],
        )
        for previous, current in zip(machine_records, machine_records[1:]):
            assert previous["end"] <= current["start"]

    for record in result.schedule:
        if record["machine_id"] == 1:
            assert record["end"] <= 20 or record["start"] >= 35

    emergency_job_id = len(jobs)
    emergency_records = [
        record for record in result.schedule if record["job_id"] == emergency_job_id
    ]
    assert len(emergency_records) == 3
    assert all(record["start"] >= 20 for record in emergency_records)

    for key, frozen_record in frozen_before_event.items():
        assert result_by_key[key] == frozen_record

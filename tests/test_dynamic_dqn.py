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
    for key, frozen_record in frozen_before_event.items():
        assert result_by_key[key] == frozen_record

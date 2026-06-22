from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from algorithms import AlgorithmResult, run_algorithm
from data_loader import available_datasets, load_dataset
from disturbance import (
    apply_disturbance_no_optimization,
    dynamic_reschedule,
    event_time,
    parse_emergency_route,
)
from metrics import calculate_metrics, schedule_deviation
from rl.actions import SEARCH_ACTIONS
from rl.checkpoint import CheckpointMetadata, load_checkpoint
from rl.dqn_agent import DQNAgent
from scheduler_core import Jobs, ScheduleRecord, num_machines_from_jobs
from services.training import TrainingConfig, TrainingReport, train_dqn
from visualization import (
    action_distribution_figure,
    convergence_figure,
    dqn_learning_figure,
    gantt_figure,
    learning_figure,
    metrics_comparison_figure,
    q_value_figure,
    schedule_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "dqn_aol_ga.pt"

for folder in (REPORT_DIR, FIGURE_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="智能车间深度强化学习调度系统 2.0",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.22);}
.system-kicker {font-size: .82rem; font-weight: 700; color: #0f766e; text-transform: uppercase;}
.system-subtitle {color: #64748b; margin-top: -.4rem; margin-bottom: 1.1rem;}
.log-box {background: #111827; color: #e5e7eb; padding: 14px; border-radius: 6px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: .84rem; line-height: 1.55; white-space: pre-wrap; max-height: 280px; overflow-y: auto;}
</style>
""",
    unsafe_allow_html=True,
)


def add_log(logs: List[str], message: str) -> None:
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def load_dqn_model(path: Path) -> Tuple[DQNAgent, CheckpointMetadata]:
    if not path.exists():
        raise FileNotFoundError(
            f"DQN model not found: {path}. Train a model in the Intelligence view first."
        )
    return load_checkpoint(path)


def model_status(path: Path) -> Tuple[str, Optional[DQNAgent], Optional[CheckpointMetadata]]:
    if not path.exists():
        return "missing", None, None
    try:
        agent, metadata = load_dqn_model(path)
        return "ready", agent, metadata
    except Exception:
        return "incompatible", None, None


def export_results(
    initial_result: AlgorithmResult,
    disturbed_schedule: Optional[List[ScheduleRecord]],
    rescheduled_result: Optional[AlgorithmResult],
    logs: List[str],
    num_machines: int,
    breakdown: Optional[Tuple[int, int, int]],
) -> None:
    schedules = {"initial": initial_result.schedule}
    if disturbed_schedule:
        schedules["disturbed"] = disturbed_schedule
    if rescheduled_result:
        schedules["rescheduled"] = rescheduled_result.schedule

    schedule_rows = []
    for label, schedule in schedules.items():
        for row in schedule_dataframe(schedule).to_dict("records"):
            row["schedule_type"] = label
            schedule_rows.append(row)
    pd.DataFrame(schedule_rows).to_csv(
        REPORT_DIR / "schedule_results.csv", index=False, encoding="utf-8-sig"
    )

    metric_rows = []
    for label, schedule in schedules.items():
        row = dict(calculate_metrics(schedule, num_machines))
        row["schedule_type"] = label
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(
        REPORT_DIR / "metrics_results.csv", index=False, encoding="utf-8-sig"
    )
    (REPORT_DIR / "event_log.txt").write_text("\n".join(logs), encoding="utf-8")
    gantt_figure(
        initial_result.schedule,
        num_machines,
        "Initial schedule",
        breakdown=breakdown,
    ).write_html(FIGURE_DIR / "initial_gantt.html")
    if rescheduled_result:
        gantt_figure(
            rescheduled_result.schedule,
            num_machines,
            "Rescheduled schedule",
            breakdown=breakdown,
        ).write_html(FIGURE_DIR / "rescheduled_gantt.html")


def run_initial(
    dataset_name: str,
    algorithm_name: str,
    fast_mode: bool,
    seed: int,
    checkpoint_path: Path,
) -> None:
    jobs, metadata = load_dataset(dataset_name)
    num_machines = int(metadata.get("machines") or num_machines_from_jobs(jobs))
    agent = None
    model_metadata = None
    if algorithm_name == "DQN-AOL-GA":
        agent, model_metadata = load_dqn_model(checkpoint_path)

    logs: List[str] = []
    add_log(logs, f"Loaded {dataset_name}: {len(jobs)} jobs, {num_machines} machines.")
    add_log(logs, f"Generating initial schedule with {algorithm_name}, seed={seed}.")
    result = run_algorithm(
        algorithm_name,
        jobs,
        num_machines,
        random_seed=seed,
        fast_mode=fast_mode,
        agent=agent,
        training=False,
    )
    add_log(logs, f"Initial schedule completed: Cmax={result.makespan}, runtime={result.runtime:.3f}s.")

    updates = {
        "dataset_name": dataset_name,
        "jobs": jobs,
        "metadata": metadata,
        "num_machines": num_machines,
        "initial_result": result,
        "disturbed_schedule": None,
        "disturbed_jobs": None,
        "rescheduled_result": None,
        "event_logs": logs,
        "deviation": 0,
        "rescheduling_count": 0,
        "dqn_metadata": model_metadata,
    }
    st.session_state.update(updates)


def run_disturbance(
    algorithm_name: str,
    fast_mode: bool,
    seed: int,
    checkpoint_path: Path,
    enable_breakdown: bool,
    breakdown_machine_display: int,
    breakdown_start: int,
    breakdown_duration: int,
    enable_emergency: bool,
    emergency_time: int,
    emergency_route_text: str,
) -> None:
    if "initial_result" not in st.session_state:
        raise ValueError("Generate an initial schedule before applying a disturbance.")
    if not enable_breakdown and not enable_emergency:
        raise ValueError("Enable a machine breakdown or emergency job first.")

    jobs: Jobs = st.session_state["jobs"]
    num_machines: int = st.session_state["num_machines"]
    initial_result: AlgorithmResult = st.session_state["initial_result"]
    logs: List[str] = list(st.session_state.get("event_logs", []))
    t_event = event_time(
        enable_breakdown, breakdown_start, enable_emergency, emergency_time
    )

    breakdown = None
    if enable_breakdown:
        breakdown = (
            breakdown_machine_display - 1,
            int(breakdown_start),
            int(breakdown_duration),
        )
        add_log(
            logs,
            f"Machine M{breakdown_machine_display} breakdown: "
            f"{breakdown_start}-{breakdown_start + breakdown_duration}.",
        )

    emergency_route = None
    if enable_emergency:
        emergency_route = parse_emergency_route(emergency_route_text)
        add_log(logs, f"Emergency job released at t={emergency_time}: {emergency_route_text}.")

    agent = None
    if algorithm_name == "DQN-AOL-GA":
        agent, _ = load_dqn_model(checkpoint_path)

    disturbed_jobs, disturbed_schedule, disturbed_logs = apply_disturbance_no_optimization(
        initial_result.schedule,
        jobs,
        num_machines,
        current_time=t_event,
        breakdown=breakdown,
        emergency_route=emergency_route,
        emergency_time=emergency_time if enable_emergency else None,
    )
    rescheduled_jobs, rescheduled_result, reschedule_logs = dynamic_reschedule(
        initial_result.schedule,
        jobs,
        num_machines,
        algorithm=algorithm_name,
        current_time=t_event,
        breakdown=breakdown,
        emergency_route=emergency_route,
        emergency_time=emergency_time if enable_emergency else None,
        random_seed=seed + 1000,
        fast_mode=fast_mode,
        agent=agent,
        training=False,
    )
    for line in disturbed_logs + reschedule_logs:
        add_log(logs, line)
    deviation = schedule_deviation(initial_result.schedule, rescheduled_result.schedule)
    add_log(logs, f"Dynamic rescheduling completed: Cmax={rescheduled_result.makespan}, deviation={deviation}.")

    st.session_state.update(
        {
            "disturbed_jobs": disturbed_jobs,
            "disturbed_schedule": disturbed_schedule,
            "rescheduled_jobs": rescheduled_jobs,
            "rescheduled_result": rescheduled_result,
            "event_logs": logs,
            "deviation": deviation,
            "rescheduling_count": int(st.session_state.get("rescheduling_count", 0)) + 1,
            "current_time": t_event,
            "breakdown": breakdown,
            "emergency_job_id": len(rescheduled_jobs) - 1
            if enable_emergency and emergency_route
            else None,
        }
    )


def display_metrics(
    result: Optional[AlgorithmResult],
    num_machines: Optional[int],
    metadata: Optional[dict],
) -> None:
    columns = st.columns(6)
    if result is None or num_machines is None:
        for column, label in zip(
            columns,
            ["Cmax", "BKS Gap", "Utilization", "Idle Time", "Runtime", "Deviation"],
        ):
            column.metric(label, "-")
        return

    metrics = calculate_metrics(result.schedule, num_machines)
    optimum = (metadata or {}).get("optimum")
    gap = "-"
    if optimum:
        gap = f"{(result.makespan - float(optimum)) / float(optimum) * 100:.2f}%"
    columns[0].metric("Cmax", result.makespan)
    columns[1].metric("BKS Gap", gap)
    columns[2].metric("Utilization", f"{float(metrics['average_utilization']) * 100:.2f}%")
    columns[3].metric("Idle Time", metrics["total_idle_time"])
    columns[4].metric("Runtime", f"{result.runtime:.3f}s")
    columns[5].metric("Deviation", st.session_state.get("deviation", 0))


st.sidebar.header("运行配置")
dataset_name = st.sidebar.selectbox("Dataset", available_datasets(), index=1)
algorithm_name = st.sidebar.selectbox(
    "Algorithm",
    ["FIFO", "GA", "SLGA", "CP-AOL-SLGA", "DQN-AOL-GA"],
    index=4,
)
seed = int(st.sidebar.number_input("Random seed", min_value=0, value=42, step=1))
fast_mode = st.sidebar.checkbox("Fast demo mode", value=True)
checkpoint_text = st.sidebar.text_input(
    "DQN checkpoint", value=str(DEFAULT_CHECKPOINT.relative_to(BASE_DIR))
)
checkpoint_path = resolve_checkpoint(checkpoint_text)
checkpoint_state, _, checkpoint_metadata = model_status(checkpoint_path)
if checkpoint_state == "ready":
    st.sidebar.success("DQN model ready")
elif checkpoint_state == "incompatible":
    st.sidebar.error("DQN model incompatible; retrain it")
else:
    st.sidebar.warning("DQN model missing")

if st.sidebar.button("Generate initial schedule", type="primary", use_container_width=True):
    try:
        with st.spinner("Optimizing schedule..."):
            run_initial(dataset_name, algorithm_name, fast_mode, seed, checkpoint_path)
    except Exception as exc:
        st.sidebar.error(str(exc))

st.sidebar.divider()
st.sidebar.subheader("扰动设置")
enable_breakdown = st.sidebar.checkbox("Machine breakdown", value=True)
breakdown_machine_display = int(
    st.sidebar.number_input("Breakdown machine", min_value=1, max_value=20, value=2)
)
breakdown_start = int(
    st.sidebar.number_input("Breakdown start", min_value=0, value=20, step=5)
)
breakdown_duration = int(
    st.sidebar.number_input("Breakdown duration", min_value=1, value=20, step=5)
)
enable_emergency = st.sidebar.checkbox("Emergency job", value=False)
emergency_time = int(
    st.sidebar.number_input("Emergency release", min_value=0, value=20, step=5)
)
emergency_route_text = st.sidebar.text_input(
    "Emergency route", value="M1:4, M3:5, M2:3"
)

if st.sidebar.button("Apply disturbance and reschedule", use_container_width=True):
    try:
        with st.spinner("Rescheduling remaining operations..."):
            run_disturbance(
                algorithm_name,
                fast_mode,
                seed,
                checkpoint_path,
                enable_breakdown,
                breakdown_machine_display,
                breakdown_start,
                breakdown_duration,
                enable_emergency,
                emergency_time,
                emergency_route_text,
            )
    except Exception as exc:
        st.sidebar.error(str(exc))

if st.sidebar.button("Export results", use_container_width=True):
    try:
        export_results(
            st.session_state["initial_result"],
            st.session_state.get("disturbed_schedule"),
            st.session_state.get("rescheduled_result"),
            st.session_state.get("event_logs", []),
            st.session_state["num_machines"],
            st.session_state.get("breakdown"),
        )
        st.sidebar.success("Exported to outputs/")
    except KeyError:
        st.sidebar.warning("Generate a schedule first")

st.markdown('<div class="system-kicker">Industrial 4.0 / Deep Reinforcement Learning</div>', unsafe_allow_html=True)
st.title("智能车间监测与动态重调度系统 2.0")
st.markdown(
    '<div class="system-subtitle">GA 负责搜索调度方案，Double DQN 根据种群状态学习选择交叉、变异与瓶颈局部搜索策略。</div>',
    unsafe_allow_html=True,
)

initial_result: Optional[AlgorithmResult] = st.session_state.get("initial_result")
shown_result: Optional[AlgorithmResult] = st.session_state.get("rescheduled_result") or initial_result
num_machines: Optional[int] = st.session_state.get("num_machines")
metadata: dict = st.session_state.get("metadata", {})
display_metrics(shown_result, num_machines, metadata)

tab_schedule, tab_intelligence, tab_disturbance = st.tabs(
    ["Scheduling", "Intelligence", "Disturbance Analysis"]
)

with tab_schedule:
    if initial_result is None:
        st.info("请在左侧选择数据集与算法，然后生成初始调度。使用 DQN-AOL-GA 前需先在 Intelligence 中训练模型。")
    else:
        data_cols = st.columns(4)
        data_cols[0].write(f"**Dataset:** {st.session_state.get('dataset_name', dataset_name)}")
        data_cols[1].write(f"**Jobs:** {len(st.session_state['jobs'])}")
        data_cols[2].write(f"**Machines:** {num_machines}")
        data_cols[3].write(f"**BKS/Optimum:** {metadata.get('optimum', '-')}")
        st.plotly_chart(
            gantt_figure(
                initial_result.schedule,
                int(num_machines),
                f"Initial Schedule - {initial_result.name} (Cmax={initial_result.makespan})",
                breakdown=st.session_state.get("breakdown"),
                current_time=st.session_state.get("current_time"),
            ),
            use_container_width=True,
        )
        with st.expander("Operation table"):
            st.dataframe(schedule_dataframe(initial_result.schedule), use_container_width=True)

with tab_intelligence:
    status_cols = st.columns(4)
    status_cols[0].metric("Model status", checkpoint_state.upper())
    status_cols[1].metric(
        "Training episodes", checkpoint_metadata.episodes if checkpoint_metadata else 0
    )
    status_cols[2].metric(
        "Training datasets",
        len(checkpoint_metadata.datasets) if checkpoint_metadata else 0,
    )
    if checkpoint_state == "ready":
        loaded_agent, _ = load_dqn_model(checkpoint_path)
        status_cols[3].metric("Current epsilon", f"{loaded_agent.epsilon:.3f}")
    else:
        status_cols[3].metric("Current epsilon", "-")

    with st.form("dqn_training_form"):
        st.subheader("Train Double DQN controller")
        train_cols = st.columns(4)
        training_datasets = tuple(
            train_cols[0].multiselect(
                "Datasets", available_datasets(), default=["FT06", "LA01", "LA02"]
            )
        )
        training_episodes = int(
            train_cols[1].number_input("Episodes", min_value=1, max_value=100, value=12)
        )
        training_population = int(
            train_cols[2].number_input("Population", min_value=8, max_value=160, value=24)
        )
        training_generations = int(
            train_cols[3].number_input("Generations", min_value=4, max_value=300, value=24)
        )
        train_submitted = st.form_submit_button("Train and save model", type="primary")

    if train_submitted:
        try:
            with st.spinner("Training DQN controller on CPU..."):
                report = train_dqn(
                    TrainingConfig(
                        datasets=training_datasets,
                        episodes=training_episodes,
                        population_size=training_population,
                        generations=training_generations,
                        base_seed=seed,
                        checkpoint_path=checkpoint_path,
                    )
                )
            st.session_state["training_report"] = report
            st.success(f"Model saved to {report.checkpoint_path}")
            st.rerun()
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    report: Optional[TrainingReport] = st.session_state.get("training_report")
    if report:
        st.plotly_chart(
            dqn_learning_figure(
                report.episode_rewards, report.losses, report.epsilon_history
            ),
            use_container_width=True,
        )
        expanded_actions = [
            action for action, count in report.action_counts.items() for _ in range(count)
        ]
        st.plotly_chart(action_distribution_figure(expanded_actions), use_container_width=True)

    if shown_result:
        st.subheader(f"Current run: {shown_result.name}")
        monitor_cols = st.columns(2)
        with monitor_cols[0]:
            st.plotly_chart(
                convergence_figure(
                    shown_result.history, f"Convergence - {shown_result.name}"
                ),
                use_container_width=True,
            )
        with monitor_cols[1]:
            if shown_result.name == "DQN-AOL-GA":
                st.plotly_chart(
                    dqn_learning_figure(
                        shown_result.reward_history,
                        shown_result.loss_history,
                        shown_result.epsilon_history,
                    ),
                    use_container_width=True,
                )
            elif shown_result.pc_history or shown_result.state_history:
                st.plotly_chart(
                    learning_figure(
                        shown_result.pc_history,
                        shown_result.pm_history,
                        shown_result.state_history,
                        shown_result.action_history,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("This baseline has no reinforcement-learning trace.")

        if shown_result.name == "DQN-AOL-GA":
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.plotly_chart(
                    action_distribution_figure(shown_result.action_history),
                    use_container_width=True,
                )
            with detail_cols[1]:
                st.plotly_chart(
                    q_value_figure(
                        shown_result.q_value_history,
                        [action.label for action in SEARCH_ACTIONS],
                    ),
                    use_container_width=True,
                )

with tab_disturbance:
    if initial_result is None:
        st.info("Generate an initial schedule before disturbance analysis.")
    else:
        inner_initial, inner_disturbed, inner_rescheduled = st.tabs(
            ["Initial", "Disturbed", "Rescheduled"]
        )
        breakdown = st.session_state.get("breakdown")
        current_time = st.session_state.get("current_time")
        emergency_job_id = st.session_state.get("emergency_job_id")

        with inner_initial:
            st.plotly_chart(
                gantt_figure(
                    initial_result.schedule,
                    int(num_machines),
                    "Initial production plan",
                    breakdown=breakdown,
                    current_time=current_time,
                ),
                use_container_width=True,
            )
        with inner_disturbed:
            disturbed_schedule = st.session_state.get("disturbed_schedule")
            if disturbed_schedule:
                disturbed_metrics = calculate_metrics(disturbed_schedule, int(num_machines))
                st.plotly_chart(
                    gantt_figure(
                        disturbed_schedule,
                        int(num_machines),
                        f"Disturbed baseline (Cmax={disturbed_metrics['makespan']})",
                        breakdown=breakdown,
                        emergency_job_id=emergency_job_id,
                        current_time=current_time,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Apply a disturbance from the sidebar.")
        with inner_rescheduled:
            rescheduled_result = st.session_state.get("rescheduled_result")
            if rescheduled_result:
                st.plotly_chart(
                    gantt_figure(
                        rescheduled_result.schedule,
                        int(num_machines),
                        f"Rescheduled plan - {rescheduled_result.name} (Cmax={rescheduled_result.makespan})",
                        breakdown=breakdown,
                        emergency_job_id=emergency_job_id,
                        current_time=current_time,
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No rescheduled plan yet.")

        if st.session_state.get("disturbed_schedule") and st.session_state.get(
            "rescheduled_result"
        ):
            metrics_map = {
                "Initial": calculate_metrics(initial_result.schedule, int(num_machines)),
                "Disturbed": calculate_metrics(
                    st.session_state["disturbed_schedule"], int(num_machines)
                ),
                "Rescheduled": calculate_metrics(
                    st.session_state["rescheduled_result"].schedule, int(num_machines)
                ),
            }
            st.plotly_chart(metrics_comparison_figure(metrics_map), use_container_width=True)
            st.dataframe(pd.DataFrame(metrics_map).T, use_container_width=True)

        st.subheader("Event log")
        logs = st.session_state.get("event_logs", [])
        safe_logs = "<br>".join(escape(line) for line in logs) if logs else "No events yet."
        st.markdown(f'<div class="log-box">{safe_logs}</div>', unsafe_allow_html=True)

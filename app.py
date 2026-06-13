from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from metrics import calculate_metrics, completed_operations, schedule_deviation
from scheduler_core import Jobs, ScheduleRecord, num_machines_from_jobs
from visualization import (
    convergence_figure,
    gantt_figure,
    learning_figure,
    metrics_comparison_figure,
    schedule_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
VIDEO_DIR = OUTPUT_DIR / "videos"
for folder in (REPORT_DIR, FIGURE_DIR, VIDEO_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="智能车间动态重调度系统",
    page_icon="🏭",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.metric-card {border: 1px solid rgba(148,163,184,.35); border-radius: 12px; padding: 12px; background: rgba(248,250,252,.85);}
.small-note {color: #64748b; font-size: 0.92rem;}
.log-box {background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap;}
</style>
""",
    unsafe_allow_html=True,
)


def add_log(logs: List[str], message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    logs.append(f"[{now}] {message}")


def export_results(
    initial_result: AlgorithmResult,
    disturbed_schedule: Optional[List[ScheduleRecord]],
    rescheduled_result: Optional[AlgorithmResult],
    logs: List[str],
    num_machines: int,
    breakdown: Optional[Tuple[int, int, int]],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    schedules = {"initial": initial_result.schedule}
    if disturbed_schedule:
        schedules["disturbed"] = disturbed_schedule
    if rescheduled_result:
        schedules["rescheduled"] = rescheduled_result.schedule

    # Schedule CSV
    schedule_rows = []
    for label, schedule in schedules.items():
        for row in schedule_dataframe(schedule).to_dict("records"):
            row["schedule_type"] = label
            schedule_rows.append(row)
    pd.DataFrame(schedule_rows).to_csv(REPORT_DIR / "schedule_results.csv", index=False, encoding="utf-8-sig")

    # Metrics CSV
    metrics_rows = []
    for label, schedule in schedules.items():
        m = calculate_metrics(schedule, num_machines)
        metrics_rows.append(
            {
                "schedule_type": label,
                "makespan": m["makespan"],
                "average_utilization": m["average_utilization"],
                "total_idle_time": m["total_idle_time"],
                "finished_operations": m["finished_operations"],
            }
        )
    pd.DataFrame(metrics_rows).to_csv(REPORT_DIR / "metrics_results.csv", index=False, encoding="utf-8-sig")

    # Event log
    (REPORT_DIR / "event_log.txt").write_text("\n".join(logs), encoding="utf-8")

    # HTML figures
    gantt_figure(initial_result.schedule, num_machines, "Initial schedule", breakdown=breakdown).write_html(FIGURE_DIR / "initial_gantt.html")
    if rescheduled_result:
        gantt_figure(rescheduled_result.schedule, num_machines, "Rescheduled schedule", breakdown=breakdown).write_html(FIGURE_DIR / "rescheduled_gantt.html")


def run_initial(dataset_name: str, algorithm_name: str, fast_mode: bool) -> None:
    logs: List[str] = []
    jobs, metadata = load_dataset(dataset_name)
    num_machines = int(metadata.get("machines") or num_machines_from_jobs(jobs))
    add_log(logs, f"Dataset {dataset_name} loaded. Jobs={len(jobs)}, Machines={num_machines}.")
    add_log(logs, f"Initial schedule generation started by {algorithm_name}.")
    result = run_algorithm(algorithm_name, jobs, num_machines, random_seed=42, fast_mode=fast_mode)
    add_log(logs, f"Initial schedule generated. Algorithm={algorithm_name}, Cmax={result.makespan}.")

    st.session_state["jobs"] = jobs
    st.session_state["metadata"] = metadata
    st.session_state["num_machines"] = num_machines
    st.session_state["initial_result"] = result
    st.session_state["disturbed_schedule"] = None
    st.session_state["disturbed_jobs"] = None
    st.session_state["rescheduled_result"] = None
    st.session_state["event_logs"] = logs
    st.session_state["deviation"] = 0
    st.session_state["rescheduling_count"] = 0


def run_disturbance(
    algorithm_name: str,
    fast_mode: bool,
    enable_breakdown: bool,
    breakdown_machine_display: int,
    breakdown_start: int,
    breakdown_duration: int,
    enable_emergency: bool,
    emergency_time: int,
    emergency_route_text: str,
) -> None:
    if "initial_result" not in st.session_state:
        st.warning("Please generate an initial schedule first.")
        return

    jobs: Jobs = st.session_state["jobs"]
    num_machines: int = st.session_state["num_machines"]
    initial_result: AlgorithmResult = st.session_state["initial_result"]
    logs: List[str] = list(st.session_state.get("event_logs", []))

    t_event = event_time(enable_breakdown, breakdown_start, enable_emergency, emergency_time)
    if not enable_breakdown and not enable_emergency:
        st.info("No disturbance is enabled.")
        return

    breakdown = None
    if enable_breakdown:
        breakdown = (breakdown_machine_display - 1, int(breakdown_start), int(breakdown_duration))
        add_log(logs, f"Machine M{breakdown_machine_display} breakdown configured: {breakdown_start}-{breakdown_start + breakdown_duration}.")

    emergency_route = None
    if enable_emergency:
        try:
            emergency_route = parse_emergency_route(emergency_route_text)
            add_log(logs, f"Emergency job configured at t={emergency_time}: {emergency_route_text}.")
        except Exception as exc:
            st.error(f"Emergency job route format error: {exc}")
            return

    add_log(logs, f"Dynamic rescheduling triggered at t={t_event}.")
    disturbed_jobs, disturbed_schedule, disturbed_logs = apply_disturbance_no_optimization(
        initial_result.schedule,
        jobs,
        num_machines,
        current_time=t_event,
        breakdown=breakdown,
        emergency_route=emergency_route,
        emergency_time=emergency_time if enable_emergency else None,
    )
    for line in disturbed_logs:
        add_log(logs, line)

    rescheduled_jobs, rescheduled_result, reschedule_logs = dynamic_reschedule(
        initial_result.schedule,
        jobs,
        num_machines,
        algorithm=algorithm_name,
        current_time=t_event,
        breakdown=breakdown,
        emergency_route=emergency_route,
        emergency_time=emergency_time if enable_emergency else None,
        random_seed=108,
        fast_mode=fast_mode,
    )
    for line in reschedule_logs:
        add_log(logs, line)

    deviation = schedule_deviation(initial_result.schedule, rescheduled_result.schedule)
    add_log(logs, f"Schedule deviation = {deviation}.")

    st.session_state["disturbed_jobs"] = disturbed_jobs
    st.session_state["disturbed_schedule"] = disturbed_schedule
    st.session_state["rescheduled_jobs"] = rescheduled_jobs
    st.session_state["rescheduled_result"] = rescheduled_result
    st.session_state["event_logs"] = logs
    st.session_state["deviation"] = deviation
    st.session_state["rescheduling_count"] = int(st.session_state.get("rescheduling_count", 0)) + 1
    st.session_state["current_time"] = t_event
    st.session_state["breakdown"] = breakdown
    st.session_state["emergency_job_id"] = len(rescheduled_jobs) - 1 if enable_emergency and emergency_route else None


def display_metrics(result: Optional[AlgorithmResult], num_machines: Optional[int]) -> None:
    if result is None or num_machines is None:
        cols = st.columns(6)
        labels = ["Cmax", "Average Utilization", "Total Idle Time", "Finished Operations", "Rescheduling Count", "Schedule Deviation"]
        for col, label in zip(cols, labels):
            col.metric(label, "-")
        return
    metrics = calculate_metrics(result.schedule, num_machines)
    cols = st.columns(6)
    cols[0].metric("Cmax", metrics["makespan"])
    cols[1].metric("Average Utilization", f"{float(metrics['average_utilization']) * 100:.2f}%")
    cols[2].metric("Total Idle Time", metrics["total_idle_time"])
    cols[3].metric("Finished Operations", metrics["finished_operations"])
    cols[4].metric("Rescheduling Count", st.session_state.get("rescheduling_count", 0))
    cols[5].metric("Schedule Deviation", st.session_state.get("deviation", 0))


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("System Controls")
dataset_name = st.sidebar.selectbox("Dataset", available_datasets(), index=1)
algorithm_name = st.sidebar.selectbox("Algorithm", ["FIFO", "GA", "SLGA", "CP-AOL-SLGA"], index=2)
fast_mode = st.sidebar.checkbox("Fast mode for classroom demo", value=True)

st.sidebar.divider()
st.sidebar.subheader("Machine Breakdown")
enable_breakdown = st.sidebar.checkbox("Enable machine breakdown", value=True)
breakdown_machine_display = st.sidebar.number_input("Breakdown machine", min_value=1, max_value=20, value=2, step=1)
breakdown_start = st.sidebar.number_input("Breakdown start time", min_value=0, value=120, step=5)
breakdown_duration = st.sidebar.number_input("Breakdown duration", min_value=1, value=60, step=5)

st.sidebar.divider()
st.sidebar.subheader("Emergency Job")
enable_emergency = st.sidebar.checkbox("Enable emergency job", value=False)
emergency_time = st.sidebar.number_input("Emergency job release time", min_value=0, value=150, step=5)
emergency_route_text = st.sidebar.text_input("Emergency route", value="M1:4, M3:5, M2:3")

st.sidebar.divider()
if st.sidebar.button("Generate Initial Schedule", type="primary"):
    run_initial(dataset_name, algorithm_name, fast_mode)

if st.sidebar.button("Apply Disturbance & Reschedule"):
    run_disturbance(
        algorithm_name,
        fast_mode,
        enable_breakdown,
        int(breakdown_machine_display),
        int(breakdown_start),
        int(breakdown_duration),
        enable_emergency,
        int(emergency_time),
        emergency_route_text,
    )

if st.sidebar.button("Export Results"):
    if "initial_result" in st.session_state:
        export_results(
            st.session_state["initial_result"],
            st.session_state.get("disturbed_schedule"),
            st.session_state.get("rescheduled_result"),
            st.session_state.get("event_logs", []),
            st.session_state["num_machines"],
            st.session_state.get("breakdown"),
        )
        st.sidebar.success("Exported to outputs/reports and outputs/figures.")
    else:
        st.sidebar.warning("Generate an initial schedule first.")

# -------------------------------
# Main page
# -------------------------------
st.title("工业4.0智能车间调度监测与动态重调度系统")
st.caption("Smart Shop Monitoring and Dynamic Rescheduling System")
st.markdown("面向作业车间调度的**计划生成、运行监测、扰动响应与动态优化**系统。")

initial_result: Optional[AlgorithmResult] = st.session_state.get("initial_result")
num_machines: Optional[int] = st.session_state.get("num_machines")
display_metrics(st.session_state.get("rescheduled_result") or initial_result, num_machines)

if initial_result is None:
    st.info("请在左侧选择数据集与算法，然后点击 Generate Initial Schedule。")
    st.stop()

metadata = st.session_state.get("metadata", {})
with st.expander("Dataset information", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**Dataset:** {dataset_name}")
    c2.write(f"**Jobs:** {metadata.get('jobs', len(st.session_state['jobs']))}")
    c3.write(f"**Machines:** {metadata.get('machines', num_machines)}")
    c4.write(f"**BKS/Optimum:** {metadata.get('optimum', '-')}")

breakdown = st.session_state.get("breakdown")
current_time = st.session_state.get("current_time")
emergency_job_id = st.session_state.get("emergency_job_id")

tab_initial, tab_disturbed, tab_rescheduled = st.tabs(["Initial Schedule", "Disturbed Schedule", "Rescheduled Schedule"])
with tab_initial:
    st.plotly_chart(
        gantt_figure(initial_result.schedule, num_machines, f"Initial Schedule - {initial_result.name} (Cmax={initial_result.makespan})", breakdown=breakdown, current_time=current_time),
        use_container_width=True,
    )
    st.dataframe(schedule_dataframe(initial_result.schedule), use_container_width=True)

with tab_disturbed:
    disturbed_schedule = st.session_state.get("disturbed_schedule")
    if disturbed_schedule:
        disturbed_metrics = calculate_metrics(disturbed_schedule, num_machines)
        st.plotly_chart(
            gantt_figure(disturbed_schedule, num_machines, f"Disturbed baseline / FIFO repair (Cmax={disturbed_metrics['makespan']})", breakdown=breakdown, emergency_job_id=emergency_job_id, current_time=current_time),
            use_container_width=True,
        )
        st.dataframe(schedule_dataframe(disturbed_schedule), use_container_width=True)
    else:
        st.info("扰动尚未应用。点击左侧 Apply Disturbance & Reschedule。")

with tab_rescheduled:
    rescheduled_result: Optional[AlgorithmResult] = st.session_state.get("rescheduled_result")
    if rescheduled_result:
        st.plotly_chart(
            gantt_figure(rescheduled_result.schedule, num_machines, f"Rescheduled Schedule - {rescheduled_result.name} (Cmax={rescheduled_result.makespan})", breakdown=breakdown, emergency_job_id=emergency_job_id, current_time=current_time),
            use_container_width=True,
        )
        st.dataframe(schedule_dataframe(rescheduled_result.schedule), use_container_width=True)
    else:
        st.info("暂未生成重调度方案。")

st.subheader("Algorithm Monitoring")
col_a, col_b = st.columns(2)
shown_result = st.session_state.get("rescheduled_result") or initial_result
with col_a:
    st.plotly_chart(convergence_figure(shown_result.history, f"Convergence curve - {shown_result.name}"), use_container_width=True)
with col_b:
    if shown_result.pc_history or shown_result.pm_history or shown_result.state_history:
        st.plotly_chart(learning_figure(shown_result.pc_history, shown_result.pm_history, shown_result.state_history, shown_result.action_history), use_container_width=True)
    else:
        st.info("FIFO or fixed GA has no Pc/Pm learning trace. Select SLGA or CP-AOL-SLGA to monitor parameter learning.")

if st.session_state.get("disturbed_schedule") and st.session_state.get("rescheduled_result"):
    st.subheader("Performance Comparison")
    metrics_map = {
        "Initial": calculate_metrics(initial_result.schedule, num_machines),
        "Disturbed": calculate_metrics(st.session_state["disturbed_schedule"], num_machines),
        "Rescheduled": calculate_metrics(st.session_state["rescheduled_result"].schedule, num_machines),
    }
    st.plotly_chart(metrics_comparison_figure(metrics_map), use_container_width=True)
    st.dataframe(pd.DataFrame(metrics_map).T, use_container_width=True)

st.subheader("Event Log")
logs = st.session_state.get("event_logs", [])
st.markdown(f"<div class='log-box'>{'<br>'.join(logs) if logs else 'No events yet.'}</div>", unsafe_allow_html=True)

with st.expander("Video demonstration note"):
    st.write(
        "本版本支持交互式网页演示。最终汇报视频建议直接对 Streamlit 页面录屏："
        "选择数据集 → 生成初始方案 → 设置机器故障/插单 → 触发重调度 → 展示三类甘特图和指标变化。"
    )

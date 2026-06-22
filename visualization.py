"""Plotly visualizations for the Streamlit monitoring system."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go

from scheduler_core import ScheduleRecord


def schedule_dataframe(schedule: Sequence[ScheduleRecord]) -> pd.DataFrame:
    rows = []
    for r in schedule:
        rows.append(
            {
                "job": f"J{r['job_id'] + 1}",
                "operation": f"O{r['operation_id'] + 1}",
                "machine": f"M{r['machine_id'] + 1}",
                "machine_id": r["machine_id"],
                "task": f"J{r['job_id'] + 1}-O{r['operation_id'] + 1}",
                "start": r["start"],
                "end": r["end"],
                "duration": r["processing_time"],
                "processing_time": r["processing_time"],
            }
        )
    return pd.DataFrame(rows)


def gantt_figure(
    schedule: Sequence[ScheduleRecord],
    num_machines: int,
    title: str,
    breakdown: Optional[Tuple[int, int, int]] = None,
    emergency_job_id: Optional[int] = None,
    current_time: Optional[int] = None,
) -> go.Figure:
    df = schedule_dataframe(schedule)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=title, height=420)
        return fig

    palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#8CD17D",
        "#B6992D", "#499894", "#86BCB6", "#D37295", "#FABFD2",
    ]
    for _, row in df.sort_values(["machine_id", "start"]).iterrows():
        job_id = int(row["job"][1:]) - 1
        is_emergency = emergency_job_id is not None and job_id == emergency_job_id
        color = "#D62728" if is_emergency else palette[job_id % len(palette)]
        fig.add_trace(
            go.Bar(
                x=[row["duration"]],
                y=[row["machine"]],
                base=[row["start"]],
                orientation="h",
                name=row["job"],
                marker=dict(color=color, line=dict(color="#222", width=1.1 if is_emergency else 0.4)),
                text=[row["task"]],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    "Task: %{text}<br>Machine: %{y}<br>Start: %{base}<br>Duration: %{x}<br>End: "
                    + str(row["end"])
                    + "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    y_labels = [f"M{i + 1}" for i in range(num_machines)]
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(y_labels)))
    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Time",
        yaxis_title="Machine",
        height=max(420, 80 * num_machines),
        plot_bgcolor="white",
        margin=dict(l=50, r=25, t=60, b=45),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    if breakdown:
        machine_id, start, duration = breakdown
        fig.add_vrect(x0=start, x1=start + duration, fillcolor="red", opacity=0.14, line_width=0, annotation_text=f"M{machine_id + 1} down", annotation_position="top left")
    if current_time is not None:
        fig.add_vline(x=current_time, line_width=2, line_dash="dash", line_color="#111827", annotation_text="Current time")
    return fig


def convergence_figure(history: Sequence[int], title: str) -> go.Figure:
    fig = go.Figure()
    if history:
        fig.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=list(history), mode="lines", name="Best Cmax"))
    fig.update_layout(title=title, xaxis_title="Generation", yaxis_title="Best Cmax", height=360, plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    return fig


def learning_figure(pc_history: Sequence[float], pm_history: Sequence[float], state_history: Sequence[int], action_history: Sequence[str] | None = None) -> go.Figure:
    fig = go.Figure()
    x = list(range(1, max(len(pc_history), len(pm_history), len(state_history), 1) + 1))
    if pc_history:
        fig.add_trace(go.Scatter(x=list(range(1, len(pc_history) + 1)), y=list(pc_history), mode="lines", name="Pc"))
    if pm_history:
        fig.add_trace(go.Scatter(x=list(range(1, len(pm_history) + 1)), y=list(pm_history), mode="lines", name="Pm"))
    if state_history:
        fig.add_trace(go.Scatter(x=list(range(1, len(state_history) + 1)), y=list(state_history), mode="lines", name="State", yaxis="y2"))
    fig.update_layout(
        title="Algorithm learning trace",
        xaxis_title="Generation",
        yaxis_title="Pc / Pm",
        yaxis2=dict(title="State", overlaying="y", side="right", showgrid=False),
        height=380,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    return fig


def dqn_learning_figure(
    rewards: Sequence[float],
    losses: Sequence[float],
    epsilons: Sequence[float],
) -> go.Figure:
    """Show the three signals needed to explain DQN training."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(rewards) + 1)),
            y=list(rewards),
            mode="lines",
            name="Reward",
            line=dict(color="#0f766e", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(losses) + 1)),
            y=list(losses),
            mode="lines",
            name="Loss",
            yaxis="y2",
            line=dict(color="#dc2626", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(epsilons) + 1)),
            y=list(epsilons),
            mode="lines",
            name="Epsilon",
            line=dict(color="#2563eb", width=1.8, dash="dot"),
        )
    )
    fig.update_layout(
        title="DQN learning signals",
        xaxis_title="Training step / generation",
        yaxis=dict(title="Reward / epsilon"),
        yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=45, r=45, t=70, b=45),
    )
    return fig


def action_distribution_figure(actions: Sequence[str]) -> go.Figure:
    counts = Counter(actions)
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            name="Selections",
            marker_color=["#0f766e", "#2563eb", "#7c3aed", "#ea580c", "#be123c"][: len(labels)],
        )
    )
    fig.update_layout(
        title="DQN action distribution",
        xaxis_title="Search strategy",
        yaxis_title="Selections",
        height=380,
        template="plotly_white",
        margin=dict(l=45, r=25, t=60, b=95),
    )
    fig.update_xaxes(tickangle=-20)
    return fig


def q_value_figure(
    q_value_history: Sequence[Sequence[float]],
    action_names: Optional[Sequence[str]] = None,
) -> go.Figure:
    fig = go.Figure()
    if q_value_history:
        width = max(len(values) for values in q_value_history)
        names = list(action_names or [f"Action {index + 1}" for index in range(width)])
        for action_index in range(width):
            values = [
                row[action_index] if action_index < len(row) else None
                for row in q_value_history
            ]
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode="lines",
                    name=names[action_index] if action_index < len(names) else f"Action {action_index + 1}",
                )
            )
    fig.update_layout(
        title="Estimated Q values",
        xaxis_title="Generation",
        yaxis_title="Q(s, a)",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.18),
        margin=dict(l=45, r=25, t=85, b=45),
    )
    return fig


def metrics_comparison_figure(metrics_map: Dict[str, Dict[str, object]]) -> go.Figure:
    methods = list(metrics_map.keys())
    cmax = [metrics_map[m]["makespan"] for m in methods]
    util = [float(metrics_map[m]["average_utilization"]) * 100 for m in methods]
    idle = [metrics_map[m]["total_idle_time"] for m in methods]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=methods, y=cmax, name="Cmax"))
    fig.add_trace(go.Bar(x=methods, y=util, name="Avg utilization (%)"))
    fig.add_trace(go.Bar(x=methods, y=idle, name="Idle time"))
    fig.update_layout(title="Schedule performance comparison", barmode="group", height=410, plot_bgcolor="white")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    return fig

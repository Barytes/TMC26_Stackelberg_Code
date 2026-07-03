"""Draw single-panel Stage-I NE-gap rerun figures with matplotlib."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METHOD_ORDER = ["Proposed", "GA", "BO-online", "MARL"]
METHOD_COLORS = {
    "Proposed": "#1f77b4",
    "GA": "#d62728",
    "BO-online": "#2ca02c",
    "MARL": "#9467bd",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str | float | int | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def _series_from_summary(
    rows: list[dict[str, str]],
    metric: str,
    methods: Iterable[str],
) -> dict[str, list[tuple[float, float, float, float]]]:
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for method in methods:
        values = []
        for row in rows:
            if row.get("method") != method:
                continue
            x = _finite_float(row.get("n_users"))
            center = _finite_float(row.get(f"{metric}_median"))
            low = _finite_float(row.get(f"{metric}_q25"))
            high = _finite_float(row.get(f"{metric}_q75"))
            if x is None or center is None or low is None or high is None:
                continue
            values.append((x, center, low, high))
        values.sort(key=lambda item: item[0])
        if values:
            out[method] = values
    return out


def _points_from_trials(
    rows: list[dict[str, str]],
    x_metric: str,
    y_metric: str,
    methods: Iterable[str],
) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for method in methods:
        values = []
        for row in rows:
            if row.get("method") != method or row.get("success") not in {"1", "True", "true"}:
                continue
            x = _finite_float(row.get(x_metric))
            y = _finite_float(row.get(y_metric))
            if x is None or y is None:
                continue
            values.append((x, y))
        if values:
            out[method] = values
    return out


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d9dee3", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#4b5563")
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors="#374151", labelsize=10)


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_summary_plot(
    summary_rows: list[dict[str, str]],
    out_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
    methods: list[str] = METHOD_ORDER,
) -> None:
    series = _series_from_summary(summary_rows, metric, methods)
    if not series:
        raise ValueError(f"No finite data for metric {metric}")

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    _style_axes(ax)

    for method in methods:
        values = series.get(method)
        if not values:
            continue
        xs = [item[0] for item in values]
        medians = [item[1] for item in values]
        lows = [item[2] for item in values]
        highs = [item[3] for item in values]
        color = METHOD_COLORS.get(method)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.14, linewidth=0)
        ax.plot(xs, medians, marker="o", linewidth=2.0, markersize=5.5, color=color, label=method)

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Number of users", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted({int(item[0]) for values in series.values() for item in values}))
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out_path)


def draw_scatter_plot(
    trial_rows: list[dict[str, str]],
    out_path: Path,
    *,
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    methods: list[str] = METHOD_ORDER,
) -> None:
    points_by_method = _points_from_trials(trial_rows, x_metric, y_metric, methods)
    if not points_by_method:
        raise ValueError(f"No finite trial data for {x_metric} vs {y_metric}")

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    _style_axes(ax)

    for method in methods:
        points = points_by_method.get(method)
        if not points:
            continue
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        ax.scatter(
            xs,
            ys,
            s=34,
            alpha=0.72,
            color=METHOD_COLORS.get(method),
            edgecolors="white",
            linewidths=0.45,
            label=method,
        )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_rows = _read_csv(out_dir / "stage1_final_grid_ne_gap_vs_users_stats.csv")
    trial_rows = _read_csv(out_dir / "stage1_final_grid_ne_gap_vs_users.csv")

    figures_dir = out_dir / "individual_figures_matplotlib"
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_final_grid_ne_gap_vs_users.png",
        metric="final_grid_ne_gap",
        title="Final grid-evaluated NE gap vs. number of users",
        ylabel="Final grid-evaluated NE gap",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_joint_revenue_vs_users.png",
        metric="joint_revenue",
        title="Joint revenue vs. number of users",
        ylabel="Joint revenue",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_runtime_vs_users.png",
        metric="runtime_sec",
        title="Runtime vs. number of users",
        ylabel="Runtime (sec)",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_total_stage2_calls_vs_users.png",
        metric="total_stage2_solver_calls",
        title="Total Stage-II solver calls vs. number of users",
        ylabel="Total Stage-II solver calls",
    )
    draw_scatter_plot(
        trial_rows,
        figures_dir / "individual_final_gap_vs_total_stage2_calls_trials.png",
        x_metric="total_stage2_solver_calls",
        y_metric="final_grid_ne_gap",
        title="Trial points: final gap vs. total Stage-II calls",
        xlabel="Total Stage-II solver calls",
        ylabel="Final grid-evaluated NE gap",
    )
    draw_scatter_plot(
        trial_rows,
        figures_dir / "individual_joint_revenue_vs_total_stage2_calls_trials.png",
        x_metric="total_stage2_solver_calls",
        y_metric="joint_revenue",
        title="Trial points: joint revenue vs. total Stage-II calls",
        xlabel="Total Stage-II solver calls",
        ylabel="Joint revenue",
    )
    print(f"Wrote matplotlib individual figures to {figures_dir}")


if __name__ == "__main__":
    main()

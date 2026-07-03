"""Replot budget-matched Stage-I comparison figures from existing CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "Proposed": "tab:blue",
    "GA": "tab:red",
    "BO": "tab:orange",
    "BO-online": "tab:purple",
    "MARL": "tab:green",
}

METHOD_LABELS = {
    "BO-online": "BO",
}

METHOD_X_OFFSET_FACTORS = {
    "GA": -0.035,
    "BO": 0.035,
    "BO-online": 0.035,
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _configure_fonts(axis_label_size: float, tick_label_size: float, legend_font_size: float) -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = tick_label_size
    plt.rcParams["axes.labelsize"] = axis_label_size
    plt.rcParams["xtick.labelsize"] = tick_label_size
    plt.rcParams["ytick.labelsize"] = tick_label_size
    plt.rcParams["legend.fontsize"] = legend_font_size
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1


def _method_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["Proposed", "GA", "BO", "BO-online", "MARL"]
    present = {row["method"] for row in rows}
    return [method for method in preferred if method in present] + sorted(present - set(preferred))


def _series(
    rows: list[dict[str, str]],
    *,
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row["method"] == method and int(float(row.get("count", "0"))) > 0],
        key=lambda row: int(float(row["n_users"])),
    )
    xs = np.asarray([int(float(row["n_users"])) for row in selected], dtype=int)
    center = np.asarray([float(row[f"{metric}_mean"]) for row in selected], dtype=float)
    std = np.asarray([float(row[f"{metric}_std"]) for row in selected], dtype=float)
    low = center - std
    high = center + std
    return xs, center, low, high


def _plot_xs(xs: np.ndarray, method: str) -> np.ndarray:
    if xs.size <= 1:
        return xs.astype(float)
    unique_xs = np.unique(xs.astype(float))
    step = float(np.min(np.diff(unique_xs))) if unique_xs.size > 1 else 0.0
    return xs.astype(float) + step * METHOD_X_OFFSET_FACTORS.get(method, 0.0)


def _plot_metric(
    *,
    rows: list[dict[str, str]],
    out_path: Path,
    metric: str,
    ylabel: str,
    axis_label_size: float,
    tick_label_size: float,
    legend_font_size: float,
) -> None:
    with plt.rc_context():
        _configure_fonts(axis_label_size, tick_label_size, legend_font_size)
        fig, ax = plt.subplots(figsize=(13.8, 8.0), dpi=180)
        for method in _method_order(rows):
            xs, center, low, high = _series(rows, method=method, metric=metric)
            if xs.size == 0:
                continue
            plot_xs = _plot_xs(xs, method)
            color = METHOD_COLORS.get(method)
            ax.plot(
                plot_xs,
                center,
                marker="o",
                linewidth=2.2,
                markersize=7.0,
                label=METHOD_LABELS.get(method, method),
                color=color,
            )
            ax.fill_between(plot_xs, low, high, alpha=0.18, color=color)
        ax.set_xlabel("Number of users")
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.tick_params(axis="both", labelsize=tick_label_size, pad=7)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="best",
            fontsize=legend_font_size,
            frameon=True,
        )
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.97)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot budget-matched Stage-I comparison figures from CSV outputs.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--axis-label-size", type=float, default=40.0)
    parser.add_argument("--tick-label-size", type=float, default=22.0)
    parser.add_argument("--legend-font-size", type=float, default=24.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = _read_csv_rows(out_dir / "stage1_budget_matched_quality_vs_users_stats.csv")
    _plot_metric(
        rows=rows,
        out_path=out_dir / "stage1_budget_matched_gap_vs_users.png",
        metric="final_grid_ne_gap",
        ylabel="NE gap",
        axis_label_size=float(args.axis_label_size),
        tick_label_size=float(args.tick_label_size),
        legend_font_size=float(args.legend_font_size),
    )
    _plot_metric(
        rows=rows,
        out_path=out_dir / "stage1_budget_matched_joint_revenue_vs_users.png",
        metric="joint_revenue",
        ylabel="Joint revenue",
        axis_label_size=float(args.axis_label_size),
        tick_label_size=float(args.tick_label_size),
        legend_font_size=float(args.legend_font_size),
    )


if __name__ == "__main__":
    main()

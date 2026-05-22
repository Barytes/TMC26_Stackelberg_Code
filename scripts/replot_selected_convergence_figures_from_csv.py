"""Replot selected convergence figures from existing CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _aligned_sequence_matrix(sequences: list[list[float]]) -> np.ndarray:
    if not sequences:
        raise ValueError("Expected at least one sequence.")
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=float)
    for idx, seq in enumerate(sequences):
        seq_arr = np.asarray(seq, dtype=float)
        arr[idx, : seq_arr.size] = seq_arr
        if 0 < seq_arr.size < max_len:
            arr[idx, seq_arr.size :] = seq_arr[-1]
    return arr


def _aligned_quantile_band(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    return (
        np.nanmedian(arr, axis=0),
        np.nanpercentile(arr, 25, axis=0),
        np.nanpercentile(arr, 75, axis=0),
    )


def _aligned_mean_band(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    center = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return center, np.maximum(center - std, 0.0), center + std


def _configure_fonts(*, axis_label_size: float, tick_label_size: float) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = tick_label_size
    plt.rcParams["axes.labelsize"] = axis_label_size
    plt.rcParams["xtick.labelsize"] = tick_label_size
    plt.rcParams["ytick.labelsize"] = tick_label_size


def _plot_a1(
    *,
    csv_path: Path,
    out_path: Path,
    axis_label_size: float,
    tick_label_size: float,
    legend_font_size: float,
    max_iterations: int,
) -> None:
    rows = _read_csv_rows(csv_path)
    by_n: dict[int, list[dict[str, str]]] = {}
    centralized_by_n: dict[int, float] = {}
    for row in rows:
        n_users = int(row["n_users"])
        by_n.setdefault(n_users, []).append(row)
        centralized_by_n[n_users] = float(row["centralized_social_cost"])

    with plt.rc_context():
        _configure_fonts(axis_label_size=axis_label_size, tick_label_size=tick_label_size)
        fig, ax = plt.subplots(figsize=(12.2, 7.2), dpi=150)
        cmap = plt.get_cmap("tab10")
        color_by_n = {n_users: cmap(idx % 10) for idx, n_users in enumerate(sorted(by_n))}
        for n_users in sorted(by_n):
            color = color_by_n[n_users]
            ordered = [
                item
                for item in sorted(by_n[n_users], key=lambda item: int(item["iteration"]))
                if int(item["iteration"]) <= max_iterations
            ]
            x = np.asarray([int(item["iteration"]) for item in ordered], dtype=int)
            y = np.asarray([float(item["social_cost"]) for item in ordered], dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=color,
                label=rf"Stage-II $|\mathcal{{I}}|={n_users}$",
            )
            ax.axhline(
                y=centralized_by_n[n_users],
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                label=rf"Optimum $|\mathcal{{I}}|={n_users}$",
            )
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Social Cost")
        ax.tick_params(axis="both", labelsize=tick_label_size)
        ax.grid(alpha=0.25)
        handles, legend_labels = ax.get_legend_handles_labels()
        ordered_handles: list[object] = []
        ordered_labels: list[str] = []
        for n_users in sorted(by_n, reverse=True):
            for prefix in ("Stage-II", "Optimum"):
                target = rf"{prefix} $|\mathcal{{I}}|={n_users}$"
                if target in legend_labels:
                    idx = legend_labels.index(target)
                    ordered_handles.append(handles[idx])
                    ordered_labels.append(legend_labels[idx])
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=1,
            fontsize=legend_font_size,
            frameon=True,
            columnspacing=1.0,
            handlelength=2.0,
        )
        ax.set_xlim(0.5, max_iterations + 0.5)
        ax.set_xticks(np.arange(2, max_iterations + 1, 2))
        fig.subplots_adjust(left=0.12, right=0.66, bottom=0.18, top=0.97)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _group_c1_series(rows: list[dict[str, str]], field: str) -> list[list[float]]:
    by_trial: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial"]), []).append(row)
    series: list[list[float]] = []
    for trial in sorted(by_trial):
        ordered = sorted(by_trial[trial], key=lambda item: int(item["iteration"]))
        series.append([float(item[field]) for item in ordered])
    return series


def _plot_c1(
    *,
    csv_path: Path,
    out_path: Path,
    axis_label_size: float,
    tick_label_size: float,
    legend_font_size: float,
    statistic: str,
) -> None:
    rows = _read_csv_rows(csv_path)
    band_fn = _aligned_mean_band if statistic == "average" else _aligned_quantile_band
    restricted_center, restricted_q25, restricted_q75 = band_fn(_group_c1_series(rows, "restricted_gap"))
    grid_center, grid_q25, grid_q75 = band_fn(_group_c1_series(rows, "grid_ne_gap"))
    stopping_tol = 0.0

    with plt.rc_context():
        _configure_fonts(axis_label_size=axis_label_size, tick_label_size=tick_label_size)
        fig, ax = plt.subplots(figsize=(13.4, 7.4), dpi=150)
        x = np.arange(1, restricted_center.size + 1)
        ax.plot(
            x,
            restricted_center,
            marker="o",
            linewidth=2.2,
            color="tab:blue",
            label=f"Restricted NE gap ({statistic})",
        )
        ax.fill_between(
            x,
            restricted_q25,
            restricted_q75,
            alpha=0.18,
            color="tab:blue",
            label="_nolegend_",
        )
        ax.plot(
            x,
            grid_center,
            marker="s",
            linewidth=2.1,
            linestyle="--",
            color="tab:orange",
            label=f"Oracle NE gap ({statistic})",
        )
        ax.fill_between(
            x,
            grid_q25,
            grid_q75,
            alpha=0.14,
            color="tab:orange",
            label="_nolegend_",
        )
        ax.axhline(stopping_tol, color="black", linestyle="--", linewidth=1.3, label="Stopping tolerance")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("NE gap")
        ax.tick_params(axis="both", labelsize=tick_label_size)
        ax.grid(alpha=0.25)
        ax.legend(
            loc="upper right",
            fontsize=legend_font_size,
            frameon=True,
            handlelength=2.0,
        )
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.97)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot selected convergence figures from existing CSV outputs.")
    parser.add_argument("--a1-dir", type=str, default=None)
    parser.add_argument("--c1-dir", type=str, default=None)
    parser.add_argument("--axis-label-size", type=float, default=40.0)
    parser.add_argument("--tick-label-size", type=float, default=31.0)
    parser.add_argument("--a1-legend-font-size", type=float, default=31.0)
    parser.add_argument("--c1-legend-font-size", type=float, default=32.0)
    parser.add_argument("--a1-max-iterations", type=int, default=12)
    parser.add_argument("--c1-statistic", type=str, default="average", choices=["median", "average"])
    parser.add_argument("--c1-output-name", type=str, default="C1_restricted_gap_trajectory.png")
    args = parser.parse_args()

    if args.a1_dir is None and args.c1_dir is None:
        raise SystemExit("At least one of --a1-dir or --c1-dir is required.")
    if args.a1_dir is not None:
        a1_dir = Path(args.a1_dir)
        _plot_a1(
            csv_path=a1_dir / "A1_stage2_social_cost_trace.csv",
            out_path=a1_dir / "A1_stage2_social_cost_trace.png",
            axis_label_size=float(args.axis_label_size),
            tick_label_size=float(args.tick_label_size),
            legend_font_size=float(args.a1_legend_font_size),
            max_iterations=int(args.a1_max_iterations),
        )
    if args.c1_dir is not None:
        c1_dir = Path(args.c1_dir)
        _plot_c1(
            csv_path=c1_dir / "C1_restricted_gap_trajectory.csv",
            out_path=c1_dir / str(args.c1_output_name),
            axis_label_size=float(args.axis_label_size),
            tick_label_size=float(args.tick_label_size),
            legend_font_size=float(args.c1_legend_font_size),
            statistic=str(args.c1_statistic),
        )


if __name__ == "__main__":
    main()

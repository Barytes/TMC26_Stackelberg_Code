"""Replot Proposed n-sweep runtime and Stage-II call figures from existing CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLOR = "tab:blue"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row["method"] == "Proposed" and int(float(row.get("count", "0"))) > 0],
        key=lambda row: int(float(row["n_users"])),
    )
    xs = np.asarray([int(float(row["n_users"])) for row in selected], dtype=int)
    center = np.asarray([float(row[f"{metric}_median"]) for row in selected], dtype=float)
    low = np.asarray([float(row[f"{metric}_q25"]) for row in selected], dtype=float)
    high = np.asarray([float(row[f"{metric}_q75"]) for row in selected], dtype=float)
    return xs, center, low, high


def _plot_metric(
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    out_path: Path,
    font_size: float,
) -> None:
    with plt.rc_context():
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 13.5
        plt.rcParams["axes.titlesize"] = 17.0
        plt.rcParams["axes.labelsize"] = 15.0
        plt.rcParams["xtick.labelsize"] = 12.8
        plt.rcParams["ytick.labelsize"] = 12.8
        plt.rcParams["legend.fontsize"] = 12.0
        plt.rcParams["axes.linewidth"] = 1.25
        plt.rcParams["xtick.major.width"] = 1.1
        plt.rcParams["ytick.major.width"] = 1.1

        xs, center, low, high = _series(rows, metric)
        fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=180)
        ax.plot(
            xs,
            center,
            marker="o",
            linewidth=1.9,
            markersize=6.0,
            label="Proposed",
            color=METHOD_COLOR,
        )
        ax.fill_between(xs, low, high, alpha=0.18, color=METHOD_COLOR)
        ax.set_xlabel("Number of users", fontsize=font_size)
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.grid(True, alpha=0.25)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot selected Proposed n-sweep figures from CSV outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--font-size", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = _read_rows(run_dir / "stage1_proposed_n_sweep_stats.csv")
    _plot_metric(
        rows,
        metric="runtime_sec",
        ylabel="Runtime (s)",
        out_path=run_dir / "stage1_runtime_vs_users.png",
        font_size=float(args.font_size),
    )
    _plot_metric(
        rows,
        metric="stage2_solver_calls",
        ylabel="Stage-II solver calls",
        out_path=run_dir / "stage1_stage2_calls_vs_users.png",
        font_size=float(args.font_size),
    )


if __name__ == "__main__":
    main()

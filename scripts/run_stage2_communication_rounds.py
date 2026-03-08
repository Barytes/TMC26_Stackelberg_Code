#!/usr/bin/env python3
"""Stage II: Communication rounds tracking and comparison.

This script compares the number of communication rounds/iterations for:
- Algorithm 2 (DG): outer iterations + inner iterations
- UBRD: best-response rounds
- VI: multiplier update iterations
- PEN: outer iterations + inner BR rounds

Output: Bar chart comparing communication rounds across methods.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.baselines import (
    baseline_stage2_ubrd,
    baseline_stage2_vi,
    baseline_stage2_penalty,
)
from tmc26_exp.stackelberg import algorithm_2_heuristic_user_selection


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "n_users", "method",
        "outer_iterations", "inner_iterations", "total_rounds"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by (n_users, method)."""
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    fields = [
        "n_users", "method",
        "outer_iters_mean", "outer_iters_std",
        "inner_iters_mean", "inner_iters_std",
        "total_rounds_mean", "total_rounds_std",
        "count",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, method) in sorted(grouped.keys()):
            bucket = grouped[(n_users, method)]
            outer = [float(r["outer_iterations"]) for r in bucket]
            inner = [float(r["inner_iterations"]) for r in bucket]
            total = [float(r["total_rounds"]) for r in bucket]

            outer_mean, outer_std = _mean_std(outer)
            inner_mean, inner_std = _mean_std(inner)
            total_mean, total_std = _mean_std(total)

            row = {
                "n_users": n_users,
                "method": method,
                "outer_iters_mean": outer_mean,
                "outer_iters_std": outer_std,
                "inner_iters_mean": inner_mean,
                "inner_iters_std": inner_std,
                "total_rounds_mean": total_mean,
                "total_rounds_std": total_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_total_rounds(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot total communication rounds vs n_users for each method."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    methods = ["Algorithm2", "UBRD", "VI", "PEN"]
    colors = {
        "Algorithm2": "#1b5e20",
        "UBRD": "#b71c1c",
        "VI": "#0d47a1",
        "PEN": "#e65100",
    }
    markers = {
        "Algorithm2": "o",
        "UBRD": "s",
        "VI": "^",
        "PEN": "v",
    }
    labels = {
        "Algorithm2": "Algorithm 2 (DG)",
        "UBRD": "UBRD",
        "VI": "VI",
        "PEN": "Penalty",
    }

    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))
        x = [int(r["n_users"]) for r in rows]
        y = [float(r["total_rounds_mean"]) for r in rows]
        yerr = [float(r["total_rounds_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=markers[method],
            color=colors[method],
            linewidth=2.0,
            capsize=4,
            label=labels[method],
            markersize=8,
        )

    ax.set_title("Stage II: Communication Rounds vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Total Communication Rounds")
    ax.set_xticks(n_users_list)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_breakdown(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot inner vs outer iterations breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    methods = ["Algorithm2", "UBRD", "VI", "PEN"]
    colors = {
        "Algorithm2": "#1b5e20",
        "UBRD": "#b71c1c",
        "VI": "#0d47a1",
        "PEN": "#e65100",
    }

    # Plot outer iterations
    ax = axes[0]
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))
        x = [int(r["n_users"]) for r in rows]
        y = [float(r["outer_iters_mean"]) for r in rows]

        ax.plot(x, y, marker='o', color=colors[method], linewidth=2.0,
                label=method, markersize=6)

    ax.set_title("Outer Iterations")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Iterations")
    ax.set_xticks(n_users_list)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    # Plot inner iterations
    ax = axes[1]
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))
        x = [int(r["n_users"]) for r in rows]
        y = [float(r["inner_iters_mean"]) for r in rows]

        ax.plot(x, y, marker='s', color=colors[method], linewidth=2.0,
                label=method, markersize=6)

    ax.set_title("Inner Iterations")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Iterations")
    ax.set_xticks(n_users_list)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Stage II: Communication Rounds Breakdown", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage II communication rounds comparison")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="50,100,200")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20263001)
    parser.add_argument("--pE", type=float, default=None)
    parser.add_argument("--pN", type=float, default=None)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    pE = float(args.pE if args.pE is not None else cfg.stackelberg.initial_pE)
    pN = float(args.pN if args.pN is not None else cfg.stackelberg.initial_pN)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage2_communication_rounds_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []

    for n_users in n_users_list:
        trial_cfg = replace(cfg, n_users=n_users)
        for trial in range(args.trials):
            seed = args.seed + 1000 * n_users + trial
            rng = np.random.default_rng(seed)
            users = sample_users(trial_cfg, rng)

            # Algorithm 2 (DG)
            alg2 = algorithm_2_heuristic_user_selection(users, pE, pN, cfg.system, cfg.stackelberg)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "Algorithm2",
                "outer_iterations": alg2.iterations,
                "inner_iterations": alg2.inner_result.iterations,
                "total_rounds": alg2.iterations + alg2.inner_result.iterations,
            })

            # UBRD
            ubrd = baseline_stage2_ubrd(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            ubrd_rounds = ubrd.meta.get("rounds", 0) if hasattr(ubrd, "meta") else 0
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "UBRD",
                "outer_iterations": ubrd_rounds,
                "inner_iterations": 0,
                "total_rounds": ubrd_rounds,
            })

            # VI
            vi = baseline_stage2_vi(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            vi_iters = vi.meta.get("iters", 0) if hasattr(vi, "meta") else 0
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "VI",
                "outer_iterations": 0,
                "inner_iterations": vi_iters,
                "total_rounds": vi_iters,
            })

            # PEN
            pen = baseline_stage2_penalty(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            pen_outer = pen.meta.get("outer_iters", 0) if hasattr(pen, "meta") else 0
            pen_inner = pen.meta.get("br_rounds", 0) if hasattr(pen, "meta") else 0
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "PEN",
                "outer_iterations": pen_outer,
                "inner_iterations": pen_inner,
                "total_rounds": pen_outer + pen_inner,
            })

    _write_raw_csv(raw_rows, run_dir / "raw_communication_rounds.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_communication_rounds.csv")
    _plot_total_rounds(summary_rows, run_dir / "communication_rounds_total.png", n_users_list)
    _plot_breakdown(summary_rows, run_dir / "communication_rounds_breakdown.png", n_users_list)

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        "metrics = outer_iterations, inner_iterations, total_rounds",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

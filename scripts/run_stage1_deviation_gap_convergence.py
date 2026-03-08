#!/usr/bin/env python3
"""Stage I: Deviation gap (epsilon) convergence plot for Algorithm 5.

This script runs Algorithm 5 (guided Stackelberg search) and records the
deviation gap (epsilon) at each search iteration to produce a convergence plot.

Output: CSV files with per-iteration data and a PNG plot showing epsilon vs iteration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "iteration", "pE", "pN", "epsilon", "offloading_size"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, float | int]]:
    """Aggregate epsilon by iteration across trials."""
    # Group by (n_users, iteration)
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), int(row["iteration"]))
        grouped.setdefault(key, []).append(row)

    fields = ["n_users", "iteration", "epsilon_mean", "epsilon_std", "count"]
    summary_rows: list[dict[str, float | int]] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, iteration) in sorted(grouped.keys()):
            bucket = grouped[(n_users, iteration)]
            epsilons = [float(x["epsilon"]) for x in bucket]
            eps_mean, eps_std = _mean_std(epsilons)
            row = {
                "n_users": n_users,
                "iteration": iteration,
                "epsilon_mean": eps_mean,
                "epsilon_std": eps_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)
    return summary_rows


def _plot(summary_rows: list[dict[str, float | int]], out_path: Path, n_users_list: list[int]) -> None:
    """Plot epsilon convergence curves for different user counts."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Group by n_users
    by_users: dict[int, list[dict[str, float | int]]] = {}
    for row in summary_rows:
        n_users = int(row["n_users"])
        by_users.setdefault(n_users, []).append(row)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_users_list)))

    for idx, n_users in enumerate(n_users_list):
        if n_users not in by_users:
            continue
        rows = sorted(by_users[n_users], key=lambda r: int(r["iteration"]))
        x = [int(r["iteration"]) for r in rows]
        y = [float(r["epsilon_mean"]) for r in rows]
        yerr = [float(r["epsilon_std"]) for r in rows]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            color=colors[idx],
            linewidth=2.0,
            capsize=3,
            label=f"n={n_users}",
        )

    ax.set_title("Stage I: Deviation Gap (ε) Convergence")
    ax.set_xlabel("Search Iteration")
    ax.set_ylabel("Deviation Gap ε")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Users", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I deviation gap convergence")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="50,100,200")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260001)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]
    if not n_users_list:
        raise ValueError("No user counts provided.")

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_deviation_gap_convergence_{timestamp}"
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

            # Run Algorithm 5
            result = algorithm_5_stackelberg_guided_search(users, cfg.system, cfg.stackelberg)

            # Record trajectory
            for step in result.trajectory:
                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "iteration": step.iteration,
                    "pE": step.pE,
                    "pN": step.pN,
                    "epsilon": step.epsilon,
                    "offloading_size": len(step.offloading_set),
                })

            # Also record final state if not already in trajectory
            if result.trajectory:
                last_iter = result.trajectory[-1].iteration
                if result.epsilon != result.trajectory[-1].epsilon:
                    raw_rows.append({
                        "trial": trial,
                        "seed": seed,
                        "n_users": n_users,
                        "iteration": last_iter + 1,
                        "pE": result.price[0],
                        "pN": result.price[1],
                        "epsilon": result.epsilon,
                        "offloading_size": len(result.offloading_set),
                    })

    _write_raw_csv(raw_rows, run_dir / "raw_stage1_convergence.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage1_convergence.csv")
    _plot(summary_rows, run_dir / "stage1_deviation_gap_convergence.png", n_users_list)

    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        "metric = deviation gap (epsilon) per iteration from Algorithm 5 trajectory",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

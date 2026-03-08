#!/usr/bin/env python3
"""Appendix A1: Final deviation gap (epsilon) vs number of users.

This script shows how the final deviation gap from Algorithm 5 scales
with system size.

Output: Line plot showing epsilon vs number of users.
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
    fields = ["trial", "seed", "n_users", "epsilon", "social_cost", "offloading_size", "search_iterations"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by n_users."""
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_users"]), []).append(row)

    summary_rows: list[dict] = []
    fields = ["n_users", "epsilon_mean", "epsilon_std", "social_cost_mean", "search_iters_mean", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n_users in sorted(grouped.keys()):
            bucket = grouped[n_users]
            eps = [float(r["epsilon"]) for r in bucket]
            costs = [float(r["social_cost"]) for r in bucket]
            iters = [int(r["search_iterations"]) for r in bucket]

            eps_mean, eps_std = _mean_std(eps)
            cost_mean, _ = _mean_std(costs)
            iter_mean, _ = _mean_std([float(i) for i in iters])

            row = {
                "n_users": n_users,
                "epsilon_mean": eps_mean,
                "epsilon_std": eps_std,
                "social_cost_mean": cost_mean,
                "search_iters_mean": iter_mean,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot(summary_rows: list[dict], out_path: Path) -> None:
    """Plot epsilon vs n_users."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y = [float(r["epsilon_mean"]) for r in summary_rows]
    yerr = [float(r["epsilon_std"]) for r in summary_rows]

    ax.errorbar(x, y, yerr=yerr, marker='o', color='#1b5e20',
                linewidth=2.0, capsize=4, markersize=8)

    ax.set_title("Appendix A1: Final Deviation Gap (ε) vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Final Epsilon (Deviation Gap)")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_log_scale(summary_rows: list[dict], out_path: Path) -> None:
    """Plot epsilon vs n_users with log scale."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y = [float(r["epsilon_mean"]) for r in summary_rows]
    yerr = [float(r["epsilon_std"]) for r in summary_rows]

    ax.errorbar(x, y, yerr=yerr, marker='o', color='#1b5e20',
                linewidth=2.0, capsize=4, markersize=8)

    ax.set_title("Appendix A1: Final Deviation Gap (ε) vs Number of Users (Log Scale)")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Final Epsilon (Deviation Gap)")
    ax.set_xticks(x)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Appendix A1: Final epsilon vs users")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="20,50,100,200,500")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20291001)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"appendix_final_epsilon_vs_users_{timestamp}"
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

            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "epsilon": result.epsilon,
                "social_cost": result.social_cost,
                "offloading_size": len(result.offloading_set),
                "search_iterations": len(result.trajectory),
            })

    _write_raw_csv(raw_rows, run_dir / "raw_final_epsilon_vs_users.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_final_epsilon_vs_users.csv")
    _plot(summary_rows, run_dir / "final_epsilon_vs_users.png")
    _plot_log_scale(summary_rows, run_dir / "final_epsilon_vs_users_log.png")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
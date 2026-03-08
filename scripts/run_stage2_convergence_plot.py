#!/usr/bin/env python3
"""Stage II: Convergence plot (iteration vs social cost).

This script tracks the social cost at each iteration of Algorithm 2 (DG) and
other Stage-II methods to visualize convergence behavior.

Output: CSV files with per-iteration data and convergence plots.
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

from tmc26_exp.config import load_config, StackelbergConfig, SystemConfig
from tmc26_exp.model import UserBatch, local_cost, theta
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_1_distributed_primal_dual,
    _build_data,
    _heuristic_score_with_t,
    _solve_fixed_set_inner_exact,
    InnerSolveResult,
)


def algorithm_2_with_trajectory(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> tuple[list[dict], dict]:
    """Run Algorithm 2 with per-iteration trajectory tracking.

    Returns: (trajectory, final_result)
    """
    data = _build_data(users)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()

    previous_ve = 0.0
    last_added: int | None = None
    iterations = 0

    trajectory: list[dict] = []

    def compute_social_cost(inner_result: InnerSolveResult, offloading_set: set[int]) -> float:
        """Compute total social cost given inner result."""
        final_set = inner_result.offloading_objective
        outside = set(range(users.n)) - set(offloading_set)
        if outside:
            return final_set + float(np.sum(data.cl[list(outside)]))
        return final_set

    for t in range(cfg.greedy_max_iters):
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is None:
            inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
        ve = inner.offloading_objective

        if t >= 1 and last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
                if inner is None:
                    inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
                ve = inner.offloading_objective

        # Record trajectory point
        social_cost = compute_social_cost(inner, offloading_set)
        trajectory.append({
            "iteration": t,
            "offloading_size": len(offloading_set),
            "offloading_objective": float(ve),
            "social_cost": float(social_cost),
        })

        candidates = sorted(active_users - offloading_set)
        if not candidates:
            iterations = t + 1
            break

        if offloading_set:
            lambda_F, lambda_B = inner.lambda_F, inner.lambda_B
        else:
            lambda_F, lambda_B = 0.0, 0.0

        best_user = min(
            candidates,
            key=lambda j: _heuristic_score_with_t(
                data,
                j,
                pE + lambda_F,
                pN + lambda_B,
                system,
            ),
        )
        best_score = _heuristic_score_with_t(data, best_user, pE + lambda_F, pN + lambda_B, system)

        if best_score < 0.0:
            previous_ve = ve
            offloading_set.add(best_user)
            last_added = best_user
            iterations = t + 1
            continue

        iterations = t + 1
        break

    # Final result
    final_inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
    if final_inner is None:
        final_inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
    final_set = final_inner.offloading_set
    outside = set(range(users.n)) - set(final_set)
    social_cost = final_inner.offloading_objective + float(np.sum(data.cl[list(outside)])) if outside else final_inner.offloading_objective

    final_result = {
        "offloading_set": tuple(sorted(final_set)),
        "social_cost": float(social_cost),
        "iterations": iterations,
        "inner_iterations": final_inner.iterations,
    }

    return trajectory, final_result


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "iteration", "offloading_size", "social_cost"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by iteration."""
    # Group by (n_users, iteration)
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), int(row["iteration"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    fields = ["n_users", "iteration", "social_cost_mean", "social_cost_std", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, iteration) in sorted(grouped.keys()):
            bucket = grouped[(n_users, iteration)]
            costs = [float(x["social_cost"]) for x in bucket]
            cost_mean, cost_std = _mean_std(costs)
            row = {
                "n_users": n_users,
                "iteration": iteration,
                "social_cost_mean": cost_mean,
                "social_cost_std": cost_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_convergence(summary_rows: list[dict], out_path: Path, n_users_list: list[int], reference_cost: float | None = None) -> None:
    """Plot social cost convergence curves."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Group by n_users
    by_users: dict[int, list[dict]] = {}
    for row in summary_rows:
        n_users = int(row["n_users"])
        by_users.setdefault(n_users, []).append(row)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_users_list)))

    for idx, n_users in enumerate(n_users_list):
        if n_users not in by_users:
            continue
        rows = sorted(by_users[n_users], key=lambda r: int(r["iteration"]))
        x = [int(r["iteration"]) for r in rows]
        y = [float(r["social_cost_mean"]) for r in rows]
        yerr = [float(r["social_cost_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o",
            color=colors[idx],
            linewidth=2.0,
            capsize=3,
            label=f"n={n_users}",
        )

    # Reference line (optimal) if provided
    if reference_cost is not None:
        ax.axhline(reference_cost, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal (CS): {reference_cost:.2f}', zorder=10)

    ax.set_title("Stage II: Algorithm 2 Convergence (Social Cost)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Social Cost")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Users", loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage II convergence plot")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="50,100,200")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
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
        run_name = f"stage2_convergence_{timestamp}"
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

            # Run Algorithm 2 with trajectory
            trajectory, final_result = algorithm_2_with_trajectory(users, pE, pN, cfg.system, cfg.stackelberg)

            # Record trajectory
            for step in trajectory:
                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "iteration": step["iteration"],
                    "offloading_size": step["offloading_size"],
                    "social_cost": step["social_cost"],
                })

    _write_raw_csv(raw_rows, run_dir / "raw_stage2_convergence.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage2_convergence.csv")
    _plot_convergence(summary_rows, run_dir / "stage2_convergence.png", n_users_list)

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

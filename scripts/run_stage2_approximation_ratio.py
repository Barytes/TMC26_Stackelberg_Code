#!/usr/bin/env python3
"""Stage II: Approximation ratio validation (Theorem 2).

This script validates the theoretical approximation ratio bound from Theorem 2.

Computes:
- Empirical ratio: V(Algorithm 2) / V(Optimal)
- Theoretical upper bound from Theorem 2

Output: Plot comparing empirical ratio to theoretical bound.
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
from tmc26_exp.model import UserBatch, local_cost, theta
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_1_distributed_primal_dual,
    algorithm_2_heuristic_user_selection,
    _build_data,
    _sorted_tuple,
)
from tmc26_exp.baselines import baseline_stage2_centralized_solver


def compute_local_cost_only(users: UserBatch) -> float:
    """Compute social cost when all users compute locally (empty offloading set)."""
    return float(np.sum(local_cost(users)))


def compute_theorem2_bound(
    users: UserBatch,
    pE: float,
    pN: float,
    system,
    cfg,
) -> tuple[float, float, int]:
    """Compute Theorem 2 approximation ratio bound.

    Returns: (bound, v_empty, optimal_offloading_size)
    """
    # Compute V(∅) - social cost with empty offloading set
    v_empty = compute_local_cost_only(users)

    # Compute optimal solution using CS solver
    opt_result = baseline_stage2_centralized_solver(users, pE, pN, system, cfg.stackelberg, cfg.baselines)
    v_optimal = opt_result.social_cost
    opt_set_size = len(opt_result.offloading_set)

    # Compute marginal gain ΔĈ_{i*} for the best first user
    # This is the maximum reduction in cost from adding one user
    data = _build_data(users)

    # Evaluate adding each single user
    best_marginal_gain = 0.0
    for i in range(users.n):
        # Cost with only user i offloading
        single_set = _sorted_tuple([i])
        inner = algorithm_1_distributed_primal_dual(users, single_set, pE, pN, system, cfg.stackelberg)
        ce_single = inner.offloading_objective
        # Local cost of user i
        cl_i = float(data.cl[i])
        # Marginal gain = local_cost_i - offloading_cost_i
        marginal_gain = cl_i - ce_single
        if marginal_gain > best_marginal_gain:
            best_marginal_gain = marginal_gain

    # If no positive marginal gain, bound is trivial
    if best_marginal_gain <= 0:
        return float('inf'), v_empty, opt_set_size

    # Theorem 2 bound: (V(∅) - ΔĈ_{i*}) / (V(∅) - |X*|ΔĈ_{i*})
    numerator = v_empty - best_marginal_gain
    denominator = v_empty - opt_set_size * best_marginal_gain

    if denominator <= 0:
        return float('inf'), v_empty, opt_set_size

    bound = numerator / denominator
    return bound, v_empty, opt_set_size


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "n_users",
        "v_algorithm2", "v_optimal", "v_empty",
        "empirical_ratio", "theorem2_bound",
        "optimal_offloading_size",
    ]
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
    fields = [
        "n_users",
        "empirical_ratio_mean", "empirical_ratio_std",
        "theorem2_bound_mean", "theorem2_bound_std",
        "v_optimal_mean", "v_empty_mean",
        "count",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n_users in sorted(grouped.keys()):
            bucket = grouped[n_users]
            emp_ratios = [float(r["empirical_ratio"]) for r in bucket if float(r["empirical_ratio"]) < float('inf')]
            bounds = [float(r["theorem2_bound"]) for r in bucket if float(r["theorem2_bound"]) < float('inf')]
            v_opts = [float(r["v_optimal"]) for r in bucket]
            v_empties = [float(r["v_empty"]) for r in bucket]

            emp_mean, emp_std = _mean_std(emp_ratios) if emp_ratios else (0.0, 0.0)
            bound_mean, bound_std = _mean_std(bounds) if bounds else (0.0, 0.0)
            vopt_mean, _ = _mean_std(v_opts)
            vempty_mean, _ = _mean_std(v_empties)

            row = {
                "n_users": n_users,
                "empirical_ratio_mean": emp_mean,
                "empirical_ratio_std": emp_std,
                "theorem2_bound_mean": bound_mean,
                "theorem2_bound_std": bound_std,
                "v_optimal_mean": vopt_mean,
                "v_empty_mean": vempty_mean,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot(summary_rows: list[dict], out_path: Path) -> None:
    """Plot empirical ratio and theoretical bound."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y_emp = [float(r["empirical_ratio_mean"]) for r in summary_rows]
    yerr_emp = [float(r["empirical_ratio_std"]) for r in summary_rows]
    y_bound = [float(r["theorem2_bound_mean"]) for r in summary_rows]

    # Plot empirical ratio
    ax.errorbar(x, y_emp, yerr=yerr_emp, marker='o', color='#1b5e20',
                linewidth=2.0, capsize=4, label='Empirical Ratio', markersize=8)

    # Plot theoretical bound
    ax.plot(x, y_bound, marker='s', color='#b71c1c',
            linewidth=2.0, linestyle='--', label='Theorem 2 Bound', markersize=8)

    # Reference line at y=1 (perfect approximation)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Perfect (ratio=1)')

    ax.set_title("Algorithm 2 Approximation Ratio vs Number of Users\n(Theorem 2 Validation)")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Approximation Ratio")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right")

    # Ensure y-axis starts from a reasonable minimum
    all_y = y_emp + y_bound
    if all_y:
        ymin = min(min(all_y) * 0.9, 0.9)
        ymax = max(max(all_y) * 1.1, 1.1)
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 2 approximation ratio validation (Theorem 2)")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="6,8,10,12,14,16")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20262001)
    parser.add_argument("--pE", type=float, default=None)
    parser.add_argument("--pN", type=float, default=None)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    # Validate user counts
    for n in n_users_list:
        if n > cfg.baselines.exact_max_users:
            raise ValueError(
                f"n_users={n} exceeds exact_max_users={cfg.baselines.exact_max_users}. "
                f"CS solver required for optimal solution."
            )

    pE = float(args.pE if args.pE is not None else cfg.stackelberg.initial_pE)
    pN = float(args.pN if args.pN is not None else cfg.stackelberg.initial_pN)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage2_approximation_ratio_{timestamp}"
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

            # Run Algorithm 2
            alg2_result = algorithm_2_heuristic_user_selection(users, pE, pN, cfg.system, cfg.stackelberg)
            v_alg2 = alg2_result.social_cost

            # Run CS solver for optimal
            cs_result = baseline_stage2_centralized_solver(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            v_optimal = cs_result.social_cost

            # Compute Theorem 2 bound
            bound, v_empty, opt_size = compute_theorem2_bound(users, pE, pN, cfg.system, cfg)

            # Empirical ratio
            empirical_ratio = v_alg2 / max(v_optimal, 1e-12)

            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "v_algorithm2": v_alg2,
                "v_optimal": v_optimal,
                "v_empty": v_empty,
                "empirical_ratio": empirical_ratio,
                "theorem2_bound": bound,
                "optimal_offloading_size": opt_size,
            })

    _write_raw_csv(raw_rows, run_dir / "raw_approximation_ratio.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_approximation_ratio.csv")
    _plot(summary_rows, run_dir / "approximation_ratio.png")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"exact_max_users = {cfg.baselines.exact_max_users}",
        "formula_empirical = V_algorithm2 / V_optimal",
        "formula_theorem2 = (V(∅) - ΔĈ_{i*}) / (V(∅) - |X*|ΔĈ_{i*})",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

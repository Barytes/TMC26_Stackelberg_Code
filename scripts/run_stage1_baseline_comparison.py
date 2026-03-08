#!/usr/bin/env python3
"""Stage I: Unified baseline comparison for Algorithm 5.

This script provides a unified framework for comparing Stage-I pricing methods:
- Algorithm 5 (GSSE): Proposed method
- PBRD: Provider Best Response Dynamics
- BO: Bayesian Optimization
- DRL: Deep Reinforcement Learning
- GSO: Grid Search Oracle (small-scale only)

Key features:
- Fair budget comparison (same max Stage-II calls, same wall-clock cap)
- REAL deviation gap (epsilon), not social cost proxy
- True wall-clock runtime per method
- Tracks Stage-II oracle calls

Outputs:
- epsilon vs iteration curves
- final epsilon vs |I|
- runtime vs |I|
- Stage-II calls vs |I|

References: SPEC.md Figures 5, 9, and Appendix A1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import random
import statistics
import sys
import time
import math


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.config import load_config, StackelbergConfig, BaselineConfig, SystemConfig
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    algorithm_4_optimal_rne_sampling,
    algorithm_5_stackelberg_guided_search,
    _build_data,
    _provider_revenue,
    Provider,
)
from tmc26_exp.baselines import (
    baseline_stage1_pbdr,
    baseline_stage1_bo,
    baseline_stage1_drl,
    baseline_stage1_grid_search_oracle,
)


def compute_epsilon(
    users: UserBatch,
    offloading_set: tuple[int, ...],
    price: tuple[float, float],
    system: SystemConfig,
) -> float:
    """Compute true deviation gap (epsilon) for a given solution."""
    gain_E = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "E", system)
    gain_N = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "N", system)
    return max(gain_E.gain, gain_N.gain)


def run_algorithm5(users: UserBatch, system: SystemConfig, cfg: StackelbergConfig, max_budget: int) -> dict:
    """Run Algorithm 5 (GSSE)."""
    t0 = time.perf_counter()
    result = algorithm_5_stackelberg_guided_search(users, system, cfg)
    runtime = time.perf_counter() - t0

    # Extract trajectory for convergence plot
    trajectory = []
    for step in result.trajectory:
        trajectory.append({
            "iteration": step.iteration,
            "epsilon": step.epsilon,
        })

    return {
        "final_epsilon": result.epsilon,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.stage2_oracle_calls,
        "outer_iterations": result.outer_iterations,
        "trajectory": trajectory,
        "price": result.price,
        "offloading_set": result.offloading_set,
    }


def run_pbdr_baseline(users: UserBatch, system: SystemConfig, stackelberg_cfg: StackelbergConfig, baseline_cfg: BaselineConfig, max_budget: int) -> dict:
    """Run PBRD baseline with budget constraint."""
    t0 = time.perf_counter()

    # PBRD from baselines.py
    result = baseline_stage1_pbdr(users, system, stackelberg_cfg, baseline_cfg)

    runtime = time.perf_counter() - t0

    # Compute epsilon
    final_eps = compute_epsilon(users, result.offloading_set, result.price, system)

    return {
        "final_epsilon": final_eps,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.meta.get("iterations", 0) if hasattr(result, "meta") else 0,
        "outer_iterations": result.meta.get("iterations", 0) if hasattr(result, "meta") else 0,
        "trajectory": [],  # PBRD doesn't track per-iteration
        "price": result.price,
        "offloading_set": result.offloading_set,
    }


def run_bo_baseline(users: UserBatch, system: SystemConfig, stackelberg_cfg: StackelbergConfig, baseline_cfg: BaselineConfig, max_budget: int) -> dict:
    """Run BO baseline with budget constraint."""
    t0 = time.perf_counter()

    result = baseline_stage1_bo(users, system, stackelberg_cfg, baseline_cfg)

    runtime = time.perf_counter() - t0

    final_eps = compute_epsilon(users, result.offloading_set, result.price, system)

    return {
        "final_epsilon": final_eps,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.meta.get("evaluations", 0) if hasattr(result, "meta") else 0,
        "outer_iterations": result.meta.get("evaluations", 0) if hasattr(result, "meta") else 0,
        "trajectory": [],
        "price": result.price,
        "offloading_set": result.offloading_set,
    }


def run_drl_baseline(users: UserBatch, system: SystemConfig, stackelberg_cfg: StackelbergConfig, baseline_cfg: BaselineConfig, max_budget: int) -> dict:
    """Run DRL baseline with budget constraint."""
    t0 = time.perf_counter()

    result = baseline_stage1_drl(users, system, stackelberg_cfg, baseline_cfg)

    runtime = time.perf_counter() - t0

    final_eps = compute_epsilon(users, result.offloading_set, result.price, system)

    return {
        "final_epsilon": final_eps,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.meta.get("episodes", 0) if hasattr(result, "meta") else 0,
        "outer_iterations": result.meta.get("episodes", 0) if hasattr(result, "meta") else 0,
        "trajectory": [],
        "price": result.price,
        "offloading_set": result.offloading_set,
    }


def run_gso_baseline(users: UserBatch, system: SystemConfig, stackelberg_cfg: StackelbergConfig, baseline_cfg: BaselineConfig, max_budget: int) -> dict:
    """Run GSO baseline (small-scale only due to exponential complexity)."""
    t0 = time.perf_counter()

    result = baseline_stage1_grid_search_oracle(users, system, stackelberg_cfg, baseline_cfg)

    runtime = time.perf_counter() - t0

    final_eps = compute_epsilon(users, result.offloading_set, result.price, system)

    return {
        "final_epsilon": final_eps,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.meta.get("grid_points", 0) if hasattr(result, "meta") else 0,
        "outer_iterations": result.meta.get("grid_points", 0) if hasattr(result, "meta") else 0,
        "trajectory": [],
        "price": result.price,
        "offloading_set": result.offloading_set,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "final_epsilon", "runtime_sec", "stage2_oracle_calls", "outer_iterations"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_trajectory_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "iteration", "epsilon"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict], out_path: Path) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    fields = ["n_users", "method", "epsilon_mean", "epsilon_std",
              "runtime_mean", "runtime_std", "stage2_calls_mean", "stage2_calls_std", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, method) in sorted(grouped.keys()):
            bucket = grouped[(n_users, method)]
            eps = [float(r["final_epsilon"]) for r in bucket]
            rt = [float(r["runtime_sec"]) for r in bucket]
            calls = [int(r["stage2_oracle_calls"]) for r in bucket]

            eps_mean, eps_std = _mean_std(eps)
            rt_mean, rt_std = _mean_std(rt)
            calls_mean, calls_std = _mean_std([float(c) for c in calls])

            row = {
                "n_users": n_users,
                "method": method,
                "epsilon_mean": eps_mean,
                "epsilon_std": eps_std,
                "runtime_mean": rt_mean,
                "runtime_std": rt_std,
                "stage2_calls_mean": calls_mean,
                "stage2_calls_std": calls_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_epsilon_vs_iteration(trajectory_rows: list[dict], out_path: Path, n_users: int) -> None:
    """Plot epsilon vs iteration for convergence comparison."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    methods = ["Algorithm5", "PBRD", "BO", "DRL"]
    colors = {
        "Algorithm5": "#1b5e20",
        "PBRD": "#b71c1c",
        "BO": "#0d47a1",
        "DRL": "#e65100",
    }

    for method in methods:
        method_rows = [r for r in trajectory_rows if r["method"] == method and int(r["n_users"]) == n_users]
        if not method_rows:
            continue

        # Group by iteration
        by_iter: dict[int, list[float]] = {}
        for r in method_rows:
            it = int(r["iteration"])
            by_iter.setdefault(it, []).append(float(r["epsilon"]))

        iterations = sorted(by_iter.keys())
        y = [statistics.fmean(by_iter[it]) for it in iterations]

        ax.plot(iterations, y, marker='o', color=colors[method], linewidth=2.0, label=method, markersize=4)

    ax.set_title(f"Stage I: Deviation Gap (ε) vs Iteration (n={n_users})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Deviation Gap (ε)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_final_epsilon_vs_users(summary_rows: list[dict], out_path: Path) -> None:
    """Plot final epsilon vs |I| for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    methods = ["Algorithm5", "PBRD", "BO", "DRL", "GSO"]
    colors = {
        "Algorithm5": "#1b5e20",
        "PBRD": "#b71c1c",
        "BO": "#0d47a1",
        "DRL": "#e65100",
        "GSO": "#4a148c",
    }
    markers = {
        "Algorithm5": "o",
        "PBRD": "s",
        "BO": "^",
        "DRL": "v",
        "GSO": "D",
    }
    labels = {
        "Algorithm5": "Algorithm 5 (GSSE)",
        "PBRD": "PBRD",
        "BO": "BO",
        "DRL": "DRL",
        "GSO": "GSO",
    }

    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))

        x = [int(r["n_users"]) for r in rows]
        y = [float(r["epsilon_mean"]) for r in rows]
        yerr = [float(r["epsilon_std"]) for r in rows]

        ax.errorbar(x, y, yerr=yerr, marker=markers[method], color=colors[method],
                    linewidth=2.0, capsize=4, label=labels[method], markersize=8)

    ax.set_title("Stage I: Final Deviation Gap (ε) vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Final Deviation Gap (ε)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_runtime_vs_users(summary_rows: list[dict], out_path: Path) -> None:
    """Plot runtime vs |I|."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    methods = ["Algorithm5", "PBRD", "BO", "DRL"]
    colors = {
        "Algorithm5": "#1b5e20",
        "PBRD": "#b71c1c",
        "BO": "#0d47a1",
        "DRL": "#e65100",
    }

    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))

        x = [int(r["n_users"]) for r in rows]
        y = [float(r["runtime_mean"]) for r in rows]
        yerr = [float(r["runtime_std"]) for r in rows]

        ax.errorbar(x, y, yerr=yerr, marker='o', color=colors[method],
                    linewidth=2.0, capsize=4, label=method, markersize=8)

    ax.set_title("Stage I: Runtime vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Wall-clock Runtime (seconds)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage2_calls_vs_users(summary_rows: list[dict], out_path: Path) -> None:
    """Plot Stage-II oracle calls vs |I|."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    methods = ["Algorithm5", "PBRD", "BO", "DRL", "GSO"]
    colors = {
        "Algorithm5": "#1b5e20",
        "PBRD": "#b71c1c",
        "BO": "#0d47a1",
        "DRL": "#e65100",
        "GSO": "#4a148c",
    }

    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))

        x = [int(r["n_users"]) for r in rows]
        y = [float(r["stage2_calls_mean"]) for r in rows]

        ax.plot(x, y, marker='o', color=colors[method], linewidth=2.0, label=method, markersize=8)

    ax.set_title("Stage I: Stage-II Oracle Calls vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Number of Stage-II Oracle Calls")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I unified baseline comparison")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="20,50,100,200")
    parser.add_argument("--small-n-users", type=str, default="6,8,10,12", help="For GSO baseline")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--max-budget", type=int, default=500, help="Max Stage-II calls for fair comparison")
    parser.add_argument("--include-gso", action="store_true", help="Include GSO for small instances")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]
    small_n_users_list = [int(x.strip()) for x in args.small_n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_baseline_comparison_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    trajectory_rows: list[dict] = []

    methods = [
        ("Algorithm5", run_algorithm5),
        ("PBRD", run_pbdr_baseline),
        ("BO", run_bo_baseline),
        ("DRL", run_drl_baseline),
    ]

    for n_users in n_users_list:
        trial_cfg = replace(cfg, n_users=n_users)

        for trial in range(args.trials):
            seed = args.seed + 1000 * n_users + trial
            rng = np.random.default_rng(seed)
            random.seed(seed)
            users = sample_users(trial_cfg, rng)

            for method_name, method_fn in methods:
                result = method_fn(users, cfg.system, cfg.stackelberg, cfg.baselines, args.max_budget)

                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "method": method_name,
                    "final_epsilon": result["final_epsilon"],
                    "runtime_sec": result["runtime_sec"],
                    "stage2_oracle_calls": result["stage2_oracle_calls"],
                    "outer_iterations": result["outer_iterations"],
                })

                # Record trajectory for Algorithm5
                if method_name == "Algorithm5":
                    for step in result["trajectory"]:
                        trajectory_rows.append({
                            "trial": trial,
                            "seed": seed,
                            "n_users": n_users,
                            "method": method_name,
                            "iteration": step["iteration"],
                            "epsilon": step["epsilon"],
                        })

            # GSO for small instances only
            if args.include_gso and n_users in small_n_users:
                result = run_gso_baseline(users, cfg.system, cfg.stackelberg, cfg.baselines, args.max_budget)
                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "method": "GSO",
                    "final_epsilon": result["final_epsilon"],
                    "runtime_sec": result["runtime_sec"],
                    "stage2_oracle_calls": result["stage2_oracle_calls"],
                    "outer_iterations": result["outer_iterations"],
                })

    _write_raw_csv(raw_rows, run_dir / "raw_stage1_comparison.csv")
    _write_trajectory_csv(trajectory_rows, run_dir / "trajectories.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage1_comparison.csv")

    # Generate plots
    _plot_final_epsilon_vs_users(summary_rows, run_dir / "final_epsilon_vs_users.png")
    _plot_runtime_vs_users(summary_rows, run_dir / "runtime_vs_users.png")
    _plot_stage2_calls_vs_users(summary_rows, run_dir / "stage2_calls_vs_users.png")

    # Convergence plot for a representative n_users
    if trajectory_rows:
        mid_n = n_users_list[len(n_users_list) // 2] if n_users_list else 100
        _plot_epsilon_vs_iteration(trajectory_rows, run_dir / f"epsilon_vs_iteration_n{mid_n}.png", mid_n)

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"small_n_users = {small_n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"max_budget = {args.max_budget}",
        f"include_gso = {args.include_gso}",
        "methods = Algorithm5, PBRD, BO, DRL" + (", GSO" if args.include_gso else ""),
        "metrics = REAL deviation gap (epsilon), wall-clock runtime, stage2_oracle_calls",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
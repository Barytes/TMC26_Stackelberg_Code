#!/usr/bin/env python3
"""Stage I: Scalability analysis - runtime and Stage-II oracle calls vs system size.

This script tracks:
1. Wall-clock runtime of Algorithm 5
2. Number of Stage-II oracle calls (algorithm_2_heuristic_user_selection invocations)
3. Number of search iterations

Output: CSV data and plots showing scalability metrics vs number of users.
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
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_5_stackelberg_guided_search,
    algorithm_2_heuristic_user_selection,
    algorithm_4_optimal_rne_sampling,
    algorithm_3_gain_approximation,
    StackelbergResult,
)


# Global counter for Stage-II calls
_stage2_call_count = 0


def _make_counting_algorithm_2(original_func):
    """Wrap algorithm_2 to count calls."""
    def wrapper(users, pE, pN, system, cfg):
        global _stage2_call_count
        _stage2_call_count += 1
        return original_func(users, pE, pN, system, cfg)
    return wrapper


def run_stage1_with_instrumentation(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> tuple[StackelbergResult, float, int]:
    """Run Algorithm 5 with instrumentation.

    Returns: (result, runtime_sec, stage2_call_count)
    """
    global _stage2_call_count
    _stage2_call_count = 0

    # Monkey-patch to count calls
    original_alg2 = algorithm_2_heuristic_user_selection
    instrumented_alg2 = _make_counting_algorithm_2(original_alg2)

    # Temporarily replace the function
    import tmc26_exp.stackelberg as stackelberg_module
    stackelberg_module.algorithm_2_heuristic_user_selection = instrumented_alg2

    try:
        t0 = time.perf_counter()
        result = algorithm_5_stackelberg_guided_search(users, system, cfg)
        runtime = time.perf_counter() - t0
        return result, runtime, _stage2_call_count
    finally:
        # Restore original function
        stackelberg_module.algorithm_2_heuristic_user_selection = original_alg2


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "n_users",
        "runtime_sec", "stage2_calls", "search_iterations",
        "final_epsilon", "final_offloading_size"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by n_users."""
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        n_users = int(row["n_users"])
        grouped.setdefault(n_users, []).append(row)

    summary_rows: list[dict] = []
    fields = [
        "n_users",
        "runtime_mean", "runtime_std",
        "stage2_calls_mean", "stage2_calls_std",
        "search_iters_mean", "search_iters_std",
        "final_epsilon_mean", "final_epsilon_std",
        "count"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n_users in sorted(grouped.keys()):
            bucket = grouped[n_users]
            runtimes = [float(r["runtime_sec"]) for r in bucket]
            calls = [int(r["stage2_calls"]) for r in bucket]
            iters = [int(r["search_iterations"]) for r in bucket]
            epsilons = [float(r["final_epsilon"]) for r in bucket]

            rt_mean, rt_std = _mean_std(runtimes)
            calls_mean, calls_std = _mean_std([float(c) for c in calls])
            iters_mean, iters_std = _mean_std([float(i) for i in iters])
            eps_mean, eps_std = _mean_std(epsilons)

            row = {
                "n_users": n_users,
                "runtime_mean": rt_mean,
                "runtime_std": rt_std,
                "stage2_calls_mean": calls_mean,
                "stage2_calls_std": calls_std,
                "search_iters_mean": iters_mean,
                "search_iters_std": iters_std,
                "final_epsilon_mean": eps_mean,
                "final_epsilon_std": eps_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_runtime(summary_rows: list[dict], out_path: Path) -> None:
    """Plot runtime vs n_users."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y = [float(r["runtime_mean"]) for r in summary_rows]
    yerr = [float(r["runtime_std"]) for r in summary_rows]

    ax.errorbar(x, y, yerr=yerr, marker='o', color='#1b5e20',
                linewidth=2.0, capsize=4, markersize=8)

    ax.set_title("Stage I: Runtime vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Wall-clock Time (seconds)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage2_calls(summary_rows: list[dict], out_path: Path) -> None:
    """Plot Stage-II oracle calls vs n_users."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y = [float(r["stage2_calls_mean"]) for r in summary_rows]
    yerr = [float(r["stage2_calls_std"]) for r in summary_rows]

    ax.errorbar(x, y, yerr=yerr, marker='s', color='#0d47a1',
                linewidth=2.0, capsize=4, markersize=8)

    ax.set_title("Stage I: Stage-II Oracle Calls vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Number of Stage-II Calls")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_search_iterations(summary_rows: list[dict], out_path: Path) -> None:
    """Plot search iterations vs n_users."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]
    y = [float(r["search_iters_mean"]) for r in summary_rows]
    yerr = [float(r["search_iters_std"]) for r in summary_rows]

    ax.errorbar(x, y, yerr=yerr, marker='^', color='#e65100',
                linewidth=2.0, capsize=4, markersize=8)

    ax.set_title("Stage I: Search Iterations vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Number of Search Iterations")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_combined(summary_rows: list[dict], out_path: Path) -> None:
    """Combined plot with dual y-axes."""
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=150)

    x = [int(r["n_users"]) for r in summary_rows]

    # Runtime on left axis
    color1 = '#1b5e20'
    y1 = [float(r["runtime_mean"]) for r in summary_rows]
    yerr1 = [float(r["runtime_std"]) for r in summary_rows]
    ax1.errorbar(x, y1, yerr=yerr1, marker='o', color=color1,
                 linewidth=2.0, capsize=4, markersize=8, label='Runtime (s)')
    ax1.set_xlabel("Number of Users")
    ax1.set_ylabel("Runtime (seconds)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale("log")

    # Stage-II calls on right axis
    ax2 = ax1.twinx()
    color2 = '#0d47a1'
    y2 = [float(r["stage2_calls_mean"]) for r in summary_rows]
    yerr2 = [float(r["stage2_calls_std"]) for r in summary_rows]
    ax2.errorbar(x, y2, yerr=yerr2, marker='s', color=color2,
                 linewidth=2.0, capsize=4, markersize=8, label='Stage-II Calls')
    ax2.set_ylabel("Stage-II Oracle Calls", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title("Stage I Scalability: Runtime and Oracle Calls")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I scalability analysis")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="20,50,100,200,500")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260005)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_scalability_{timestamp}"
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

            # Run with instrumentation
            result, runtime, stage2_calls = run_stage1_with_instrumentation(
                users, cfg.system, cfg.stackelberg
            )

            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "runtime_sec": runtime,
                "stage2_calls": stage2_calls,
                "search_iterations": len(result.trajectory),
                "final_epsilon": result.epsilon,
                "final_offloading_size": len(result.offloading_set),
            })

            print(f"  n={n_users}, trial={trial}: runtime={runtime:.3f}s, calls={stage2_calls}")

    _write_raw_csv(raw_rows, run_dir / "raw_stage1_scalability.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage1_scalability.csv")

    # Generate plots
    _plot_runtime(summary_rows, run_dir / "stage1_scalability_runtime.png")
    _plot_stage2_calls(summary_rows, run_dir / "stage1_scalability_oracle_calls.png")
    _plot_search_iterations(summary_rows, run_dir / "stage1_scalability_iterations.png")
    _plot_combined(summary_rows, run_dir / "stage1_scalability_combined.png")

    # Write metadata
    total_runtime = sum(float(r["runtime_sec"]) for r in raw_rows)
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"total_experiments = {len(raw_rows)}",
        f"total_runtime = {total_runtime:.2f}s",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"\nDone. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage I: Algorithm 3 gain approximation accuracy validation.

This script compares the approximated best-response gain from Algorithm 3
against the exact gain computed via exhaustive enumeration (for small instances).

Output: Scatter plot comparing approximated vs exact gain, with identity line.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.config import load_config, SystemConfig
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_3_gain_approximation,
    _build_data,
    _provider_revenue,
    _sorted_tuple,
    Provider,
)


def compute_exact_best_response_gain(
    users: UserBatch,
    current_set: Iterable[int],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> tuple[float, tuple[int, ...]]:
    """Compute exact best-response gain via exhaustive enumeration.

    For small instances, enumerate all possible offloading sets to find
    the true optimal deviation target and its gain.

    Returns: (exact_gain, optimal_set)
    """
    data = _build_data(users)
    X = _sorted_tuple(current_set)
    current_revenue = _provider_revenue(data, X, pE, pN, provider, system)

    n = users.n
    best_gain = 0.0
    best_set = X

    # Enumerate all possible subsets of users
    # For small n (<=16), this is feasible (2^16 = 65536)
    from itertools import combinations

    for size in range(n + 1):
        for subset in combinations(range(n), size):
            Y = _sorted_tuple(subset)
            candidate_revenue = _provider_revenue(data, Y, pE, pN, provider, system)
            gain = candidate_revenue - current_revenue
            if gain > best_gain:
                best_gain = float(gain)
                best_set = Y

    return best_gain, best_set


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "n_users", "provider", "pE", "pN",
        "approx_gain", "exact_gain", "abs_error", "rel_error",
        "approx_best_set", "exact_best_set", "candidate_count"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> dict:
    """Compute summary statistics."""
    abs_errors = [float(r["abs_error"]) for r in rows]
    rel_errors = [float(r["rel_error"]) for r in rows if float(r["exact_gain"]) > 1e-8]

    summary = {
        "count": len(rows),
        "abs_error_mean": statistics.fmean(abs_errors) if abs_errors else 0.0,
        "abs_error_std": statistics.stdev(abs_errors) if len(abs_errors) > 1 else 0.0,
        "rel_error_mean": statistics.fmean(rel_errors) if rel_errors else 0.0,
        "rel_error_std": statistics.stdev(rel_errors) if len(rel_errors) > 1 else 0.0,
    }

    fields = ["count", "abs_error_mean", "abs_error_std", "rel_error_mean", "rel_error_std"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(summary)

    return summary


def _plot_scatter(rows: list[dict[str, object]], out_path: Path) -> None:
    """Plot scatter of approximated vs exact gain."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # Separate by provider
    esp_rows = [r for r in rows if r["provider"] == "E"]
    nsp_rows = [r for r in rows if r["provider"] == "N"]

    # Plot ESP points
    if esp_rows:
        x_esp = [float(r["exact_gain"]) for r in esp_rows]
        y_esp = [float(r["approx_gain"]) for r in esp_rows]
        ax.scatter(x_esp, y_esp, c="#1b5e20", alpha=0.5, s=30, label="ESP", zorder=3)

    # Plot NSP points
    if nsp_rows:
        x_nsp = [float(r["exact_gain"]) for r in nsp_rows]
        y_nsp = [float(r["approx_gain"]) for r in nsp_rows]
        ax.scatter(x_nsp, y_nsp, c="#b71c1c", alpha=0.5, s=30, label="NSP", zorder=3)

    # Identity line (y = x)
    all_gains = [float(r["exact_gain"]) for r in rows] + [float(r["approx_gain"]) for r in rows]
    if all_gains:
        max_val = max(max(all_gains), 1e-8)
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label="y=x (perfect)", zorder=2)
        ax.set_xlim(-0.05 * max_val, 1.05 * max_val)
        ax.set_ylim(-0.05 * max_val, 1.05 * max_val)

    ax.set_title("Algorithm 3 Gain Approximation Accuracy")
    ax.set_xlabel("Exact Gain (exhaustive enumeration)")
    ax.set_ylabel("Approximated Gain (Algorithm 3)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_error_distribution(rows: list[dict[str, object]], out_path: Path) -> None:
    """Plot distribution of relative errors."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    rel_errors = [float(r["rel_error"]) * 100 for r in rows if float(r["exact_gain"]) > 1e-8]

    if rel_errors:
        ax.hist(rel_errors, bins=30, edgecolor='black', alpha=0.7, color='#0d47a1')
        ax.axvline(statistics.fmean(rel_errors), color='r', linestyle='--', linewidth=2,
                   label=f"Mean: {statistics.fmean(rel_errors):.2f}%")

    ax.set_title("Relative Error Distribution")
    ax.set_xlabel("Relative Error (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, axis='y')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 3 gain approximation accuracy")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="6,8,10,12")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--samples-per-trial", type=int, default=5,
                        help="Number of random price/offloading samples per trial")
    parser.add_argument("--seed", type=int, default=20260003)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    # Validate user counts are within exact computation limit
    for n in n_users_list:
        if n > cfg.baselines.exact_max_users:
            raise ValueError(
                f"n_users={n} exceeds exact_max_users={cfg.baselines.exact_max_users}. "
                f"Exhaustive enumeration is only feasible for small instances."
            )

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_gain_accuracy_{timestamp}"
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

            for sample in range(args.samples_per_trial):
                # Sample random prices
                pE = rng.uniform(cfg.system.cE * 1.1, cfg.system.cE * 3.0) if hasattr(cfg.system, 'cE') else rng.uniform(0.5, 3.0)
                pN = rng.uniform(cfg.system.cN * 1.1, cfg.system.cN * 3.0) if hasattr(cfg.system, 'cN') else rng.uniform(0.5, 3.0)

                # Sample random offloading set
                offloading_prob = rng.uniform(0.2, 0.8)
                current_set = tuple(i for i in range(n_users) if rng.random() < offloading_prob)

                for provider in ["E", "N"]:
                    # Compute approximated gain
                    approx_result = algorithm_3_gain_approximation(
                        users, current_set, pE, pN, provider, cfg.system
                    )

                    # Compute exact gain
                    exact_gain, exact_best_set = compute_exact_best_response_gain(
                        users, current_set, pE, pN, provider, cfg.system
                    )

                    # Compute errors
                    abs_error = abs(approx_result.gain - exact_gain)
                    rel_error = abs_error / max(exact_gain, 1e-8)

                    raw_rows.append({
                        "trial": trial,
                        "seed": seed,
                        "n_users": n_users,
                        "provider": provider,
                        "pE": pE,
                        "pN": pN,
                        "approx_gain": approx_result.gain,
                        "exact_gain": exact_gain,
                        "abs_error": abs_error,
                        "rel_error": rel_error,
                        "approx_best_set": str(approx_result.best_set),
                        "exact_best_set": str(exact_best_set),
                        "candidate_count": approx_result.candidate_count,
                    })

    _write_raw_csv(raw_rows, run_dir / "raw_gain_accuracy.csv")
    summary = _write_summary_csv(raw_rows, run_dir / "summary_gain_accuracy.csv")
    _plot_scatter(raw_rows, run_dir / "gain_accuracy_scatter.png")
    _plot_error_distribution(raw_rows, run_dir / "gain_accuracy_error_dist.png")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"samples_per_trial = {args.samples_per_trial}",
        f"seed = {args.seed}",
        f"total_samples = {len(raw_rows)}",
        f"abs_error_mean = {summary['abs_error_mean']:.6f}",
        f"rel_error_mean = {summary['rel_error_mean']:.6f}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")
    print(f"  Mean absolute error: {summary['abs_error_mean']:.6f}")
    print(f"  Mean relative error: {summary['rel_error_mean']*100:.2f}%")


if __name__ == "__main__":
    main()

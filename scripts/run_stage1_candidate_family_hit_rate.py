#!/usr/bin/env python3
"""Stage I: Candidate family N(p) effectiveness (hit rate).

This script validates the construction of the candidate family N(p) by measuring
how often the exact optimal deviation target is contained in the candidate family.

Output: Bar chart showing hit rate vs number of users.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from itertools import combinations


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
    _candidate_family,
    _provider_revenue,
    _sorted_tuple,
    Provider,
)


def find_exact_best_response_target(
    users: UserBatch,
    current_set: tuple[int, ...],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> tuple[int, ...]:
    """Find the exact best deviation target via exhaustive enumeration."""
    data = _build_data(users)
    X = current_set
    current_revenue = _provider_revenue(data, X, pE, pN, provider, system)

    n = users.n
    best_gain = 0.0
    best_set = X

    # Enumerate all possible subsets
    for size in range(n + 1):
        for subset in combinations(range(n), size):
            Y = _sorted_tuple(subset)
            candidate_revenue = _provider_revenue(data, Y, pE, pN, provider, system)
            gain = candidate_revenue - current_revenue
            if gain > best_gain:
                best_gain = float(gain)
                best_set = Y

    return best_set


def compute_candidate_hit_rate(
    users: UserBatch,
    current_set: tuple[int, ...],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> tuple[bool, int, tuple[int, ...], tuple[int, ...]]:
    """Check if exact optimal target is in candidate family.

    Returns: (hit, candidate_count, exact_target, approx_target)
    """
    data = _build_data(users)

    # Get candidate family
    family = _candidate_family(data, current_set, pE, pN, system)
    family_sets = set(family)

    # Find exact optimal target
    exact_target = find_exact_best_response_target(users, current_set, pE, pN, provider, system)

    # Check if exact target is in family
    hit = exact_target in family_sets

    # Get approximated target from Algorithm 3
    approx_result = algorithm_3_gain_approximation(users, current_set, pE, pN, provider, system)

    return hit, len(family), exact_target, approx_result.best_set


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "n_users", "provider", "pE", "pN",
        "hit", "candidate_count", "exact_target", "approx_target"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate hit rate by n_users and provider."""
    # Group by (n_users, provider)
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["provider"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    fields = ["n_users", "provider", "hit_rate_mean", "hit_rate_std", "candidate_count_mean", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, provider) in sorted(grouped.keys()):
            bucket = grouped[(n_users, provider)]
            hits = [1.0 if r["hit"] else 0.0 for r in bucket]
            cand_counts = [int(r["candidate_count"]) for r in bucket]

            hit_mean, hit_std = _mean_std(hits)
            cand_mean, _ = _mean_std([float(c) for c in cand_counts])

            row = {
                "n_users": n_users,
                "provider": provider,
                "hit_rate_mean": hit_mean * 100,  # Convert to percentage
                "hit_rate_std": hit_std * 100,
                "candidate_count_mean": cand_mean,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_hit_rate(summary_rows: list[dict], out_path: Path) -> None:
    """Plot hit rate vs n_users for both providers."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Separate by provider
    esp_rows = [r for r in summary_rows if r["provider"] == "E"]
    nsp_rows = [r for r in summary_rows if r["provider"] == "N"]

    # Plot ESP
    if esp_rows:
        x_esp = [int(r["n_users"]) for r in esp_rows]
        y_esp = [float(r["hit_rate_mean"]) for r in esp_rows]
        yerr_esp = [float(r["hit_rate_std"]) for r in esp_rows]
        ax.errorbar(x_esp, y_esp, yerr=yerr_esp, marker='o', color='#1b5e20',
                    linewidth=2.0, capsize=4, label='ESP', markersize=8)

    # Plot NSP
    if nsp_rows:
        x_nsp = [int(r["n_users"]) for r in nsp_rows]
        y_nsp = [float(r["hit_rate_mean"]) for r in nsp_rows]
        yerr_nsp = [float(r["hit_rate_std"]) for r in nsp_rows]
        ax.errorbar(x_nsp, y_nsp, yerr=yerr_nsp, marker='s', color='#b71c1c',
                    linewidth=2.0, capsize=4, label='NSP', markersize=8)

    # Reference lines
    ax.axhline(100, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect (100%)')
    ax.axhline(80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Good (80%)')
    ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance (50%)')

    ax.set_title("Candidate Family N(p) Hit Rate")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_candidate_count(summary_rows: list[dict], out_path: Path) -> None:
    """Plot average candidate family size vs n_users."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    esp_rows = [r for r in summary_rows if r["provider"] == "E"]
    nsp_rows = [r for r in summary_rows if r["provider"] == "N"]

    if esp_rows:
        x = [int(r["n_users"]) for r in esp_rows]
        y = [float(r["candidate_count_mean"]) for r in esp_rows]
        ax.plot(x, y, marker='o', color='#1b5e20', linewidth=2.0, label='ESP', markersize=8)

    if nsp_rows:
        x = [int(r["n_users"]) for r in nsp_rows]
        y = [float(r["candidate_count_mean"]) for r in nsp_rows]
        ax.plot(x, y, marker='s', color='#b71c1c', linewidth=2.0, label='NSP', markersize=8)

    ax.set_title("Candidate Family Size")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Average Candidate Count")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate family N(p) hit rate validation")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="6,8,10,12,14")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260004)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    # Validate user counts
    for n in n_users_list:
        if n > cfg.baselines.exact_max_users:
            raise ValueError(
                f"n_users={n} exceeds exact_max_users={cfg.baselines.exact_max_users}. "
                f"Exact enumeration is only feasible for small instances."
            )

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_candidate_hit_rate_{timestamp}"
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

            # Sample random prices
            pE = rng.uniform(0.5, 3.0)
            pN = rng.uniform(0.5, 3.0)

            # Sample random offloading set
            offloading_prob = rng.uniform(0.2, 0.8)
            current_set = tuple(i for i in range(n_users) if rng.random() < offloading_prob)

            for provider in ["E", "N"]:
                hit, cand_count, exact_target, approx_target = compute_candidate_hit_rate(
                    users, current_set, pE, pN, provider, cfg.system
                )

                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "provider": provider,
                    "pE": pE,
                    "pN": pN,
                    "hit": hit,
                    "candidate_count": cand_count,
                    "exact_target": str(exact_target),
                    "approx_target": str(approx_target),
                })

    _write_raw_csv(raw_rows, run_dir / "raw_candidate_hit_rate.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_candidate_hit_rate.csv")
    _plot_hit_rate(summary_rows, run_dir / "candidate_hit_rate.png")
    _plot_candidate_count(summary_rows, run_dir / "candidate_family_size.png")

    # Compute overall hit rate
    overall_hits = sum(1 for r in raw_rows if r["hit"])
    overall_hit_rate = overall_hits / len(raw_rows) * 100 if raw_rows else 0

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"total_samples = {len(raw_rows)}",
        f"overall_hit_rate = {overall_hit_rate:.2f}%",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")
    print(f"  Overall hit rate: {overall_hit_rate:.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ablation: Guided search ablation for Algorithm 5.

This script compares different variants of Algorithm 5 to validate that:
1. The exact deviation targets Y_E^*(p) and Y_N^*(p) are genuinely useful
2. The neighborhood fallback contributes beyond using only deviation targets
3. The deviation-target guidance improves search beyond random exploration

Variants compared (as per SPEC.md Figure 15):
- Full Algorithm 5: exact deviation-target prioritization + neighborhood fallback
- No deviation-target prioritization: random neighbor from N(p)
- Neighborhood-only: only uses N(p), no deviation targets
- Deviation-target-only: only uses Y_E^* / Y_N^*, no neighborhood fallback
- Random-restart local search: weak baseline

Output metrics (REAL deviation gap, not social cost proxy):
- final deviation gap / final epsilon
- runtime
- Stage-II oracle calls
- outer iterations

Output: Grouped bar charts comparing variants.
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

from tmc26_exp.config import load_config, StackelbergConfig, SystemConfig
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    algorithm_4_optimal_rne_sampling,
    _build_data,
    _candidate_family,
    _provider_revenue,
    _sorted_tuple,
    Provider,
    RNEResult,
    GainApproxResult,
    StackelbergResult,
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


def run_full_algorithm_5(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> dict:
    """Run full Algorithm 5 with deviation-target prioritization + neighborhood fallback."""
    from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search

    t0 = time.perf_counter()
    result = algorithm_5_stackelberg_guided_search(users, system, cfg)
    runtime = time.perf_counter() - t0

    return {
        "epsilon": result.epsilon,
        "runtime_sec": runtime,
        "stage2_oracle_calls": result.stage2_oracle_calls,
        "outer_iterations": result.outer_iterations,
        "evaluated_candidates": result.evaluated_candidates,
        "social_cost": result.social_cost,
    }


def run_no_prioritization(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> dict:
    """Run variant: no deviation-target prioritization, random neighbor from N(p)."""
    stage2_oracle_calls = 0
    evaluated_candidates = 0

    initial_price = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))

    t0 = time.perf_counter()

    init_stage2 = algorithm_2_heuristic_user_selection(users, initial_price[0], initial_price[1], system, cfg)
    stage2_oracle_calls += 1
    current_set = init_stage2.offloading_set
    current_rne = algorithm_4_optimal_rne_sampling(users, current_set, initial_price, system, cfg)
    current_price = current_rne.price

    visited: set[tuple[int, ...]] = {current_set}
    outer_iterations = 0

    for t in range(cfg.search_max_iters):
        outer_iterations = t + 1
        current_eps = compute_epsilon(users, current_set, current_price, system)

        # Get neighborhood N(p) but pick randomly instead of prioritized
        data = _build_data(users)
        neighborhood = _candidate_family(data, current_set, current_price[0], current_price[1], system)

        # Filter out visited and current
        candidates = [c for c in neighborhood if c != current_set and c not in visited]
        if not candidates:
            break

        # Random selection (no prioritization)
        random.shuffle(candidates)

        best_candidate = None
        best_rne = None
        best_eps = current_eps

        for candidate_set in candidates:
            evaluated_candidates += 1
            candidate_rne = algorithm_4_optimal_rne_sampling(users, candidate_set, current_price, system, cfg)

            if candidate_rne.epsilon + cfg.search_improvement_tol < best_eps:
                best_eps = candidate_rne.epsilon
                best_candidate = candidate_set
                best_rne = candidate_rne
                break  # Take first improvement (random order)

        if best_candidate is None:
            break

        visited.add(best_candidate)
        current_set = best_candidate
        current_price = best_rne.price

    final_stage2 = algorithm_2_heuristic_user_selection(users, current_price[0], current_price[1], system, cfg)
    stage2_oracle_calls += 1
    final_eps = compute_epsilon(users, final_stage2.offloading_set, current_price, system)

    return {
        "epsilon": final_eps,
        "runtime_sec": time.perf_counter() - t0,
        "stage2_oracle_calls": stage2_oracle_calls,
        "outer_iterations": outer_iterations,
        "evaluated_candidates": evaluated_candidates,
        "social_cost": final_stage2.social_cost,
    }


def run_neighborhood_only(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> dict:
    """Run variant: neighborhood-only search, no deviation targets at all."""
    stage2_oracle_calls = 0
    evaluated_candidates = 0

    initial_price = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))

    t0 = time.perf_counter()

    init_stage2 = algorithm_2_heuristic_user_selection(users, initial_price[0], initial_price[1], system, cfg)
    stage2_oracle_calls += 1
    current_set = init_stage2.offloading_set
    current_rne = algorithm_4_optimal_rne_sampling(users, current_set, initial_price, system, cfg)
    current_price = current_rne.price

    visited: set[tuple[int, ...]] = {current_set}
    outer_iterations = 0

    for t in range(cfg.search_max_iters):
        outer_iterations = t + 1

        # Compute epsilon via exhaustive evaluation (no Algorithm 3 gain approximation)
        data = _build_data(users)
        neighborhood = _candidate_family(data, current_set, current_price[0], current_price[1], system)

        candidates = [c for c in neighborhood if c != current_set and c not in visited]
        if not candidates:
            break

        best_candidate = None
        best_rne = None
        best_eps = float('inf')

        # Evaluate all candidates exhaustively
        for candidate_set in candidates:
            evaluated_candidates += 1
            candidate_rne = algorithm_4_optimal_rne_sampling(users, candidate_set, current_price, system, cfg)
            candidate_eps = candidate_rne.epsilon

            if candidate_eps + cfg.search_improvement_tol < best_eps:
                best_eps = candidate_eps
                best_candidate = candidate_set
                best_rne = candidate_rne

        current_eps = compute_epsilon(users, current_set, current_price, system)
        if best_candidate is None or best_eps >= current_eps - cfg.search_improvement_tol:
            break

        visited.add(best_candidate)
        current_set = best_candidate
        current_price = best_rne.price

    final_stage2 = algorithm_2_heuristic_user_selection(users, current_price[0], current_price[1], system, cfg)
    stage2_oracle_calls += 1
    final_eps = compute_epsilon(users, final_stage2.offloading_set, current_price, system)

    return {
        "epsilon": final_eps,
        "runtime_sec": time.perf_counter() - t0,
        "stage2_oracle_calls": stage2_oracle_calls,
        "outer_iterations": outer_iterations,
        "evaluated_candidates": evaluated_candidates,
        "social_cost": final_stage2.social_cost,
    }


def run_deviation_target_only(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> dict:
    """Run variant: only uses Y_E^* / Y_N^*, no neighborhood fallback."""
    stage2_oracle_calls = 0
    evaluated_candidates = 0

    initial_price = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))

    t0 = time.perf_counter()

    init_stage2 = algorithm_2_heuristic_user_selection(users, initial_price[0], initial_price[1], system, cfg)
    stage2_oracle_calls += 1
    current_set = init_stage2.offloading_set
    current_rne = algorithm_4_optimal_rne_sampling(users, current_set, initial_price, system, cfg)
    current_price = current_rne.price

    visited: set[tuple[int, ...]] = {current_set}
    outer_iterations = 0

    for t in range(cfg.search_max_iters):
        outer_iterations = t + 1

        gain_E = algorithm_3_gain_approximation(users, current_set, current_price[0], current_price[1], "E", system)
        gain_N = algorithm_3_gain_approximation(users, current_set, current_price[0], current_price[1], "N", system)
        current_eps = max(gain_E.gain, gain_N.gain)

        # Only evaluate exact deviation targets
        exact_targets: list[tuple[int, ...]] = []
        for cand in (gain_E.best_set, gain_N.best_set):
            if cand not in exact_targets and cand != current_set:
                exact_targets.append(cand)

        best_candidate = None
        best_rne = None
        best_eps = current_eps

        for candidate_set in exact_targets:
            evaluated_candidates += 1
            candidate_rne = algorithm_4_optimal_rne_sampling(users, candidate_set, current_price, system, cfg)

            if candidate_rne.epsilon + cfg.search_improvement_tol < best_eps:
                best_eps = candidate_rne.epsilon
                best_candidate = candidate_set
                best_rne = candidate_rne

        if best_candidate is None or best_candidate in visited:
            break  # No neighborhood fallback!

        visited.add(best_candidate)
        current_set = best_candidate
        current_price = best_rne.price

    final_stage2 = algorithm_2_heuristic_user_selection(users, current_price[0], current_price[1], system, cfg)
    stage2_oracle_calls += 1
    final_eps = compute_epsilon(users, final_stage2.offloading_set, current_price, system)

    return {
        "epsilon": final_eps,
        "runtime_sec": time.perf_counter() - t0,
        "stage2_oracle_calls": stage2_oracle_calls,
        "outer_iterations": outer_iterations,
        "evaluated_candidates": evaluated_candidates,
        "social_cost": final_stage2.social_cost,
    }


def run_random_restart(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
    max_restarts: int = 10,
) -> dict:
    """Run weak baseline: random restart local search."""
    stage2_oracle_calls = 0
    evaluated_candidates = 0

    t0 = time.perf_counter()

    best_eps = float('inf')
    best_price = (system.cE, system.cN)
    best_set = tuple()

    for restart in range(max_restarts):
        # Random starting price
        pE = random.uniform(system.cE * 1.1, system.cE * 5.0)
        pN = random.uniform(system.cN * 1.1, system.cN * 5.0)

        init_stage2 = algorithm_2_heuristic_user_selection(users, pE, pN, system, cfg)
        stage2_oracle_calls += 1
        current_set = init_stage2.offloading_set
        current_price = (pE, pN)

        current_eps = compute_epsilon(users, current_set, current_price, system)

        if current_eps < best_eps:
            best_eps = current_eps
            best_price = current_price
            best_set = current_set

        evaluated_candidates += 1

    return {
        "epsilon": best_eps,
        "runtime_sec": time.perf_counter() - t0,
        "stage2_oracle_calls": stage2_oracle_calls,
        "outer_iterations": max_restarts,
        "evaluated_candidates": evaluated_candidates,
        "social_cost": None,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "variant", "epsilon", "runtime_sec", "stage2_oracle_calls", "outer_iterations"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict], out_path: Path) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    fields = ["n_users", "variant", "epsilon_mean", "epsilon_std",
              "runtime_mean", "runtime_std", "stage2_calls_mean", "stage2_calls_std", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, variant) in sorted(grouped.keys()):
            bucket = grouped[(n_users, variant)]
            eps = [float(r["epsilon"]) for r in bucket]
            rt = [float(r["runtime_sec"]) for r in bucket]
            calls = [int(r["stage2_oracle_calls"]) for r in bucket]

            eps_mean, eps_std = _mean_std(eps)
            rt_mean, rt_std = _mean_std(rt)
            calls_mean, calls_std = _mean_std([float(c) for c in calls])

            row = {
                "n_users": n_users,
                "variant": variant,
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


def _plot_epsilon_comparison(summary_rows: list[dict], out_path: Path) -> None:
    """Plot final epsilon comparison across variants."""
    n_users_list = sorted(set(int(r["n_users"]) for r in summary_rows))
    variants = ["Full", "NoPrio", "NeighborhoodOnly", "DevTargetOnly", "RandomRestart"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    x = np.arange(len(n_users_list))
    width = 0.15

    colors = {
        "Full": "#1b5e20",
        "NoPrio": "#b71c1c",
        "NeighborhoodOnly": "#0d47a1",
        "DevTargetOnly": "#e65100",
        "RandomRestart": "#757575",
    }

    for i, variant in enumerate(variants):
        rows = [r for r in summary_rows if r["variant"] == variant]
        if not rows:
            continue

        values_map = {int(r["n_users"]): r for r in rows}
        y = [values_map.get(n, {}).get("epsilon_mean", 0) for n in n_users_list]
        yerr = [values_map.get(n, {}).get("epsilon_std", 0) for n in n_users_list]

        ax.bar(x + i * width, y, width, label=variant, color=colors[variant], alpha=0.8, yerr=yerr)

    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Final Deviation Gap (ε)")
    ax.set_title("Ablation: Algorithm 5 Variants - Final Deviation Gap Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(n_users_list)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, axis='y')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_runtime_comparison(summary_rows: list[dict], out_path: Path) -> None:
    """Plot runtime comparison across variants."""
    n_users_list = sorted(set(int(r["n_users"]) for r in summary_rows))
    variants = ["Full", "NoPrio", "NeighborhoodOnly", "DevTargetOnly", "RandomRestart"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    x = np.arange(len(n_users_list))
    width = 0.15

    colors = {
        "Full": "#1b5e20",
        "NoPrio": "#b71c1c",
        "NeighborhoodOnly": "#0d47a1",
        "DevTargetOnly": "#e65100",
        "RandomRestart": "#757575",
    }

    for i, variant in enumerate(variants):
        rows = [r for r in summary_rows if r["variant"] == variant]
        if not rows:
            continue

        values_map = {int(r["n_users"]): r for r in rows}
        y = [values_map.get(n, {}).get("runtime_mean", 0) for n in n_users_list]
        yerr = [values_map.get(n, {}).get("runtime_std", 0) for n in n_users_list]

        ax.bar(x + i * width, y, width, label=variant, color=colors[variant], alpha=0.8, yerr=yerr)

    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Ablation: Algorithm 5 Variants - Runtime Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(n_users_list)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, axis='y')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage2_calls_comparison(summary_rows: list[dict], out_path: Path) -> None:
    """Plot Stage-II oracle calls comparison."""
    n_users_list = sorted(set(int(r["n_users"]) for r in summary_rows))
    variants = ["Full", "NoPrio", "NeighborhoodOnly", "DevTargetOnly", "RandomRestart"]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    x = np.arange(len(n_users_list))
    width = 0.15

    colors = {
        "Full": "#1b5e20",
        "NoPrio": "#b71c1c",
        "NeighborhoodOnly": "#0d47a1",
        "DevTargetOnly": "#e65100",
        "RandomRestart": "#757575",
    }

    for i, variant in enumerate(variants):
        rows = [r for r in summary_rows if r["variant"] == variant]
        if not rows:
            continue

        values_map = {int(r["n_users"]): r for r in rows}
        y = [values_map.get(n, {}).get("stage2_calls_mean", 0) for n in n_users_list]

        ax.bar(x + i * width, y, width, label=variant, color=colors[variant], alpha=0.8)

    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Stage-II Oracle Calls")
    ax.set_title("Ablation: Algorithm 5 Variants - Stage-II Oracle Calls Comparison")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(n_users_list)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, axis='y')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: Algorithm 5 guided search variants")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="20,50,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20282001)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--max-restarts", type=int, default=10, help="Max restarts for random baseline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ablation_guided_search_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []

    variants = [
        ("Full", run_full_algorithm_5),
        ("NoPrio", run_no_prioritization),
        ("NeighborhoodOnly", run_neighborhood_only),
        ("DevTargetOnly", run_deviation_target_only),
    ]

    for n_users in n_users_list:
        trial_cfg = replace(cfg, n_users=n_users)

        for trial in range(args.trials):
            seed = args.seed + 1000 * n_users + trial
            rng = np.random.default_rng(seed)
            random.seed(seed)
            users = sample_users(trial_cfg, rng)

            for variant_name, variant_fn in variants:
                result = variant_fn(users, cfg.system, cfg.stackelberg)
                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "variant": variant_name,
                    "epsilon": result["epsilon"],
                    "runtime_sec": result["runtime_sec"],
                    "stage2_oracle_calls": result["stage2_oracle_calls"],
                    "outer_iterations": result["outer_iterations"],
                })

            # Random restart baseline
            result = run_random_restart(users, cfg.system, cfg.stackelberg, args.max_restarts)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "variant": "RandomRestart",
                "epsilon": result["epsilon"],
                "runtime_sec": result["runtime_sec"],
                "stage2_oracle_calls": result["stage2_oracle_calls"],
                "outer_iterations": result["outer_iterations"],
            })

    _write_raw_csv(raw_rows, run_dir / "raw_guided_search_ablation.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_guided_search_ablation.csv")
    _plot_epsilon_comparison(summary_rows, run_dir / "ablation_epsilon.png")
    _plot_runtime_comparison(summary_rows, run_dir / "ablation_runtime.png")
    _plot_stage2_calls_comparison(summary_rows, run_dir / "ablation_stage2_calls.png")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"max_restarts = {args.max_restarts}",
        "variants = Full, NoPrio, NeighborhoodOnly, DevTargetOnly, RandomRestart",
        "metrics = epsilon (REAL deviation gap), runtime, stage2_oracle_calls",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
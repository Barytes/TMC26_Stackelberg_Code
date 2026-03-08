#!/usr/bin/env python3
"""Stage I exploration: full-neighborhood search + gain-estimator calibration.

Outputs:
- Algorithm 3 approximation audit (boundary vs refined_price vs exact best response)
- Algorithm 5 comparison (baseline two_stage vs full_search variants)
- Aggregate metrics: iterations, final epsilon, distance to true-SE proxy, runtime
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.baselines import baseline_stage1_grid_search_oracle
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    _build_data,
    _candidate_family,
    _provider_revenue,
    _sorted_tuple,
    algorithm_3_gain_approximation,
    algorithm_5_stackelberg_guided_search,
)


def _exact_best_gain(users, cur_set, pE, pN, provider, system):
    data = _build_data(users)
    X = _sorted_tuple(cur_set)
    current_revenue = _provider_revenue(data, X, pE, pN, provider, system)
    best_gain = 0.0
    best_set = X
    for k in range(users.n + 1):
        for subset in combinations(range(users.n), k):
            Y = _sorted_tuple(subset)
            gain = _provider_revenue(data, Y, pE, pN, provider, system) - current_revenue
            if gain > best_gain:
                best_gain = float(gain)
                best_set = Y
    return best_gain, best_set


def _mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def _plot_metric(summary_rows, metric, out_path, title):
    labels = [r["variant"] for r in summary_rows]
    vals = [float(r[metric]) for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    bars = ax.bar(labels, vals, color=["#546e7a", "#1565c0", "#2e7d32"][: len(labels)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Stage I full-neighborhood exploration")
    ap.add_argument("--config", type=str, default="configs/default.toml")
    ap.add_argument("--n-users", type=str, default="8,12,16")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--audit-samples-per-trial", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260309)
    ap.add_argument("--run-name", type=str, default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        run_name = f"stage1_fullsearch_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline_two_stage_boundary", "two_stage", "boundary"),
        ("full_search_boundary", "full_search", "boundary"),
        ("full_search_refined", "full_search", "refined_price"),
    ]

    audit_rows = []
    comp_rows = []

    for n_users in n_users_list:
        trial_cfg = replace(cfg, n_users=n_users)
        for trial in range(args.trials):
            seed = args.seed + 10000 * n_users + trial
            rng = np.random.default_rng(seed)
            users = sample_users(trial_cfg, rng)

            # true SE proxy (same across variants)
            se = baseline_stage1_grid_search_oracle(users, cfg.system, cfg.stackelberg, cfg.baselines).price

            # Algorithm-3 audit samples
            for _ in range(args.audit_samples_per_trial):
                pE = float(rng.uniform(cfg.system.cE * 1.05, cfg.system.cE * 3.0))
                pN = float(rng.uniform(cfg.system.cN * 1.05, cfg.system.cN * 3.0))
                prob = float(rng.uniform(0.2, 0.8))
                cur = tuple(i for i in range(n_users) if rng.random() < prob)
                fam = _candidate_family(_build_data(users), _sorted_tuple(cur), pE, pN, cfg.system)

                for provider in ["E", "N"]:
                    exact_gain, exact_set = _exact_best_gain(users, cur, pE, pN, provider, cfg.system)
                    for estimator in ["boundary", "refined_price"]:
                        approx = algorithm_3_gain_approximation(
                            users, cur, pE, pN, provider, cfg.system, estimator_variant=estimator
                        )
                        audit_rows.append({
                            "n_users": n_users,
                            "trial": trial,
                            "provider": provider,
                            "estimator": estimator,
                            "pE": pE,
                            "pN": pN,
                            "approx_gain": approx.gain,
                            "exact_gain": exact_gain,
                            "abs_error": abs(approx.gain - exact_gain),
                            "rel_error": abs(approx.gain - exact_gain) / max(exact_gain, 1e-8),
                            "family_size": len(fam),
                            "family_hit": int(exact_set in fam),
                            "exact_best_set": str(exact_set),
                            "approx_best_set": str(approx.best_set),
                        })

            for name, mode, estimator in variants:
                stack_cfg = replace(
                    cfg.stackelberg,
                    stage1_neighborhood_mode=mode,
                    gain_estimator_variant=estimator,
                )
                t0 = time.perf_counter()
                out = algorithm_5_stackelberg_guided_search(users, cfg.system, stack_cfg)
                rt = time.perf_counter() - t0
                dist = math.sqrt((out.price[0] - se[0]) ** 2 + (out.price[1] - se[1]) ** 2)
                comp_rows.append({
                    "n_users": n_users,
                    "trial": trial,
                    "variant": name,
                    "iterations": out.outer_iterations,
                    "final_epsilon": out.epsilon,
                    "dist_to_true_se": dist,
                    "runtime_sec": rt,
                    "evaluated_candidates": out.evaluated_candidates,
                    "final_pE": out.price[0],
                    "final_pN": out.price[1],
                    "stopping_reason": out.stopping_reason,
                })

    with (run_dir / "audit_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader(); w.writerows(audit_rows)

    with (run_dir / "comparison_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
        w.writeheader(); w.writerows(comp_rows)

    # audit summary
    audit_summary = []
    for estimator in ["boundary", "refined_price"]:
        rows = [r for r in audit_rows if r["estimator"] == estimator]
        audit_summary.append({
            "estimator": estimator,
            "count": len(rows),
            "abs_error_mean": _mean([r["abs_error"] for r in rows]),
            "rel_error_mean": _mean([r["rel_error"] for r in rows if r["exact_gain"] > 1e-8]),
            "family_hit_rate": _mean([r["family_hit"] for r in rows]),
            "abs_error_p90": float(np.percentile([r["abs_error"] for r in rows], 90)) if rows else 0.0,
        })

    with (run_dir / "audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit_summary[0].keys()))
        w.writeheader(); w.writerows(audit_summary)

    # comparison summary
    summary = []
    for name, _, _ in variants:
        rows = [r for r in comp_rows if r["variant"] == name]
        summary.append({
            "variant": name,
            "count": len(rows),
            "iterations_mean": _mean([r["iterations"] for r in rows]),
            "final_epsilon_mean": _mean([r["final_epsilon"] for r in rows]),
            "dist_to_true_se_mean": _mean([r["dist_to_true_se"] for r in rows]),
            "runtime_sec_mean": _mean([r["runtime_sec"] for r in rows]),
            "evaluated_candidates_mean": _mean([r["evaluated_candidates"] for r in rows]),
        })

    with (run_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    _plot_metric(summary, "final_epsilon_mean", run_dir / "compare_final_epsilon.png", "Stage I: Final epsilon")
    _plot_metric(summary, "dist_to_true_se_mean", run_dir / "compare_dist_to_true_se.png", "Stage I: Distance to true-SE proxy")
    _plot_metric(summary, "runtime_sec_mean", run_dir / "compare_runtime.png", "Stage I: Runtime")
    _plot_metric(summary, "iterations_mean", run_dir / "compare_iterations.png", "Stage I: Iterations")

    (run_dir / "run_meta.txt").write_text(
        "\n".join([
            f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
            f"config = {args.config}",
            f"n_users = {n_users_list}",
            f"trials = {args.trials}",
            f"audit_samples_per_trial = {args.audit_samples_per_trial}",
            f"seed = {args.seed}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"Done. Results: {run_dir}")


if __name__ == "__main__":
    main()

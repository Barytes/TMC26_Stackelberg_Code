#!/usr/bin/env python3
"""Algorithm 2 social-cost gap vs the exact centralized optimum.

For each sampled instance, the script computes:

- `V_algorithm2`: social cost returned by Algorithm 2
- `V_optimal`: exact optimal social cost from the centralized solver
- `paper_gap_pct_signed = (V_optimal - V_algorithm2) / V_optimal * 100`
- `relative_error_pct = (V_algorithm2 - V_optimal) / V_optimal * 100`

The second quantity is the conventional nonnegative relative error used for
plotting, while the first follows the sign convention stated in the request.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from _stage2_metric_helpers import mean_std, parse_user_counts, run_stage2_methods

ROOT = SCRIPT_DIR.parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial",
        "seed",
        "n_users",
        "algorithm2_social_cost",
        "optimal_social_cost",
        "paper_gap_pct_signed",
        "relative_error_pct",
        "algorithm2_offloading_size",
        "optimal_offloading_size",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["n_users"]), []).append(row)

    fields = [
        "n_users",
        "paper_gap_pct_signed_mean",
        "paper_gap_pct_signed_std",
        "relative_error_pct_mean",
        "relative_error_pct_std",
        "algorithm2_social_cost_mean",
        "optimal_social_cost_mean",
        "count",
    ]
    summary_rows: list[dict[str, float | int]] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n_users in sorted(grouped):
            bucket = grouped[n_users]
            paper_gap = [float(x["paper_gap_pct_signed"]) for x in bucket]
            rel_err = [float(x["relative_error_pct"]) for x in bucket]
            alg2_vals = [float(x["algorithm2_social_cost"]) for x in bucket]
            opt_vals = [float(x["optimal_social_cost"]) for x in bucket]
            paper_gap_m, paper_gap_s = mean_std(paper_gap)
            rel_err_m, rel_err_s = mean_std(rel_err)
            alg2_m, _ = mean_std(alg2_vals)
            opt_m, _ = mean_std(opt_vals)
            row = {
                "n_users": n_users,
                "paper_gap_pct_signed_mean": paper_gap_m,
                "paper_gap_pct_signed_std": paper_gap_s,
                "relative_error_pct_mean": rel_err_m,
                "relative_error_pct_std": rel_err_s,
                "algorithm2_social_cost_mean": alg2_m,
                "optimal_social_cost_mean": opt_m,
                "count": len(bucket),
            }
            summary_rows.append(row)
            writer.writerow(row)
    return summary_rows


def _plot(summary_rows: list[dict[str, float | int]], out_path: Path) -> None:
    x = [int(row["n_users"]) for row in summary_rows]
    y = [float(row["relative_error_pct_mean"]) for row in summary_rows]
    yerr = [float(row["relative_error_pct_std"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        color="#1b5e20",
        linewidth=2.0,
        capsize=4,
        label="Algorithm 2 vs CS",
    )
    ax.set_title("Algorithm 2 Social-cost Relative Error vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Relative Error (%)")
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 2 social-cost gap vs centralized optimum")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument(
        "--user-counts",
        type=str,
        default="6,8,10,12,14,16",
        help="Use only user counts within exact_max_users, since this script requires CS at every point.",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20266000)
    parser.add_argument("--pE", type=float, default=None)
    parser.add_argument("--pN", type=float, default=None)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")

    cfg = load_config(args.config)
    user_counts = parse_user_counts(args.user_counts)
    if any(n > cfg.baselines.exact_max_users for n in user_counts):
        raise ValueError(
            f"All user counts must be <= exact_max_users ({cfg.baselines.exact_max_users}) for this script."
        )
    pE = float(args.pE if args.pE is not None else cfg.stackelberg.initial_pE)
    pN = float(args.pN if args.pN is not None else cfg.stackelberg.initial_pN)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"algorithm2_social_cost_gap_vs_users_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    for n_users in user_counts:
        trial_cfg = replace(cfg, n_users=n_users)
        for trial in range(args.trials):
            seed = args.seed + 1000 * n_users + trial
            rng = np.random.default_rng(seed)
            users = sample_users(trial_cfg, rng)
            runs = run_stage2_methods(users, pE, pN, cfg)
            alg2 = runs["Algorithm2"]
            cs = runs["CS"]
            v_opt = max(cs.social_cost, 1e-12)
            paper_gap = (cs.social_cost - alg2.social_cost) / v_opt * 100.0
            rel_err = (alg2.social_cost - cs.social_cost) / v_opt * 100.0
            raw_rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "algorithm2_social_cost": alg2.social_cost,
                    "optimal_social_cost": cs.social_cost,
                    "paper_gap_pct_signed": paper_gap,
                    "relative_error_pct": rel_err,
                    "algorithm2_offloading_size": alg2.offloading_size,
                    "optimal_offloading_size": cs.offloading_size,
                }
            )

    _write_raw_csv(raw_rows, run_dir / "raw_algorithm2_social_cost_gap_vs_users.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_algorithm2_social_cost_gap_vs_users.csv")
    _plot(summary_rows, run_dir / "algorithm2_social_cost_gap_vs_users.png")

    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"user_counts = {user_counts}",
        f"trials = {args.trials}",
        f"exact_max_users = {cfg.baselines.exact_max_users}",
        "paper_gap_pct_signed = (V_optimal - V_algorithm2) / V_optimal * 100",
        "relative_error_pct = (V_algorithm2 - V_optimal) / V_optimal * 100",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

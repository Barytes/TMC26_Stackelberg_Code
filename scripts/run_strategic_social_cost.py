#!/usr/bin/env python3
"""Strategic Settings: User social cost vs number of users.

This script compares user social cost across different strategic settings:
- GSSE (proposed): Algorithm 5 Stackelberg search
- MarketEquilibrium: Tatonnement-style price adjustment
- SingleSP: Single leader (ESP only)
- RandomOffloading: Random baseline

Output: Line plot comparing social cost vs number of users.
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
from tmc26_exp.baselines import (
    baseline_market_equilibrium,
    baseline_single_sp,
    baseline_random_offloading,
)
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search


def run_gsse(users, system, stackelberg_cfg):
    """Run GSSE (proposed method) - Algorithm 5."""
    result = algorithm_5_stackelberg_guided_search(users, system, stackelberg_cfg)
    return result


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "social_cost", "offloading_size"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by (n_users, method)."""
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    fields = ["n_users", "method", "social_cost_mean", "social_cost_std", "count"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, method) in sorted(grouped.keys()):
            bucket = grouped[(n_users, method)]
            costs = [float(x["social_cost"]) for x in bucket]
            cost_mean, cost_std = _mean_std(costs)
            row = {
                "n_users": n_users,
                "method": method,
                "social_cost_mean": cost_mean,
                "social_cost_std": cost_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot social cost comparison."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    methods = ["GSSE", "MarketEquilibrium", "SingleSP", "RandomOffloading"]
    colors = {
        "GSSE": "#1b5e20",
        "MarketEquilibrium": "#0d47a1",
        "SingleSP": "#e65100",
        "RandomOffloading": "#757575",
    }
    markers = {
        "GSSE": "o",
        "MarketEquilibrium": "^",
        "SingleSP": "s",
        "RandomOffloading": "x",
    }
    labels = {
        "GSSE": "GSSE (Proposed)",
        "MarketEquilibrium": "Market Equilibrium",
        "SingleSP": "Single SP",
        "RandomOffloading": "Random",
    }

    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))
        x = [int(r["n_users"]) for r in rows]
        y = [float(r["social_cost_mean"]) for r in rows]
        yerr = [float(r["social_cost_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=markers[method],
            color=colors[method],
            linewidth=2.0,
            capsize=4,
            label=labels[method],
            markersize=8,
        )

    ax.set_title("Strategic Settings: User Social Cost vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Social Cost")
    ax.set_xticks(n_users_list)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic settings: user social cost comparison")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=str, default="20,50,100,200")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20271001)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"strategic_social_cost_{timestamp}"
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

            # GSSE (Algorithm 5)
            gsse = run_gsse(users, cfg.system, cfg.stackelberg)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "GSSE",
                "social_cost": gsse.social_cost,
                "offloading_size": len(gsse.offloading_set),
            })

            # Market Equilibrium
            me = baseline_market_equilibrium(users, cfg.system, cfg.stackelberg, cfg.baselines)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "MarketEquilibrium",
                "social_cost": me.social_cost,
                "offloading_size": len(me.offloading_set),
            })

            # Single SP (ESP leader only)
            ssp = baseline_single_sp(users, cfg.system, cfg.stackelberg, cfg.baselines)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "SingleSP",
                "social_cost": ssp.social_cost,
                "offloading_size": len(ssp.offloading_set),
            })

            # Random Offloading
            rand = baseline_random_offloading(users, cfg.system, cfg.stackelberg, cfg.baselines)
            raw_rows.append({
                "trial": trial,
                "seed": seed,
                "n_users": n_users,
                "method": "RandomOffloading",
                "social_cost": rand.social_cost,
                "offloading_size": len(rand.offloading_set),
            })

    _write_raw_csv(raw_rows, run_dir / "raw_strategic_social_cost.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_strategic_social_cost.csv")
    _plot(summary_rows, run_dir / "strategic_social_cost.png", n_users_list)

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        "methods = GSSE, MarketEquilibrium, SingleSP, RandomOffloading",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
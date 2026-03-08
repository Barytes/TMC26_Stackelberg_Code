#!/usr/bin/env python3
"""Strategic Settings: Pareto tradeoff scatter plot.

This script creates a scatter plot showing the tradeoff between:
- X-axis: Social cost (user welfare proxy)
- Y-axis: Joint provider revenue

Each point represents one method on one trial instance.

Output: Scatter plot with Pareto frontier visualization.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
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


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "social_cost", "joint_revenue"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_scatter(rows: list[dict], out_path: Path) -> None:
    """Plot Pareto tradeoff scatter."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

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
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        x = [float(r["social_cost"]) for r in method_rows]
        y = [float(r["joint_revenue"]) for r in method_rows]

        ax.scatter(
            x, y,
            c=colors[method],
            marker=markers[method],
            alpha=0.6,
            s=40,
            label=labels[method],
            zorder=3,
        )

    # Find and plot approximate Pareto frontier
    all_points = [(r["social_cost"], r["joint_revenue"]) for r in rows]
    if all_points:
        # Sort by social cost
        sorted_points = sorted(set(all_points), key=lambda p: p[0])
        pareto_front = []
        max_revenue = -float('inf')
        for sc, rev in sorted_points:
            if rev > max_revenue:
                pareto_front.append((sc, rev))
                max_revenue = rev

        if len(pareto_front) > 1:
            pareto_x = [p[0] for p in pareto_front]
            pareto_y = [p[1] for p in pareto_front]
            ax.plot(pareto_x, pareto_y, 'k--', linewidth=1.5, alpha=0.7,
                    label='Pareto Frontier', zorder=2)

    ax.set_title("Strategic Settings: Pareto Tradeoff\n(User Social Cost vs Joint Provider Revenue)")
    ax.set_xlabel("Social Cost (lower is better for users)")
    ax.set_ylabel("Joint Revenue (higher is better for providers)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_with_density(rows: list[dict], out_path: Path) -> None:
    """Plot with density contours for each method."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

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
        method_rows = [r for r in rows if r["method"] == method]
        if not method_rows:
            continue
        x = [float(r["social_cost"]) for r in method_rows]
        y = [float(r["joint_revenue"]) for r in method_rows]

        # Plot scatter
        ax.scatter(
            x, y,
            c=colors[method],
            marker=markers[method],
            alpha=0.5,
            s=30,
            label=labels[method],
            zorder=3,
        )

        # Add convex hull for visibility
        if len(x) >= 3:
            from scipy.spatial import ConvexHull
            points = np.array(list(zip(x, y)))
            try:
                hull = ConvexHull(points)
                hull_points = np.append(hull.vertices, hull.vertices[0])
                ax.fill(points[hull_points, 0], points[hull_points, 1],
                       color=colors[method], alpha=0.1, zorder=1)
            except:
                pass

    ax.set_title("Strategic Settings: Pareto Tradeoff Analysis\n(with convex hull for each method)")
    ax.set_xlabel("Social Cost (lower is better)")
    ax.set_ylabel("Joint Revenue (higher is better)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic settings: Pareto tradeoff scatter")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20272001)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_users = args.n_users

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"strategic_pareto_tradeoff_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []

    trial_cfg = replace(cfg, n_users=n_users)
    for trial in range(args.trials):
        seed = args.seed + trial
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
            "joint_revenue": gsse.esp_revenue + gsse.nsp_revenue,
        })

        # Market Equilibrium
        me = baseline_market_equilibrium(users, cfg.system, cfg.stackelberg, cfg.baselines)
        raw_rows.append({
            "trial": trial,
            "seed": seed,
            "n_users": n_users,
            "method": "MarketEquilibrium",
            "social_cost": me.social_cost,
            "joint_revenue": me.esp_revenue + me.nsp_revenue,
        })

        # Single SP
        ssp = baseline_single_sp(users, cfg.system, cfg.stackelberg, cfg.baselines)
        raw_rows.append({
            "trial": trial,
            "seed": seed,
            "n_users": n_users,
            "method": "SingleSP",
            "social_cost": ssp.social_cost,
            "joint_revenue": ssp.esp_revenue + ssp.nsp_revenue,
        })

        # Random Offloading
        rand = baseline_random_offloading(users, cfg.system, cfg.stackelberg, cfg.baselines)
        raw_rows.append({
            "trial": trial,
            "seed": seed,
            "n_users": n_users,
            "method": "RandomOffloading",
            "social_cost": rand.social_cost,
            "joint_revenue": rand.esp_revenue + rand.nsp_revenue,
        })

    _write_raw_csv(raw_rows, run_dir / "raw_pareto_tradeoff.csv")
    _plot_scatter(raw_rows, run_dir / "pareto_tradeoff.png")

    # Try to import scipy for density plot
    try:
        _plot_with_density(raw_rows, run_dir / "pareto_tradeoff_density.png")
    except ImportError:
        print("  (scipy not available, skipping density plot)")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {n_users}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        "methods = GSSE, MarketEquilibrium, SingleSP, RandomOffloading",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
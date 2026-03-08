#!/usr/bin/env python3
"""Strategic Settings: Revenue and Utilization vs F/B capacity ratio.

This script analyzes sensitivity of provider revenue and capacity utilization
to different F (computation) and B (bandwidth) capacity values.

Output: Heatmaps and line plots showing revenue/utilization vs F/B ratio.
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

from tmc26_exp.config import load_config, SystemConfig
from tmc26_exp.simulator import sample_users
from tmc26_exp.baselines import baseline_market_equilibrium
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search


def run_gsse_with_custom_system(users, system_config, stackelberg_cfg):
    """Run GSSE with custom system configuration."""
    from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search
    result = algorithm_5_stackelberg_guided_search(users, system_config, stackelberg_cfg)
    return result


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "trial", "seed", "F", "B", "F_B_ratio",
        "joint_revenue", "esp_revenue", "nsp_revenue",
        "utilization_F", "utilization_B",
        "social_cost", "offloading_size"
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot_revenue_heatmap(rows: list[dict], out_path: Path) -> None:
    """Plot revenue heatmap as function of F and B."""
    # Group by F and B
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        key = (float(row["F"]), float(row["B"]))
        grouped.setdefault(key, []).append(row)

    # Get unique F and B values
    F_values = sorted(set(k[0] for k in grouped.keys()))
    B_values = sorted(set(k[1] for k in grouped.keys()))

    # Create grid
    revenue_grid = np.zeros((len(B_values), len(F_values)))
    for i, B in enumerate(B_values):
        for j, F in enumerate(F_values):
            if (F, B) in grouped:
                bucket = grouped[(F, B)]
                revenues = [float(r["joint_revenue"]) for r in bucket]
                revenue_grid[i, j] = statistics.fmean(revenues)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    F_grid, B_grid = np.meshgrid(F_values, B_values)

    im = ax.pcolormesh(F_grid, B_grid, revenue_grid, cmap="viridis", shading='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Joint Revenue")

    ax.set_title("Joint Revenue vs F and B Capacity")
    ax.set_xlabel("F (Computation Capacity)")
    ax.set_ylabel("B (Bandwidth Capacity)")
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_utilization_heatmap(rows: list[dict], out_path: Path) -> None:
    """Plot utilization heatmap."""
    grouped: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        key = (float(row["F"]), float(row["B"]))
        grouped.setdefault(key, []).append(row)

    F_values = sorted(set(k[0] for k in grouped.keys()))
    B_values = sorted(set(k[1] for k in grouped.keys()))

    util_F_grid = np.zeros((len(B_values), len(F_values)))
    util_B_grid = np.zeros((len(B_values), len(F_values)))

    for i, B in enumerate(B_values):
        for j, F in enumerate(F_values):
            if (F, B) in grouped:
                bucket = grouped[(F, B)]
                util_F = [float(r["utilization_F"]) for r in bucket]
                util_B = [float(r["utilization_B"]) for r in bucket]
                util_F_grid[i, j] = statistics.fmean(util_F)
                util_B_grid[i, j] = statistics.fmean(util_B)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    F_grid, B_grid = np.meshgrid(F_values, B_values)

    # F utilization
    im1 = axes[0].pcolormesh(F_grid, B_grid, util_F_grid * 100, cmap="Blues", shading='auto')
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label("Utilization (%)")
    axes[0].set_title("Computation (F) Utilization")
    axes[0].set_xlabel("F")
    axes[0].set_ylabel("B")
    axes[0].set_aspect('equal', adjustable='box')

    # B utilization
    im2 = axes[1].pcolormesh(F_grid, B_grid, util_B_grid * 100, cmap="Oranges", shading='auto')
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label("Utilization (%)")
    axes[1].set_title("Bandwidth (B) Utilization")
    axes[1].set_xlabel("F")
    axes[1].set_ylabel("B")
    axes[1].set_aspect('equal', adjustable='box')

    fig.suptitle("Capacity Utilization vs F/B", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_fb_ratio_sensitivity(rows: list[dict], out_path: Path) -> None:
    """Plot metrics vs F/B ratio."""
    # Group by F_B_ratio
    grouped: dict[float, list[dict]] = {}
    for row in rows:
        ratio = float(row["F_B_ratio"])
        grouped.setdefault(ratio, []).append(row)

    ratios = sorted(grouped.keys())
    revenue_means = []
    revenue_stds = []
    util_F_means = []
    util_B_means = []

    for ratio in ratios:
        bucket = grouped[ratio]
        revs = [float(r["joint_revenue"]) for r in bucket]
        util_F = [float(r["utilization_F"]) for r in bucket]
        util_B = [float(r["utilization_B"]) for r in bucket]

        rev_mean, rev_std = _mean_std(revs)
        revenue_means.append(rev_mean)
        revenue_stds.append(rev_std)
        util_F_means.append(statistics.fmean(util_F) * 100)
        util_B_means.append(statistics.fmean(util_B) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Revenue vs F/B ratio
    axes[0].errorbar(ratios, revenue_means, yerr=revenue_stds, marker='o',
                     color='#1b5e20', linewidth=2.0, capsize=4)
    axes[0].set_title("Joint Revenue vs F/B Ratio")
    axes[0].set_xlabel("F/B Ratio")
    axes[0].set_ylabel("Joint Revenue")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # Utilization vs F/B ratio
    axes[1].plot(ratios, util_F_means, marker='o', color='#0d47a1',
                 linewidth=2.0, label='F Utilization')
    axes[1].plot(ratios, util_B_means, marker='s', color='#e65100',
                 linewidth=2.0, label='B Utilization')
    axes[1].set_title("Capacity Utilization vs F/B Ratio")
    axes[1].set_xlabel("F/B Ratio")
    axes[1].set_ylabel("Utilization (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic settings: F/B capacity sensitivity")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--F-values", type=str, default="50,100,200")
    parser.add_argument("--B-values", type=str, default="20,40,80")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20273001)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    F_values = [int(x.strip()) for x in args.F_values.split(",") if x.strip()]
    B_values = [int(x.strip()) for x in args.B_values.split(",") if x.strip()]
    n_users = args.n_users

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"strategic_fb_sensitivity_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []

    # Base system configuration
    base_F = cfg.system.F
    base_B = cfg.system.B

    trial_cfg = replace(cfg, n_users=n_users)

    for F in F_values:
        for B in B_values:
            # Create custom system config
            custom_system = SystemConfig(
                F=F,
                B=B,
                cE=cfg.system.cE,
                cN=cfg.system.cN,
            )

            for trial in range(args.trials):
                seed = args.seed + 1000 * (F * 100 + B) + trial
                rng = np.random.default_rng(seed)
                users = sample_users(trial_cfg, rng)

                # Run GSSE with custom system
                result = run_gsse_with_custom_system(users, custom_system, cfg.stackelberg)

                # Compute utilization
                inner = result.inner_result
                total_f = float(np.sum(inner.f))
                total_b = float(np.sum(inner.b))
                util_F = total_f / F if F > 0 else 0
                util_B = total_b / B if B > 0 else 0

                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "F": F,
                    "B": B,
                    "F_B_ratio": F / B,
                    "joint_revenue": result.esp_revenue + result.nsp_revenue,
                    "esp_revenue": result.esp_revenue,
                    "nsp_revenue": result.nsp_revenue,
                    "utilization_F": util_F,
                    "utilization_B": util_B,
                    "social_cost": result.social_cost,
                    "offloading_size": len(result.offloading_set),
                })

    _write_raw_csv(raw_rows, run_dir / "raw_fb_sensitivity.csv")
    _plot_revenue_heatmap(raw_rows, run_dir / "fb_revenue_heatmap.png")
    _plot_utilization_heatmap(raw_rows, run_dir / "fb_utilization_heatmap.png")
    _plot_fb_ratio_sensitivity(raw_rows, run_dir / "fb_ratio_sensitivity.png")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"F_values = {F_values}",
        f"B_values = {B_values}",
        f"n_users = {n_users}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
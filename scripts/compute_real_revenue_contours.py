"""
Compute real ESP/NSP revenue contours using DG solver for Stage II user responses.

This script evaluates the actual revenue at each (pE, pN) price point by solving
the Stage II user offloading game with the DG (Algorithm 2 - Distributed Greedy)
baseline, then computing the realized revenue from actual resource allocations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tmc26_exp.baselines import run_stage2_solver
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.plotting import plot_surface_contour, plot_surface_heatmap


def evaluate_real_revenue_surface(
    config,
    revenue_type: str,  # "esp" or "nsp"
):
    """
    Evaluate real revenue surface using DG solver for Stage II.

    For each (pE, pN) point, runs the DG solver to get actual user offloading
    decisions and resource allocations, then computes the realized revenue.
    """
    rng = np.random.default_rng(config.seed)
    pE_values = np.linspace(config.pE.min, config.pE.max, config.pE.points)
    pN_values = np.linspace(config.pN.min, config.pN.max, config.pN.points)

    values_trials = np.zeros((config.n_trials, pN_values.size, pE_values.size), dtype=float)

    for t in range(config.n_trials):
        print(f"  Trial {t+1}/{config.n_trials}")
        users = sample_users(config, rng)

        for n_idx, pN in enumerate(pN_values):
            for e_idx, pE in enumerate(pE_values):
                # Use DG solver to get actual user response
                outcome = run_stage2_solver(
                    "DG",
                    users,
                    float(pE),
                    float(pN),
                    config.system,
                    config.stackelberg,
                    config.baselines,
                )

                # Extract actual revenue
                if revenue_type == "esp":
                    values_trials[t, n_idx, e_idx] = outcome.esp_revenue
                else:  # nsp
                    values_trials[t, n_idx, e_idx] = outcome.nsp_revenue

    # Create surface result similar to MetricSurface
    class RealRevenueSurface:
        def __init__(self, name, label, pE_values, pN_values, mean_values, std_values):
            self.metric_name = name
            self.metric_label = label
            self.pE_values = pE_values
            self.pN_values = pN_values
            self.mean_values = mean_values
            self.std_values = std_values

    return RealRevenueSurface(
        metric_name=f"{revenue_type}_real_revenue",
        metric_label=f"{'ESP' if revenue_type == 'esp' else 'NSP'} Real Revenue (DG Solver)",
        pE_values=pE_values,
        pN_values=pN_values,
        mean_values=np.mean(values_trials, axis=0),
        std_values=np.std(values_trials, axis=0),
    )


def save_surface_csv(surface, out_path: Path) -> None:
    """Save metric surface to CSV."""
    rows: list[str] = ["pE,pN,value_mean,value_std"]
    for n_idx, pN in enumerate(surface.pN_values):
        for e_idx, pE in enumerate(surface.pE_values):
            rows.append(
                f"{pE:.10g},{pN:.10g},{surface.mean_values[n_idx, e_idx]:.10g},{surface.std_values[n_idx, e_idx]:.10g}"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compute real revenue contours using CS solver"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/real_revenue_contours",
        help="Output directory (default: outputs/real_revenue_contours)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing real revenue surfaces...")
    print(f"  pE range: [{cfg.pE.min}, {cfg.pE.max}], {cfg.pE.points} points")
    print(f"  pN range: [{cfg.pN.min}, {cfg.pN.max}], {cfg.pN.points} points")
    print(f"  Trials: {cfg.n_trials}")
    print(f"  Users: {cfg.n_users}")
    print()

    # Compute ESP real revenue
    print("Computing ESP real revenue...")
    esp_surface = evaluate_real_revenue_surface(cfg, "esp")
    save_surface_csv(esp_surface, run_dir / "esp_real_revenue.csv")
    plot_surface_heatmap(esp_surface, run_dir / "esp_real_revenue_heatmap.png")
    plot_surface_contour(esp_surface, run_dir / "esp_real_revenue_contour.png")
    print(f"  ESP revenue range: [{esp_surface.mean_values.min():.4f}, {esp_surface.mean_values.max():.4f}]")
    print()

    # Compute NSP real revenue
    print("Computing NSP real revenue...")
    nsp_surface = evaluate_real_revenue_surface(cfg, "nsp")
    save_surface_csv(nsp_surface, run_dir / "nsp_real_revenue.csv")
    plot_surface_heatmap(nsp_surface, run_dir / "nsp_real_revenue_heatmap.png")
    plot_surface_contour(nsp_surface, run_dir / "nsp_real_revenue_contour.png")
    print(f"  NSP revenue range: [{nsp_surface.mean_values.min():.4f}, {nsp_surface.mean_values.max():.4f}]")
    print()

    # Write summary
    summary_text = f"""Real Revenue Contours (CS Solver)
=================================

Configuration:
  Config file: {args.config}
  Output directory: {run_dir}
  pE range: [{cfg.pE.min}, {cfg.pE.max}], {cfg.pE.points} points
  pN range: [{cfg.pN.min}, {cfg.pN.max}], {cfg.pN.points} points
  Trials: {cfg.n_trials}
  Users per trial: {cfg.n_users}
  Seed: {cfg.seed}

Results:
  ESP real revenue range: [{esp_surface.mean_values.min():.6f}, {esp_surface.mean_values.max():.6f}]
  NSP real revenue range: [{nsp_surface.mean_values.min():.6f}, {nsp_surface.mean_values.max():.6f}]

Files generated:
  - esp_real_revenue.csv
  - esp_real_revenue_heatmap.png
  - esp_real_revenue_contour.png
  - nsp_real_revenue.csv
  - nsp_real_revenue_heatmap.png
  - nsp_real_revenue_contour.png

Note: These contours show REAL revenue computed using the DG (Algorithm 2 - Distributed Greedy)
baseline for Stage II user responses. Unlike the "potential revenue" proxies in the
default run, these values account for actual user offloading decisions and capacity
constraints. The DG solver provides near-optimal solutions much faster than CS.
"""

    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

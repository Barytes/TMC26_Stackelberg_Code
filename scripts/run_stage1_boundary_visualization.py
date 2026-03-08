#!/usr/bin/env python3
"""Stage I: Boundary visualization with price trajectory overlay.

This script:
1. Loads or computes revenue contours on the (pE, pN) plane
2. Computes deviation gap surfaces
3. Runs Algorithm 5 to get price trajectory
4. Plots the combined deviation gap contour with trajectory overlay

Output: Contour plot with equilibrium trajectory marked.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search, _build_data, _margin_for_user
from tmc26_exp.baselines import (
    baseline_stage1_pbdr,
    baseline_stage1_bo,
    baseline_stage1_drl,
    baseline_stage1_grid_search_oracle,
    _stage2_solver,
)


def load_revenue_surface(csv_path: Path):
    """Load revenue surface from CSV file.

    CSV format: pE,pN,value_mean,value_std
    Returns: (pE_values, pN_values, revenue_values)
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    pE_values = np.unique(data[:, 0])
    pN_values = np.unique(data[:, 1])
    revenue = data[:, 2].reshape(pN_values.size, pE_values.size)
    return pE_values, pN_values, revenue


def compute_deviation_gaps(esp_csv: Path, nsp_csv: Path):
    """Compute deviation gaps from existing revenue contours.

    Returns: (pE_values, pN_values, esp_gap, nsp_gap, combined_gap)
    """
    pE_values, pN_values, esp_revenue = load_revenue_surface(esp_csv)
    _, _, nsp_revenue = load_revenue_surface(nsp_csv)

    # ESP deviation gap: for each pN, find max ESP revenue over all pE
    esp_max_per_pN = np.max(esp_revenue, axis=1, keepdims=True)
    esp_gap = esp_max_per_pN - esp_revenue

    # NSP deviation gap: for each pE, find max NSP revenue over all pN
    nsp_max_per_pE = np.max(nsp_revenue, axis=0, keepdims=True)
    nsp_gap = nsp_max_per_pE - nsp_revenue

    # Combined gap: max of both
    combined_gap = np.maximum(esp_gap, nsp_gap)

    return pE_values, pN_values, esp_gap, nsp_gap, combined_gap


def run_and_get_trajectory(users, system, stackelberg_cfg):
    """Run Algorithm 5 and extract price trajectory."""
    result = algorithm_5_stackelberg_guided_search(users, system, stackelberg_cfg)

    # Extract trajectory points
    trajectory = []
    for step in result.trajectory:
        trajectory.append({
            "iteration": step.iteration,
            "pE": step.pE,
            "pN": step.pN,
            "epsilon": step.epsilon,
            "dist_to_se": getattr(step, "dist_to_se", float("nan")),
            "epsilon_delta": getattr(step, "epsilon_delta", float("nan")),
        })

    # Add final point
    trajectory.append({
        "iteration": len(result.trajectory),
        "pE": result.price[0],
        "pN": result.price[1],
        "epsilon": result.epsilon,
        "dist_to_se": trajectory[-1]["dist_to_se"] if trajectory else float("nan"),
        "epsilon_delta": float("nan"),
    })

    return trajectory, result


def _extract_upper_boundary_mask(data, offloading_set: tuple[int, ...], pE_values: np.ndarray, pN_values: np.ndarray, system) -> np.ndarray:
    """Upper boundary mask for fixed offloading set: min_i m_i(p, X) ~= 0 from positive side."""
    if not offloading_set:
        return np.zeros((pN_values.size, pE_values.size), dtype=bool)
    tol = 1e-2
    mask = np.zeros((pN_values.size, pE_values.size), dtype=bool)
    for yi, pN in enumerate(pN_values):
        for xi, pE in enumerate(pE_values):
            margins = [
                _margin_for_user(data, offloading_set, i, float(pE), float(pN), system)
                for i in offloading_set
            ]
            m_min = min(margins)
            if 0.0 <= m_min <= tol:
                mask[yi, xi] = True
    return mask


def _pbdr_trajectory(users, system, stack_cfg, base_cfg):
    traj = []
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    traj.append((float(pE), float(pN)))
    for _ in range(base_cfg.pbdr_max_iters):
        gridE = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.pbdr_grid_points)
        outsE = [_stage2_solver(base_cfg.stage2_solver_for_pricing, users, float(x), pN, system, stack_cfg, base_cfg) for x in gridE]
        pE = float(max(outsE, key=lambda o: o.esp_revenue).price[0])
        gridN = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.pbdr_grid_points)
        outsN = [_stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, float(x), system, stack_cfg, base_cfg) for x in gridN]
        pN = float(max(outsN, key=lambda o: o.nsp_revenue).price[1])
        traj.append((pE, pN))
        if len(traj) >= 2 and abs(traj[-1][0]-traj[-2][0]) + abs(traj[-1][1]-traj[-2][1]) <= base_cfg.pbdr_tol:
            break
    return traj


def _bo_trajectory(users, system, stack_cfg, base_cfg):
    rng = np.random.default_rng(base_cfg.random_seed + 101)
    traj = []
    best = None
    best_val = -1e18
    for _ in range(base_cfg.bo_init_points + base_cfg.bo_iters):
        pE = float(rng.uniform(system.cE, base_cfg.max_price_E))
        pN = float(rng.uniform(system.cN, base_cfg.max_price_N))
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        val = out.esp_revenue + out.nsp_revenue - 0.01 * out.social_cost
        if val > best_val:
            best_val = val
            best = (pE, pN)
        traj.append((float(best[0]), float(best[1])))
    return traj


def _drl_trajectory(users, system, stack_cfg, base_cfg):
    grid_e = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.drl_price_levels)
    grid_n = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.drl_price_levels)
    actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    rng = np.random.default_rng(base_cfg.random_seed + 303)
    i = int(rng.integers(0, grid_e.size)); j = int(rng.integers(0, grid_n.size))
    traj = [(float(grid_e[i]), float(grid_n[j]))]
    for _ in range(min(80, base_cfg.drl_episodes * max(1, base_cfg.drl_steps_per_episode // 2))):
        a = actions[int(rng.integers(0, len(actions)))]
        i = min(max(i + a[0], 0), grid_e.size - 1)
        j = min(max(j + a[1], 0), grid_n.size - 1)
        traj.append((float(grid_e[i]), float(grid_n[j])))
    return traj


def plot_boundary_with_trajectory(
    pE_values: np.ndarray,
    pN_values: np.ndarray,
    combined_gap: np.ndarray,
    trajectory: list[dict],
    baseline_points: dict[str, tuple[float, float]],
    baseline_trajectories: dict[str, list[tuple[float, float]]],
    true_se: tuple[float, float],
    upper_boundary_mask: np.ndarray,
    out_path: Path,
    n_users: int,
) -> None:
    """Plot deviation-gap contour + theorem boundary + trajectories / baseline endpoints."""
    pE_grid, pN_grid = np.meshgrid(pE_values, pN_values)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)

    contourf = ax.contourf(pE_grid, pN_grid, combined_gap, levels=20, cmap="hot", alpha=0.8)
    contour = ax.contour(pE_grid, pN_grid, combined_gap, levels=10, colors="white", linewidths=0.5, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=7, fmt="%.1f")

    # Overlay theoretical upper boundary (Theorem 3 / Corollary 1 intent)
    by, bx = np.where(upper_boundary_mask)
    if bx.size > 0:
        ax.scatter(pE_values[bx], pN_values[by], s=7, c="#1565c0", marker=".", alpha=0.7,
                   label="Upper boundary ∂P_X*", zorder=7)

    traj_pE = [t["pE"] for t in trajectory]
    traj_pN = [t["pN"] for t in trajectory]
    ax.plot(traj_pE, traj_pN, 'c-', linewidth=2.5, marker='o', markersize=5,
            markerfacecolor='cyan', markeredgecolor='black', markeredgewidth=0.8,
            label='Algorithm 5 trajectory', zorder=10)
    ax.plot(traj_pE[0], traj_pN[0], 'go', markersize=10, markeredgecolor='darkgreen', markeredgewidth=1.5,
            label='Start', zorder=11)
    ax.plot(traj_pE[-1], traj_pN[-1], 'r*', markersize=14, markeredgecolor='darkred', markeredgewidth=1.0,
            label='Algorithm 5 final', zorder=11)

    style = {"PBRD": "s", "BO": "^", "DRL": "D"}
    colors = {"PBRD": "#8e24aa", "BO": "#f9a825", "DRL": "#2e7d32"}
    for name, traj in baseline_trajectories.items():
        if not traj:
            continue
        xs = [x for x, _ in traj]
        ys = [y for _, y in traj]
        ax.plot(xs, ys, linestyle='--', linewidth=1.2, alpha=0.9, color=colors.get(name, 'gray'),
                label=f"{name} trajectory", zorder=8)
    for name, (pE, pN) in baseline_points.items():
        ax.scatter([pE], [pN], s=70, marker=style.get(name, "x"), c=colors.get(name, "black"),
                   edgecolors="black", linewidths=0.7, label=f"{name} final", zorder=12)

    ax.scatter([true_se[0]], [true_se[1]], s=140, marker='X', c='#ff1744', edgecolors='black', linewidths=1.0,
               label='True SE (grid-search oracle)', zorder=13)

    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label("Combined Deviation Gap (ε)")

    ax.set_title(f"Stage I Figure 6: Boundary/Gap/Trajectory Overlay\n(n={n_users} users)")
    ax.set_xlabel("pE (ESP price)")
    ax.set_ylabel("pN (NSP price)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_trajectory_csv(trajectory: list[dict], result, out_path: Path) -> None:
    """Write trajectory data to CSV."""
    fields = ["iteration", "pE", "pN", "epsilon", "dist_to_se", "epsilon_delta"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for step in trajectory:
            w.writerow({k: step[k] for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage I boundary visualization with trajectory")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--esp-csv", type=str, default="outputs/real_revenue_contours/esp_real_revenue.csv")
    parser.add_argument("--nsp-csv", type=str, default="outputs/real_revenue_contours/nsp_real_revenue.csv")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1, help="Number of trials (default 1 for visualization)")
    parser.add_argument("--seed", type=int, default=20260002)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    esp_csv = Path(args.esp_csv)
    nsp_csv = Path(args.nsp_csv)

    # Check if revenue contours exist
    use_computed_contours = esp_csv.exists() and nsp_csv.exists()

    cfg = load_config(args.config)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage1_boundary_visualization_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trial_cfg = replace(cfg, n_users=args.n_users)

    for trial in range(args.trials):
        seed = args.seed + trial
        rng = np.random.default_rng(seed)
        users = sample_users(trial_cfg, rng)

        # Run Algorithm 5 and get trajectory
        trajectory, result = run_and_get_trajectory(users, cfg.system, cfg.stackelberg)

        # Selected baselines (final points for overlay)
        pbdr = baseline_stage1_pbdr(users, cfg.system, cfg.stackelberg, cfg.baselines)
        bo = baseline_stage1_bo(users, cfg.system, cfg.stackelberg, cfg.baselines)
        drl = baseline_stage1_drl(users, cfg.system, cfg.stackelberg, cfg.baselines)
        baseline_points = {
            "PBRD": pbdr.price,
            "BO": bo.price,
            "DRL": drl.price,
        }
        baseline_trajectories = {
            "PBRD": _pbdr_trajectory(users, cfg.system, cfg.stackelberg, cfg.baselines),
            "BO": _bo_trajectory(users, cfg.system, cfg.stackelberg, cfg.baselines),
            "DRL": _drl_trajectory(users, cfg.system, cfg.stackelberg, cfg.baselines),
        }
        true_se = baseline_stage1_grid_search_oracle(users, cfg.system, cfg.stackelberg, cfg.baselines).price

        # Write trajectory CSV
        _write_trajectory_csv(trajectory, result, run_dir / f"trajectory_trial{trial}.csv")

        # Export baseline trajectories
        for bname, btraj in baseline_trajectories.items():
            out_csv = run_dir / f"baseline_trajectory_{bname.lower()}_trial{trial}.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["iteration", "pE", "pN"])
                for k, (x, y) in enumerate(btraj):
                    w.writerow([k, float(x), float(y)])

        diag_txt = run_dir / f"algorithm5_diagnostics_trial{trial}.txt"
        eps_values = [float(x["epsilon"]) for x in trajectory]
        dist_values = [float(x.get("dist_to_se", np.nan)) for x in trajectory]
        lines = [
            f"search_steps = {len(result.trajectory)}",
            f"outer_iterations = {result.outer_iterations}",
            f"stage2_oracle_calls = {result.stage2_oracle_calls}",
            f"evaluated_candidates = {result.evaluated_candidates}",
            f"stopping_reason = {getattr(result, 'stopping_reason', 'unknown')}",
            f"epsilon_progression = {eps_values}",
            f"distance_to_se_progression = {dist_values}",
        ]
        diag_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # If revenue contours available, create visualization
        if use_computed_contours:
            print(f"Loading revenue contours from: {esp_csv.parent}")
            pE_values, pN_values, esp_gap, nsp_gap, combined_gap = compute_deviation_gaps(esp_csv, nsp_csv)

            data = _build_data(users)
            upper_boundary_mask = _extract_upper_boundary_mask(data, result.offloading_set, pE_values, pN_values, cfg.system)

            plot_boundary_with_trajectory(
                pE_values,
                pN_values,
                combined_gap,
                trajectory,
                baseline_points,
                baseline_trajectories,
                true_se,
                upper_boundary_mask,
                run_dir / f"boundary_visualization_trial{trial}.png",
                args.n_users,
            )

            # Persist heatmap/contour surface values
            gap_csv = run_dir / f"combined_gap_surface_trial{trial}.csv"
            with gap_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["pE", "pN", "combined_gap"])
                for yi, pN in enumerate(pN_values):
                    for xi, pE in enumerate(pE_values):
                        w.writerow([float(pE), float(pN), float(combined_gap[yi, xi])])

            # Quantitative summary: distance of final points to nearest boundary point
            by, bx = np.where(upper_boundary_mask)
            boundary_pts = np.column_stack([pE_values[bx], pN_values[by]]) if bx.size > 0 else np.zeros((0, 2))
            summary_rows = []
            quant_lines = []
            if boundary_pts.shape[0] > 0:
                for name, (pE, pN) in {"Algorithm5": result.price, "TrueSE": true_se, **baseline_points}.items():
                    d = np.min(np.sqrt((boundary_pts[:, 0] - pE) ** 2 + (boundary_pts[:, 1] - pN) ** 2))
                    quant_lines.append(f"{name}_distance_to_boundary = {float(d):.6g}")
                    summary_rows.append({
                        "method": name,
                        "pE": float(pE),
                        "pN": float(pN),
                        "distance_to_boundary": float(d),
                    })
            else:
                quant_lines.append("boundary_points = 0")

            (run_dir / f"boundary_summary_trial{trial}.txt").write_text("\n".join(quant_lines) + "\n", encoding="utf-8")

            summary_csv = run_dir / f"boundary_summary_trial{trial}.csv"
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["method", "pE", "pN", "distance_to_boundary"])
                w.writeheader()
                w.writerows(summary_rows)

            summary_json = run_dir / f"boundary_summary_trial{trial}.json"
            summary_json.write_text(json.dumps({
                "trial": trial,
                "n_users": args.n_users,
                "boundary_points": int(boundary_pts.shape[0]),
                "methods": summary_rows,
            }, indent=2), encoding="utf-8")

            print(f"  Created visualization + heatmap surface + boundary summaries for trial {trial}")
        else:
            print(f"Revenue contours not found at: {esp_csv}")
            print("  Run compute_real_revenue_contours.py first to generate contours.")
            print("  Trajectory data saved without visualization.")

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"n_users = {args.n_users}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        f"esp_csv = {args.esp_csv}",
        f"nsp_csv = {args.nsp_csv}",
        f"contours_available = {use_computed_contours}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"\nDone. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

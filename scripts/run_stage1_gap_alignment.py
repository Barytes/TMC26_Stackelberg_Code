#!/usr/bin/env python3
"""Stage I: epsilon landscape alignment (Algorithm-3 hat-G vs real-revenue gap).

Computes on the same grid and same user instance:
- epsilon_hat(pE,pN) = max{hat G_E, hat G_N} from Algorithm 3 estimator
  (for both estimator variants: boundary / refined_price)
- epsilon_real(pE,pN) from real revenues (best unilateral improvement)

Outputs heatmaps, delta maps, summaries, and a concise markdown report payload.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.baselines import run_stage2_solver
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import algorithm_3_gain_approximation


def _save_surface_csv(path: Path, pE: np.ndarray, pN: np.ndarray, z: np.ndarray, name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pN\\pE", *[f"{x:.10g}" for x in pE]])
        for j, pN_val in enumerate(pN):
            w.writerow([f"{pN_val:.10g}", *[f"{z[j, i]:.12g}" for i in range(len(pE))]])


def _argmin_point(z: np.ndarray, pE: np.ndarray, pN: np.ndarray) -> dict[str, float | int]:
    j, i = np.unravel_index(np.argmin(z), z.shape)
    return {
        "i": int(i),
        "j": int(j),
        "pE": float(pE[i]),
        "pN": float(pN[j]),
        "value": float(z[j, i]),
    }


def _plot_heatmap(ax, z: np.ndarray, pE: np.ndarray, pN: np.ndarray, title: str, cmap: str, vmin=None, vmax=None):
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[float(pE[0]), float(pE[-1]), float(pN[0]), float(pN[-1])],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    return im


def _plot_variant_bundle(
    out_dir: Path,
    variant: str,
    pE: np.ndarray,
    pN: np.ndarray,
    epsilon_hat: np.ndarray,
    epsilon_real: np.ndarray,
    se_proxy: dict[str, float | int],
    hat_argmin: dict[str, float | int],
) -> None:
    delta = epsilon_hat - epsilon_real
    vmax_eps = float(max(np.max(epsilon_hat), np.max(epsilon_real), 1e-12))
    vmax_abs_delta = float(max(np.max(np.abs(delta)), 1e-12))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=160)

    im0 = _plot_heatmap(
        axes[0, 0],
        epsilon_hat,
        pE,
        pN,
        f"epsilon_hat ({variant})",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_eps,
    )
    im1 = _plot_heatmap(
        axes[0, 1],
        epsilon_real,
        pE,
        pN,
        "epsilon_real (from real revenue)",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_eps,
    )
    im2 = _plot_heatmap(
        axes[1, 0],
        delta,
        pE,
        pN,
        f"delta = epsilon_hat - epsilon_real ({variant})",
        cmap="coolwarm",
        vmin=-vmax_abs_delta,
        vmax=vmax_abs_delta,
    )

    # Optional contour comparison overlay
    cs_real = axes[1, 1].contour(pE, pN, epsilon_real, levels=12, cmap="Blues", linewidths=1.2)
    cs_hat = axes[1, 1].contour(pE, pN, epsilon_hat, levels=12, cmap="Oranges", linewidths=1.0, linestyles="--")
    axes[1, 1].clabel(cs_real, inline=True, fontsize=7, fmt="%.2g")
    axes[1, 1].set_title(f"Contour overlay (blue=real, orange={variant})")
    axes[1, 1].set_xlabel("pE")
    axes[1, 1].set_ylabel("pN")

    # Mark unified SE proxy (argmin of epsilon_real) on all maps
    for ax in axes.ravel():
        ax.scatter([se_proxy["pE"]], [se_proxy["pN"]], marker="*", s=120, c="red", edgecolors="white", linewidths=0.8)
        ax.scatter([hat_argmin["pE"]], [hat_argmin["pN"]], marker="x", s=70, c="black", linewidths=1.2)

    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)

    fig.suptitle(
        f"Stage I epsilon landscape alignment ({variant})\n"
        "red *= unified SE proxy (argmin epsilon_real), black x = argmin epsilon_hat",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"landscape_alignment_{variant}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage I epsilon landscape alignment")
    ap.add_argument("--config", type=str, default="configs/default.toml")
    ap.add_argument("--n-users", type=int, default=None, help="Override number of users")
    ap.add_argument("--seed", type=int, default=20260309)
    ap.add_argument("--grid-points", type=int, default=None, help="Override square grid points")
    ap.add_argument("--max-price-E", type=float, default=None)
    ap.add_argument("--max-price-N", type=float, default=None)
    ap.add_argument("--run-name", type=str, default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n_users is not None:
        cfg = replace(cfg, n_users=args.n_users)

    grid_points = args.grid_points or cfg.baselines.gso_grid_points
    max_price_E = args.max_price_E if args.max_price_E is not None else cfg.baselines.max_price_E
    max_price_N = args.max_price_N if args.max_price_N is not None else cfg.baselines.max_price_N

    pE_values = np.linspace(cfg.system.cE, max_price_E, grid_points)
    pN_values = np.linspace(cfg.system.cN, max_price_N, grid_points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"stage1_gap_alignment_{timestamp}"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    users = sample_users(cfg, rng)

    shape = (len(pN_values), len(pE_values))
    esp_rev = np.zeros(shape, dtype=float)
    nsp_rev = np.zeros(shape, dtype=float)
    epsilon_hat_boundary = np.zeros(shape, dtype=float)
    epsilon_hat_refined = np.zeros(shape, dtype=float)

    for j, pN in enumerate(pN_values):
        for i, pE in enumerate(pE_values):
            out = run_stage2_solver(
                cfg.baselines.stage2_solver_for_pricing,
                users,
                float(pE),
                float(pN),
                cfg.system,
                cfg.stackelberg,
                cfg.baselines,
            )
            esp_rev[j, i] = out.esp_revenue
            nsp_rev[j, i] = out.nsp_revenue

            gE_b = algorithm_3_gain_approximation(
                users,
                out.offloading_set,
                float(pE),
                float(pN),
                "E",
                cfg.system,
                estimator_variant="boundary",
            ).gain
            gN_b = algorithm_3_gain_approximation(
                users,
                out.offloading_set,
                float(pE),
                float(pN),
                "N",
                cfg.system,
                estimator_variant="boundary",
            ).gain
            epsilon_hat_boundary[j, i] = max(float(gE_b), float(gN_b))

            gE_r = algorithm_3_gain_approximation(
                users,
                out.offloading_set,
                float(pE),
                float(pN),
                "E",
                cfg.system,
                estimator_variant="refined_price",
            ).gain
            gN_r = algorithm_3_gain_approximation(
                users,
                out.offloading_set,
                float(pE),
                float(pN),
                "N",
                cfg.system,
                estimator_variant="refined_price",
            ).gain
            epsilon_hat_refined[j, i] = max(float(gE_r), float(gN_r))

    eps_E_real = np.max(esp_rev, axis=1, keepdims=True) - esp_rev
    eps_N_real = np.max(nsp_rev, axis=0, keepdims=True) - nsp_rev
    epsilon_real = np.maximum(eps_E_real, eps_N_real)

    se_proxy = _argmin_point(epsilon_real, pE_values, pN_values)
    boundary_argmin = _argmin_point(epsilon_hat_boundary, pE_values, pN_values)
    refined_argmin = _argmin_point(epsilon_hat_refined, pE_values, pN_values)

    def summarize(e_hat: np.ndarray, variant_name: str, argmin_hat: dict[str, float | int]) -> dict[str, float | int | str]:
        diff = e_hat - epsilon_real
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(np.abs(diff)))
        grid_dist = float(np.hypot(float(argmin_hat["i"]) - float(se_proxy["i"]), float(argmin_hat["j"]) - float(se_proxy["j"])))
        price_dist = float(np.hypot(float(argmin_hat["pE"]) - float(se_proxy["pE"]), float(argmin_hat["pN"]) - float(se_proxy["pN"])))
        return {
            "variant": variant_name,
            "mae": mae,
            "rmse": rmse,
            "max_abs_diff": max_abs,
            "argmin_hat_pE": float(argmin_hat["pE"]),
            "argmin_hat_pN": float(argmin_hat["pN"]),
            "argmin_hat_value": float(argmin_hat["value"]),
            "argmin_real_pE": float(se_proxy["pE"]),
            "argmin_real_pN": float(se_proxy["pN"]),
            "argmin_real_value": float(se_proxy["value"]),
            "argmin_grid_distance": grid_dist,
            "argmin_price_distance": price_dist,
        }

    summary_rows = [
        summarize(epsilon_hat_boundary, "boundary", boundary_argmin),
        summarize(epsilon_hat_refined, "refined_price", refined_argmin),
    ]

    _save_surface_csv(out_dir / "epsilon_real.csv", pE_values, pN_values, epsilon_real, "epsilon_real")
    _save_surface_csv(out_dir / "epsilon_hat_boundary.csv", pE_values, pN_values, epsilon_hat_boundary, "epsilon_hat_boundary")
    _save_surface_csv(out_dir / "epsilon_hat_refined_price.csv", pE_values, pN_values, epsilon_hat_refined, "epsilon_hat_refined_price")
    _save_surface_csv(out_dir / "delta_boundary_minus_real.csv", pE_values, pN_values, epsilon_hat_boundary - epsilon_real, "delta_boundary_minus_real")
    _save_surface_csv(out_dir / "delta_refined_minus_real.csv", pE_values, pN_values, epsilon_hat_refined - epsilon_real, "delta_refined_minus_real")

    _plot_variant_bundle(out_dir, "boundary", pE_values, pN_values, epsilon_hat_boundary, epsilon_real, se_proxy, boundary_argmin)
    _plot_variant_bundle(out_dir, "refined_price", pE_values, pN_values, epsilon_hat_refined, epsilon_real, se_proxy, refined_argmin)

    with (out_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    metadata = {
        "run_name": run_name,
        "seed": args.seed,
        "n_users": cfg.n_users,
        "stage2_solver": cfg.baselines.stage2_solver_for_pricing,
        "grid_points": int(grid_points),
        "pE_range": [float(pE_values[0]), float(pE_values[-1])],
        "pN_range": [float(pN_values[0]), float(pN_values[-1])],
        "unified_se_proxy": se_proxy,
        "summary": summary_rows,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[done] outputs: {out_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

"""Legacy auxiliary plotting helper for discrete PBRD trajectories on a heatmap CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.baselines import run_discrete_br_dynamics


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return value


def _unit_float(raw: str) -> float:
    value = float(raw)
    if not (0.0 < value < 1.0):
        raise argparse.ArgumentTypeError("Value must be in (0, 1).")
    return value


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    required = {"pE", "pN", "esp_revenue", "nsp_revenue", "eps"}
    if not required.issubset(set(rows[0].keys() if rows else [])):
        raise ValueError(f"CSV missing required columns: {required}")

    pE_vals = sorted({float(r["pE"]) for r in rows})
    pN_vals = sorted({float(r["pN"]) for r in rows})
    pE_grid = np.asarray(pE_vals, dtype=float)
    pN_grid = np.asarray(pN_vals, dtype=float)
    n_rows, n_cols = pN_grid.size, pE_grid.size

    e_map = {v: i for i, v in enumerate(pE_vals)}
    n_map = {v: j for j, v in enumerate(pN_vals)}
    esp_rev = np.full((n_rows, n_cols), np.nan, dtype=float)
    nsp_rev = np.full((n_rows, n_cols), np.nan, dtype=float)
    eps = np.full((n_rows, n_cols), np.nan, dtype=float)

    for r in rows:
        pE = float(r["pE"])
        pN = float(r["pN"])
        i = e_map[pE]
        j = n_map[pN]
        esp_rev[j, i] = float(r["esp_revenue"])
        nsp_rev[j, i] = float(r["nsp_revenue"])
        eps[j, i] = float(r["eps"])

    if np.any(~np.isfinite(esp_rev)) or np.any(~np.isfinite(nsp_rev)) or np.any(~np.isfinite(eps)):
        raise ValueError("CSV grid is incomplete or contains invalid values.")

    return pE_grid, pN_grid, esp_rev, nsp_rev, eps


def _nearest_idx(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _nearest_positive_idx(grid: np.ndarray, target: float) -> int:
    positive_idx = np.flatnonzero(grid > 0.0)
    if positive_idx.size == 0:
        return _nearest_idx(grid, target)
    pos_vals = grid[positive_idx]
    local = int(np.argmin(np.abs(pos_vals - target)))
    return int(positive_idx[local])


def _plot_eps_with_trajectory(
    eps: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    trajectory: list[tuple[int, int]],
    mode: str,
    eps_tol: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    im = ax.imshow(
        eps,
        origin="lower",
        aspect="auto",
        extent=[float(pE_grid[0]), float(pE_grid[-1]), float(pN_grid[0]), float(pN_grid[-1])],
        cmap="magma",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("epsilon")

    # Mark SE set by epsilon tolerance.
    se_mask = eps <= eps_tol
    se_n_idx, se_e_idx = np.nonzero(se_mask)
    if se_n_idx.size > 0:
        ax.scatter(
            pE_grid[se_e_idx],
            pN_grid[se_n_idx],
            s=12,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
            label="SE set (eps<=tol)",
        )

    traj = np.asarray(trajectory, dtype=int)
    traj_pE = pE_grid[traj[:, 1]]
    traj_pN = pN_grid[traj[:, 0]]
    ax.plot(traj_pE, traj_pN, color="cyan", linewidth=1.4, alpha=0.85, label=f"PBDR path ({mode})")
    if traj.shape[0] > 2:
        mid = traj[1:-1]
        ax.scatter(
            pE_grid[mid[:, 1]],
            pN_grid[mid[:, 0]],
            c=np.arange(1, traj.shape[0] - 1),
            cmap="winter",
            s=24,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )
    ax.scatter([traj_pE[0]], [traj_pN[0]], s=70, c="lime", edgecolors="black", linewidths=0.6, label="start")
    ax.scatter([traj_pE[-1]], [traj_pN[-1]], s=120, marker="*", c="red", edgecolors="black", linewidths=0.6, label="end")

    ax.set_title("PBDR trajectory on epsilon heatmap")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_trajectory_csv(
    out_path: Path,
    trajectory: list[tuple[int, int]],
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    eps: np.ndarray,
) -> None:
    rows = ["step,pE,pN,eps"]
    for k, (j, i) in enumerate(trajectory):
        rows.append(f"{k},{float(pE_grid[i]):.10g},{float(pN_grid[j]):.10g},{float(eps[j, i]):.10g}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot discrete PBDR trajectory on epsilon heatmap from a grid CSV.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to price_grid_metrics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/pbdr_traj_<mode>_<timestamp>).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["alternating", "greedy"],
        default="alternating",
        help="BR dynamics mode.",
    )
    parser.add_argument("--max-iters", type=_positive_int, default=200, help="Max BR dynamics iterations.")
    parser.add_argument(
        "--start-pE",
        type=float,
        default=0.1,
        help="Start pE target (nearest positive grid point will be used; default 0.1).",
    )
    parser.add_argument(
        "--start-pN",
        type=float,
        default=0.1,
        help="Start pN target (nearest positive grid point will be used; default 0.1).",
    )
    parser.add_argument("--eps-tol", type=_nonnegative_float, default=1e-12, help="SE tolerance: eps<=tol.")
    parser.add_argument(
        "--low-eps-quantile",
        type=_unit_float,
        default=0.1,
        help="Quantile threshold for low-epsilon region check.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    pE_grid, pN_grid, esp_rev, nsp_rev, eps = _load_grid_csv(csv_path)

    start_i = _nearest_positive_idx(pE_grid, float(args.start_pE))
    start_j = _nearest_positive_idx(pN_grid, float(args.start_pN))
    trajectory = run_discrete_br_dynamics(
        esp_rev=esp_rev,
        nsp_rev=nsp_rev,
        start_idx=(start_j, start_i),
        max_iters=int(args.max_iters),
        mode=args.mode,
    )

    out_dir = resolve_out_dir(f"pbdr_traj_{args.mode}", args.out_dir)

    fig_path = out_dir / "grid_ne_gap_heatmap_pbdr_trajectory.png"
    _plot_eps_with_trajectory(
        eps=eps,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        trajectory=trajectory,
        mode=args.mode,
        eps_tol=float(args.eps_tol),
        out_path=fig_path,
    )
    _save_trajectory_csv(out_dir / "pbdr_trajectory.csv", trajectory, pE_grid, pN_grid, eps)

    low_eps_thr = float(np.quantile(eps, float(args.low_eps_quantile)))
    visited_eps = np.asarray([eps[j, i] for j, i in trajectory], dtype=float)
    end_j, end_i = trajectory[-1]
    summary_lines = [
        f"csv = {csv_path}",
        f"mode = {args.mode}",
        f"start_pE = {float(pE_grid[start_i]):.10g}",
        f"start_pN = {float(pN_grid[start_j]):.10g}",
        f"steps = {len(trajectory) - 1}",
        f"visited_points = {len(trajectory)}",
        f"final_pE = {float(pE_grid[end_i]):.10g}",
        f"final_pN = {float(pN_grid[end_j]):.10g}",
        f"final_eps = {float(eps[end_j, end_i]):.10g}",
        f"converged_to_se_set = {bool(eps[end_j, end_i] <= float(args.eps_tol))}",
        f"low_eps_quantile = {float(args.low_eps_quantile):.4g}",
        f"low_eps_threshold = {low_eps_thr:.10g}",
        f"entered_low_eps_region = {bool(np.any(visited_eps <= low_eps_thr))}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

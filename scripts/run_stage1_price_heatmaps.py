"""Block B auxiliary script: Stage I revenue and gap heatmap diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.baselines import evaluate_stage1_price_grid
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _min_two_int(raw: str) -> int:
    value = int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError("Value must be >= 2.")
    return value


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


def _select_equilibrium_representative(
    eq_mask: np.ndarray,
    eps: np.ndarray,
    joint_rev: np.ndarray,
    social_cost: np.ndarray,
) -> tuple[int, int]:
    _ = joint_rev, social_cost  # kept for call-site compatibility
    if np.any(eq_mask):
        masked_eps = np.where(eq_mask, eps, np.inf)
        return tuple(int(x) for x in np.unravel_index(int(np.argmin(masked_eps)), eps.shape))
    return tuple(int(x) for x in np.unravel_index(int(np.argmin(eps)), eps.shape))


def _overlay_equilibrium_markers(
    ax,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    eq_mask: np.ndarray,
    representative: tuple[int, int],
) -> None:
    eq_n_idx, eq_e_idx = np.nonzero(eq_mask)
    if eq_n_idx.size > 0:
        ax.scatter(
            pE_grid[eq_e_idx],
            pN_grid[eq_n_idx],
            s=14,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
            label="SE set (eps<=tol)",
        )
    rep_n, rep_e = representative
    ax.scatter(
        [float(pE_grid[rep_e])],
        [float(pN_grid[rep_n])],
        s=140,
        marker="*",
        c="lime",
        edgecolors="black",
        linewidths=0.8,
        label="SE representative",
        zorder=4,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)


def _plot_heatmap(
    values: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    eq_mask: np.ndarray | None = None,
    representative: tuple[int, int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[float(pE_grid[0]), float(pE_grid[-1]), float(pN_grid[0]), float(pN_grid[-1])],
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    if eq_mask is not None and representative is not None:
        _overlay_equilibrium_markers(ax, pE_grid, pN_grid, eq_mask, representative)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_grid_csv(
    out_path: Path,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    esp_rev: np.ndarray,
    nsp_rev: np.ndarray,
    grid_ne_gap: np.ndarray,
    legacy_gain_proxy: np.ndarray,
) -> None:
    rows = ["pE,pN,esp_revenue,nsp_revenue,joint_revenue,grid_ne_gap,legacy_gain_proxy"]
    for j, pN in enumerate(pN_grid):
        for i, pE in enumerate(pE_grid):
            rows.append(
                f"{float(pE):.10g},{float(pN):.10g},{float(esp_rev[j, i]):.10g},{float(nsp_rev[j, i]):.10g},{float(esp_rev[j, i] + nsp_rev[j, i]):.10g},{float(grid_ne_gap[j, i]):.10g},{float(legacy_gain_proxy[j, i]):.10g}"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Stage-I revenue and restricted-gap heatmaps on [0,pEmax] x [0,pNmax]."
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--n-users", type=_positive_int, default=None, help="Optional user-count override.")
    parser.add_argument("--pEmax", type=_positive_float, required=True, help="Upper bound for pE axis.")
    parser.add_argument("--pNmax", type=_positive_float, required=True, help="Upper bound for pN axis.")
    parser.add_argument("--pE-points", type=_min_two_int, default=81, help="Number of pE grid points.")
    parser.add_argument("--pN-points", type=_min_two_int, default=81, help="Number of pN grid points.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed override.")
    parser.add_argument(
        "--stage2-method",
        type=str,
        default=None,
        choices=["CS", "UBRD", "VI", "PEN", "DG"],
        help="Optional Stage-II solver override.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_stage1_price_heatmaps_<timestamp>).",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to print grid-evaluation progress (default: true).",
    )
    parser.add_argument(
        "--progress-step",
        type=_positive_int,
        default=10,
        help="Print once every N evaluated price points (default: 10).",
    )
    parser.add_argument(
        "--eps-tol",
        type=_nonnegative_float,
        default=1e-12,
        help="Tolerance for equilibrium mask: eps<=tol.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_users is not None:
        cfg = replace(cfg, n_users=int(args.n_users))
    seed = cfg.seed if args.seed is None else int(args.seed)
    rng = np.random.default_rng(seed)
    users = sample_users(cfg, rng)

    out_dir = resolve_out_dir("run_stage1_price_heatmaps", args.out_dir)

    started_at = time.perf_counter()

    def progress_cb(done: int, total: int, pE: float, pN: float) -> None:
        if not args.show_progress:
            return
        step = max(1, int(args.progress_step))
        if done % step != 0 and done != total:
            return
        ratio = 100.0 * done / max(total, 1)
        elapsed = time.perf_counter() - started_at
        print(
            f"\rEvaluating grid: {done}/{total} ({ratio:.1f}%) elapsed={elapsed:.1f}s pE={pE:.4g} pN={pN:.4g}",
            end="",
            flush=True,
        )
        if done == total:
            print()

    grid = evaluate_stage1_price_grid(
        users=users,
        system=cfg.system,
        stack_cfg=cfg.stackelberg,
        base_cfg=cfg.baselines,
        pE_min=0.0,
        pE_max=float(args.pEmax),
        pN_min=0.0,
        pN_max=float(args.pNmax),
        pE_points=int(args.pE_points),
        pN_points=int(args.pN_points),
        stage2_method=args.stage2_method,
        progress_cb=progress_cb,
    )
    joint_rev = grid.esp_rev + grid.nsp_rev
    social_cost = np.array(
        [[float(out.social_cost) for out in row] for row in grid.outcomes],
        dtype=float,
    )
    legacy_gain_proxy = np.array(
        [[float(out.legacy_gain_proxy) for out in row] for row in grid.outcomes],
        dtype=float,
    )
    eq_mask = grid.grid_ne_gap <= float(args.eps_tol)
    rep = _select_equilibrium_representative(eq_mask, grid.grid_ne_gap, joint_rev, social_cost)

    _plot_heatmap(
        values=grid.esp_rev,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="ESP Revenue Heatmap",
        cbar_label="ESP Revenue",
        out_path=out_dir / "esp_revenue_heatmap.png",
        cmap="viridis",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=grid.nsp_rev,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="NSP Revenue Heatmap",
        cbar_label="NSP Revenue",
        out_path=out_dir / "nsp_revenue_heatmap.png",
        cmap="plasma",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=joint_rev,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="Joint Revenue (ESP+NSP) Heatmap",
        cbar_label="ESP+NSP Revenue",
        out_path=out_dir / "joint_revenue_heatmap.png",
        cmap="cividis",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=grid.grid_ne_gap,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="Grid-NE-Gap Heatmap",
        cbar_label="grid_ne_gap",
        out_path=out_dir / "grid_ne_gap_heatmap.png",
        cmap="magma",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=legacy_gain_proxy,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="Legacy Gain Proxy Heatmap",
        cbar_label="legacy_gain_proxy",
        out_path=out_dir / "legacy_gain_proxy_heatmap.png",
        cmap="inferno",
        eq_mask=eq_mask,
        representative=rep,
    )
    _write_grid_csv(
        out_path=out_dir / "price_grid_metrics.csv",
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        esp_rev=grid.esp_rev,
        nsp_rev=grid.nsp_rev,
        grid_ne_gap=grid.grid_ne_gap,
        legacy_gain_proxy=legacy_gain_proxy,
    )

    min_j, min_i = np.unravel_index(int(np.argmin(grid.grid_ne_gap)), grid.grid_ne_gap.shape)
    summary_lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"stage2_method = {args.stage2_method or cfg.baselines.stage2_solver_for_pricing}",
        f"pE_range = [0.0, {float(args.pEmax)}], points = {int(args.pE_points)}",
        f"pN_range = [0.0, {float(args.pNmax)}], points = {int(args.pN_points)}",
        f"eps_tol = {float(args.eps_tol):.10g}",
        f"equilibrium_count = {int(np.count_nonzero(eq_mask))}",
        f"min_grid_ne_gap = {float(grid.grid_ne_gap[min_j, min_i]):.10g}",
        f"argmin_pE = {float(grid.pE_grid[min_i]):.10g}",
        f"argmin_pN = {float(grid.pN_grid[min_j]):.10g}",
        f"representative_pE = {float(grid.pE_grid[rep[1]]):.10g}",
        f"representative_pN = {float(grid.pN_grid[rep[0]]):.10g}",
        f"representative_grid_ne_gap = {float(grid.grid_ne_gap[rep[0], rep[1]]):.10g}",
        f"representative_legacy_gain_proxy = {float(legacy_gain_proxy[rep[0], rep[1]]):.10g}",
        f"representative_joint_revenue = {float(joint_rev[rep[0], rep[1]]):.10g}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()

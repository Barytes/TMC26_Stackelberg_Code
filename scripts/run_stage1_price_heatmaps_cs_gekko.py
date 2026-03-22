"""Block B auxiliary script: exact-CS-assisted Stage I heatmap diagnostics."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.baselines import _solve_centralized_minlp
from tmc26_exp.config import BaselineConfig, load_config
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users

_ROW_USERS: UserBatch | None = None
_ROW_SYSTEM = None
_ROW_STACK = None
_ROW_BASE = None
_ROW_PES: np.ndarray | None = None


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


def _build_gekko_baseline_cfg(base_cfg: BaselineConfig, args: argparse.Namespace) -> BaselineConfig:
    return replace(
        base_cfg,
        cs_use_minlp=True,
        gekko_time_limit=int(args.gekko_time_limit),
        gekko_max_iter=int(args.gekko_max_iter),
        gekko_mip_gap=float(args.gekko_mip_gap),
    )


def _init_row_solver(users: UserBatch, system, stack_cfg, base_cfg: BaselineConfig, pE_grid: np.ndarray) -> None:
    global _ROW_USERS, _ROW_SYSTEM, _ROW_STACK, _ROW_BASE, _ROW_PES
    _ROW_USERS = users
    _ROW_SYSTEM = system
    _ROW_STACK = stack_cfg
    _ROW_BASE = base_cfg
    _ROW_PES = pE_grid


def _solve_one_row(row_input: tuple[int, float]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j, pN = row_input
    assert _ROW_USERS is not None
    assert _ROW_SYSTEM is not None
    assert _ROW_STACK is not None
    assert _ROW_BASE is not None
    assert _ROW_PES is not None

    esp_row = np.full(_ROW_PES.size, np.nan, dtype=float)
    nsp_row = np.full(_ROW_PES.size, np.nan, dtype=float)
    social_row = np.full(_ROW_PES.size, np.nan, dtype=float)
    success_row = np.zeros(_ROW_PES.size, dtype=bool)

    for i, pE in enumerate(_ROW_PES):
        out = _solve_centralized_minlp(
            users=_ROW_USERS,
            pE=float(pE),
            pN=float(pN),
            system=_ROW_SYSTEM,
            stack_cfg=_ROW_STACK,
            base_cfg=_ROW_BASE,
        )
        esp_row[i] = float(out.esp_revenue)
        nsp_row[i] = float(out.nsp_revenue)
        social_row[i] = float(out.social_cost)
        success_row[i] = bool(out.meta.get("success", True))

    return j, esp_row, nsp_row, social_row, success_row


def _select_equilibrium_representative(eq_mask: np.ndarray, eps: np.ndarray) -> tuple[int, int]:
    if np.any(eq_mask):
        masked_eps = np.where(eq_mask, eps, np.inf)
        return tuple(int(x) for x in np.unravel_index(int(np.argmin(masked_eps)), eps.shape))

    finite = np.isfinite(eps)
    if not np.any(finite):
        raise RuntimeError("No finite epsilon values produced by GEKKO on the grid.")
    finite_eps = np.where(finite, eps, np.inf)
    return tuple(int(x) for x in np.unravel_index(int(np.argmin(finite_eps)), eps.shape))


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
    masked_values = np.ma.masked_invalid(values)
    im = ax.imshow(
        masked_values,
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
    joint_rev: np.ndarray,
    eps: np.ndarray,
    social_cost: np.ndarray,
    success_mask: np.ndarray,
) -> None:
    rows = [
        "pE,pN,esp_revenue,nsp_revenue,joint_revenue,eps,social_cost,centralized_success",
    ]
    for j, pN in enumerate(pN_grid):
        for i, pE in enumerate(pE_grid):
            rows.append(
                f"{float(pE):.10g},{float(pN):.10g},"
                f"{float(esp_rev[j, i]):.10g},{float(nsp_rev[j, i]):.10g},"
                f"{float(joint_rev[j, i]):.10g},{float(eps[j, i]):.10g},"
                f"{float(social_cost[j, i]):.10g},{int(success_mask[j, i])}"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ESP/NSP/joint revenue and epsilon heatmaps on [0,pEmax] x [0,pNmax], "
            "with Stage-II centralized solver solved by GEKKO (MINLP)."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--n-users", type=_positive_int, default=20, help="Number of sampled users.")
    parser.add_argument("--pEmax", type=_positive_float, required=True, help="Upper bound for pE axis.")
    parser.add_argument("--pNmax", type=_positive_float, required=True, help="Upper bound for pN axis.")
    parser.add_argument("--pE-points", type=_min_two_int, default=150, help="Number of pE grid points.")
    parser.add_argument("--pN-points", type=_min_two_int, default=150, help="Number of pN grid points.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed override.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_stage1_price_heatmaps_cs_gekko_<timestamp>).",
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
    parser.add_argument(
        "--gekko-time-limit",
        type=_positive_int,
        default=60,
        help="GEKKO max_time for each solve (seconds).",
    )
    parser.add_argument(
        "--gekko-max-iter",
        type=_positive_int,
        default=200,
        help="GEKKO minlp_maximum_iterations for each solve.",
    )
    parser.add_argument(
        "--gekko-mip-gap",
        type=_positive_float,
        default=1e-3,
        help="GEKKO minlp_gap_tol for each solve.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Number of worker processes for row-parallel GEKKO solves (default: 1).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_users = replace(cfg, n_users=int(args.n_users))
    seed = cfg.seed if args.seed is None else int(args.seed)
    rng = np.random.default_rng(seed)
    users = sample_users(cfg_users, rng)
    base_cfg = _build_gekko_baseline_cfg(cfg.baselines, args)

    out_dir = resolve_out_dir("run_stage1_price_heatmaps_cs_gekko", args.out_dir)

    pE_grid = np.linspace(0.0, float(args.pEmax), int(args.pE_points))
    pN_grid = np.linspace(0.0, float(args.pNmax), int(args.pN_points))

    esp_rev = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    nsp_rev = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    social_cost = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    success_mask = np.zeros((pN_grid.size, pE_grid.size), dtype=bool)

    started_at = time.perf_counter()
    total_points = pN_grid.size * pE_grid.size
    done_points = 0

    if int(args.workers) <= 1:
        for j, pN in enumerate(pN_grid):
            for i, pE in enumerate(pE_grid):
                out = _solve_centralized_minlp(
                    users=users,
                    pE=float(pE),
                    pN=float(pN),
                    system=cfg.system,
                    stack_cfg=cfg.stackelberg,
                    base_cfg=base_cfg,
                )
                esp_rev[j, i] = float(out.esp_revenue)
                nsp_rev[j, i] = float(out.nsp_revenue)
                social_cost[j, i] = float(out.social_cost)
                success_mask[j, i] = bool(out.meta.get("success", True))

                done_points += 1
                if args.show_progress:
                    step = max(1, int(args.progress_step))
                    if done_points % step == 0 or done_points == total_points:
                        ratio = 100.0 * done_points / max(total_points, 1)
                        elapsed = time.perf_counter() - started_at
                        success_count = int(np.count_nonzero(success_mask))
                        fail_count = done_points - success_count
                        print(
                            f"\rEvaluating GEKKO grid: {done_points}/{total_points} ({ratio:.1f}%) "
                            f"elapsed={elapsed:.1f}s pE={pE:.4g} pN={pN:.4g} ok={success_count} fail={fail_count}",
                            end="",
                            flush=True,
                        )
    else:
        done_rows = 0
        row_jobs = [(j, float(pN)) for j, pN in enumerate(pN_grid)]
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_row_solver,
            initargs=(users, cfg.system, cfg.stackelberg, base_cfg, pE_grid),
        ) as ex:
            futures = [ex.submit(_solve_one_row, job) for job in row_jobs]
            for fut in as_completed(futures):
                j, esp_row, nsp_row, social_row, success_row = fut.result()
                esp_rev[j, :] = esp_row
                nsp_rev[j, :] = nsp_row
                social_cost[j, :] = social_row
                success_mask[j, :] = success_row

                done_rows += 1
                done_points = done_rows * pE_grid.size
                if args.show_progress:
                    ratio = 100.0 * done_points / max(total_points, 1)
                    elapsed = time.perf_counter() - started_at
                    success_count = int(np.count_nonzero(success_mask))
                    fail_count = done_points - success_count
                    last_pN = float(pN_grid[j])
                    print(
                        f"\rEvaluating GEKKO grid: {done_points}/{total_points} ({ratio:.1f}%) "
                        f"elapsed={elapsed:.1f}s row_pN={last_pN:.4g} ok={success_count} fail={fail_count}",
                        end="",
                        flush=True,
                    )
    if args.show_progress:
        print()

    joint_rev = esp_rev + nsp_rev
    with np.errstate(invalid="ignore"):
        esp_max_per_pN = np.nanmax(esp_rev, axis=1, keepdims=True)
        nsp_max_per_pE = np.nanmax(nsp_rev, axis=0, keepdims=True)
    eps_E = esp_max_per_pN - esp_rev
    eps_N = nsp_max_per_pE - nsp_rev
    eps = np.maximum(eps_E, eps_N)
    eps = np.where(np.isfinite(esp_rev) & np.isfinite(nsp_rev), eps, np.nan)

    eq_mask = np.isfinite(eps) & (eps <= float(args.eps_tol))
    rep = _select_equilibrium_representative(eq_mask, eps)

    _plot_heatmap(
        values=esp_rev,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        title="ESP Revenue Heatmap (CS=GEKKO)",
        cbar_label="ESP Revenue",
        out_path=out_dir / "esp_revenue_heatmap.png",
        cmap="viridis",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=nsp_rev,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        title="NSP Revenue Heatmap (CS=GEKKO)",
        cbar_label="NSP Revenue",
        out_path=out_dir / "nsp_revenue_heatmap.png",
        cmap="plasma",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=joint_rev,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        title="Joint Revenue (ESP+NSP) Heatmap (CS=GEKKO)",
        cbar_label="ESP+NSP Revenue",
        out_path=out_dir / "joint_revenue_heatmap.png",
        cmap="cividis",
        eq_mask=eq_mask,
        representative=rep,
    )
    _plot_heatmap(
        values=eps,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        title="Grid-NE-Gap Heatmap (CS=GEKKO)",
        cbar_label="grid_ne_gap",
        out_path=out_dir / "grid_ne_gap_heatmap.png",
        cmap="magma",
        eq_mask=eq_mask,
        representative=rep,
    )
    _write_grid_csv(
        out_path=out_dir / "price_grid_metrics.csv",
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        esp_rev=esp_rev,
        nsp_rev=nsp_rev,
        joint_rev=joint_rev,
        eps=eps,
        social_cost=social_cost,
        success_mask=success_mask,
    )

    finite_eps = np.isfinite(eps)
    if np.any(finite_eps):
        min_j, min_i = np.unravel_index(int(np.nanargmin(eps)), eps.shape)
        min_eps = float(eps[min_j, min_i])
        argmin_pE = float(pE_grid[min_i])
        argmin_pN = float(pN_grid[min_j])
    else:
        min_eps = float("nan")
        argmin_pE = float("nan")
        argmin_pN = float("nan")

    success_count = int(np.count_nonzero(success_mask))
    total_count = int(success_mask.size)
    fail_count = total_count - success_count
    summary_lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"n_users = {int(args.n_users)}",
        "stage2_method = CS_GEKKO_MINLP",
        f"pE_range = [0.0, {float(args.pEmax)}], points = {int(args.pE_points)}",
        f"pN_range = [0.0, {float(args.pNmax)}], points = {int(args.pN_points)}",
        f"eps_tol = {float(args.eps_tol):.10g}",
        f"equilibrium_count = {int(np.count_nonzero(eq_mask))}",
        f"min_eps = {min_eps:.10g}",
        f"argmin_pE = {argmin_pE:.10g}",
        f"argmin_pN = {argmin_pN:.10g}",
        f"representative_pE = {float(pE_grid[rep[1]]):.10g}",
        f"representative_pN = {float(pN_grid[rep[0]]):.10g}",
        f"representative_eps = {float(eps[rep[0], rep[1]]):.10g}",
        f"representative_joint_revenue = {float(joint_rev[rep[0], rep[1]]):.10g}",
        f"centralized_success_points = {success_count}",
        f"centralized_failed_points = {fail_count}",
        f"centralized_success_rate = {float(success_count / max(total_count, 1)):.6f}",
        f"gekko_time_limit = {int(args.gekko_time_limit)}",
        f"gekko_max_iter = {int(args.gekko_max_iter)}",
        f"gekko_mip_gap = {float(args.gekko_mip_gap):.10g}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()

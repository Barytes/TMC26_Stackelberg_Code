from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.baselines import evaluate_stage1_price_grid
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import run_stage1_solver


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return value


def _min_two_int(raw: str) -> int:
    value = int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError("Value must be >= 2.")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return value


def _nearest_idx(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _load_summary_kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    required = {"pE", "pN", "eps"}
    if not required.issubset(set(rows[0].keys() if rows else [])):
        raise ValueError(f"CSV missing required columns: {required}")

    pE_vals = sorted({float(r["pE"]) for r in rows})
    pN_vals = sorted({float(r["pN"]) for r in rows})
    pE_grid = np.asarray(pE_vals, dtype=float)
    pN_grid = np.asarray(pN_vals, dtype=float)
    eps = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)

    e_map = {v: i for i, v in enumerate(pE_vals)}
    n_map = {v: j for j, v in enumerate(pN_vals)}
    for r in rows:
        i = e_map[float(r["pE"])]
        j = n_map[float(r["pN"])]
        eps[j, i] = float(r["eps"])
    if np.any(~np.isfinite(eps)):
        raise ValueError("CSV epsilon grid is incomplete.")
    return pE_grid, pN_grid, eps


def _trajectory_points(result) -> list[tuple[float, float]]:
    points = [(float(step.pE), float(step.pN)) for step in result.trajectory]
    final_p = (float(result.price[0]), float(result.price[1]))
    if not points:
        points.append(final_p)
        return points
    last = points[-1]
    if abs(last[0] - final_p[0]) + abs(last[1] - final_p[1]) > 1e-12:
        points.append(final_p)
    return points


def _plot_eps_with_trajectory(
    eps: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    trajectory: list[tuple[float, float]],
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
            alpha=0.85,
            label="SE set (eps<=tol)",
        )

    pE_path = np.asarray([p[0] for p in trajectory], dtype=float)
    pN_path = np.asarray([p[1] for p in trajectory], dtype=float)
    ax.plot(pE_path, pN_path, color="cyan", linewidth=1.4, alpha=0.9, label="VBBR-BRD path")
    ax.scatter(
        pE_path,
        pN_path,
        c=np.arange(len(trajectory)),
        cmap="winter",
        s=26,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )
    ax.scatter([pE_path[0]], [pN_path[0]], s=75, c="lime", edgecolors="black", linewidths=0.6, label="start")
    ax.scatter([pE_path[-1]], [pN_path[-1]], s=120, marker="*", c="red", edgecolors="black", linewidths=0.6, label="end")

    ax.set_title("VBBR-BRD trajectory on epsilon heatmap")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_trajectory_csv(
    out_path: Path,
    trajectory: list[tuple[float, float]],
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    eps: np.ndarray,
) -> None:
    rows = ["step,pE,pN,nearest_grid_pE,nearest_grid_pN,nearest_grid_eps"]
    for step, (pE, pN) in enumerate(trajectory):
        i = _nearest_idx(pE_grid, pE)
        j = _nearest_idx(pN_grid, pN)
        rows.append(
            f"{step},{pE:.10g},{pN:.10g},{float(pE_grid[i]):.10g},{float(pN_grid[j]):.10g},{float(eps[j, i]):.10g}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot VBBR-BRD Stage-I trajectory on epsilon heatmap."
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed override.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional existing price_grid_metrics.csv. If set, reuse this epsilon grid instead of recomputing heatmap.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Optional summary.txt path for default seed/config/eps_tol when --csv is used.",
    )
    parser.add_argument("--pEmax", type=_positive_float, default=None, help="Upper bound for pE axis.")
    parser.add_argument("--pNmax", type=_positive_float, default=None, help="Upper bound for pN axis.")
    parser.add_argument("--pE-points", type=_min_two_int, default=81, help="Number of pE grid points.")
    parser.add_argument("--pN-points", type=_min_two_int, default=81, help="Number of pN grid points.")
    parser.add_argument(
        "--stage2-method",
        type=str,
        default=None,
        choices=["CS", "UBRD", "VI", "PEN", "DG"],
        help="Optional Stage-II solver override for heatmap generation.",
    )
    parser.add_argument(
        "--search-max-iters",
        type=_positive_int,
        default=None,
        help="Optional override for VBBR Stage-I search_max_iters.",
    )
    parser.add_argument(
        "--eps-tol",
        type=_nonnegative_float,
        default=None,
        help="SE tolerance: eps<=tol. Default uses summary eps_tol if available, else 1e-12.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_stage1_vbbr_traj_on_heatmap_<timestamp>).",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to print heatmap-grid evaluation progress (default: true).",
    )
    parser.add_argument(
        "--progress-step",
        type=_positive_int,
        default=50,
        help="Print once every N grid points (default: 50).",
    )
    parser.add_argument("--vbbr-local-r", type=int, default=None, help="Optional override for vbbr_local_R.")
    parser.add_argument("--vbbr-local-s", type=int, default=None, help="Optional override for vbbr_local_S.")
    parser.add_argument("--vbbr-local-budget", type=int, default=None, help="Optional override for vbbr_local_budget.")
    parser.add_argument("--vbbr-top-m", type=_positive_int, default=None, help="Optional override for vbbr_top_m.")
    parser.add_argument(
        "--vbbr-oracle-max-rounds",
        type=_positive_int,
        default=None,
        help="Optional override for vbbr_oracle_max_rounds.",
    )
    parser.add_argument(
        "--vbbr-no-improve-patience",
        type=_positive_int,
        default=None,
        help="Optional override for vbbr_no_improve_patience.",
    )
    parser.add_argument(
        "--vbbr-damping-alpha",
        type=_positive_float,
        default=None,
        help="Optional override for vbbr_damping_alpha (0<alpha<=1).",
    )
    parser.add_argument(
        "--vbbr-outer-update-mode",
        type=str,
        default=None,
        choices=["gain_max", "gain_min", "esp_first", "nsp_first"],
        help="Optional outer-update mode for VBBR.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary is not None else None
    if summary_path is None and args.csv is not None:
        csv_path = Path(args.csv)
        candidate = csv_path.parent / "summary.txt"
        if candidate.exists():
            summary_path = candidate
    summary = _load_summary_kv(summary_path) if summary_path is not None else {}

    config_path = str(args.config)
    if args.config == "configs/default.toml" and "config" in summary:
        config_path = summary["config"]
    cfg = load_config(config_path)

    seed = int(args.seed) if args.seed is not None else int(summary.get("seed", cfg.seed))
    pEmax = float(args.pEmax) if args.pEmax is not None else float(cfg.baselines.max_price_E)
    pNmax = float(args.pNmax) if args.pNmax is not None else float(cfg.baselines.max_price_N)
    eps_tol = float(args.eps_tol) if args.eps_tol is not None else float(summary.get("eps_tol", 1e-12))
    rng = np.random.default_rng(seed)
    users = sample_users(cfg, rng)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs")
    default_out_dir = output_root / f"run_stage1_vbbr_traj_on_heatmap_{timestamp}"
    if args.out_dir is None:
        out_dir = default_out_dir
    else:
        requested = Path(args.out_dir)
        if requested.is_absolute():
            out_dir = requested
        elif requested.parts and requested.parts[0] == output_root.name:
            out_dir = requested
        else:
            out_dir = output_root / requested
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv is None:
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
                f"\rEvaluating eps-heatmap grid: {done}/{total} ({ratio:.1f}%) elapsed={elapsed:.1f}s pE={pE:.4g} pN={pN:.4g}",
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
            pE_max=pEmax,
            pN_min=0.0,
            pN_max=pNmax,
            pE_points=int(args.pE_points),
            pN_points=int(args.pN_points),
            stage2_method=args.stage2_method,
            progress_cb=progress_cb,
        )
        pE_grid = grid.pE_grid
        pN_grid = grid.pN_grid
        eps = grid.eps
        heatmap_source = "computed"
    else:
        csv_path = Path(args.csv)
        pE_grid, pN_grid, eps = _load_grid_csv(csv_path)
        heatmap_source = str(csv_path)

    stack_cfg = replace(cfg.stackelberg, stage1_solver_variant="vbbr_brd")
    vbbr_updates: dict[str, int | float] = {}
    if args.search_max_iters is not None:
        vbbr_updates["search_max_iters"] = int(args.search_max_iters)
    if args.vbbr_local_r is not None:
        if args.vbbr_local_r < 0:
            raise ValueError("--vbbr-local-r must be >= 0")
        vbbr_updates["vbbr_local_R"] = int(args.vbbr_local_r)
    if args.vbbr_local_s is not None:
        if args.vbbr_local_s < 0:
            raise ValueError("--vbbr-local-s must be >= 0")
        vbbr_updates["vbbr_local_S"] = int(args.vbbr_local_s)
    if args.vbbr_local_budget is not None:
        if args.vbbr_local_budget < 0:
            raise ValueError("--vbbr-local-budget must be >= 0")
        vbbr_updates["vbbr_local_budget"] = int(args.vbbr_local_budget)
    if args.vbbr_top_m is not None:
        vbbr_updates["vbbr_top_m"] = int(args.vbbr_top_m)
    if args.vbbr_oracle_max_rounds is not None:
        vbbr_updates["vbbr_oracle_max_rounds"] = int(args.vbbr_oracle_max_rounds)
    if args.vbbr_no_improve_patience is not None:
        vbbr_updates["vbbr_no_improve_patience"] = int(args.vbbr_no_improve_patience)
    if args.vbbr_damping_alpha is not None:
        if args.vbbr_damping_alpha > 1.0:
            raise ValueError("--vbbr-damping-alpha must satisfy 0 < alpha <= 1")
        vbbr_updates["vbbr_damping_alpha"] = float(args.vbbr_damping_alpha)
    if args.vbbr_outer_update_mode is not None:
        vbbr_updates["vbbr_outer_update_mode"] = str(args.vbbr_outer_update_mode)
    if vbbr_updates:
        stack_cfg = replace(stack_cfg, **vbbr_updates)
    result = run_stage1_solver(users, cfg.system, stack_cfg)
    trajectory = _trajectory_points(result)

    fig_path = out_dir / "eps_heatmap_vbbr_trajectory.png"
    _plot_eps_with_trajectory(
        eps=eps,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        trajectory=trajectory,
        eps_tol=eps_tol,
        out_path=fig_path,
    )
    _save_trajectory_csv(out_dir / "vbbr_trajectory.csv", trajectory, pE_grid, pN_grid, eps)

    min_j, min_i = np.unravel_index(int(np.argmin(eps)), eps.shape)
    end_pE, end_pN = trajectory[-1]
    end_i = _nearest_idx(pE_grid, end_pE)
    end_j = _nearest_idx(pN_grid, end_pN)
    out_of_bounds_points = int(
        sum(
            1
            for pE, pN in trajectory
            if (pE < float(pE_grid[0]) or pE > float(pE_grid[-1]) or pN < float(pN_grid[0]) or pN > float(pN_grid[-1]))
        )
    )
    summary_lines = [
        f"config = {config_path}",
        f"seed = {seed}",
        f"heatmap_source = {heatmap_source}",
        f"stage1_solver_variant = vbbr_brd",
        f"vbbr_local_R = {stack_cfg.vbbr_local_R}",
        f"vbbr_local_S = {stack_cfg.vbbr_local_S}",
        f"vbbr_local_budget = {stack_cfg.vbbr_local_budget}",
        f"vbbr_top_m = {stack_cfg.vbbr_top_m}",
        f"vbbr_oracle_max_rounds = {stack_cfg.vbbr_oracle_max_rounds}",
        f"vbbr_no_improve_patience = {stack_cfg.vbbr_no_improve_patience}",
        f"vbbr_damping_alpha = {stack_cfg.vbbr_damping_alpha}",
        f"vbbr_outer_update_mode = {stack_cfg.vbbr_outer_update_mode}",
        f"stage2_method_for_heatmap = {args.stage2_method or cfg.baselines.stage2_solver_for_pricing}",
        f"pE_range = [{float(pE_grid[0])}, {float(pE_grid[-1])}], points = {int(pE_grid.size)}",
        f"pN_range = [{float(pN_grid[0])}, {float(pN_grid[-1])}], points = {int(pN_grid.size)}",
        f"trajectory_points = {len(trajectory)}",
        f"trajectory_points_out_of_heatmap = {out_of_bounds_points}",
        f"start_pE = {trajectory[0][0]:.10g}",
        f"start_pN = {trajectory[0][1]:.10g}",
        f"final_pE = {end_pE:.10g}",
        f"final_pN = {end_pN:.10g}",
        f"final_eps_on_nearest_grid = {float(eps[end_j, end_i]):.10g}",
        f"global_min_eps = {float(eps[min_j, min_i]):.10g}",
        f"global_argmin_pE = {float(pE_grid[min_i]):.10g}",
        f"global_argmin_pN = {float(pN_grid[min_j]):.10g}",
        f"stackelberg_stopping_reason = {result.stopping_reason}",
        f"stackelberg_outer_iterations = {result.outer_iterations}",
        f"stackelberg_stage2_oracle_calls = {result.stage2_oracle_calls}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

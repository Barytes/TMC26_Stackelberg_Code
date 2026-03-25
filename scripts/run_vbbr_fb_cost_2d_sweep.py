from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", str(Path(tempfile.gettempdir()) / "tmc26_cache")))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _figure_wrapper_utils import resolve_out_dir, write_csv_rows
from tmc26_exp.config import ExperimentConfig, load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import solve_stage1_pricing


def _parse_ratio_list(raw: str, *, name: str) -> list[float]:
    ratios = [float(item) for item in raw.split(",") if item.strip()]
    if not ratios:
        raise ValueError(f"{name} must contain at least one positive value.")
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError(f"All values in {name} must be positive.")
    return sorted(set(ratios))


def _load_cfg(config_path: str, n_users: int) -> ExperimentConfig:
    cfg = load_config(config_path)
    return replace(cfg, n_users=int(n_users))


def _trial_seed(seed: int, n_users: int, trial: int) -> int:
    return int(seed) + 10007 * int(n_users) + int(trial)


def _sample_users_for_trial(cfg: ExperimentConfig, seed: int, trial: int):
    rng = np.random.default_rng(_trial_seed(seed, cfg.n_users, trial))
    return sample_users(cfg, rng)


def _fb_from_ratio(product: float, ratio: float) -> tuple[float, float]:
    return float(math.sqrt(product * ratio)), float(math.sqrt(product / ratio))


def _grid_stats(
    rows: list[dict[str, object]],
    *,
    fb_ratios: list[float],
    cost_ratios: list[float],
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean_grid = np.full((len(cost_ratios), len(fb_ratios)), np.nan, dtype=float)
    std_grid = np.full_like(mean_grid, np.nan)
    for i, cost_ratio in enumerate(cost_ratios):
        for j, fb_ratio in enumerate(fb_ratios):
            vals = np.asarray(
                [
                    float(row[y_key])
                    for row in rows
                    if float(row["cost_ratio"]) == float(cost_ratio)
                    and float(row["fb_ratio"]) == float(fb_ratio)
                    and np.isfinite(float(row[y_key]))
                ],
                dtype=float,
            )
            if vals.size == 0:
                continue
            mean_grid[i, j] = float(np.mean(vals))
            std_grid[i, j] = float(np.std(vals))
    return mean_grid, std_grid


def _write_grid_summary_csv(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    fb_ratios: list[float],
    cost_ratios: list[float],
    metric_keys: list[str],
) -> None:
    summary_rows: list[dict[str, object]] = []
    for cost_ratio in cost_ratios:
        for fb_ratio in fb_ratios:
            bucket = [
                row
                for row in rows
                if float(row["cost_ratio"]) == float(cost_ratio) and float(row["fb_ratio"]) == float(fb_ratio)
            ]
            if not bucket:
                continue
            payload: dict[str, object] = {
                "fb_ratio": float(fb_ratio),
                "cost_ratio": float(cost_ratio),
                "F": float(bucket[0]["F"]),
                "B": float(bucket[0]["B"]),
                "cE": float(bucket[0]["cE"]),
                "cN": float(bucket[0]["cN"]),
                "trials": len(bucket),
            }
            for key in metric_keys:
                vals = np.asarray([float(row[key]) for row in bucket], dtype=float)
                payload[f"{key}_mean"] = float(np.mean(vals))
                payload[f"{key}_std"] = float(np.std(vals))
            summary_rows.append(payload)

    fieldnames = ["fb_ratio", "cost_ratio", "F", "B", "cE", "cN", "trials"]
    for key in metric_keys:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    write_csv_rows(out_path, fieldnames, summary_rows)


def _plot_heatmap_panels(
    rows: list[dict[str, object]],
    *,
    fb_ratios: list[float],
    cost_ratios: list[float],
    out_path: Path,
    title: str,
    panels: list[tuple[str, str, str]],
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(13.6, 5.2), dpi=170)
    if len(panels) == 1:
        axes = [axes]

    x_labels = [f"{ratio:.3g}" for ratio in fb_ratios]
    y_labels = [f"{ratio:.3g}" for ratio in cost_ratios]

    for ax, (metric_key, colorbar_label, cmap) in zip(axes, panels):
        mean_grid, _ = _grid_stats(rows, fb_ratios=fb_ratios, cost_ratios=cost_ratios, y_key=metric_key)
        im = ax.imshow(mean_grid, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(fb_ratios)))
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_yticks(np.arange(len(cost_ratios)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("F/B")
        ax.set_ylabel("cE/cN")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    fixed_cE: float,
    fb_ratios: list[float],
    cost_ratios: list[float],
    rows: list[dict[str, object]],
    runtime_sec: float,
) -> None:
    social_mean, _ = _grid_stats(rows, fb_ratios=fb_ratios, cost_ratios=cost_ratios, y_key="social_cost")
    revenue_mean, _ = _grid_stats(rows, fb_ratios=fb_ratios, cost_ratios=cost_ratios, y_key="joint_revenue")
    offload_mean, _ = _grid_stats(rows, fb_ratios=fb_ratios, cost_ratios=cost_ratios, y_key="offloading_ratio")

    min_social_idx = tuple(int(x) for x in np.unravel_index(np.nanargmin(social_mean), social_mean.shape))
    max_revenue_idx = tuple(int(x) for x in np.unravel_index(np.nanargmax(revenue_mean), revenue_mean.shape))
    max_offload_idx = tuple(int(x) for x in np.unravel_index(np.nanargmax(offload_mean), offload_mean.shape))

    lines = [
        f"config = {args.config}",
        f"n_users = {args.n_users}",
        f"seed = {args.seed}",
        f"trials = {args.trials}",
        "scan_mode = fixed_product_F_over_B_and_fixed_cE_over_cN",
        f"base_F = {cfg.system.F:.10g}",
        f"base_B = {cfg.system.B:.10g}",
        f"fixed_product = {cfg.system.F * cfg.system.B:.10g}",
        f"fixed_cE = {fixed_cE:.10g}",
        f"base_cN = {cfg.system.cN:.10g}",
        f"fb_ratio_list = {','.join(f'{ratio:.10g}' for ratio in fb_ratios)}",
        f"cost_ratio_list = {','.join(f'{ratio:.10g}' for ratio in cost_ratios)}",
        "same_user_realization_across_grid_within_trial = true",
        f"rows = {len(rows)}",
        f"runtime_sec = {runtime_sec:.10g}",
        f"min_mean_social_cost_fb_ratio = {fb_ratios[min_social_idx[1]]:.10g}",
        f"min_mean_social_cost_cost_ratio = {cost_ratios[min_social_idx[0]]:.10g}",
        f"max_mean_joint_revenue_fb_ratio = {fb_ratios[max_revenue_idx[1]]:.10g}",
        f"max_mean_joint_revenue_cost_ratio = {cost_ratios[max_revenue_idx[0]]:.10g}",
        f"max_mean_offloading_ratio_fb_ratio = {fb_ratios[max_offload_idx[1]]:.10g}",
        f"max_mean_offloading_ratio_cost_ratio = {cost_ratios[max_offload_idx[0]]:.10g}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2D VBBR sweep over resource asymmetry F/B and provider-cost ratio cE/cN, "
            "collecting equilibrium metrics and plotting heatmaps."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--fb-ratio-list", type=str, default="0.5,1,1.5,2,2.5,3,4")
    parser.add_argument("--cost-ratio-list", type=str, default="0.001,0.01,0.03,0.1,0.3,1,3,10,100,1000")
    parser.add_argument("--fixed-cE", type=float, default=None)
    parser.add_argument(
        "--solver-variant",
        type=str,
        default="vbbr_brd",
        choices=["vbbr_brd", "paper_iterative_pricing", "topk_brd"],
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    fb_ratios = _parse_ratio_list(args.fb_ratio_list, name="fb-ratio-list")
    cost_ratios = _parse_ratio_list(args.cost_ratio_list, name="cost-ratio-list")
    out_dir = resolve_out_dir("run_vbbr_fb_cost_2d_sweep", args.out_dir)
    cfg = _load_cfg(args.config, args.n_users)
    stack_cfg = replace(cfg.stackelberg, stage1_solver_variant=str(args.solver_variant))
    fixed_cE = float(cfg.system.cE if args.fixed_cE is None else args.fixed_cE)
    product = float(cfg.system.F * cfg.system.B)

    rows: list[dict[str, object]] = []
    t_total_start = time.perf_counter()
    for trial in range(1, int(args.trials) + 1):
        users = _sample_users_for_trial(cfg, args.seed, trial)
        seed_for_trial = _trial_seed(args.seed, cfg.n_users, trial)
        for cost_ratio in cost_ratios:
            cN_val = float(fixed_cE / cost_ratio)
            for fb_ratio in fb_ratios:
                F_val, B_val = _fb_from_ratio(product, fb_ratio)
                system = replace(cfg.system, F=F_val, B=B_val, cE=fixed_cE, cN=cN_val)
                t0 = time.perf_counter()
                result = solve_stage1_pricing(users, system, stack_cfg)
                runtime_sec = time.perf_counter() - t0

                inner = result.inner_result
                offloading_size = int(len(result.offloading_set))
                comp_utilization = float(np.sum(inner.f) / system.F)
                band_utilization = float(np.sum(inner.b) / system.B)
                final_pE = float(result.price[0])
                final_pN = float(result.price[1])
                esp_revenue = float(result.esp_revenue)
                nsp_revenue = float(result.nsp_revenue)

                rows.append(
                    {
                        "trial": int(trial),
                        "trial_seed": int(seed_for_trial),
                        "n_users": int(cfg.n_users),
                        "fb_ratio": float(fb_ratio),
                        "cost_ratio": float(cost_ratio),
                        "F": float(F_val),
                        "B": float(B_val),
                        "cE": float(system.cE),
                        "cN": float(system.cN),
                        "final_pE": final_pE,
                        "final_pN": final_pN,
                        "price_margin_E": float(final_pE - system.cE),
                        "price_margin_N": float(final_pN - system.cN),
                        "comp_utilization": comp_utilization,
                        "band_utilization": band_utilization,
                        "offloading_ratio": float(offloading_size / cfg.n_users if cfg.n_users > 0 else float("nan")),
                        "offloading_size": offloading_size,
                        "social_cost": float(result.social_cost),
                        "esp_revenue": esp_revenue,
                        "nsp_revenue": nsp_revenue,
                        "joint_revenue": float(esp_revenue + nsp_revenue),
                        "restricted_gap": float(result.restricted_gap),
                        "outer_iterations": int(result.outer_iterations),
                        "stage2_oracle_calls": int(result.stage2_oracle_calls),
                        "runtime_sec": float(runtime_sec),
                        "stage1_method": str(result.stage1_method),
                        "stopping_reason": str(result.stopping_reason),
                    }
                )
    runtime_total_sec = time.perf_counter() - t_total_start

    metric_keys = [
        "price_margin_E",
        "price_margin_N",
        "comp_utilization",
        "band_utilization",
        "offloading_ratio",
        "offloading_size",
        "social_cost",
        "joint_revenue",
        "restricted_gap",
        "runtime_sec",
    ]

    write_csv_rows(
        out_dir / "vbbr_fb_cost_2d_metrics.csv",
        [
            "trial",
            "trial_seed",
            "n_users",
            "fb_ratio",
            "cost_ratio",
            "F",
            "B",
            "cE",
            "cN",
            "final_pE",
            "final_pN",
            "price_margin_E",
            "price_margin_N",
            "comp_utilization",
            "band_utilization",
            "offloading_ratio",
            "offloading_size",
            "social_cost",
            "esp_revenue",
            "nsp_revenue",
            "joint_revenue",
            "restricted_gap",
            "outer_iterations",
            "stage2_oracle_calls",
            "runtime_sec",
            "stage1_method",
            "stopping_reason",
        ],
        rows,
    )
    _write_grid_summary_csv(
        out_dir / "vbbr_fb_cost_2d_grid_summary.csv",
        rows=rows,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        metric_keys=metric_keys,
    )

    _plot_heatmap_panels(
        rows,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        out_path=out_dir / "vbbr_fb_cost_2d_price_margins.png",
        title="2D sweep: equilibrium price margins",
        panels=[
            ("price_margin_E", "pE* - cE", "viridis"),
            ("price_margin_N", "pN* - cN", "magma"),
        ],
    )
    _plot_heatmap_panels(
        rows,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        out_path=out_dir / "vbbr_fb_cost_2d_utilization.png",
        title="2D sweep: resource utilization",
        panels=[
            ("comp_utilization", "Computation utilization", "YlGn"),
            ("band_utilization", "Bandwidth utilization", "YlOrRd"),
        ],
    )
    _plot_heatmap_panels(
        rows,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        out_path=out_dir / "vbbr_fb_cost_2d_offloading.png",
        title="2D sweep: offloading outcomes",
        panels=[
            ("offloading_ratio", "Offloading ratio", "PuBu"),
            ("offloading_size", "Number of offloading users", "BuPu"),
        ],
    )
    _plot_heatmap_panels(
        rows,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        out_path=out_dir / "vbbr_fb_cost_2d_welfare_revenue.png",
        title="2D sweep: social cost and joint revenue",
        panels=[
            ("social_cost", "Total user social cost", "cividis"),
            ("joint_revenue", "Joint provider revenue", "plasma"),
        ],
    )

    _write_summary(
        out_dir / "vbbr_fb_cost_2d_summary.txt",
        args=args,
        cfg=cfg,
        fixed_cE=fixed_cE,
        fb_ratios=fb_ratios,
        cost_ratios=cost_ratios,
        rows=rows,
        runtime_sec=runtime_total_sec,
    )


if __name__ == "__main__":
    main()

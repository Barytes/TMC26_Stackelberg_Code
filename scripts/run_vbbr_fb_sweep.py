from __future__ import annotations

import argparse
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


def _parse_ratio_list(raw: str) -> list[float]:
    ratios = [float(item) for item in raw.split(",") if item.strip()]
    if not ratios:
        raise ValueError("ratio-list must contain at least one positive value.")
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("All F/B ratios must be positive.")
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
    F_val = math.sqrt(product * ratio)
    B_val = math.sqrt(product / ratio)
    return float(F_val), float(B_val)


def _series_stats(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
) -> list[tuple[float, float, float]]:
    x_values = sorted({float(row[x_key]) for row in rows})
    stats: list[tuple[float, float, float]] = []
    for x_val in x_values:
        vals = np.asarray(
            [
                float(row[y_key])
                for row in rows
                if float(row[x_key]) == float(x_val) and np.isfinite(float(row[y_key]))
            ],
            dtype=float,
        )
        if vals.size == 0:
            stats.append((float(x_val), float("nan"), float("nan")))
        else:
            stats.append((float(x_val), float(np.mean(vals)), float(np.std(vals))))
    return stats


def _add_base_ratio_marker(ax: plt.Axes, base_ratio: float) -> None:
    ax.axvline(base_ratio, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)


def _plot_multi_series(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    base_ratio: float,
    y_specs: list[tuple[str, str, str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=160)
    for y_key, label, color, marker in y_specs:
        stats = _series_stats(rows, x_key="ratio", y_key=y_key)
        x = np.asarray([item[0] for item in stats], dtype=float)
        y = np.asarray([item[1] for item in stats], dtype=float)
        e = np.asarray([item[2] for item in stats], dtype=float)
        if np.all(~np.isfinite(y)):
            continue
        ax.plot(x, y, color=color, marker=marker, linewidth=2.0, markersize=5.8, label=label)
        if np.any(np.isfinite(e) & (e > 0.0)):
            ax.fill_between(x, y - e, y + e, color=color, alpha=0.14)
    _add_base_ratio_marker(ax, base_ratio)
    ax.set_xlabel("Resource asymmetry F/B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_single_metric_panels(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    base_ratio: float,
    panels: list[tuple[str, str, str, str]],
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(12.8, 4.8), dpi=160)
    if len(panels) == 1:
        axes = [axes]
    for ax, (y_key, ylabel, color, marker) in zip(axes, panels):
        stats = _series_stats(rows, x_key="ratio", y_key=y_key)
        x = np.asarray([item[0] for item in stats], dtype=float)
        y = np.asarray([item[1] for item in stats], dtype=float)
        e = np.asarray([item[2] for item in stats], dtype=float)
        if np.any(np.isfinite(y)):
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, markersize=5.8)
            if np.any(np.isfinite(e) & (e > 0.0)):
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.14)
        _add_base_ratio_marker(ax, base_ratio)
        ax.set_xlabel("Resource asymmetry F/B")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    solver_variant: str,
    ratios: list[float],
    rows: list[dict[str, object]],
    runtime_sec: float,
) -> None:
    social_stats = _series_stats(rows, x_key="ratio", y_key="social_cost")
    revenue_stats = _series_stats(rows, x_key="ratio", y_key="joint_revenue")
    offload_stats = _series_stats(rows, x_key="ratio", y_key="offloading_ratio")

    min_social_ratio = min(social_stats, key=lambda item: item[1])[0]
    max_revenue_ratio = max(revenue_stats, key=lambda item: item[1])[0]
    max_offload_ratio = max(offload_stats, key=lambda item: item[1])[0]

    lines = [
        f"config = {args.config}",
        f"n_users = {args.n_users}",
        f"seed = {args.seed}",
        f"trials = {args.trials}",
        f"solver_variant = {solver_variant}",
        "ratio_mode = fixed_product",
        f"base_F = {cfg.system.F:.10g}",
        f"base_B = {cfg.system.B:.10g}",
        f"base_ratio = {cfg.system.F / cfg.system.B:.10g}",
        f"fixed_product = {cfg.system.F * cfg.system.B:.10g}",
        f"ratio_list = {','.join(f'{ratio:.10g}' for ratio in ratios)}",
        "same_user_realization_across_ratios_within_trial = true",
        f"rows = {len(rows)}",
        f"runtime_sec = {runtime_sec:.10g}",
        f"min_mean_social_cost_ratio = {min_social_ratio:.10g}",
        f"max_mean_joint_revenue_ratio = {max_revenue_ratio:.10g}",
        f"max_mean_offloading_ratio_ratio = {max_offload_ratio:.10g}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run VBBR-based Stage I pricing on n=50-style instances while sweeping "
            "resource asymmetry F/B and plotting key equilibrium metrics."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ratio-list", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--solver-variant",
        type=str,
        default="vbbr_brd",
        choices=["vbbr_brd", "paper_iterative_pricing", "topk_brd"],
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    ratios = _parse_ratio_list(args.ratio_list)
    out_dir = resolve_out_dir("run_vbbr_fb_sweep", args.out_dir)
    cfg = _load_cfg(args.config, args.n_users)
    stack_cfg = replace(cfg.stackelberg, stage1_solver_variant=str(args.solver_variant))
    product = float(cfg.system.F * cfg.system.B)
    base_ratio = float(cfg.system.F / cfg.system.B)

    rows: list[dict[str, object]] = []
    t_total_start = time.perf_counter()
    for trial in range(1, int(args.trials) + 1):
        users = _sample_users_for_trial(cfg, args.seed, trial)
        seed_for_trial = _trial_seed(args.seed, cfg.n_users, trial)
        for ratio in ratios:
            F_val, B_val = _fb_from_ratio(product, ratio)
            system = replace(cfg.system, F=F_val, B=B_val)
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
                    "ratio": float(ratio),
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

    csv_path = out_dir / "vbbr_fb_sweep_metrics.csv"
    write_csv_rows(
        csv_path,
        [
            "trial",
            "trial_seed",
            "n_users",
            "ratio",
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

    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_price_margins.png",
        title="VBBR sweep: equilibrium price margins vs. F/B",
        ylabel="Price margin",
        base_ratio=base_ratio,
        y_specs=[
            ("price_margin_E", "pE* - cE", "tab:blue", "o"),
            ("price_margin_N", "pN* - cN", "tab:orange", "s"),
        ],
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_utilization.png",
        title="VBBR sweep: resource utilization vs. F/B",
        ylabel="Utilization",
        base_ratio=base_ratio,
        y_specs=[
            ("comp_utilization", "Computation utilization", "tab:green", "o"),
            ("band_utilization", "Bandwidth utilization", "tab:red", "s"),
        ],
    )
    _plot_single_metric_panels(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_offloading.png",
        title="VBBR sweep: offloading outcomes vs. F/B",
        base_ratio=base_ratio,
        panels=[
            ("offloading_ratio", "Offloading ratio", "tab:purple", "o"),
            ("offloading_size", "Number of offloading users", "tab:brown", "s"),
        ],
    )
    _plot_single_metric_panels(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_welfare_revenue.png",
        title="VBBR sweep: social cost and joint revenue vs. F/B",
        base_ratio=base_ratio,
        panels=[
            ("social_cost", "Total user social cost", "tab:blue", "o"),
            ("joint_revenue", "Joint provider revenue", "tab:orange", "s"),
        ],
    )

    _write_summary(
        out_dir / "vbbr_fb_sweep_summary.txt",
        args=args,
        cfg=cfg,
        solver_variant=str(args.solver_variant),
        ratios=ratios,
        rows=rows,
        runtime_sec=runtime_total_sec,
    )


if __name__ == "__main__":
    main()

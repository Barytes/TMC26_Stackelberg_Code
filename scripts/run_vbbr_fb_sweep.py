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
from matplotlib import font_manager
import numpy as np

from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
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


def _configure_fonts(language: str, font_scale: float) -> None:
    if language == "zh":
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",
        ]
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        chosen = [name for name in candidates if name in available]
        if chosen:
            existing = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = chosen + [name for name in existing if name not in chosen]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 13.5 * font_scale
    plt.rcParams["axes.titlesize"] = 17.0 * font_scale
    plt.rcParams["axes.labelsize"] = 15.0 * font_scale
    plt.rcParams["xtick.labelsize"] = 12.8 * font_scale
    plt.rcParams["ytick.labelsize"] = 12.8 * font_scale
    plt.rcParams["legend.fontsize"] = 12.0 * font_scale
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1


def _style_axis_black(ax: plt.Axes, *, include_x: bool = True, include_y: bool = True) -> None:
    if include_x:
        ax.xaxis.label.set_color("black")
        ax.tick_params(axis="x", colors="black")
    if include_y:
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="y", colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")


def _tick_label_text(value: float) -> str:
    return f"{float(value):g}"


def _equally_spaced_positions(values: np.ndarray, ticks: list[float]) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    tick_arr = np.asarray(ticks, dtype=float)
    for idx, value in enumerate(values):
        matches = np.where(np.isclose(tick_arr, float(value), rtol=0.0, atol=1e-9))[0]
        if matches.size:
            out[idx] = float(matches[0])
    return out


def _plot_multi_series(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    base_ratio: float,
    y_specs: list[tuple[str, str, str, str]],
    xlabel: str = "Resource asymmetry F/B",
    language: str = "en",
    font_scale: float = 1.0,
    x_ticks: list[float] | None = None,
    marker_ratio: float | None = None,
    equal_spacing: bool = False,
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig_width = 12.2 if x_ticks and len(x_ticks) > 8 else 9.6
        fig, ax = plt.subplots(figsize=(fig_width, 6.2), dpi=180)
        for y_key, label, color, marker in y_specs:
            stats = _series_stats(rows, x_key="ratio", y_key=y_key)
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            x_plot = _equally_spaced_positions(x, x_ticks) if (equal_spacing and x_ticks) else x
            if np.all(~np.isfinite(y)):
                continue
            mask = np.isfinite(x_plot) & np.isfinite(y)
            if not np.any(mask):
                continue
            ax.plot(x_plot[mask], y[mask], color=color, marker=marker, linewidth=2.0, markersize=6.0, label=label)
            if np.any(np.isfinite(e) & (e > 0.0)):
                ax.fill_between(x_plot[mask], (y - e)[mask], (y + e)[mask], color=color, alpha=0.14)
        marker_x = float(base_ratio if marker_ratio is None else marker_ratio)
        if equal_spacing and x_ticks:
            marker_positions = _equally_spaced_positions(np.asarray([marker_x], dtype=float), x_ticks)
            if np.isfinite(marker_positions[0]):
                marker_x = float(marker_positions[0])
        _add_base_ratio_marker(ax, marker_x)
        if x_ticks:
            tick_positions = np.arange(len(x_ticks), dtype=float) if equal_spacing else x_ticks
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([_tick_label_text(tick) for tick in x_ticks])
            if equal_spacing:
                ax.set_xlim(-0.35, len(x_ticks) - 0.65)
            else:
                ax.set_xlim(min(x_ticks) - 0.03, max(x_ticks) + 0.03)
            if len(x_ticks) > 8:
                ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        legend_kwargs: dict[str, object] = {"loc": legend_loc}
        if legend_bbox_to_anchor is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
        ax.legend(**legend_kwargs)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_metric_with_band(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    y_key: str,
    label: str,
    color: str,
    marker: str,
    x_ticks: list[float] | None = None,
    linestyle: str = "-",
    zorder: float = 2.0,
    markerfacecolor: str | None = None,
    band_mode: str = "std",
    equal_spacing: bool = False,
) -> bool:
    stats = _series_stats(rows, x_key="ratio", y_key=y_key)
    x = np.asarray([item[0] for item in stats], dtype=float)
    y = np.asarray([item[1] for item in stats], dtype=float)
    e = np.asarray([item[2] for item in stats], dtype=float)
    if band_mode == "sem":
        counts = np.asarray(
            [
                sum(
                    1
                    for row in rows
                    if float(row["ratio"]) == float(x_val) and np.isfinite(float(row[y_key]))
                )
                for x_val in x
            ],
            dtype=float,
        )
        e = np.where(counts > 0.0, e / np.sqrt(counts), 0.0)
    elif band_mode != "std":
        raise ValueError(f"Unsupported band_mode: {band_mode}")
    x_plot = _equally_spaced_positions(x, x_ticks) if (equal_spacing and x_ticks) else x
    if np.all(~np.isfinite(y)):
        return False
    mask = np.isfinite(x_plot) & np.isfinite(y)
    if not np.any(mask):
        return False
    ax.plot(
        x_plot[mask],
        y[mask],
        color=color,
        marker=marker,
        linewidth=2.0,
        markersize=6.0,
        linestyle=linestyle,
        markerfacecolor=(markerfacecolor if markerfacecolor is not None else color),
        markeredgecolor=color,
        markeredgewidth=1.1,
        label=label,
        zorder=zorder,
    )
    if np.any(np.isfinite(e) & (e > 0.0)):
        ax.fill_between(x_plot[mask], (y - e)[mask], (y + e)[mask], color=color, alpha=0.14, zorder=max(zorder - 1.0, 0.0))
    return True


def _axis_limits_from_stats(
    stats: list[tuple[float, float, float]],
    *,
    lower_pad_frac: float = 0.18,
    upper_pad_frac: float = 0.18,
) -> tuple[float, float] | None:
    values = np.asarray([item[1] for item in stats], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_max) * 0.12, 1.0)
        lower = y_min - pad
        upper = y_max + pad
    else:
        span = y_max - y_min
        lower = y_min - lower_pad_frac * span
        upper = y_max + upper_pad_frac * span
    return lower, upper


def _set_metric_axis_limits(
    ax: plt.Axes,
    stats: list[tuple[float, float, float]],
    *,
    lower_pad_frac: float = 0.18,
    upper_pad_frac: float = 0.18,
) -> None:
    limits = _axis_limits_from_stats(stats, lower_pad_frac=lower_pad_frac, upper_pad_frac=upper_pad_frac)
    if limits is None:
        return
    lower, upper = limits
    if lower >= 0.0:
        lower = max(0.0, lower)
    ax.set_ylim(lower, upper)


def _plot_dual_axis(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    base_ratio: float,
    left_spec: tuple[str, str, str, str],
    right_spec: tuple[str, str, str, str],
    language: str,
    font_scale: float = 1.35,
    left_band_mode: str = "std",
    right_band_mode: str = "std",
    x_ticks: list[float] | None = None,
    marker_ratio: float | None = None,
    equal_spacing: bool = False,
    right_linestyle: str = "-",
    zero_based_axes: bool = False,
    legend_loc: str = "center",
    legend_bbox_to_anchor: tuple[float, float] | None = (0.73, 0.52),
    legend_ncol: int = 1,
    left_pad_fracs: tuple[float, float] | None = None,
    right_pad_fracs: tuple[float, float] | None = None,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        plt.rcParams["axes.labelsize"] = 15.5 * font_scale
        plt.rcParams["legend.fontsize"] = 12.8 * font_scale

        fig_width = 12.2 if x_ticks and len(x_ticks) > 8 else 9.8
        fig, ax_left = plt.subplots(figsize=(fig_width, 6.4), dpi=180)
        ax_right = ax_left.twinx()

        left_key, left_label, left_color, left_marker = left_spec
        right_key, right_label, right_color, right_marker = right_spec

        left_drawn = _plot_metric_with_band(
            ax_left,
            rows,
            y_key=left_key,
            label=left_label,
            color=left_color,
            marker=left_marker,
            x_ticks=x_ticks,
            linestyle="--",
            zorder=4.0,
            markerfacecolor="white",
            band_mode=left_band_mode,
            equal_spacing=equal_spacing,
        )
        right_drawn = _plot_metric_with_band(
            ax_right,
            rows,
            y_key=right_key,
            label=right_label,
            color=right_color,
            marker=right_marker,
            x_ticks=x_ticks,
            linestyle=right_linestyle,
            zorder=3.0,
            band_mode=right_band_mode,
            equal_spacing=equal_spacing,
        )

        marker_x = float(base_ratio if marker_ratio is None else marker_ratio)
        if equal_spacing and x_ticks:
            marker_positions = _equally_spaced_positions(np.asarray([marker_x], dtype=float), x_ticks)
            if np.isfinite(marker_positions[0]):
                marker_x = float(marker_positions[0])
        _add_base_ratio_marker(ax_left, marker_x)

        if x_ticks:
            tick_positions = np.arange(len(x_ticks), dtype=float) if equal_spacing else x_ticks
            ax_left.set_xticks(tick_positions)
            ax_left.set_xticklabels([_tick_label_text(tick) for tick in x_ticks])
            if equal_spacing:
                ax_left.set_xlim(-0.35, len(x_ticks) - 0.65)
            else:
                ax_left.set_xlim(min(x_ticks) - 0.03, max(x_ticks) + 0.03)
            if len(x_ticks) > 8:
                ax_left.tick_params(axis="x", labelrotation=0)

        ax_left.set_xlabel(xlabel, color="black")
        ax_left.set_ylabel(left_label, color="black")
        ax_right.set_ylabel(right_label, color="black")
        _style_axis_black(ax_left, include_x=True, include_y=True)
        _style_axis_black(ax_right, include_x=False, include_y=True)
        left_stats = _series_stats(rows, x_key="ratio", y_key=left_key)
        right_stats = _series_stats(rows, x_key="ratio", y_key=right_key)
        if zero_based_axes:
            left_vals = np.asarray([item[1] for item in left_stats], dtype=float)
            right_vals = np.asarray([item[1] for item in right_stats], dtype=float)
            left_finite = left_vals[np.isfinite(left_vals)]
            right_finite = right_vals[np.isfinite(right_vals)]
            ax_left.set_ylim(bottom=0.0)
            ax_right.set_ylim(bottom=0.0)
            if left_finite.size > 0:
                left_upper = min(1.0, max(0.3, float(np.max(left_finite) * 1.25)))
                ax_left.set_ylim(top=left_upper)
            if right_finite.size > 0:
                right_upper = max(float(np.max(right_finite) + 2.0), float(np.max(right_finite) * 1.15))
                ax_right.set_ylim(top=right_upper)
        else:
            left_lower_pad, left_upper_pad = left_pad_fracs if left_pad_fracs is not None else (0.18, 0.18)
            right_lower_pad, right_upper_pad = right_pad_fracs if right_pad_fracs is not None else (0.18, 0.18)
            _set_metric_axis_limits(
                ax_left,
                left_stats,
                lower_pad_frac=left_lower_pad,
                upper_pad_frac=left_upper_pad,
            )
            _set_metric_axis_limits(
                ax_right,
                right_stats,
                lower_pad_frac=right_lower_pad,
                upper_pad_frac=right_upper_pad,
            )

        ax_left.set_title(title)
        ax_left.grid(alpha=0.25)

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        if left_drawn or right_drawn:
            legend_kwargs: dict[str, object] = {
                "loc": legend_loc,
                "ncol": legend_ncol,
                "columnspacing": 1.2,
                "handlelength": 2.8,
                "frameon": True,
            }
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            ax_left.legend(handles_left + handles_right, labels_left + labels_right, **legend_kwargs)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
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


def _load_metric_rows(path: Path) -> list[dict[str, object]]:
    int_keys = {
        "trial",
        "trial_seed",
        "n_users",
        "offloading_size",
        "outer_iterations",
        "stage2_oracle_calls",
    }
    float_keys = {
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
        "social_cost",
        "esp_revenue",
        "nsp_revenue",
        "joint_revenue",
        "restricted_gap",
        "runtime_sec",
    }
    rows: list[dict[str, object]] = []
    for raw in load_csv_rows(path):
        row: dict[str, object] = {}
        for key, value in raw.items():
            if key in int_keys:
                row[key] = int(value)
            elif key in float_keys:
                row[key] = float(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


def _load_summary_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        meta[key.strip()] = value.strip()
    return meta


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
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Existing vbbr_fb_sweep_metrics.csv used to replot without rerunning the sweep.",
    )
    parser.add_argument(
        "--base-ratio",
        type=float,
        default=None,
        help="Reference F/B ratio for the vertical marker when replotting from an existing CSV.",
    )
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    if args.out_dir is None and metrics_csv is not None:
        out_dir = metrics_csv.resolve().parent
    else:
        out_dir = resolve_out_dir("run_vbbr_fb_sweep", args.out_dir)

    rows: list[dict[str, object]]
    runtime_total_sec: float | None = None
    summary_meta = _load_summary_meta(metrics_csv.with_name("vbbr_fb_sweep_summary.txt")) if metrics_csv is not None else {}

    if metrics_csv is not None:
        rows = _load_metric_rows(metrics_csv)
        if not rows:
            raise ValueError(f"No rows found in metrics CSV: {metrics_csv}")
        ratios = sorted({float(row["ratio"]) for row in rows})
        if args.base_ratio is not None:
            base_ratio = float(args.base_ratio)
        elif "base_ratio" in summary_meta:
            base_ratio = float(summary_meta["base_ratio"])
        else:
            base_ratio = 1.0
    else:
        ratios = _parse_ratio_list(args.ratio_list)
        cfg = _load_cfg(args.config, args.n_users)
        stack_cfg = replace(cfg.stackelberg, stage1_solver_variant=str(args.solver_variant))
        product = float(cfg.system.F * cfg.system.B)
        base_ratio = float(cfg.system.F / cfg.system.B)

        rows = []
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

    preferred_x_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    x_ticks = (
        preferred_x_ticks
        if len(ratios) == len(preferred_x_ticks)
        and all(any(np.isclose(float(ratio), tick) for ratio in ratios) for tick in preferred_x_ticks)
        else ratios
    )
    equal_spacing = True
    marker_ratio = 1.0

    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_price_margins.png",
        title="Equilibrium unit profit vs. F/B",
        xlabel="Resource asymmetry F/B",
        ylabel="Equilibrium unit profit",
        base_ratio=base_ratio,
        y_specs=[
            ("price_margin_E", r"$p_E^* - c_E$", "tab:blue", "o"),
            ("price_margin_N", r"$p_N^* - c_N$", "tab:orange", "s"),
        ],
        language="en",
        font_scale=1.35,
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_price_margins_zh.png",
        title="\u5747\u8861\u5355\u4f4d\u5229\u6da6\u4e0e F/B \u7684\u5173\u7cfb",
        xlabel="\u8d44\u6e90\u4e0d\u5bf9\u79f0\u6bd4 F/B",
        ylabel="\u5747\u8861\u5355\u4f4d\u5229\u6da6",
        base_ratio=base_ratio,
        y_specs=[
            ("price_margin_E", r"$p_E^* - c_E$", "tab:blue", "o"),
            ("price_margin_N", r"$p_N^* - c_N$", "tab:orange", "s"),
        ],
        language="zh",
        font_scale=1.35,
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_utilization.png",
        title="Resource utilization vs. F/B",
        xlabel="Resource asymmetry F/B",
        ylabel="Utilization",
        base_ratio=base_ratio,
        y_specs=[
            ("comp_utilization", "Computation utilization", "tab:green", "o"),
            ("band_utilization", "Bandwidth utilization", "tab:red", "s"),
        ],
        language="en",
        font_scale=1.35,
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        legend_loc="lower left",
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_utilization_zh.png",
        title="\u8d44\u6e90\u5229\u7528\u7387\u4e0e F/B \u7684\u5173\u7cfb",
        xlabel="\u8d44\u6e90\u4e0d\u5bf9\u79f0\u6bd4 F/B",
        ylabel="\u5229\u7528\u7387",
        base_ratio=base_ratio,
        y_specs=[
            ("comp_utilization", "\u8ba1\u7b97\u8d44\u6e90\u5229\u7528\u7387", "tab:green", "o"),
            ("band_utilization", "\u5e26\u5bbd\u5229\u7528\u7387", "tab:red", "s"),
        ],
        language="zh",
        font_scale=1.35,
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        legend_loc="lower left",
    )
    _plot_dual_axis(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_offloading.png",
        title="Offloading outcomes vs. F/B",
        xlabel="Resource asymmetry F/B",
        base_ratio=base_ratio,
        left_spec=("offloading_ratio", "Offloading ratio", "tab:purple", "o"),
        right_spec=("offloading_size", "Number of offloading users", "tab:green", "s"),
        language="en",
        font_scale=1.35,
        left_band_mode="sem",
        right_band_mode="sem",
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        zero_based_axes=True,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )
    _plot_dual_axis(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_offloading_zh.png",
        title="\u5378\u8f7d\u7ed3\u679c\u4e0e F/B \u7684\u5173\u7cfb",
        xlabel="\u8d44\u6e90\u4e0d\u5bf9\u79f0\u6bd4 F/B",
        base_ratio=base_ratio,
        left_spec=("offloading_ratio", "\u5378\u8f7d\u6bd4\u4f8b", "tab:purple", "o"),
        right_spec=("offloading_size", "\u5378\u8f7d\u7528\u6237\u6570", "tab:green", "s"),
        language="zh",
        font_scale=1.35,
        left_band_mode="sem",
        right_band_mode="sem",
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        zero_based_axes=True,
        legend_loc="upper right",
        legend_bbox_to_anchor=None,
    )
    _plot_dual_axis(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_welfare_revenue.png",
        title="Social cost and joint revenue vs. F/B",
        xlabel="Resource asymmetry F/B",
        base_ratio=base_ratio,
        left_spec=("social_cost", "Total user social cost", "tab:blue", "o"),
        right_spec=("joint_revenue", "Service provider joint revenue", "tab:orange", "s"),
        language="en",
        font_scale=1.35,
        left_band_mode="sem",
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        legend_loc="lower left",
        legend_bbox_to_anchor=(0.02, 0.03),
        left_pad_fracs=(0.34, 0.16),
        right_pad_fracs=(0.34, 0.16),
    )
    _plot_dual_axis(
        rows,
        out_path=out_dir / "vbbr_fb_sweep_welfare_revenue_zh.png",
        title="\u793e\u4f1a\u6210\u672c\u4e0e\u8054\u5408\u6536\u76ca\u968f F/B \u7684\u53d8\u5316",
        xlabel="\u8d44\u6e90\u4e0d\u5bf9\u79f0\u6bd4 F/B",
        base_ratio=base_ratio,
        left_spec=("social_cost", "\u7528\u6237\u603b\u793e\u4f1a\u6210\u672c", "tab:blue", "o"),
        right_spec=("joint_revenue", "\u670d\u52a1\u63d0\u4f9b\u5546\u8054\u5408\u6536\u76ca", "tab:orange", "s"),
        language="zh",
        font_scale=1.35,
        left_band_mode="sem",
        x_ticks=x_ticks,
        marker_ratio=marker_ratio,
        equal_spacing=equal_spacing,
        legend_loc="lower left",
        legend_bbox_to_anchor=(0.02, 0.03),
        left_pad_fracs=(0.34, 0.16),
        right_pad_fracs=(0.34, 0.16),
    )

    if metrics_csv is None and runtime_total_sec is not None:
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

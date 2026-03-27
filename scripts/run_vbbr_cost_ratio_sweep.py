from __future__ import annotations

import argparse
from dataclasses import replace
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
        raise ValueError("All cE/cN ratios must be positive.")
    return sorted(set(ratios))


def _load_cfg(config_path: str, n_users: int) -> ExperimentConfig:
    cfg = load_config(config_path)
    return replace(cfg, n_users=int(n_users))


def _trial_seed(seed: int, n_users: int, trial: int) -> int:
    return int(seed) + 10007 * int(n_users) + int(trial)


def _sample_users_for_trial(cfg: ExperimentConfig, seed: int, trial: int):
    rng = np.random.default_rng(_trial_seed(seed, cfg.n_users, trial))
    return sample_users(cfg, rng)


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


def _plot_metric_with_band(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    *,
    y_key: str,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    zorder: float = 2.0,
    markerfacecolor: str | None = None,
    band_mode: str = "std",
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
    if np.all(~np.isfinite(y)):
        return False
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        linewidth=2.0,
        markersize=5.8,
        linestyle=linestyle,
        markerfacecolor=(markerfacecolor if markerfacecolor is not None else color),
        markeredgecolor=color,
        markeredgewidth=1.1,
        label=label,
        zorder=zorder,
    )
    if np.any(np.isfinite(e) & (e > 0.0)):
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.14, zorder=max(zorder - 1.0, 0.0))
    return True


def _plot_multi_series(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    base_ratio: float,
    xscale: str,
    y_specs: list[tuple[str, str, str, str]],
    xlabel: str = r"cost ratio $c_E/c_N$",
    language: str = "en",
    font_scale: float = 1.0,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=180)
        for y_key, label, color, marker in y_specs:
            stats = _series_stats(rows, x_key="ratio", y_key=y_key)
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            if np.all(~np.isfinite(y)):
                continue
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, markersize=6.0, label=label)
            if np.any(np.isfinite(e) & (e > 0.0)):
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.14)
        _add_base_ratio_marker(ax, base_ratio)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_offloading_dual_axis(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    base_ratio: float,
    xscale: str,
    left_spec: tuple[str, str, str, str],
    right_spec: tuple[str, str, str, str],
    language: str,
    font_scale: float = 1.0,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig, ax_left = plt.subplots(figsize=(9.8, 6.4), dpi=180)
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
            linestyle="--",
            zorder=4.0,
            markerfacecolor="white",
        )
        right_drawn = _plot_metric_with_band(
            ax_right,
            rows,
            y_key=right_key,
            label=right_label,
            color=right_color,
            marker=right_marker,
            linestyle="-",
            zorder=3.0,
        )

        _add_base_ratio_marker(ax_left, base_ratio)
        ax_left.set_xscale(xscale)
        ax_left.set_xlabel(xlabel, color="black")
        ax_left.set_ylabel(left_label, color="black")
        ax_right.set_ylabel(right_label, color="black")
        _style_axis_black(ax_left, include_x=True, include_y=True)
        _style_axis_black(ax_right, include_x=False, include_y=True)
        left_stats = _series_stats(rows, x_key="ratio", y_key=left_key)
        right_stats = _series_stats(rows, x_key="ratio", y_key=right_key)
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
        ax_left.set_title(title)
        ax_left.grid(alpha=0.25)

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        if left_drawn or right_drawn:
            ax_left.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_single_metric_panels(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    base_ratio: float,
    xscale: str,
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
        ax.set_xscale(xscale)
        ax.set_xlabel("Provider cost ratio cE/cN")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _set_metric_axis_limits(ax: plt.Axes, stats: list[tuple[float, float, float]]) -> None:
    values = np.asarray([item[1] for item in stats], dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if np.isclose(y_min, y_max):
        pad = max(abs(y_max) * 0.12, 1.0)
        lower = y_min - pad
        upper = y_max + pad
    else:
        span = y_max - y_min
        lower = y_min - 0.18 * span
        upper = y_max + 0.18 * span
    if lower >= 0.0:
        lower = max(0.0, lower)
    ax.set_ylim(lower, upper)


def _plot_welfare_revenue_dual_axis(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    base_ratio: float,
    xscale: str,
    left_spec: tuple[str, str, str, str],
    right_spec: tuple[str, str, str, str],
    language: str,
    font_scale: float = 1.38,
    left_band_mode: str = "std",
    right_band_mode: str = "std",
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        plt.rcParams["axes.labelsize"] = 15.5 * font_scale
        plt.rcParams["legend.fontsize"] = 12.8 * font_scale

        fig, ax_left = plt.subplots(figsize=(9.8, 6.4), dpi=180)
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
            linestyle="--",
            zorder=4.0,
            markerfacecolor="white",
            band_mode=left_band_mode,
        )
        right_drawn = _plot_metric_with_band(
            ax_right,
            rows,
            y_key=right_key,
            label=right_label,
            color=right_color,
            marker=right_marker,
            linestyle="-",
            zorder=3.0,
            band_mode=right_band_mode,
        )

        _add_base_ratio_marker(ax_left, base_ratio)
        ax_left.set_xscale(xscale)
        ax_left.set_xlabel(xlabel, color="black")
        ax_left.set_ylabel(left_label, color="black")
        ax_right.set_ylabel(right_label, color="black")
        _style_axis_black(ax_left, include_x=True, include_y=True)
        _style_axis_black(ax_right, include_x=False, include_y=True)

        _set_metric_axis_limits(ax_left, _series_stats(rows, x_key="ratio", y_key=left_key))
        _set_metric_axis_limits(ax_right, _series_stats(rows, x_key="ratio", y_key=right_key))

        ax_left.set_title(title)
        ax_left.grid(alpha=0.25)

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        if left_drawn or right_drawn:
            ax_left.legend(
                handles_left + handles_right,
                labels_left + labels_right,
                loc="center",
                bbox_to_anchor=(0.73, 0.52),
                ncol=1,
                columnspacing=1.2,
                handlelength=2.8,
                frameon=True,
            )
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    fixed_cE: float,
    solver_variant: str,
    ratios: list[float],
    rows: list[dict[str, object]],
    runtime_sec: float,
    xscale: str,
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
        "scan_mode = fixed_cE_scan_cN",
        f"fixed_cE = {fixed_cE:.10g}",
        f"base_cN = {cfg.system.cN:.10g}",
        f"base_ratio = {fixed_cE / cfg.system.cN:.10g}",
        f"ratio_list = {','.join(f'{ratio:.10g}' for ratio in ratios)}",
        f"xscale = {xscale}",
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
            "Run VBBR-based Stage I pricing on n=50-style instances while fixing cE, "
            "sweeping cN, and plotting key equilibrium metrics against cE/cN."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ratio-list", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--solver-variant",
        type=str,
        default="vbbr_brd",
        choices=["vbbr_brd", "paper_iterative_pricing", "topk_brd"],
    )
    parser.add_argument("--fixed-cE", type=float, default=None)
    parser.add_argument("--xscale", type=str, default="auto", choices=["auto", "linear", "log"])
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="Existing vbbr_cost_ratio_sweep_metrics.csv used to replot without rerunning the sweep.",
    )
    parser.add_argument(
        "--base-ratio",
        type=float,
        default=None,
        help="Reference cE/cN ratio for the vertical marker when replotting from an existing CSV.",
    )
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    if args.out_dir is None and metrics_csv is not None:
        out_dir = metrics_csv.resolve().parent
    else:
        out_dir = resolve_out_dir("run_vbbr_cost_ratio_sweep", args.out_dir)

    rows: list[dict[str, object]]
    runtime_total_sec: float | None = None
    summary_meta = (
        _load_summary_meta(metrics_csv.with_name("vbbr_cost_ratio_sweep_summary.txt")) if metrics_csv is not None else {}
    )

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
        if args.xscale == "auto":
            xscale = summary_meta.get("xscale", "auto")
            if xscale == "auto":
                ratio_span = max(ratios) / min(ratios)
                xscale = "log" if ratio_span >= 20.0 else "linear"
        else:
            xscale = str(args.xscale)
    else:
        ratios = _parse_ratio_list(args.ratio_list)
        cfg = _load_cfg(args.config, args.n_users)
        stack_cfg = replace(cfg.stackelberg, stage1_solver_variant=str(args.solver_variant))
        fixed_cE = float(cfg.system.cE if args.fixed_cE is None else args.fixed_cE)
        base_ratio = float(fixed_cE / cfg.system.cN)
        ratio_span = max(ratios) / min(ratios)
        xscale = "log" if (args.xscale == "auto" and ratio_span >= 20.0) else str(args.xscale)
        if xscale == "auto":
            xscale = "linear"

        rows = []
        t_total_start = time.perf_counter()
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users_for_trial(cfg, args.seed, trial)
            seed_for_trial = _trial_seed(args.seed, cfg.n_users, trial)
            for ratio in ratios:
                cN_val = float(fixed_cE / ratio)
                system = replace(cfg.system, cE=fixed_cE, cN=cN_val)
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
                        "cE": float(system.cE),
                        "cN": float(system.cN),
                        "final_pE": final_pE,
                        "final_pN": final_pN,
                        "price_margin_E": float(final_pE - system.cE),
                        "price_margin_N": float(final_pN - system.cN),
                        "comp_utilization": comp_utilization,
                        "band_utilization": band_utilization,
                        "offloading_ratio": float(
                            offloading_size / cfg.n_users if cfg.n_users > 0 else float("nan")
                        ),
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

        csv_path = out_dir / "vbbr_cost_ratio_sweep_metrics.csv"
        write_csv_rows(
            csv_path,
            [
                "trial",
                "trial_seed",
                "n_users",
                "ratio",
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
        out_path=out_dir / "vbbr_cost_ratio_sweep_price_margins.png",
        title=r"Equilibrium unit profit vs. $c_E/c_N$",
        xlabel=r"cost ratio $c_E/c_N$",
        ylabel="Equilibrium unit profit",
        base_ratio=base_ratio,
        xscale=xscale,
        y_specs=[
            ("price_margin_E", r"$p_E^* - c_E$", "tab:blue", "o"),
            ("price_margin_N", r"$p_N^* - c_N$", "tab:orange", "s"),
        ],
        language="en",
        font_scale=1.35,
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_price_margins_zh.png",
        title="\u5747\u8861\u5355\u4f4d\u5229\u6da6\u4e0e $c_E/c_N$ \u7684\u5173\u7cfb",
        xlabel="\u6210\u672c\u6bd4 $c_E/c_N$",
        ylabel="\u5747\u8861\u5355\u4f4d\u5229\u6da6",
        base_ratio=base_ratio,
        xscale=xscale,
        y_specs=[
            ("price_margin_E", r"$p_E^* - c_E$", "tab:blue", "o"),
            ("price_margin_N", r"$p_N^* - c_N$", "tab:orange", "s"),
        ],
        language="zh",
        font_scale=1.35,
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_utilization.png",
        title=r"Resource utilization vs. $c_E/c_N$",
        xlabel=r"cost ratio $c_E/c_N$",
        ylabel="Utilization",
        base_ratio=base_ratio,
        xscale=xscale,
        y_specs=[
            ("comp_utilization", "Computation utilization", "tab:green", "o"),
            ("band_utilization", "Bandwidth utilization", "tab:red", "s"),
        ],
        language="en",
        font_scale=1.35,
    )
    _plot_multi_series(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_utilization_zh.png",
        title="\u8d44\u6e90\u5229\u7528\u7387\u4e0e $c_E/c_N$ \u7684\u5173\u7cfb",
        xlabel="\u6210\u672c\u6bd4 $c_E/c_N$",
        ylabel="\u5229\u7528\u7387",
        base_ratio=base_ratio,
        xscale=xscale,
        y_specs=[
            ("comp_utilization", "\u8ba1\u7b97\u8d44\u6e90\u5229\u7528\u7387", "tab:green", "o"),
            ("band_utilization", "\u5e26\u5bbd\u5229\u7528\u7387", "tab:red", "s"),
        ],
        language="zh",
        font_scale=1.35,
    )
    _plot_offloading_dual_axis(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_offloading.png",
        title=r"User offloading outcome vs. $c_E/c_N$",
        xlabel=r"Service Provider cost ratio $c_E/c_N$",
        base_ratio=base_ratio,
        xscale=xscale,
        left_spec=("offloading_ratio", "Offloading ratio", "tab:purple", "o"),
        right_spec=("offloading_size", "Number of offloading users", "tab:brown", "s"),
        language="en",
        font_scale=1.35,
    )
    _plot_offloading_dual_axis(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_offloading_zh.png",
        title="\u7528\u6237\u5378\u8f7d\u7ed3\u679c\u4e0e $c_E/c_N$ \u7684\u5173\u7cfb",
        xlabel="\u670d\u52a1\u63d0\u4f9b\u5546\u6210\u672c\u6bd4 $c_E/c_N$",
        base_ratio=base_ratio,
        xscale=xscale,
        left_spec=("offloading_ratio", "\u5378\u8f7d\u6bd4\u4f8b", "tab:purple", "o"),
        right_spec=("offloading_size", "\u5378\u8f7d\u7528\u6237\u6570", "tab:brown", "s"),
        language="zh",
        font_scale=1.35,
    )
    _plot_single_metric_panels(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_welfare_revenue.png",
        title="VBBR sweep: social cost and joint revenue vs. cE/cN",
        base_ratio=base_ratio,
        xscale=xscale,
        panels=[
            ("social_cost", "Total user social cost", "tab:blue", "o"),
            ("joint_revenue", "Joint provider revenue", "tab:orange", "s"),
        ],
    )
    _plot_welfare_revenue_dual_axis(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_welfare_revenue.png",
        title="Social cost and joint revenue vs. cE/cN",
        xlabel=r"Provider cost ratio $c_E/c_N$",
        base_ratio=base_ratio,
        xscale=xscale,
        left_spec=("social_cost", "Total user social cost", "tab:blue", "o"),
        right_spec=("joint_revenue", "Joint provider revenue", "tab:orange", "s"),
        language="en",
        font_scale=1.38,
        left_band_mode="sem",
    )
    _plot_welfare_revenue_dual_axis(
        rows,
        out_path=out_dir / "vbbr_cost_ratio_sweep_welfare_revenue_zh.png",
        title="\u793e\u4f1a\u6210\u672c\u4e0e\u8054\u5408\u6536\u76ca\u968f $c_E/c_N$ \u7684\u53d8\u5316",
        xlabel="\u670d\u52a1\u63d0\u4f9b\u5546\u6210\u672c\u6bd4 $c_E/c_N$",
        base_ratio=base_ratio,
        xscale=xscale,
        left_spec=("social_cost", "\u7528\u6237\u603b\u793e\u4f1a\u6210\u672c", "tab:blue", "o"),
        right_spec=("joint_revenue", "\u8054\u5408\u670d\u52a1\u5546\u6536\u76ca", "tab:orange", "s"),
        language="zh",
        font_scale=1.38,
        left_band_mode="sem",
    )

    if metrics_csv is None and runtime_total_sec is not None:
        _write_summary(
            out_dir / "vbbr_cost_ratio_sweep_summary.txt",
            args=args,
            cfg=cfg,
            fixed_cE=fixed_cE,
            solver_variant=str(args.solver_variant),
            ratios=ratios,
            rows=rows,
            runtime_sec=runtime_total_sec,
            xscale=xscale,
        )


if __name__ == "__main__":
    main()

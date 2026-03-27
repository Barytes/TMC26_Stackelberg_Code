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
from matplotlib.lines import Line2D
import numpy as np

from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
from tmc26_exp.config import ExperimentConfig, load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import solve_stage1_pricing


def _parse_value_list(raw: str, *, name: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one positive value.")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"All values in {name} must be positive.")
    return sorted(set(values))


def _load_cfg(config_path: str, n_users: int) -> ExperimentConfig:
    cfg = load_config(config_path)
    return replace(cfg, n_users=int(n_users))


def _trial_seed(seed: int, n_users: int, trial: int) -> int:
    return int(seed) + 10007 * int(n_users) + int(trial)


def _sample_users_for_trial(cfg: ExperimentConfig, seed: int, trial: int):
    rng = np.random.default_rng(_trial_seed(seed, cfg.n_users, trial))
    return sample_users(cfg, rng)


def _same_float(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-8))


def _match_value(value: float, candidates: list[float]) -> float | None:
    for candidate in candidates:
        if _same_float(float(value), float(candidate)):
            return float(candidate)
    return None


def _default_f_b_lists(cfg: ExperimentConfig) -> tuple[list[float], list[float], list[float]]:
    ratio_anchors = [0.1, 0.4, 1.0, 2.5, 4.0]
    product = float(cfg.system.F * cfg.system.B)
    f_values = sorted(float(np.sqrt(product * ratio)) for ratio in ratio_anchors)
    b_values = sorted(float(np.sqrt(product / ratio)) for ratio in ratio_anchors)
    return f_values, b_values, ratio_anchors


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
    plt.rcParams["xtick.labelsize"] = 12.5 * font_scale
    plt.rcParams["ytick.labelsize"] = 12.5 * font_scale
    plt.rcParams["legend.fontsize"] = 11.5 * font_scale
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


def _format_tick(value: float) -> str:
    return f"{float(value):g}"


def _metric_row_key(row: dict[str, object]) -> tuple[int, float, float]:
    return (int(row["trial"]), float(row["F"]), float(row["B"]))


def _sort_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (int(row["trial"]), float(row["B"]), float(row["F"])))


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


def _validate_reuse_summary(
    meta: dict[str, str],
    *,
    config_path: str,
    n_users: int,
    seed: int,
    reuse_metrics_csv: Path,
) -> None:
    if not meta:
        return
    mismatches: list[str] = []
    if "config" in meta and Path(meta["config"]).as_posix() != Path(config_path).as_posix():
        mismatches.append(f"config={meta['config']}")
    if "n_users" in meta and int(meta["n_users"]) != int(n_users):
        mismatches.append(f"n_users={meta['n_users']}")
    if "seed" in meta and int(meta["seed"]) != int(seed):
        mismatches.append(f"seed={meta['seed']}")
    if mismatches:
        raise ValueError(
            f"Reuse metrics file {reuse_metrics_csv} does not match the requested run settings: "
            + ", ".join(mismatches)
        )


def _select_reusable_rows(
    rows: list[dict[str, object]],
    *,
    cfg: ExperimentConfig,
    seed: int,
    trials: int,
    f_values: list[float],
    b_values: list[float],
) -> list[dict[str, object]]:
    expected_trial_seeds = {
        int(trial): int(_trial_seed(seed, cfg.n_users, trial))
        for trial in range(1, int(trials) + 1)
    }
    selected_by_key: dict[tuple[int, float, float], dict[str, object]] = {}
    for row in rows:
        trial = int(row["trial"])
        if trial not in expected_trial_seeds:
            continue
        if int(row["n_users"]) != int(cfg.n_users):
            continue
        if int(row["trial_seed"]) != int(expected_trial_seeds[trial]):
            continue
        if not _same_float(float(row["cE"]), float(cfg.system.cE)):
            continue
        if not _same_float(float(row["cN"]), float(cfg.system.cN)):
            continue
        matched_f = _match_value(float(row["F"]), f_values)
        matched_b = _match_value(float(row["B"]), b_values)
        if matched_f is None or matched_b is None:
            continue
        key = (int(trial), float(matched_f), float(matched_b))
        if key in selected_by_key:
            continue
        normalized = dict(row)
        normalized["F"] = float(matched_f)
        normalized["B"] = float(matched_b)
        selected_by_key[key] = normalized
    return _sort_metric_rows(list(selected_by_key.values()))


def _series_stats(
    rows: list[dict[str, object]],
    *,
    x_values: list[float],
    group_values: list[float],
    x_key: str,
    group_key: str,
    y_key: str,
) -> dict[float, list[tuple[float, float, float]]]:
    out: dict[float, list[tuple[float, float, float]]] = {}
    for group_val in group_values:
        stats: list[tuple[float, float, float]] = []
        for x_val in x_values:
            vals = np.asarray(
                [
                    float(row[y_key])
                    for row in rows
                    if _same_float(float(row[x_key]), x_val)
                    and _same_float(float(row[group_key]), group_val)
                    and np.isfinite(float(row[y_key]))
                ],
                dtype=float,
            )
            if vals.size == 0:
                stats.append((float(x_val), float("nan"), float("nan")))
            else:
                stats.append((float(x_val), float(np.mean(vals)), float(np.std(vals))))
        out[float(group_val)] = stats
    return out


def _mean_grid(
    rows: list[dict[str, object]],
    *,
    f_values: list[float],
    b_values: list[float],
    y_key: str,
) -> np.ndarray:
    grid = np.full((len(b_values), len(f_values)), np.nan, dtype=float)
    for b_idx, b_val in enumerate(b_values):
        for f_idx, f_val in enumerate(f_values):
            vals = np.asarray(
                [
                    float(row[y_key])
                    for row in rows
                    if _same_float(float(row["F"]), f_val)
                    and _same_float(float(row["B"]), b_val)
                    and np.isfinite(float(row[y_key]))
                ],
                dtype=float,
            )
            if vals.size == 0:
                continue
            grid[b_idx, f_idx] = float(np.mean(vals))
    return grid


def _format_heatmap_cell(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 1000.0 or (0.0 < magnitude < 0.01):
        return f"{float(value):.2e}"
    if magnitude >= 100.0:
        return f"{float(value):.1f}"
    if magnitude >= 10.0:
        return f"{float(value):.2f}"
    return f"{float(value):.3f}"


def _plot_metric_heatmap(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    colorbar_label: str,
    x_values: list[float],
    b_values: list[float],
    y_key: str,
    cmap: str,
    language: str,
    font_scale: float = 1.25,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig, ax = plt.subplots(figsize=(8.6, 7.1), dpi=190)
        grid = _mean_grid(rows, f_values=x_values, b_values=b_values, y_key=y_key)
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            raise ValueError(f"No finite values found for heatmap metric: {y_key}")

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_xticklabels([_format_tick(value) for value in x_values])
        ax.set_yticks(np.arange(len(b_values)))
        ax.set_yticklabels([_format_tick(value) for value in b_values])
        if language == "zh":
            ax.set_xlabel("\u8ba1\u7b97\u8d44\u6e90\u5bb9\u91cf F")
            ax.set_ylabel("\u5e26\u5bbd\u5bb9\u91cf B")
        else:
            ax.set_xlabel("Computation capacity F")
            ax.set_ylabel("Bandwidth capacity B")
        ax.set_title(title)
        _style_axis_black(ax, include_x=True, include_y=True)

        span = float(np.max(finite) - np.min(finite))
        threshold = float(np.min(finite) + 0.58 * span) if span > 0.0 else float(np.min(finite))
        for b_idx in range(len(b_values)):
            for f_idx in range(len(x_values)):
                value = float(grid[b_idx, f_idx])
                if not np.isfinite(value):
                    continue
                text_color = "white" if value >= threshold else "black"
                ax.text(
                    f_idx,
                    b_idx,
                    _format_heatmap_cell(value),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10.0 * font_scale,
                )

        cbar = fig.colorbar(im, ax=ax, pad=0.03)
        cbar.set_label(colorbar_label)
        fig.tight_layout(pad=0.9)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _set_axis_limits_from_values(ax: plt.Axes, values: list[float], *, zero_based: bool = False) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return
    if zero_based:
        upper = float(np.max(finite))
        ax.set_ylim(0.0, max(0.3, upper * 1.18))
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


def _legend_handles_for_b(
    b_values: list[float],
    colors: list[tuple[float, float, float, float]],
) -> list[Line2D]:
    return [
        Line2D([0], [0], color=colors[idx], linewidth=2.2, marker="o", markersize=6.0, label=f"B={_format_tick(b_val)}")
        for idx, b_val in enumerate(b_values)
    ]


def _plot_grouped_multi_series(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: list[float],
    b_values: list[float],
    y_specs: list[tuple[str, str, str, str, bool]],
    language: str,
    font_scale: float = 1.35,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig, ax = plt.subplots(figsize=(11.8, 6.9), dpi=180)
        x_positions = np.arange(len(x_values), dtype=float)
        colors = list(plt.cm.viridis(np.linspace(0.12, 0.88, len(b_values))))

        all_values: list[float] = []
        for b_idx, b_val in enumerate(b_values):
            color = colors[b_idx]
            for y_key, _, linestyle, marker, hollow in y_specs:
                stats = _series_stats(rows, x_values=x_values, group_values=[b_val], x_key="F", group_key="B", y_key=y_key)[float(b_val)]
                y = np.asarray([item[1] for item in stats], dtype=float)
                mask = np.isfinite(y)
                if not np.any(mask):
                    continue
                ax.plot(
                    x_positions[mask],
                    y[mask],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5.8,
                    markerfacecolor=("white" if hollow else color),
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                )
                all_values.extend(float(value) for value in y[mask])

        ax.set_xticks(x_positions)
        ax.set_xticklabels([_format_tick(value) for value in x_values])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        _style_axis_black(ax, include_x=True, include_y=True)
        _set_axis_limits_from_values(ax, all_values)

        b_handles = _legend_handles_for_b(b_values, colors)
        if language == "zh":
            metric_title = "指标"
            b_title = "带宽容量 B"
        else:
            metric_title = "Metric"
            b_title = "Bandwidth capacity B"
        metric_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.2,
                linestyle=linestyle,
                marker=marker,
                markersize=6.0,
                markerfacecolor=("white" if hollow else "black"),
                markeredgecolor="black",
                label=label,
            )
            for _, label, linestyle, marker, hollow in y_specs
        ]
        legend_b = ax.legend(
            handles=b_handles,
            title=b_title,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            borderaxespad=0.0,
        )
        legend_metric = ax.legend(
            handles=metric_handles,
            title=metric_title,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            frameon=True,
            borderaxespad=0.0,
        )
        ax.add_artist(legend_b)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_grouped_dual_axis(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    title: str,
    xlabel: str,
    x_values: list[float],
    b_values: list[float],
    left_spec: tuple[str, str, str, str, bool],
    right_spec: tuple[str, str, str, str, bool],
    language: str,
    font_scale: float = 1.35,
    zero_based: bool = False,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        fig, ax_left = plt.subplots(figsize=(11.8, 6.9), dpi=180)
        ax_right = ax_left.twinx()
        x_positions = np.arange(len(x_values), dtype=float)
        colors = list(plt.cm.viridis(np.linspace(0.12, 0.88, len(b_values))))

        left_key, left_label, left_style, left_marker, left_hollow = left_spec
        right_key, right_label, right_style, right_marker, right_hollow = right_spec
        left_values_all: list[float] = []
        right_values_all: list[float] = []

        for b_idx, b_val in enumerate(b_values):
            color = colors[b_idx]
            left_stats = _series_stats(rows, x_values=x_values, group_values=[b_val], x_key="F", group_key="B", y_key=left_key)[float(b_val)]
            right_stats = _series_stats(rows, x_values=x_values, group_values=[b_val], x_key="F", group_key="B", y_key=right_key)[float(b_val)]
            y_left = np.asarray([item[1] for item in left_stats], dtype=float)
            y_right = np.asarray([item[1] for item in right_stats], dtype=float)
            mask_left = np.isfinite(y_left)
            mask_right = np.isfinite(y_right)
            if np.any(mask_left):
                ax_left.plot(
                    x_positions[mask_left],
                    y_left[mask_left],
                    color=color,
                    linestyle=left_style,
                    marker=left_marker,
                    linewidth=2.0,
                    markersize=5.8,
                    markerfacecolor=("white" if left_hollow else color),
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                )
                left_values_all.extend(float(value) for value in y_left[mask_left])
            if np.any(mask_right):
                ax_right.plot(
                    x_positions[mask_right],
                    y_right[mask_right],
                    color=color,
                    linestyle=right_style,
                    marker=right_marker,
                    linewidth=2.0,
                    markersize=5.8,
                    markerfacecolor=("white" if right_hollow else color),
                    markeredgecolor=color,
                    markeredgewidth=1.1,
                )
                right_values_all.extend(float(value) for value in y_right[mask_right])

        ax_left.set_xticks(x_positions)
        ax_left.set_xticklabels([_format_tick(value) for value in x_values])
        ax_left.set_xlabel(xlabel)
        ax_left.set_ylabel(left_label)
        ax_right.set_ylabel(right_label)
        ax_left.set_title(title)
        ax_left.grid(alpha=0.25)
        _style_axis_black(ax_left, include_x=True, include_y=True)
        _style_axis_black(ax_right, include_x=False, include_y=True)
        _set_axis_limits_from_values(ax_left, left_values_all, zero_based=zero_based)
        _set_axis_limits_from_values(ax_right, right_values_all, zero_based=zero_based)

        b_handles = _legend_handles_for_b(b_values, colors)
        if language == "zh":
            metric_title = "指标"
            b_title = "带宽容量 B"
        else:
            metric_title = "Metric"
            b_title = "Bandwidth capacity B"
        metric_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.2,
                linestyle=left_style,
                marker=left_marker,
                markersize=6.0,
                markerfacecolor=("white" if left_hollow else "black"),
                markeredgecolor="black",
                label=left_label,
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.2,
                linestyle=right_style,
                marker=right_marker,
                markersize=6.0,
                markerfacecolor=("white" if right_hollow else "black"),
                markeredgecolor="black",
                label=right_label,
            ),
        ]
        legend_b = ax_left.legend(
            handles=b_handles,
            title=b_title,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            borderaxespad=0.0,
        )
        legend_metric = ax_left.legend(
            handles=metric_handles,
            title=metric_title,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            frameon=True,
            borderaxespad=0.0,
        )
        ax_left.add_artist(legend_b)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _write_summary(
    out_path: Path,
    *,
    args: argparse.Namespace,
    f_values: list[float],
    b_values: list[float],
    rows: list[dict[str, object]],
    runtime_sec: float,
    derived_ratio_anchors: list[float] | None,
    reused_rows: int = 0,
    new_rows: int = 0,
    reuse_metrics_csv: Path | None = None,
) -> None:
    lines = [
        f"config = {args.config}",
        f"n_users = {args.n_users}",
        f"seed = {args.seed}",
        f"trials = {args.trials}",
        "scan_mode = direct_F_B_grid",
        f"F_list = {','.join(f'{value:.10g}' for value in f_values)}",
        f"B_list = {','.join(f'{value:.10g}' for value in b_values)}",
        f"target_grid_points = {len(f_values) * len(b_values)}",
        f"expected_rows = {len(f_values) * len(b_values) * int(args.trials)}",
        f"rows = {len(rows)}",
        f"runtime_sec = {runtime_sec:.10g}",
        "same_user_realization_across_grid_within_trial = true",
        f"reused_rows = {int(reused_rows)}",
        f"new_rows = {int(new_rows)}",
    ]
    if reuse_metrics_csv is not None:
        lines.append(f"reuse_metrics_csv = {reuse_metrics_csv}")
    if derived_ratio_anchors is not None:
        lines.append(f"derived_from_fb_ratio_anchors = {','.join(f'{value:.10g}' for value in derived_ratio_anchors)}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grid_summary_csv(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    f_values: list[float],
    b_values: list[float],
    metric_keys: list[str],
) -> None:
    summary_rows: list[dict[str, object]] = []
    for b_val in b_values:
        for f_val in f_values:
            bucket = [
                row
                for row in rows
                if _same_float(float(row["F"]), f_val) and _same_float(float(row["B"]), b_val)
            ]
            if not bucket:
                continue
            payload: dict[str, object] = {
                "F": float(f_val),
                "B": float(b_val),
                "trials": len(bucket),
            }
            for key in metric_keys:
                vals = np.asarray([float(row[key]) for row in bucket], dtype=float)
                payload[f"{key}_mean"] = float(np.mean(vals))
                payload[f"{key}_std"] = float(np.std(vals))
            summary_rows.append(payload)
    fieldnames = ["F", "B", "trials"]
    for key in metric_keys:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])
    write_csv_rows(out_path, fieldnames, summary_rows)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct 2D VBBR sweep over F and B using default Stackelberg settings, "
            "then plot single-figure line charts with F on the x-axis and one curve per B."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--F-list", type=str, default=None)
    parser.add_argument("--B-list", type=str, default=None)
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
        help="Existing vbbr_fb_grid_sweep_metrics.csv used to replot without rerunning the grid sweep.",
    )
    parser.add_argument(
        "--reuse-metrics-csv",
        type=str,
        default=None,
        help="Existing vbbr_fb_grid_sweep_metrics.csv reused to skip already-computed trial/F/B points.",
    )
    args = parser.parse_args()
    if args.metrics_csv is not None and args.reuse_metrics_csv is not None:
        raise ValueError("--metrics-csv and --reuse-metrics-csv cannot be used together.")

    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else None
    reuse_metrics_csv = Path(args.reuse_metrics_csv) if args.reuse_metrics_csv else None
    if args.out_dir is None and metrics_csv is not None:
        out_dir = metrics_csv.resolve().parent
    else:
        out_dir = resolve_out_dir("run_vbbr_fb_grid_sweep", args.out_dir)

    rows: list[dict[str, object]]
    runtime_total_sec: float | None = None
    derived_ratio_anchors: list[float] | None = None
    reused_row_count = 0
    new_row_count = 0

    if metrics_csv is not None:
        rows = _load_metric_rows(metrics_csv)
        if not rows:
            raise ValueError(f"No rows found in metrics CSV: {metrics_csv}")
        f_values = sorted({float(row["F"]) for row in rows})
        b_values = sorted({float(row["B"]) for row in rows})
    else:
        cfg = _load_cfg(args.config, args.n_users)
        stack_cfg = replace(cfg.stackelberg, stage1_solver_variant=str(args.solver_variant))
        if args.F_list is None or args.B_list is None:
            default_f_values, default_b_values, derived_ratio_anchors = _default_f_b_lists(cfg)
            f_values = default_f_values if args.F_list is None else _parse_value_list(args.F_list, name="F-list")
            b_values = default_b_values if args.B_list is None else _parse_value_list(args.B_list, name="B-list")
        else:
            f_values = _parse_value_list(args.F_list, name="F-list")
            b_values = _parse_value_list(args.B_list, name="B-list")

        rows = []
        if reuse_metrics_csv is not None:
            reuse_meta = _load_summary_meta(reuse_metrics_csv.with_name("vbbr_fb_grid_sweep_summary.txt"))
            _validate_reuse_summary(
                reuse_meta,
                config_path=args.config,
                n_users=cfg.n_users,
                seed=args.seed,
                reuse_metrics_csv=reuse_metrics_csv,
            )
            rows = _select_reusable_rows(
                _load_metric_rows(reuse_metrics_csv),
                cfg=cfg,
                seed=args.seed,
                trials=args.trials,
                f_values=f_values,
                b_values=b_values,
            )
            reused_row_count = len(rows)
        existing_keys = {_metric_row_key(row) for row in rows}
        t_total_start = time.perf_counter()
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users_for_trial(cfg, args.seed, trial)
            seed_for_trial = _trial_seed(args.seed, cfg.n_users, trial)
            for b_val in b_values:
                for f_val in f_values:
                    key = (int(trial), float(f_val), float(b_val))
                    if key in existing_keys:
                        continue
                    system = replace(cfg.system, F=float(f_val), B=float(b_val))
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
                            "F": float(f_val),
                            "B": float(b_val),
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
                    existing_keys.add(key)
                    new_row_count += 1
        runtime_total_sec = time.perf_counter() - t_total_start
        rows = _sort_metric_rows(rows)

        write_csv_rows(
            out_dir / "vbbr_fb_grid_sweep_metrics.csv",
            [
                "trial",
                "trial_seed",
                "n_users",
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
            out_dir / "vbbr_fb_grid_sweep_grid_summary.csv",
            rows=rows,
            f_values=f_values,
            b_values=b_values,
            metric_keys=[
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
            ],
        )

    heatmap_specs = [
        {
            "key": "price_margin_E",
            "stem": "esp_unit_profit_heatmap",
            "title_en": "Mean ESP unit profit heatmap",
            "title_zh": "\u5e73\u5747 ESP \u5355\u4f4d\u5229\u6da6\u70ed\u56fe",
            "label_en": "ESP unit profit",
            "label_zh": "ESP\u5355\u4f4d\u5229\u6da6",
            "cmap": "viridis",
        },
        {
            "key": "price_margin_N",
            "stem": "nsp_unit_profit_heatmap",
            "title_en": "Mean NSP unit profit heatmap",
            "title_zh": "\u5e73\u5747 NSP \u5355\u4f4d\u5229\u6da6\u70ed\u56fe",
            "label_en": "NSP unit profit",
            "label_zh": "NSP\u5355\u4f4d\u5229\u6da6",
            "cmap": "magma",
        },
        {
            "key": "comp_utilization",
            "stem": "comp_utilization_heatmap",
            "title_en": "Mean computation utilization heatmap",
            "title_zh": "\u5e73\u5747\u8ba1\u7b97\u8d44\u6e90\u5229\u7528\u7387\u70ed\u56fe",
            "label_en": "Computation utilization",
            "label_zh": "\u8ba1\u7b97\u8d44\u6e90\u5229\u7528\u7387",
            "cmap": "YlGn",
        },
        {
            "key": "band_utilization",
            "stem": "band_utilization_heatmap",
            "title_en": "Mean bandwidth utilization heatmap",
            "title_zh": "\u5e73\u5747\u5e26\u5bbd\u5229\u7528\u7387\u70ed\u56fe",
            "label_en": "Bandwidth utilization",
            "label_zh": "\u5e26\u5bbd\u5229\u7528\u7387",
            "cmap": "YlOrRd",
        },
        {
            "key": "offloading_ratio",
            "stem": "offloading_ratio_heatmap",
            "title_en": "Mean offloading ratio heatmap",
            "title_zh": "\u5e73\u5747\u5378\u8f7d\u6bd4\u4f8b\u70ed\u56fe",
            "label_en": "Offloading ratio",
            "label_zh": "\u5378\u8f7d\u6bd4\u4f8b",
            "cmap": "PuBu",
        },
        {
            "key": "offloading_size",
            "stem": "offloading_users_heatmap",
            "title_en": "Mean number of offloading users heatmap",
            "title_zh": "\u5e73\u5747\u5378\u8f7d\u7528\u6237\u6570\u70ed\u56fe",
            "label_en": "Number of offloading users",
            "label_zh": "\u5378\u8f7d\u7528\u6237\u6570",
            "cmap": "BuPu",
        },
        {
            "key": "social_cost",
            "stem": "social_cost_heatmap",
            "title_en": "Mean social cost heatmap",
            "title_zh": "\u5e73\u5747\u793e\u4f1a\u6210\u672c\u70ed\u56fe",
            "label_en": "Total user social cost",
            "label_zh": "\u7528\u6237\u603b\u793e\u4f1a\u6210\u672c",
            "cmap": "cividis",
        },
        {
            "key": "joint_revenue",
            "stem": "joint_revenue_heatmap",
            "title_en": "Mean joint revenue heatmap",
            "title_zh": "\u5e73\u5747\u8054\u5408\u6536\u76ca\u70ed\u56fe",
            "label_en": "Joint provider revenue",
            "label_zh": "\u8054\u5408\u670d\u52a1\u5546\u6536\u76ca",
            "cmap": "plasma",
        },
    ]
    for spec in heatmap_specs:
        _plot_metric_heatmap(
            rows,
            out_path=out_dir / f"vbbr_fb_grid_sweep_{spec['stem']}.png",
            title=str(spec["title_en"]),
            colorbar_label=str(spec["label_en"]),
            x_values=f_values,
            b_values=b_values,
            y_key=str(spec["key"]),
            cmap=str(spec["cmap"]),
            language="en",
        )
        _plot_metric_heatmap(
            rows,
            out_path=out_dir / f"vbbr_fb_grid_sweep_{spec['stem']}_zh.png",
            title=str(spec["title_zh"]),
            colorbar_label=str(spec["label_zh"]),
            x_values=f_values,
            b_values=b_values,
            y_key=str(spec["key"]),
            cmap=str(spec["cmap"]),
            language="zh",
        )

    if metrics_csv is None and runtime_total_sec is not None:
        _write_summary(
            out_dir / "vbbr_fb_grid_sweep_summary.txt",
            args=args,
            f_values=f_values,
            b_values=b_values,
            rows=rows,
            runtime_sec=runtime_total_sec,
            derived_ratio_anchors=derived_ratio_anchors,
            reused_rows=reused_row_count,
            new_rows=new_row_count,
            reuse_metrics_csv=reuse_metrics_csv,
        )


if __name__ == "__main__":
    main()

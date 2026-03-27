from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import os
from pathlib import Path
import sys

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", "/tmp/tmc26_cache"))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

from _figure_missing_impl import (
    _load_cfg,
    _load_grid_ne_gap_surface,
    _nearest_grid_ne_gap,
    _run_stage1_method,
    _sample_users,
    _write_summary,
)
from _figure_wrapper_utils import resolve_out_dir, write_csv_rows
from tmc26_exp.baselines import BaselineOutcome, _grid_ne_gap_audit, _price_cache_key

TRIAL_FIELDS = [
    "method",
    "n_users",
    "trial",
    "success",
    "final_pE",
    "final_pN",
    "offloading_size",
    "restricted_gap",
    "final_grid_ne_gap",
    "final_grid_ne_gap_source",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "runtime_sec",
    "stage2_solver_calls",
    "audit_stage2_solver_calls",
    "total_stage2_solver_calls",
    "error",
]

SUMMARY_METRICS = [
    "final_grid_ne_gap",
    "joint_revenue",
    "runtime_sec",
    "stage2_solver_calls",
    "audit_stage2_solver_calls",
    "total_stage2_solver_calls",
]

METHOD_COLORS = {
    "Proposed": "tab:blue",
    "GA": "tab:red",
    "BO": "tab:orange",
    "BO-online": "tab:purple",
    "MARL": "tab:green",
}


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

METHOD_MARKERS = {
    "Proposed": "o",
    "GA": "s",
    "BO": "D",
    "BO-online": "D",
    "MARL": "^",
}

METHOD_SHORT_LABELS = {
    "Proposed": "P",
    "GA": "GA",
    "BO": "BO",
    "BO-online": "BO",
    "MARL": "M",
}

N_MARKER_CYCLE = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _parse_n_users_list(raw: str) -> list[int]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n-users-list cannot be empty.")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Each n in n-users-list must be > 0.")
    return values


def _parse_methods(raw: str) -> list[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("methods cannot be empty.")
    allowed = {
        "proposed": "Proposed",
        "ga": "GA",
        "bo": "BO",
        "bo-online": "BO-online",
        "bo_online": "BO-online",
        "marl": "MARL",
    }
    methods: list[str] = []
    for item in items:
        key = item.lower()
        if key not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported method: {item}. Allowed: Proposed, GA, BO, BO-online, MARL.")
        methods.append(allowed[key])
    return methods


def _resolve_gap_heatmap_csv_path(template: str | None, n_users: int) -> Path | None:
    if template is None:
        return None
    candidate = Path(str(template).format(n=int(n_users)))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    return candidate if candidate.exists() else None


def _load_gap_heatmap_surfaces(
    n_list: list[int],
    gap_heatmap_csv_template: str | None,
) -> dict[int, tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
    surfaces: dict[int, tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = {}
    if gap_heatmap_csv_template is None:
        return surfaces
    for n_users in sorted({int(n) for n in n_list}):
        csv_path = _resolve_gap_heatmap_csv_path(gap_heatmap_csv_template, n_users)
        if csv_path is None:
            continue
        pE_grid, pN_grid, grid_ne_gap = _load_grid_ne_gap_surface(csv_path)
        surfaces[int(n_users)] = (csv_path, pE_grid, pN_grid, grid_ne_gap)
    return surfaces


def _build_current_outcome(
    method: str,
    price: tuple[float, float],
    offloading_set: tuple[int, ...],
    social_cost: float,
    esp_revenue: float,
    nsp_revenue: float,
) -> BaselineOutcome:
    return BaselineOutcome(
        name=str(method),
        price=(float(price[0]), float(price[1])),
        offloading_set=tuple(int(x) for x in offloading_set),
        social_cost=float(social_cost),
        esp_revenue=float(esp_revenue),
        nsp_revenue=float(nsp_revenue),
        grid_ne_gap=float("nan"),
        legacy_gain_proxy=float("nan"),
        meta={},
    )


def _apply_baseline_overrides(
    base_cfg,
    *,
    bo_candidate_pool: int | None,
    bo_iters: int | None,
    ga_population_size: int | None,
    ga_generations: int | None,
    marl_price_levels: int | None,
    marl_episodes: int | None,
    marl_steps_per_episode: int | None,
):
    updates: dict[str, int] = {}
    if bo_candidate_pool is not None:
        updates["bo_candidate_pool"] = max(1, int(bo_candidate_pool))
    if bo_iters is not None:
        updates["bo_iters"] = max(0, int(bo_iters))
    if ga_population_size is not None:
        updates["ga_population_size"] = max(2, int(ga_population_size))
    if ga_generations is not None:
        updates["ga_generations"] = max(0, int(ga_generations))
    if marl_price_levels is not None:
        updates["marl_price_levels"] = max(2, int(marl_price_levels))
    if marl_episodes is not None:
        updates["marl_episodes"] = max(1, int(marl_episodes))
    if marl_steps_per_episode is not None:
        updates["marl_steps_per_episode"] = max(1, int(marl_steps_per_episode))
    return replace(base_cfg, **updates) if updates else base_cfg


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_trials(rows: list[dict[str, object]], methods: list[str], n_list: list[int]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        for n in n_list:
            subset = [
                row
                for row in rows
                if str(row["method"]) == method and int(row["n_users"]) == int(n)
            ]
            success_rows = [row for row in subset if int(row["success"]) == 1]
            summary: dict[str, object] = {
                "method": method,
                "n_users": int(n),
                "count": int(len(success_rows)),
                "failure_count": int(len(subset) - len(success_rows)),
            }
            for metric in SUMMARY_METRICS:
                vals = np.asarray(
                    [
                        float(row[metric])
                        for row in success_rows
                        if row.get(metric, "") not in {"", None} and np.isfinite(float(row[metric]))
                    ],
                    dtype=float,
                )
                stats = _finite_stats(vals)
                for stat_name, stat_value in stats.items():
                    summary[f"{metric}_{stat_name}"] = float(stat_value)
            summary_rows.append(summary)
    return summary_rows


def _summary_fieldnames() -> list[str]:
    fields = ["method", "n_users", "count", "failure_count"]
    for metric in SUMMARY_METRICS:
        for suffix in ["mean", "std", "median", "q25", "q75", "min", "max"]:
            fields.append(f"{metric}_{suffix}")
    return fields


def load_summary_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_trial_rows(csv_path: Path) -> list[dict[str, object]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _extract_metric_series(
    summary_rows: list[dict[str, object]],
    *,
    method: str,
    metric: str,
    statistic: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [row for row in summary_rows if str(row["method"]) == method]
    xs = np.asarray([int(row["n_users"]) for row in rows], dtype=float)
    if statistic == "mean_std":
        center = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=float)
        spread = np.asarray([float(row[f"{metric}_std"]) for row in rows], dtype=float)
        low = center - spread
        high = center + spread
    else:
        center = np.asarray([float(row[f"{metric}_median"]) for row in rows], dtype=float)
        low = np.asarray([float(row[f"{metric}_q25"]) for row in rows], dtype=float)
        high = np.asarray([float(row[f"{metric}_q75"]) for row in rows], dtype=float)
    return xs, center, low, high


def _summary_center_value(row: dict[str, object], metric: str, statistic: str) -> float:
    suffix = "median" if statistic == "median_iqr" else "mean"
    return float(row[f"{metric}_{suffix}"])


def compute_empirical_non_dominated_frontier(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    x_direction: str,
    y_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if x_direction != "min":
        raise ValueError("Only x_direction='min' is supported.")
    if y_direction not in {"min", "max"}:
        raise ValueError("y_direction must be 'min' or 'max'.")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    non_dominated = np.zeros(xs.shape[0], dtype=bool)
    valid_indices = np.flatnonzero(valid)
    for i in valid_indices:
        dominated = False
        for j in valid_indices:
            if i == j:
                continue
            x_better = xs[j] <= xs[i]
            if y_direction == "min":
                y_better = ys[j] <= ys[i]
                strictly_better = (xs[j] < xs[i]) or (ys[j] < ys[i])
            else:
                y_better = ys[j] >= ys[i]
                strictly_better = (xs[j] < xs[i]) or (ys[j] > ys[i])
            if x_better and y_better and strictly_better:
                dominated = True
                break
        non_dominated[i] = not dominated
    frontier_indices = np.flatnonzero(non_dominated)
    if frontier_indices.size == 0:
        return non_dominated, frontier_indices
    secondary = ys[frontier_indices] if y_direction == "min" else -ys[frontier_indices]
    order = np.lexsort((secondary, xs[frontier_indices]))
    frontier_indices = frontier_indices[order]
    return non_dominated, frontier_indices


def _tradeoff_points(
    summary_rows: list[dict[str, object]],
    *,
    methods: list[str],
    x_metric: str,
    y_metric: str,
    statistic: str,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for method in methods:
        for row in summary_rows:
            if str(row["method"]) != method:
                continue
            if int(float(row.get("count", 0))) <= 0:
                continue
            x_value = _summary_center_value(row, x_metric, statistic)
            y_value = _summary_center_value(row, y_metric, statistic)
            if not (np.isfinite(x_value) and np.isfinite(y_value) and x_value > 0):
                continue
            n_users = int(float(row["n_users"]))
            points.append(
                {
                    "method": method,
                    "n_users": n_users,
                    "x": float(x_value),
                    "y": float(y_value),
                    "label": f"{METHOD_SHORT_LABELS.get(method, method)}-{n_users}",
                }
            )
    return points


def _trial_tradeoff_points(
    trial_rows: list[dict[str, object]],
    *,
    methods: list[str],
    x_metric: str,
    y_metric: str,
    n_filter: int | None = None,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for method in methods:
        for row in trial_rows:
            if str(row["method"]) != method:
                continue
            if int(float(row.get("success", 0))) != 1:
                continue
            n_users = int(float(row["n_users"]))
            if n_filter is not None and n_users != int(n_filter):
                continue
            x_value = float(row[x_metric])
            y_value = float(row[y_metric])
            if not (np.isfinite(x_value) and np.isfinite(y_value) and x_value > 0):
                continue
            points.append(
                {
                    "method": method,
                    "n_users": n_users,
                    "trial": int(float(row["trial"])),
                    "x": x_value,
                    "y": y_value,
                    "label": f"{METHOD_SHORT_LABELS.get(method, method)}-{n_users}-T{int(float(row['trial']))}",
                }
            )
    return points


def _n_style_map(points: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    n_values = sorted({int(point["n_users"]) for point in points})
    if not n_values:
        return {}
    if len(n_values) == 1:
        return {n_values[0]: {"marker": "o", "size": 62.0, "markersize": 7.8}}
    min_size = 54.0
    max_size = 82.0
    style_map: dict[int, dict[str, object]] = {}
    for idx, n_value in enumerate(n_values):
        frac = idx / max(1, len(n_values) - 1)
        style_map[n_value] = {
            "marker": N_MARKER_CYCLE[idx % len(N_MARKER_CYCLE)],
            "size": float(min_size + frac * (max_size - min_size)),
            "markersize": float(7.0 + frac * 2.2),
        }
    return style_map


def _trial_tradeoff_legend_handles(
    *,
    methods: list[str],
    points: list[dict[str, object]],
    method_label_overrides: dict[str, str] | None = None,
) -> tuple[list[Line2D], list[Line2D]]:
    method_label_overrides = method_label_overrides or {}
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS.get(method, "tab:gray"),
            markeredgecolor="black",
            markeredgewidth=0.9,
            markersize=7.8,
            linestyle="None",
            label=method_label_overrides.get(method, method),
        )
        for method in methods
    ]
    style_map = _n_style_map(points)
    n_handles = [
        Line2D(
            [0],
            [0],
            marker=str(style_map[n_value]["marker"]),
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=float(style_map[n_value]["markersize"]),
            linestyle="None",
            label=f"n={n_value}",
        )
        for n_value in sorted(style_map)
    ]
    return method_handles, n_handles


def _print_frontier_summary(panel_name: str, points: list[dict[str, object]], frontier_indices: np.ndarray) -> None:
    tuples = [
        (
            str(points[idx]["method"]),
            int(points[idx]["n_users"]),
            float(points[idx]["x"]),
            float(points[idx]["y"]),
        )
        for idx in frontier_indices
    ]
    print(f"[{panel_name}] total_points={len(points)} non_dominated_points={len(frontier_indices)} frontier={tuples}")


def _plot_tradeoff_panel(
    ax: plt.Axes,
    *,
    points: list[dict[str, object]],
    methods: list[str],
    x_label: str,
    y_label: str,
    panel_title: str,
    y_direction: str,
) -> None:
    xs = np.asarray([float(point["x"]) for point in points], dtype=float)
    ys = np.asarray([float(point["y"]) for point in points], dtype=float)
    non_dominated_mask, frontier_indices = compute_empirical_non_dominated_frontier(
        xs,
        ys,
        x_direction="min",
        y_direction=y_direction,
    )
    _print_frontier_summary(panel_title, points, frontier_indices)
    for method in methods:
        method_indices = [idx for idx, point in enumerate(points) if str(point["method"]) == method]
        if not method_indices:
            continue
        color = METHOD_COLORS.get(method, "tab:gray")
        marker = METHOD_MARKERS.get(method, "o")
        dominated_indices = [idx for idx in method_indices if not bool(non_dominated_mask[idx])]
        frontier_method_indices = [idx for idx in method_indices if bool(non_dominated_mask[idx])]
        if dominated_indices:
            ax.scatter(
                xs[dominated_indices],
                ys[dominated_indices],
                s=44,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.28,
                zorder=2,
            )
        if frontier_method_indices:
            ax.scatter(
                xs[frontier_method_indices],
                ys[frontier_method_indices],
                s=82,
                marker=marker,
                facecolors=color,
                edgecolors="black",
                linewidths=1.4,
                alpha=0.95,
                zorder=4,
            )
    if frontier_indices.size:
        ax.step(
            xs[frontier_indices],
            ys[frontier_indices],
            where="post",
            color="0.15",
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
        )
        for idx in frontier_indices:
            ax.annotate(
                str(points[idx]["label"]),
                (xs[idx], ys[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="0.15",
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(panel_title)
    ax.grid(True, alpha=0.25)


def plot_tradeoff_frontier_figure(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    statistic: str,
    y_metric: str,
    y_label: str,
    y_direction: str,
    figure_title: str,
    panel_title_prefix: str,
) -> None:
    runtime_points = _tradeoff_points(
        summary_rows,
        methods=methods,
        x_metric="runtime_sec",
        y_metric=y_metric,
        statistic=statistic,
    )
    calls_points = _tradeoff_points(
        summary_rows,
        methods=methods,
        x_metric="stage2_solver_calls",
        y_metric=y_metric,
        statistic=statistic,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), dpi=150)
    _plot_tradeoff_panel(
        axes[0],
        points=runtime_points,
        methods=methods,
        x_label="Wall-clock runtime (s, log scale)",
        y_label=y_label,
        panel_title=f"{panel_title_prefix} vs runtime\nEmpirical non-dominated frontier",
        y_direction=y_direction,
    )
    _plot_tradeoff_panel(
        axes[1],
        points=calls_points,
        methods=methods,
        x_label="Stage-II solver calls (log scale)",
        y_label=y_label,
        panel_title=f"{panel_title_prefix} vs Stage-II calls\nEmpirical non-dominated frontier",
        y_direction=y_direction,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS.get(method, "o"),
            color="none",
            markerfacecolor=METHOD_COLORS.get(method, "tab:gray"),
            markeredgecolor="black",
            markeredgewidth=1.1,
            markersize=7.5,
            linestyle="None",
            label=method,
        )
        for method in methods
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.15",
            linewidth=1.8,
            label="Empirical non-dominated frontier",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(len(legend_handles), 5),
        frameon=True,
    )
    fig.suptitle(figure_title, y=1.08)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    fig.savefig(out_path)
    plt.close(fig)


def _plot_tradeoff_trial_panel(
    ax: plt.Axes,
    *,
    points: list[dict[str, object]],
    methods: list[str],
    x_label: str,
    y_label: str,
    panel_title: str,
) -> None:
    style_map = _n_style_map(points)
    n_values = sorted(style_map)
    for method in methods:
        color = METHOD_COLORS.get(method, "tab:gray")
        for n_value in n_values:
            subset = [
                point
                for point in points
                if str(point["method"]) == method and int(point["n_users"]) == int(n_value)
            ]
            if not subset:
                continue
            xs = np.asarray([float(point["x"]) for point in subset], dtype=float)
            ys = np.asarray([float(point["y"]) for point in subset], dtype=float)
            ax.scatter(
                xs,
                ys,
                s=float(style_map[n_value]["size"]),
                marker=str(style_map[n_value]["marker"]),
                facecolors=color,
                edgecolors="black",
                linewidths=0.85,
                alpha=0.74,
                zorder=3,
            )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(panel_title)
    ax.grid(True, alpha=0.25)


def plot_tradeoff_trial_figure(
    trial_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    y_metric: str,
    y_label: str,
    figure_title: str,
    panel_title_prefix: str,
    n_filter: int | None = None,
    method_label_overrides: dict[str, str] | None = None,
) -> None:
    runtime_points = _trial_tradeoff_points(
        trial_rows,
        methods=methods,
        x_metric="runtime_sec",
        y_metric=y_metric,
        n_filter=n_filter,
    )
    calls_points = _trial_tradeoff_points(
        trial_rows,
        methods=methods,
        x_metric="stage2_solver_calls",
        y_metric=y_metric,
        n_filter=n_filter,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), dpi=150)
    suffix = " (n=30 trials)" if n_filter is not None else " (all trial points)"
    _plot_tradeoff_trial_panel(
        axes[0],
        points=runtime_points,
        methods=methods,
        x_label="Wall-clock runtime (s, log scale)",
        y_label=y_label,
        panel_title=f"{panel_title_prefix} vs runtime{suffix}",
    )
    _plot_tradeoff_trial_panel(
        axes[1],
        points=calls_points,
        methods=methods,
        x_label="Stage-II solver calls (log scale)",
        y_label=y_label,
        panel_title=f"{panel_title_prefix} vs Stage-II calls{suffix}",
    )
    all_points = runtime_points + calls_points
    method_handles, n_handles = _trial_tradeoff_legend_handles(
        methods=methods,
        points=all_points,
        method_label_overrides=method_label_overrides,
    )
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.33, 1.01),
        ncol=min(len(method_handles), 4),
        frameon=True,
        title="Method",
    )
    if len(n_handles) > 1:
        fig.legend(
            handles=n_handles,
            loc="upper center",
            bbox_to_anchor=(0.80, 1.01),
            ncol=min(len(n_handles), 5),
            frameon=True,
            title="n",
        )
    fig.suptitle(figure_title, y=1.06)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def plot_tradeoff_trial_single_panel(
    trial_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    x_metric: str,
    x_label: str,
    y_metric: str,
    y_label: str,
    title: str,
    n_filter: int | None = None,
    method_label_overrides: dict[str, str] | None = None,
) -> None:
    points = _trial_tradeoff_points(
        trial_rows,
        methods=methods,
        x_metric=x_metric,
        y_metric=y_metric,
        n_filter=n_filter,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    _plot_tradeoff_trial_panel(
        ax,
        points=points,
        methods=methods,
        x_label=x_label,
        y_label=y_label,
        panel_title=title,
    )
    method_handles, n_handles = _trial_tradeoff_legend_handles(
        methods=methods,
        points=points,
        method_label_overrides=method_label_overrides,
    )
    method_legend = ax.legend(
        handles=method_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        title="Method",
        fontsize=8.5,
        title_fontsize=9.5,
    )
    ax.add_artist(method_legend)
    if len(n_handles) > 1:
        ax.legend(
            handles=n_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.02),
            frameon=True,
            title="n",
            fontsize=8.5,
            title_fontsize=9.5,
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_summary(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    metric: str,
    statistic: str,
    ylabel: str,
    title: str,
    logy: bool = False,
    xlabel: str = "Number of users",
    show_legend: bool = True,
    language: str = "en",
    font_scale: float = 1.0,
    method_label_overrides: dict[str, str] | None = None,
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    legend_fontsize: float | None = None,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        method_label_overrides = method_label_overrides or {}
        fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=180)
        for method in methods:
            xs, center, low, high = _extract_metric_series(summary_rows, method=method, metric=metric, statistic=statistic)
            if xs.size == 0:
                continue
            color = METHOD_COLORS.get(method)
            ax.plot(xs, center, marker="o", linewidth=1.9, markersize=6.0, label=method_label_overrides.get(method, method), color=color)
            ax.fill_between(xs, low, high, alpha=0.18, color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if show_legend:
            legend_kwargs: dict[str, object] = {
                "loc": legend_loc,
            }
            if legend_bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
            if legend_fontsize is not None:
                legend_kwargs["fontsize"] = legend_fontsize
            ax.legend(**legend_kwargs)
        fig.tight_layout(pad=1.0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def plot_gap_summary(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    statistic: str,
) -> None:
    plot_metric_summary(
        summary_rows,
        out_path,
        methods=methods,
        metric="final_grid_ne_gap",
        statistic=statistic,
        ylabel="Final grid-evaluated NE gap",
        title="Final grid-evaluated NE gap vs. number of users",
    )


def plot_stage2_calls_broken_axis(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    *,
    methods: list[str],
    statistic: str,
    title: str = "Stage-II solver calls vs. number of users",
    xlabel: str = "Number of users",
    ylabel: str = "Stage-II solver calls",
    language: str = "en",
    font_scale: float = 1.0,
    method_label_overrides: dict[str, str] | None = None,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor: tuple[float, float] = (0.98, 0.53),
    legend_fontsize: float | None = None,
    ylabel_fontsize: float | None = None,
    ytick_label_scale: float = 1.0,
) -> None:
    with plt.rc_context():
        _configure_fonts(language, font_scale)
        method_label_overrides = method_label_overrides or {}
        method_centers: list[tuple[str, float]] = []
        series_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for method in methods:
            xs, center, low, high = _extract_metric_series(summary_rows, method=method, metric="stage2_solver_calls", statistic=statistic)
            if xs.size == 0:
                continue
            series_by_method[method] = (xs, center, low, high)
            finite_center = center[np.isfinite(center) & (center > 0)]
            if finite_center.size:
                method_centers.append((method, float(np.median(finite_center))))
        if len(method_centers) < 2:
            plot_metric_summary(
                summary_rows,
                out_path,
                methods=methods,
                metric="stage2_solver_calls",
                statistic=statistic,
                ylabel=ylabel,
                title=title,
                xlabel=xlabel,
                language=language,
                font_scale=font_scale,
                method_label_overrides=method_label_overrides,
            )
            return

        ordered = sorted(method_centers, key=lambda item: item[1])
        ratios = [ordered[i + 1][1] / max(ordered[i][1], 1e-12) for i in range(len(ordered) - 1)]
        split_idx = int(np.argmax(ratios))
        if ratios[split_idx] < 3.0:
            plot_metric_summary(
                summary_rows,
                out_path,
                methods=methods,
                metric="stage2_solver_calls",
                statistic=statistic,
                ylabel=ylabel,
                title=title,
                xlabel=xlabel,
                logy=True,
                language=language,
                font_scale=font_scale,
                method_label_overrides=method_label_overrides,
            )
            return

        low_methods = {method for method, _ in ordered[: split_idx + 1]}
        high_methods = {method for method, _ in ordered[split_idx + 1 :]}
        low_highs = np.asarray(
            [float(np.nanmax(series_by_method[m][3])) for m in low_methods if m in series_by_method],
            dtype=float,
        )
        high_lows = np.asarray(
            [float(np.nanmin(series_by_method[m][2])) for m in high_methods if m in series_by_method],
            dtype=float,
        )
        low_cap = float(np.nanmax(low_highs)) if low_highs.size else 1.0
        high_floor = float(np.nanmin(high_lows)) if high_lows.size else low_cap * 3.0
        low_ylim_top = max(low_cap * 1.12, 1.0)
        high_ylim_bottom = max(high_floor * 0.92, low_ylim_top * 1.4)

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(8.8, 6.6),
            dpi=150,
            gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.05},
        )
        for method in methods:
            if method not in series_by_method:
                continue
            xs, center, low, high = series_by_method[method]
            color = METHOD_COLORS.get(method)
            for ax in (ax_top, ax_bottom):
                ax.plot(xs, center, marker="o", linewidth=1.8, label=method_label_overrides.get(method, method), color=color)
                ax.fill_between(xs, low, high, alpha=0.18, color=color)

        ax_top.set_ylim(high_ylim_bottom, max(float(np.nanmax([np.nanmax(v[3]) for v in series_by_method.values()])), high_ylim_bottom) * 1.05)
        ax_bottom.set_ylim(0.0, low_ylim_top)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()
        if ytick_label_scale != 1.0:
            ytick_size = float(plt.rcParams["ytick.labelsize"]) * float(ytick_label_scale)
            ax_top.tick_params(axis="y", labelsize=ytick_size)
            ax_bottom.tick_params(axis="y", labelsize=ytick_size)
        for ax in (ax_top, ax_bottom):
            ax.grid(True, alpha=0.25)
        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        ax_top.set_title(title)
        ax_bottom.set_xlabel(xlabel)
        if ylabel_fontsize is not None:
            fig.supylabel(ylabel, fontsize=ylabel_fontsize)
        else:
            fig.supylabel(ylabel)
        handles, labels = ax_top.get_legend_handles_labels()
        legend_kwargs: dict[str, object] = {
            "loc": legend_loc,
            "bbox_to_anchor": legend_bbox_to_anchor,
        }
        if legend_fontsize is not None:
            legend_kwargs["fontsize"] = legend_fontsize
        fig.legend(handles, labels, **legend_kwargs)
        fig.tight_layout(rect=(0.04, 0.03, 0.94, 0.98))
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def compute_trial_rows(
    *,
    config_path: str,
    seed: int,
    n_list: list[int],
    trials: int,
    methods: list[str],
    final_audit_grid_points: int | None = None,
    gap_heatmap_csv_template: str | None = None,
    bo_candidate_pool: int | None = None,
    bo_iters: int | None = None,
    ga_population_size: int | None = None,
    ga_generations: int | None = None,
    marl_price_levels: int | None = None,
    marl_episodes: int | None = None,
    marl_steps_per_episode: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = _load_cfg(config_path)
    audit_points = max(2, int(final_audit_grid_points if final_audit_grid_points is not None else cfg.baselines.gso_grid_points))
    pE_audit_grid = np.linspace(float(cfg.system.cE), float(cfg.baselines.max_price_E), audit_points)
    pN_audit_grid = np.linspace(float(cfg.system.cN), float(cfg.baselines.max_price_N), audit_points)
    heatmap_surfaces = _load_gap_heatmap_surfaces(n_list, gap_heatmap_csv_template)
    rows: list[dict[str, object]] = []
    failures = 0

    for n in n_list:
        heatmap_surface = heatmap_surfaces.get(int(n))
        for trial in range(1, trials + 1):
            users = _sample_users(cfg, n, seed, trial)
            for method in methods:
                internal_method = "Proposed" if method == "Proposed" else method
                run_base_cfg = cfg.baselines
                if method != "Proposed":
                    run_base_cfg = _apply_baseline_overrides(
                        run_base_cfg,
                        bo_candidate_pool=bo_candidate_pool,
                        bo_iters=bo_iters,
                        ga_population_size=ga_population_size,
                        ga_generations=ga_generations,
                        marl_price_levels=marl_price_levels,
                        marl_episodes=marl_episodes,
                        marl_steps_per_episode=marl_steps_per_episode,
                    )
                try:
                    price, offloading_set, restricted_gap, esp_revenue, nsp_revenue, meta = _run_stage1_method(
                        users,
                        cfg.system,
                        cfg.stackelberg,
                        run_base_cfg,
                        internal_method,
                    )
                    current_out = _build_current_outcome(
                        method=method,
                        price=price,
                        offloading_set=offloading_set,
                        social_cost=float(meta.get("social_cost", float("nan"))),
                        esp_revenue=float(esp_revenue),
                        nsp_revenue=float(nsp_revenue),
                    )
                    stage2_cache = {_price_cache_key(float(price[0]), float(price[1])): current_out}
                    if method == "Proposed" and abs(float(restricted_gap)) <= 1e-15:
                        final_grid_gap = 0.0
                        audit_stage2_calls = 0
                        gap_source = "restricted_gap_zero_certified"
                    elif heatmap_surface is not None:
                        _, heatmap_pE_grid, heatmap_pN_grid, heatmap_grid_ne_gap = heatmap_surface
                        final_grid_gap = _nearest_grid_ne_gap(
                            float(price[0]),
                            float(price[1]),
                            pE_grid=heatmap_pE_grid,
                            pN_grid=heatmap_pN_grid,
                            grid_ne_gap=heatmap_grid_ne_gap,
                        )
                        audit_stage2_calls = 0
                        gap_source = "heatmap_csv_nearest"
                    else:
                        final_grid_gap = _grid_ne_gap_audit(
                            current_out,
                            users,
                            cfg.system,
                            cfg.stackelberg,
                            run_base_cfg,
                            stage2_cache,
                            pE_audit_grid,
                            pN_audit_grid,
                        )
                        audit_stage2_calls = max(0, int(len(stage2_cache) - 1))
                        gap_source = "audit_grid"
                    stage2_solver_calls = int(meta.get("stage2_calls", 0))
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "success": 1,
                            "final_pE": float(price[0]),
                            "final_pN": float(price[1]),
                            "offloading_size": int(len(offloading_set)),
                            "restricted_gap": float(restricted_gap),
                            "final_grid_ne_gap": float(final_grid_gap),
                            "final_grid_ne_gap_source": str(gap_source),
                            "esp_revenue": float(esp_revenue),
                            "nsp_revenue": float(nsp_revenue),
                            "joint_revenue": float(esp_revenue + nsp_revenue),
                            "runtime_sec": float(meta.get("runtime_sec", float("nan"))),
                            "stage2_solver_calls": int(stage2_solver_calls),
                            "audit_stage2_solver_calls": int(audit_stage2_calls),
                            "total_stage2_solver_calls": int(stage2_solver_calls + audit_stage2_calls),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    failures += 1
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "success": 0,
                            "final_pE": float("nan"),
                            "final_pN": float("nan"),
                            "offloading_size": -1,
                            "restricted_gap": float("nan"),
                            "final_grid_ne_gap": float("nan"),
                            "final_grid_ne_gap_source": "",
                            "esp_revenue": float("nan"),
                            "nsp_revenue": float("nan"),
                            "joint_revenue": float("nan"),
                            "runtime_sec": float("nan"),
                            "stage2_solver_calls": float("nan"),
                            "audit_stage2_solver_calls": float("nan"),
                            "total_stage2_solver_calls": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    meta = {
        "audit_grid_points": int(audit_points),
        "search_objective_grid_points": int(cfg.baselines.gso_grid_points),
        "stage2_method_for_pricing": str(cfg.baselines.stage2_solver_for_pricing),
        "gap_heatmap_template": "" if gap_heatmap_csv_template is None else str(gap_heatmap_csv_template),
        "gap_heatmap_reused_n_list": ",".join(str(n) for n in sorted(heatmap_surfaces)),
        "failures": int(failures),
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final Stage-I solution quality across user scales via direct grid-NE-gap audits "
            "at each method's returned price."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=_parse_n_users_list, default="10,15,20,25,30")
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument("--methods", type=_parse_methods, default="Proposed,GA,BO,MARL")
    parser.add_argument("--statistic", choices=["mean_std", "median_iqr"], default="median_iqr")
    parser.add_argument("--final-audit-grid-points", type=_positive_int, default=None)
    parser.add_argument(
        "--gap-heatmap-csv-template",
        type=str,
        default=None,
        help=(
            "Optional path template like outputs/.../n{n}/price_grid_metrics.csv. "
            "If a file exists for a given n, final_grid_ne_gap reuses the nearest heatmap value "
            "instead of running a fresh audit grid."
        ),
    )
    parser.add_argument("--bo-candidate-pool", type=_positive_int, default=None)
    parser.add_argument("--bo-iters", type=int, default=None)
    parser.add_argument("--ga-population-size", type=_positive_int, default=None)
    parser.add_argument("--ga-generations", type=int, default=None)
    parser.add_argument("--marl-price-levels", type=_positive_int, default=None)
    parser.add_argument("--marl-episodes", type=_positive_int, default=None)
    parser.add_argument("--marl-steps-per-episode", type=_positive_int, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--trials-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if isinstance(args.n_users_list, str):
        n_list = _parse_n_users_list(args.n_users_list)
    else:
        n_list = list(args.n_users_list)
    if isinstance(args.methods, str):
        methods = _parse_methods(args.methods)
    else:
        methods = list(args.methods)

    summary_csv_override = Path(args.summary_csv).resolve() if args.summary_csv else None
    trials_csv_override = Path(args.trials_csv).resolve() if args.trials_csv else None
    if summary_csv_override is not None:
        summary_rows = load_summary_rows(summary_csv_override)
        trial_rows = load_trial_rows(trials_csv_override) if trials_csv_override is not None else []
        out_dir = Path(args.out_dir).resolve() if args.out_dir else summary_csv_override.parent
    else:
        out_dir = resolve_out_dir("run_stage1_final_grid_ne_gap_vs_users", args.out_dir)
        rows, meta = compute_trial_rows(
            config_path=str(args.config),
            seed=int(args.seed),
            n_list=n_list,
            trials=int(args.trials),
            methods=methods,
            final_audit_grid_points=args.final_audit_grid_points,
            gap_heatmap_csv_template=args.gap_heatmap_csv_template,
            bo_candidate_pool=args.bo_candidate_pool,
            bo_iters=args.bo_iters,
            ga_population_size=args.ga_population_size,
            ga_generations=args.ga_generations,
            marl_price_levels=args.marl_price_levels,
            marl_episodes=args.marl_episodes,
            marl_steps_per_episode=args.marl_steps_per_episode,
        )
        summary_rows = summarize_trials(rows, methods, n_list)
        trial_rows = rows

        trials_csv = out_dir / "stage1_final_grid_ne_gap_vs_users.csv"
        summary_csv = out_dir / "stage1_final_grid_ne_gap_vs_users_stats.csv"
        fig_path = out_dir / "stage1_final_grid_ne_gap_vs_users.png"
        summary_path = out_dir / "stage1_final_grid_ne_gap_vs_users_summary.txt"

        write_csv_rows(trials_csv, TRIAL_FIELDS, rows)
        write_csv_rows(summary_csv, _summary_fieldnames(), summary_rows)
        plot_gap_summary(summary_rows, fig_path, methods=methods, statistic=str(args.statistic))
        plot_metric_summary(
            summary_rows,
            out_dir / "stage1_runtime_vs_users.png",
            methods=methods,
            metric="runtime_sec",
            statistic=str(args.statistic),
            ylabel="Stage-I runtime (s)",
            title="Stage-I runtime vs. number of users",
        )
        plot_metric_summary(
            summary_rows,
            out_dir / "stage1_runtime_vs_users_log.png",
            methods=methods,
            metric="runtime_sec",
            statistic=str(args.statistic),
            ylabel="Stage-I runtime (s)",
            title="Stage-I runtime vs. number of users",
            logy=True,
        )
        plot_metric_summary(
            summary_rows,
            out_dir / "stage1_stage2_calls_vs_users.png",
            methods=methods,
            metric="stage2_solver_calls",
            statistic=str(args.statistic),
            ylabel="Stage-II solver calls",
            title="Stage-II solver calls vs. number of users",
        )
        _write_summary(
            summary_path,
            [
                f"config = {args.config}",
                f"seed = {args.seed}",
                f"trials = {args.trials}",
                f"n_users_list = {','.join(str(x) for x in n_list)}",
                f"methods = {','.join(methods)}",
                f"plot_statistic = {args.statistic}",
                f"audit_grid_points = {meta['audit_grid_points']}",
                f"search_objective_grid_points = {meta['search_objective_grid_points']}",
                f"stage2_method_for_pricing = {meta['stage2_method_for_pricing']}",
                f"gap_heatmap_csv_template = {meta['gap_heatmap_template']}",
                f"gap_heatmap_reused_n_list = {meta['gap_heatmap_reused_n_list']}",
                f"bo_candidate_pool = {'' if args.bo_candidate_pool is None else int(args.bo_candidate_pool)}",
                f"bo_iters = {'' if args.bo_iters is None else int(args.bo_iters)}",
                f"ga_population_size = {'' if args.ga_population_size is None else int(args.ga_population_size)}",
                f"ga_generations = {'' if args.ga_generations is None else int(args.ga_generations)}",
                f"marl_price_levels = {'' if args.marl_price_levels is None else int(args.marl_price_levels)}",
                f"marl_episodes = {'' if args.marl_episodes is None else int(args.marl_episodes)}",
                f"marl_steps_per_episode = {'' if args.marl_steps_per_episode is None else int(args.marl_steps_per_episode)}",
                "final_grid_ne_gap_definition = max unilateral provider revenue improvement on the audit price grid with the other provider price fixed",
                "grid_ne_gap_evaluation_mode = Proposed runs with restricted_gap==0 are certified to have final_grid_ne_gap=0 because the audit grid is a subset of unilateral deviations; otherwise a matching gap-heatmap CSV is reused if available, and direct audit is the fallback",
                "recorded_metrics = final_grid_ne_gap,joint_revenue,runtime_sec,stage2_solver_calls,audit_stage2_solver_calls,total_stage2_solver_calls",
                f"failed_runs = {meta['failures']}",
            ],
        )
    present_methods = {str(row["method"]) for row in summary_rows}
    plot_methods = [method for method in methods if method in present_methods]
    plot_metric_summary(
        summary_rows,
        out_dir / "stage1_runtime_vs_users.png",
        methods=plot_methods,
        metric="runtime_sec",
        statistic=str(args.statistic),
        ylabel="Stage-I runtime (s)",
        title="Stage-I runtime vs. number of users",
    )
    plot_metric_summary(
        summary_rows,
        out_dir / "stage1_runtime_vs_users_log.png",
        methods=plot_methods,
        metric="runtime_sec",
        statistic=str(args.statistic),
        ylabel="Stage-I runtime (s)",
        title="Stage-I runtime vs. number of users",
        logy=True,
    )
    plot_metric_summary(
        summary_rows,
        out_dir / "stage1_stage2_calls_vs_users.png",
        methods=plot_methods,
        metric="stage2_solver_calls",
        statistic=str(args.statistic),
        ylabel="Stage-II solver calls",
        title="Stage-II solver calls vs. number of users",
    )
    plot_tradeoff_frontier_figure(
        summary_rows,
        out_dir / "stage1_tradeoff_gap_frontier.png",
        methods=plot_methods,
        statistic=str(args.statistic),
        y_metric="final_grid_ne_gap",
        y_label="Final grid-evaluated NE gap",
        y_direction="min",
        figure_title="Stage-I quality-efficiency tradeoff: final gap vs search cost",
        panel_title_prefix="Final gap",
    )
    plot_tradeoff_frontier_figure(
        summary_rows,
        out_dir / "stage1_tradeoff_revenue_frontier.png",
        methods=plot_methods,
        statistic=str(args.statistic),
        y_metric="joint_revenue",
        y_label="Joint provider revenue",
        y_direction="max",
        figure_title="Stage-I tradeoff: joint revenue vs search cost",
        panel_title_prefix="Joint revenue",
    )
    if trial_rows:
        plot_tradeoff_trial_figure(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_n30.png",
            methods=plot_methods,
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            figure_title="Stage-I quality-efficiency tradeoff: final gap vs search cost (trial points, n=30)",
            panel_title_prefix="Final gap",
            n_filter=30,
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_n30_runtime.png",
            methods=plot_methods,
            x_metric="runtime_sec",
            x_label="Wall-clock runtime (s, log scale)",
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            title="Final gap vs runtime (n=30 trials)",
            n_filter=30,
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_n30_stage2_calls.png",
            methods=plot_methods,
            x_metric="stage2_solver_calls",
            x_label="Stage-II solver calls (log scale)",
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            title="Final gap vs Stage-II calls (n=30 trials)",
            n_filter=30,
        )
        plot_tradeoff_trial_figure(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_n30.png",
            methods=plot_methods,
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            figure_title="Stage-I tradeoff: joint revenue vs search cost (trial points, n=30)",
            panel_title_prefix="Joint revenue",
            n_filter=30,
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_n30_runtime.png",
            methods=plot_methods,
            x_metric="runtime_sec",
            x_label="Wall-clock runtime (s, log scale)",
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            title="Joint revenue vs runtime (n=30 trials)",
            n_filter=30,
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_n30_stage2_calls.png",
            methods=plot_methods,
            x_metric="stage2_solver_calls",
            x_label="Stage-II solver calls (log scale)",
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            title="Joint revenue vs Stage-II calls (n=30 trials)",
            n_filter=30,
        )
        plot_tradeoff_trial_figure(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_all.png",
            methods=plot_methods,
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            figure_title="Stage-I quality-efficiency tradeoff: final gap vs search cost (all trial points)",
            panel_title_prefix="Final gap",
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_all_runtime.png",
            methods=plot_methods,
            x_metric="runtime_sec",
            x_label="Wall-clock runtime (s, log scale)",
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            title="Final gap vs runtime (all trial points)",
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_gap_trials_all_stage2_calls.png",
            methods=plot_methods,
            x_metric="stage2_solver_calls",
            x_label="Stage-II solver calls (log scale)",
            y_metric="final_grid_ne_gap",
            y_label="Final grid-evaluated NE gap",
            title="Final gap vs Stage-II calls (all trial points)",
        )
        plot_tradeoff_trial_figure(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_all.png",
            methods=plot_methods,
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            figure_title="Stage-I tradeoff: joint revenue vs search cost (all trial points)",
            panel_title_prefix="Joint revenue",
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_all_runtime.png",
            methods=plot_methods,
            x_metric="runtime_sec",
            x_label="Wall-clock runtime (s, log scale)",
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            title="Joint revenue vs runtime (all trial points)",
        )
        plot_tradeoff_trial_single_panel(
            trial_rows,
            out_dir / "stage1_tradeoff_revenue_trials_all_stage2_calls.png",
            methods=plot_methods,
            x_metric="stage2_solver_calls",
            x_label="Stage-II solver calls (log scale)",
            y_metric="joint_revenue",
            y_label="Joint provider revenue",
            title="Joint revenue vs Stage-II calls (all trial points)",
        )


if __name__ == "__main__":
    main()

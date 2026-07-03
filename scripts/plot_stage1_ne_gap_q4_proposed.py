"""Plot Stage-I metrics with Q4/budget8 as Proposed.

The figures are single-panel matplotlib plots with a small inset only when a
metric has strong outliers. Lines show mean values and shaded bands show one
standard deviation across trials.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", "/tmp/tmc26_cache"))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OLD_RUN_CSV = (
    ROOT
    / "outputs/2. comp stackelberg baseline/"
    / "run_stage1_final_grid_ne_gap_vs_users_20260608_n50_250_t10/"
    / "stage1_final_grid_ne_gap_vs_users.csv"
)
DEFAULT_Q4_RUN_CSV = (
    ROOT
    / "outputs/2. comp stackelberg baseline/"
    / "run_stage1_vbbr_q4_budget8_true_grid_ne_gap_vs_users_20260618_n50_250_t10/"
    / "stage1_final_grid_ne_gap_vs_users.csv"
)
DEFAULT_Q4_AUDIT_CSV = (
    ROOT
    / "outputs/2. comp stackelberg baseline/"
    / "run_stage1_vbbr_q4_budget8_true_grid_ne_gap_vs_users_20260618_n50_250_t10/"
    / "proposed_true_grid_ne_gap_audit_120.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs/2. comp stackelberg baseline/"
    / "run_stage1_vbbr_q4_budget8_true_grid_ne_gap_vs_users_20260618_n50_250_t10/"
    / "q4_proposed_single_figures"
)

METHOD_ORDER = ["Proposed", "GA", "BO-online", "MARL"]
METHOD_COLORS = {
    "Proposed": "#1f77b4",
    "GA": "#d62728",
    "BO-online": "#2ca02c",
    "MARL": "#9467bd",
}
METHOD_MARKERS = {
    "Proposed": "o",
    "GA": "s",
    "BO-online": "D",
    "MARL": "^",
}
METHOD_LABELS = {
    "BO-online": "BO",
}

FRONTIER_OUT_NAME = "stage1_quality_cost_frontier_runtime_ne_gap.png"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    old_field: str
    q4_field: str | None
    audit_field: str | None
    ylabel: str
    out_name: str
    focus_methods: tuple[str, ...]
    allow_inset: bool = True
    log_y: bool = False
    broken_y: bool = False
    highlight_proposed: bool = False
    large_fonts: bool = False
    paper_compact: bool = False


METRICS = [
    MetricSpec(
        key="ne_gap",
        old_field="final_grid_ne_gap",
        q4_field=None,
        audit_field="true_grid_ne_gap",
        ylabel="NE gap",
        out_name="stage1_ne_gap_q4_proposed_mean_std.png",
        focus_methods=("Proposed", "GA", "BO-online"),
        paper_compact=True,
    ),
    MetricSpec(
        key="joint_revenue",
        old_field="joint_revenue",
        q4_field="joint_revenue",
        audit_field=None,
        ylabel="Joint revenue",
        out_name="stage1_joint_revenue_q4_proposed_mean_std.png",
        focus_methods=("Proposed", "GA", "BO-online", "MARL"),
    ),
    MetricSpec(
        key="runtime_sec",
        old_field="runtime_sec",
        q4_field="runtime_sec",
        audit_field=None,
        ylabel="Runtime (s)",
        out_name="stage1_runtime_vs_users_log.png",
        focus_methods=("Proposed",),
        allow_inset=False,
        log_y=True,
        paper_compact=True,
    ),
    MetricSpec(
        key="total_stage2_solver_calls",
        old_field="total_stage2_solver_calls",
        q4_field="total_stage2_solver_calls",
        audit_field=None,
        ylabel="Stage-II solver calls",
        out_name="stage1_stage2_calls_vs_users.png",
        focus_methods=("Proposed",),
        allow_inset=False,
        broken_y=True,
        paper_compact=True,
    ),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _finite_int(raw: str | None) -> int | None:
    value = _finite_float(raw)
    return int(value) if value is not None else None


def _collect_metric_values(
    *,
    old_rows: list[dict[str, str]],
    q4_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    spec: MetricSpec,
) -> dict[tuple[str, int], list[float]]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)

    for row in old_rows:
        method = str(row.get("method", "")).strip()
        if method == "Proposed" or method not in METHOD_ORDER:
            continue
        if row.get("success") not in {"1", "True", "true"}:
            continue
        n_users = _finite_int(row.get("n_users"))
        value = _finite_float(row.get(spec.old_field))
        if n_users is None or value is None:
            continue
        values[(method, n_users)].append(value)

    if spec.audit_field is not None:
        for row in audit_rows:
            n_users = _finite_int(row.get("n_users"))
            value = _finite_float(row.get(spec.audit_field))
            if n_users is None or value is None:
                continue
            values[("Proposed", n_users)].append(value)
    elif spec.q4_field is not None:
        for row in q4_rows:
            if row.get("method") != "Proposed" or row.get("success") not in {"1", "True", "true"}:
                continue
            n_users = _finite_int(row.get("n_users"))
            value = _finite_float(row.get(spec.q4_field))
            if n_users is None or value is None:
                continue
            values[("Proposed", n_users)].append(value)

    return dict(values)


def _metric_summary_rows(
    values: dict[tuple[str, int], list[float]],
    *,
    spec: MetricSpec,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        for n_users in sorted(n for (m, n) in values if m == method):
            arr = np.asarray(values[(method, n_users)], dtype=float)
            source = "q4_proposed"
            if method != "Proposed":
                source = "old_rerun_baseline"
            elif spec.audit_field is not None:
                source = "q4_proposed_ne_gap_audit"
            rows.append(
                {
                    "metric": spec.key,
                    "method": method,
                    "n_users": str(int(n_users)),
                    "count": str(int(arr.size)),
                    "mean": f"{float(np.mean(arr)):.12g}",
                    "std": f"{float(np.std(arr)):.12g}",
                    "min": f"{float(np.min(arr)):.12g}",
                    "max": f"{float(np.max(arr)):.12g}",
                    "source": source,
                }
            )
    return rows


def _write_summary(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "method", "n_users", "count", "mean", "std", "min", "max", "source"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _style_matplotlib(*, large_fonts: bool = False, paper_compact: bool = False) -> None:
    if large_fonts:
        axis_size = 40
        tick_size = 40
        legend_size = 40
    elif paper_compact:
        axis_size = 40
        tick_size = 22
        legend_size = 24
    else:
        axis_size = 21
        tick_size = 17
        legend_size = 16
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": tick_size,
            "axes.titlesize": axis_size,
            "axes.labelsize": axis_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": 1.25 if paper_compact else 1.35,
            "xtick.major.width": 1.1 if paper_compact else 1.2,
            "ytick.major.width": 1.1 if paper_compact else 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )


def _series(summary_rows: list[dict[str, str]], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows = [row for row in summary_rows if row["method"] == method]
    if not rows:
        return None
    rows.sort(key=lambda row: int(row["n_users"]))
    xs = np.asarray([int(row["n_users"]) for row in rows], dtype=float)
    means = np.asarray([float(row["mean"]) for row in rows], dtype=float)
    stds = np.asarray([float(row["std"]) for row in rows], dtype=float)
    return xs, means, stds


def _draw_series(
    ax: plt.Axes,
    summary_rows: list[dict[str, str]],
    *,
    show_labels: bool,
    alpha: float = 0.16,
    lower_floor: float = 0.0,
    highlight_proposed: bool = False,
    large_markers: bool = False,
    marker_size: float | None = None,
    line_width: float | None = None,
) -> None:
    resolved_marker_size = marker_size
    if resolved_marker_size is None and large_markers:
        resolved_marker_size = 15.0
    for method in METHOD_ORDER:
        series = _series(summary_rows, method)
        if series is None:
            continue
        xs, means, stds = series
        lower = np.maximum(means - stds, lower_floor)
        upper = means + stds
        color = METHOD_COLORS[method]
        is_highlight = highlight_proposed and method == "Proposed"
        ax.fill_between(xs, lower, upper, color=color, alpha=0.24 if is_highlight else alpha, linewidth=0)
        line = ax.plot(
            xs,
            means,
            color=color,
            marker=METHOD_MARKERS[method],
            linewidth=line_width if line_width is not None else (4.4 if is_highlight else (3.0 if show_labels else 1.8)),
            markersize=resolved_marker_size if resolved_marker_size is not None else (12.0 if is_highlight else (8.8 if show_labels else 4.2)),
            label=METHOD_LABELS.get(method, method) if show_labels else None,
            zorder=5 if is_highlight else 3,
        )
        if is_highlight:
            line[0].set_path_effects(
                [path_effects.Stroke(linewidth=6.8, foreground="white"), path_effects.Normal()]
            )


def _full_ymax(summary_rows: list[dict[str, str]]) -> float:
    max_y = 0.0
    for method in METHOD_ORDER:
        series = _series(summary_rows, method)
        if series is None:
            continue
        _xs, means, stds = series
        max_y = max(max_y, float(np.max(means + stds)))
    return max_y


def _focus_ymax(summary_rows: list[dict[str, str]], focus_methods: tuple[str, ...]) -> float:
    max_y = 0.0
    for method in focus_methods:
        series = _series(summary_rows, method)
        if series is None:
            continue
        _xs, means, stds = series
        max_y = max(max_y, float(np.max(means + stds)))
    return max_y


def _positive_lower_floor(summary_rows: list[dict[str, str]]) -> float:
    min_positive = math.inf
    for method in METHOD_ORDER:
        series = _series(summary_rows, method)
        if series is None:
            continue
        _xs, means, stds = series
        positive = means - stds
        positive = positive[positive > 0]
        if positive.size:
            min_positive = min(min_positive, float(np.min(positive)))
    if not math.isfinite(min_positive):
        return 1e-3
    return max(min_positive * 0.5, 1e-3)


def _style_axis(ax: plt.Axes, *, full_frame: bool = False) -> None:
    ax.grid(True, which="major", color="#d6dbe1", linewidth=1.0, alpha=0.25 if full_frame else 0.9)
    ax.set_axisbelow(True)
    if full_frame:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.25)
        return
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_broken_y(summary_rows: list[dict[str, str]], spec: MetricSpec, out_path: Path) -> None:
    _style_matplotlib(large_fonts=spec.large_fonts, paper_compact=spec.paper_compact)
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(18.5, 13.0) if spec.large_fonts else ((13.8, 8.0) if spec.paper_compact else (11.2, 7.4)),
        dpi=180 if spec.paper_compact else 220,
        gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.08},
    )

    compact_marker_size = 7.0 if spec.paper_compact else None
    compact_line_width = 2.2 if spec.paper_compact else None
    _draw_series(
        ax_top,
        summary_rows,
        show_labels=True,
        alpha=0.16,
        large_markers=spec.large_fonts,
        marker_size=compact_marker_size,
        line_width=compact_line_width,
    )
    _draw_series(
        ax_bottom,
        summary_rows,
        show_labels=False,
        alpha=0.16,
        large_markers=spec.large_fonts,
        marker_size=compact_marker_size,
        line_width=compact_line_width,
    )

    low_methods = ("Proposed", "MARL")
    high_methods = ("GA", "BO-online")
    low_ymax = max(_focus_ymax(summary_rows, low_methods) * 1.18, 1.0)
    high_ymin = max(_focus_ymax(summary_rows, ("BO-online",)) * 0.82, low_ymax * 1.35)
    high_ymax = _focus_ymax(summary_rows, high_methods) * 1.14

    ax_bottom.set_ylim(bottom=0.0, top=low_ymax)
    ax_top.set_ylim(bottom=high_ymin, top=high_ymax)
    ax_bottom.set_xlabel("Number of users")
    ax_bottom.set_xticks(sorted({int(row["n_users"]) for row in summary_rows}))
    fig.text(
        0.075 if spec.paper_compact else 0.055,
        0.48,
        spec.ylabel,
        va="center",
        rotation="vertical",
        fontsize=40 if spec.large_fonts or spec.paper_compact else 21,
    )

    for ax in (ax_top, ax_bottom):
        _style_axis(ax, full_frame=spec.paper_compact)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False)

    kwargs = dict(marker=[(-1, -0.7), (1, 0.7)], markersize=13, linestyle="none", color="black", mec="black", mew=1.2, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    if spec.paper_compact:
        ax_top.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.56),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
    else:
        ax_top.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.44 if spec.large_fonts else 1.34),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0 if spec.large_fonts else 1.4,
            borderaxespad=0.0,
        )

    fig.subplots_adjust(
        left=0.19 if spec.large_fonts else (0.22 if spec.paper_compact else 0.15),
        right=0.98,
        bottom=0.18 if spec.large_fonts else (0.16 if spec.paper_compact else 0.15),
        top=0.74 if spec.large_fonts else (0.76 if spec.paper_compact else 0.80),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.paper_compact:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    else:
        fig.savefig(out_path)
    plt.close(fig)


def _plot_mean_std(summary_rows: list[dict[str, str]], spec: MetricSpec, out_path: Path) -> None:
    if spec.broken_y:
        _plot_broken_y(summary_rows, spec, out_path)
        return

    _style_matplotlib(large_fonts=spec.large_fonts, paper_compact=spec.paper_compact)
    fig, ax = plt.subplots(
        figsize=(18.5, 12.0) if spec.large_fonts else ((13.8, 8.0) if spec.paper_compact else (11.2, 7.4)),
        dpi=180 if spec.paper_compact else 220,
    )

    full_ymax = _full_ymax(summary_rows)
    focus_ymax = _focus_ymax(summary_rows, spec.focus_methods)
    use_inset = spec.allow_inset and full_ymax > max(focus_ymax * 1.65, focus_ymax + 1e-9)
    main_ymax = focus_ymax if use_inset else full_ymax

    lower_floor = _positive_lower_floor(summary_rows) if spec.log_y else 0.0
    _draw_series(
        ax,
        summary_rows,
        show_labels=True,
        lower_floor=lower_floor,
        highlight_proposed=spec.highlight_proposed,
        large_markers=spec.large_fonts,
        marker_size=7.0 if spec.paper_compact else None,
        line_width=2.2 if spec.paper_compact else None,
    )

    ax.set_xlabel("Number of users")
    ax.set_ylabel(spec.ylabel)
    ax.set_xticks(sorted({int(row["n_users"]) for row in summary_rows}))
    if spec.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=lower_floor, top=main_ymax * 1.45 if main_ymax > 0 else 1.0)
        ax.grid(True, which="minor", color="#edf0f3", linewidth=0.7, alpha=0.8)
    else:
        ax.set_ylim(bottom=0.0, top=main_ymax * 1.18 if main_ymax > 0 else 1.0)
    _style_axis(ax, full_frame=spec.paper_compact)
    if spec.paper_compact:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20 if spec.large_fonts else 1.12),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0 if spec.large_fonts else 1.4,
            borderaxespad=0.0,
        )

    if use_inset:
        if spec.paper_compact and spec.key == "ne_gap":
            inset_width = "24%"
            inset_height = "24%"
            inset_borderpad = 1.0
            inset_title_size = 10
            inset_tick_size = 7.5
        else:
            inset_width = "36%"
            inset_height = "36%"
            inset_borderpad = 1.15
            inset_title_size = 11
            inset_tick_size = 8.5
        inset = inset_axes(ax, width=inset_width, height=inset_height, loc="upper left", borderpad=inset_borderpad)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.94)
        _draw_series(inset, summary_rows, show_labels=False, alpha=0.10)
        inset.set_title("Full range", fontsize=inset_title_size, pad=3)
        inset.set_xticks(sorted({int(row["n_users"]) for row in summary_rows})[::2])
        inset.set_ylim(bottom=0.0, top=full_ymax * 1.12)
        inset.grid(True, color="#d6dbe1", linewidth=0.7, alpha=0.8)
        inset.tick_params(labelsize=inset_tick_size, width=0.8, length=3, pad=1.5)
        inset.yaxis.tick_right()
        inset.yaxis.set_label_position("right")
        inset.spines["top"].set_visible(False)
        inset.spines["left"].set_color("#6b7280")
        inset.spines["bottom"].set_color("#6b7280")
        inset.spines["right"].set_color("#6b7280")

    fig.subplots_adjust(
        left=0.19 if spec.large_fonts else (0.18 if spec.paper_compact else 0.15),
        right=0.98,
        bottom=0.18 if spec.large_fonts else (0.16 if spec.paper_compact else 0.15),
        top=0.76 if spec.large_fonts else (0.78 if spec.paper_compact else 0.84),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.paper_compact:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    else:
        fig.savefig(out_path)
    plt.close(fig)


def _clean_old_images(out_dir: Path) -> list[Path]:
    removed: list[Path] = []
    if not out_dir.exists():
        return removed
    for path in sorted(out_dir.glob("*.png")):
        path.unlink()
        removed.append(path)
    return removed


def _summary_by_metric_method_n(
    summary_rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    indexed: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in summary_rows:
        metric = row["metric"]
        method = row["method"]
        n_users = int(row["n_users"])
        indexed[(metric, method, n_users)] = {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }
    return indexed


def _plot_quality_cost_frontier(summary_rows: list[dict[str, str]], out_path: Path) -> None:
    """Plot runtime-vs-NE-gap frontier from the generated mean/std summary."""
    _style_matplotlib(paper_compact=True)
    indexed = _summary_by_metric_method_n(summary_rows)
    n_users_values = sorted(
        n_users
        for metric, method, n_users in indexed
        if metric == "ne_gap" and method == "Proposed"
    )

    fig, ax = plt.subplots(figsize=(13.8, 8.0), dpi=180)
    ne_gap_upper = [
        values["mean"] + values["std"]
        for (metric, _method, _n_users), values in indexed.items()
        if metric == "ne_gap"
    ]
    main_ymax = max(ne_gap_upper) * 1.08 if ne_gap_upper else 1.0

    def collect_method(method: str) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
        xs: list[float] = []
        ys: list[float] = []
        xerr: list[float] = []
        yerr: list[float] = []
        point_ns: list[int] = []
        for n_users in n_users_values:
            runtime = indexed.get(("runtime_sec", method, n_users))
            ne_gap = indexed.get(("ne_gap", method, n_users))
            if runtime is None or ne_gap is None:
                continue
            xs.append(runtime["mean"])
            ys.append(ne_gap["mean"])
            xerr.append(runtime["std"])
            yerr.append(ne_gap["std"])
            point_ns.append(n_users)
        return xs, ys, xerr, yerr, point_ns

    for method in METHOD_ORDER:
        xs, ys, xerr, yerr, point_ns = collect_method(method)
        if not xs:
            continue

        color = METHOD_COLORS[method]
        is_proposed = method == "Proposed"
        container = ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            yerr=yerr,
            color=color,
            marker=METHOD_MARKERS[method],
            markersize=8.0 if is_proposed else 6.5,
            linewidth=2.4 if is_proposed else 1.8,
            elinewidth=1.1,
            capsize=3.0,
            alpha=0.95,
            label=METHOD_LABELS.get(method, method),
            zorder=5 if is_proposed else 3,
        )
        if is_proposed:
            container.lines[0].set_path_effects(
                [path_effects.Stroke(linewidth=4.6, foreground="white"), path_effects.Normal()]
            )

        for x, y, n_users in zip(xs, ys, point_ns):
            if n_users not in {50, 250}:
                continue
            offset = {
                "Proposed": (5, 4),
                "GA": (5, 4),
                "BO-online": (5, 4),
            }.get(method, (5, 4))
            ax.annotate(
                str(n_users),
                xy=(x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=10,
                color=color,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("NE gap")
    ax.set_ylim(bottom=0.0, top=main_ymax)
    _style_axis(ax, full_frame=True)
    ax.grid(True, which="minor", axis="x", color="#edf0f3", linewidth=0.7, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    label_order = ["Proposed", "GA", "BO", "MARL"]
    ordered = [
        (handles[labels.index(label)], label)
        for label in label_order
        if label in labels
    ]
    legend_handles = [handle for handle, _label in ordered]
    legend_labels = [label for _handle, label in ordered]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage-I metric figures using Q4/budget8 as Proposed.")
    parser.add_argument("--old-run-csv", type=Path, default=DEFAULT_OLD_RUN_CSV)
    parser.add_argument("--q4-run-csv", type=Path, default=DEFAULT_Q4_RUN_CSV)
    parser.add_argument("--q4-audit-csv", type=Path, default=DEFAULT_Q4_AUDIT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Remove old PNGs in the output figure directory before plotting.")
    args = parser.parse_args()

    out_dir = args.out_dir
    removed = _clean_old_images(out_dir) if args.clean else []
    old_rows = _read_csv(args.old_run_csv)
    q4_rows = _read_csv(args.q4_run_csv)
    audit_rows = _read_csv(args.q4_audit_csv)

    all_summary_rows: list[dict[str, str]] = []
    figure_paths: list[Path] = []
    for spec in METRICS:
        values = _collect_metric_values(old_rows=old_rows, q4_rows=q4_rows, audit_rows=audit_rows, spec=spec)
        summary_rows = _metric_summary_rows(values, spec=spec)
        if not summary_rows:
            raise SystemExit(f"No rows found for metric {spec.key}")
        all_summary_rows.extend(summary_rows)
        figure_path = out_dir / spec.out_name
        _plot_mean_std(summary_rows, spec, figure_path)
        figure_paths.append(figure_path)

    summary_csv = out_dir / "stage1_q4_proposed_mean_std_summary.csv"
    _write_summary(all_summary_rows, summary_csv)
    frontier_path = out_dir / FRONTIER_OUT_NAME
    _plot_quality_cost_frontier(all_summary_rows, frontier_path)
    figure_paths.append(frontier_path)

    print(f"removed_old_pngs={len(removed)}")
    print(f"summary_csv={summary_csv}")
    for path in figure_paths:
        print(f"figure_png={path}")


if __name__ == "__main__":
    main()

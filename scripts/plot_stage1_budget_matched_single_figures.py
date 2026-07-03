"""Plot budget-matched Stage-I metrics as single matplotlib figures."""

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
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN_CSV = (
    ROOT
    / "outputs/2. comp stackelberg baseline/"
    / "run_stage1_budget_matched_quality_vs_users_20260608_n50_250_t10/"
    / "stage1_budget_matched_quality_vs_users.csv"
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
    / "run_stage1_budget_matched_quality_vs_users_20260608_n50_250_t10/"
    / "budget_matched_single_figures"
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


@dataclass(frozen=True)
class MetricSpec:
    key: str
    field: str
    ylabel: str
    out_name: str
    focus_methods: tuple[str, ...]


METRICS = [
    MetricSpec(
        key="ne_gap",
        field="final_grid_ne_gap",
        ylabel="NE gap",
        out_name="stage1_budget_matched_ne_gap_mean_std.png",
        focus_methods=("Proposed",),
    ),
    MetricSpec(
        key="joint_revenue",
        field="joint_revenue",
        ylabel="Joint revenue",
        out_name="stage1_budget_matched_joint_revenue_mean_std.png",
        focus_methods=("Proposed", "GA", "BO-online", "MARL"),
    ),
    MetricSpec(
        key="runtime_sec",
        field="runtime_sec",
        ylabel="Runtime (s)",
        out_name="stage1_budget_matched_runtime_mean_std.png",
        focus_methods=("Proposed",),
    ),
    MetricSpec(
        key="total_stage2_solver_calls",
        field="total_stage2_solver_calls",
        ylabel="Stage-II solver calls",
        out_name="stage1_budget_matched_stage2_solver_calls_mean_std.png",
        focus_methods=("Proposed",),
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
    rows: list[dict[str, str]],
    spec: MetricSpec,
) -> dict[tuple[str, int], list[float]]:
    values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method not in METHOD_ORDER:
            continue
        if row.get("success") not in {"1", "True", "true"}:
            continue
        n_users = _finite_int(row.get("n_users"))
        value = _finite_float(row.get(spec.field))
        if n_users is None or value is None:
            continue
        values[(method, n_users)].append(value)
    return dict(values)


def _override_proposed_ne_gap_with_q4_audit(
    values: dict[tuple[str, int], list[float]],
    audit_rows: list[dict[str, str]],
) -> dict[tuple[str, int], list[float]]:
    overridden: dict[tuple[str, int], list[float]] = {
        key: list(series) for key, series in values.items() if key[0] != "Proposed"
    }
    for row in audit_rows:
        n_users = _finite_int(row.get("n_users"))
        value = _finite_float(row.get("true_grid_ne_gap"))
        if n_users is None or value is None:
            continue
        overridden.setdefault(("Proposed", n_users), []).append(value)
    return overridden


def _metric_summary_rows(
    values: dict[tuple[str, int], list[float]],
    *,
    spec: MetricSpec,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in METHOD_ORDER:
        for n_users in sorted(n for (m, n) in values if m == method):
            arr = np.asarray(values[(method, n_users)], dtype=float)
            source = "budget_matched"
            if spec.key == "ne_gap" and method == "Proposed":
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


def _style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.titlesize": 22,
            "axes.labelsize": 21,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "axes.linewidth": 1.35,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
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
    methods: tuple[str, ...] = tuple(METHOD_ORDER),
) -> None:
    for method in methods:
        series = _series(summary_rows, method)
        if series is None:
            continue
        xs, means, stds = series
        lower = np.maximum(means - stds, 0.0)
        upper = means + stds
        color = METHOD_COLORS[method]
        ax.fill_between(xs, lower, upper, color=color, alpha=alpha, linewidth=0)
        ax.plot(
            xs,
            means,
            color=color,
            marker=METHOD_MARKERS[method],
            linewidth=3.2 if show_labels else 1.8,
            markersize=9.5 if show_labels else 4.2,
            label=method if show_labels else None,
        )


def _add_legend_proxies(ax: plt.Axes, drawn_methods: tuple[str, ...]) -> None:
    for method in METHOD_ORDER:
        if method in drawn_methods:
            continue
        ax.plot(
            [],
            [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=3.2,
            markersize=9.5,
            label=method,
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


def _plot_mean_std(summary_rows: list[dict[str, str]], spec: MetricSpec, out_path: Path) -> None:
    _style_matplotlib()
    fig, ax = plt.subplots(figsize=(11.2, 7.4), dpi=220)

    full_ymax = _full_ymax(summary_rows)
    focus_ymax = _focus_ymax(summary_rows, spec.focus_methods)
    use_inset = full_ymax > max(focus_ymax * 1.65, focus_ymax + 1e-9)
    main_ymax = focus_ymax if use_inset else full_ymax

    main_methods = spec.focus_methods if use_inset else tuple(METHOD_ORDER)
    _draw_series(ax, summary_rows, show_labels=True, methods=main_methods)
    if use_inset:
        _add_legend_proxies(ax, main_methods)

    ax.set_xlabel("Number of users")
    ax.set_ylabel(spec.ylabel)
    ax.set_xticks(sorted({int(row["n_users"]) for row in summary_rows}))
    top_pad = 1.08 if use_inset else 1.18
    ax.set_ylim(bottom=0.0, top=main_ymax * top_pad if main_ymax > 0 else 1.0)
    ax.grid(True, which="major", color="#d6dbe1", linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    paired = sorted(zip(handles, labels, strict=True), key=lambda item: order.get(item[1], len(order)))
    sorted_handles = [handle for handle, _label in paired]
    sorted_labels = [label for _handle, label in paired]
    ax.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
        borderaxespad=0.0,
    )

    if use_inset:
        inset = inset_axes(ax, width="36%", height="36%", loc="upper left", borderpad=1.15)
        inset.patch.set_facecolor("white")
        inset.patch.set_alpha(0.94)
        _draw_series(inset, summary_rows, show_labels=False, alpha=0.10)
        inset.set_title("Full range", fontsize=11, pad=3)
        inset.set_xticks(sorted({int(row["n_users"]) for row in summary_rows})[::2])
        inset.set_ylim(bottom=0.0, top=full_ymax * 1.12)
        inset.grid(True, color="#d6dbe1", linewidth=0.7, alpha=0.8)
        inset.tick_params(labelsize=8.5, width=0.8, length=3, pad=1.5)
        inset.yaxis.tick_right()
        inset.yaxis.set_label_position("right")
        inset.spines["top"].set_visible(False)
        inset.spines["left"].set_color("#6b7280")
        inset.spines["bottom"].set_color("#6b7280")
        inset.spines["right"].set_color("#6b7280")

    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.84)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot budget-matched Stage-I metric figures.")
    parser.add_argument("--run-csv", type=Path, default=DEFAULT_RUN_CSV)
    parser.add_argument("--q4-audit-csv", type=Path, default=DEFAULT_Q4_AUDIT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--clean", action="store_true", help="Remove old PNGs in the output figure directory before plotting.")
    args = parser.parse_args()

    removed = _clean_old_images(args.out_dir) if args.clean else []
    rows = _read_csv(args.run_csv)
    q4_audit_rows = _read_csv(args.q4_audit_csv)

    all_summary_rows: list[dict[str, str]] = []
    figure_paths: list[Path] = []
    for spec in METRICS:
        values = _collect_metric_values(rows, spec)
        if spec.key == "ne_gap":
            values = _override_proposed_ne_gap_with_q4_audit(values, q4_audit_rows)
        summary_rows = _metric_summary_rows(values, spec=spec)
        if not summary_rows:
            raise SystemExit(f"No rows found for metric {spec.key}")
        all_summary_rows.extend(summary_rows)
        figure_path = args.out_dir / spec.out_name
        _plot_mean_std(summary_rows, spec, figure_path)
        figure_paths.append(figure_path)

    summary_csv = args.out_dir / "stage1_budget_matched_mean_std_summary.csv"
    _write_summary(all_summary_rows, summary_csv)

    print(f"removed_old_pngs={len(removed)}")
    print(f"summary_csv={summary_csv}")
    for path in figure_paths:
        print(f"figure_png={path}")


if __name__ == "__main__":
    main()

"""Replot fixed Stage-I summary figures from existing statistics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "Proposed": "tab:blue",
    "GA": "tab:red",
    "BO": "tab:orange",
    "BO-online": "tab:purple",
    "MARL": "tab:green",
}

METHOD_LABELS = {
    "BO-online": "BO",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _configure_fonts(axis_label_size: float, tick_label_size: float, legend_font_size: float) -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = tick_label_size
    plt.rcParams["axes.labelsize"] = axis_label_size
    plt.rcParams["xtick.labelsize"] = tick_label_size
    plt.rcParams["ytick.labelsize"] = tick_label_size
    plt.rcParams["legend.fontsize"] = legend_font_size
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1


def _method_order(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["Proposed", "GA", "BO", "BO-online", "MARL"]
    present = {row["method"] for row in rows}
    return [method for method in preferred if method in present] + sorted(present - set(preferred))


def _series(
    rows: list[dict[str, str]],
    *,
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in rows if row["method"] == method and int(float(row.get("count", "0"))) > 0],
        key=lambda row: int(float(row["n_users"])),
    )
    xs = np.asarray([int(float(row["n_users"])) for row in selected], dtype=int)
    center = np.asarray([float(row[f"{metric}_median"]) for row in selected], dtype=float)
    low = np.asarray([float(row[f"{metric}_q25"]) for row in selected], dtype=float)
    high = np.asarray([float(row[f"{metric}_q75"]) for row in selected], dtype=float)
    return xs, center, low, high


def _plot_metric(
    *,
    rows: list[dict[str, str]],
    out_path: Path,
    metric: str,
    ylabel: str,
    axis_label_size: float,
    tick_label_size: float,
    legend_font_size: float,
    logy: bool = False,
) -> None:
    with plt.rc_context():
        _configure_fonts(axis_label_size, tick_label_size, legend_font_size)
        fig, ax = plt.subplots(figsize=(12.2, 8.0), dpi=180)
        for method in _method_order(rows):
            xs, center, low, high = _series(rows, method=method, metric=metric)
            if xs.size == 0:
                continue
            color = METHOD_COLORS.get(method)
            ax.plot(
                xs,
                center,
                marker="o",
                linewidth=2.2,
                markersize=7.0,
                label=METHOD_LABELS.get(method, method),
                color=color,
            )
            ax.fill_between(xs, low, high, alpha=0.18, color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Number of users")
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.tick_params(axis="both", labelsize=tick_label_size, pad=7)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=legend_font_size,
            frameon=True,
        )
        fig.subplots_adjust(left=0.18, right=0.74, bottom=0.16, top=0.97)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_stage2_calls_broken_axis(
    *,
    rows: list[dict[str, str]],
    out_path: Path,
    axis_label_size: float,
    tick_label_size: float,
    legend_font_size: float,
) -> None:
    with plt.rc_context():
        _configure_fonts(axis_label_size, tick_label_size, legend_font_size)
        series_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        method_centers: list[tuple[str, float]] = []
        for method in _method_order(rows):
            xs, center, low, high = _series(rows, method=method, metric="stage2_solver_calls")
            if xs.size == 0:
                continue
            series_by_method[method] = (xs, center, low, high)
            finite_center = center[np.isfinite(center) & (center > 0)]
            if finite_center.size:
                method_centers.append((method, float(np.median(finite_center))))

        ordered = sorted(method_centers, key=lambda item: item[1])
        ratios = [ordered[i + 1][1] / max(ordered[i][1], 1e-12) for i in range(len(ordered) - 1)]
        split_idx = int(np.argmax(ratios)) if ratios else 0
        if not ratios or ratios[split_idx] < 3.0:
            _plot_metric(
                rows=rows,
                out_path=out_path,
                metric="stage2_solver_calls",
                ylabel="Stage-II solver calls",
                axis_label_size=axis_label_size,
                tick_label_size=tick_label_size,
                legend_font_size=legend_font_size,
            )
            return

        low_methods = {method for method, _ in ordered[: split_idx + 1]}
        high_methods = {method for method, _ in ordered[split_idx + 1 :]}
        low_highs = np.asarray(
            [float(np.nanmax(series_by_method[method][3])) for method in low_methods],
            dtype=float,
        )
        high_lows = np.asarray(
            [float(np.nanmin(series_by_method[method][2])) for method in high_methods],
            dtype=float,
        )
        low_cap = float(np.nanmax(low_highs)) if low_highs.size else 1.0
        high_floor = float(np.nanmin(high_lows)) if high_lows.size else low_cap * 3.0
        low_ylim_top = max(low_cap * 1.12, 1.0)
        high_ylim_bottom = max(high_floor * 0.92, low_ylim_top * 1.4)
        high_ylim_top = max(
            float(np.nanmax([np.nanmax(values[3]) for values in series_by_method.values()])),
            high_ylim_bottom,
        ) * 1.05

        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12.2, 8.0),
            dpi=180,
            gridspec_kw={"height_ratios": [2.2, 1.3], "hspace": 0.05},
        )
        for method in _method_order(rows):
            if method not in series_by_method:
                continue
            xs, center, low, high = series_by_method[method]
            color = METHOD_COLORS.get(method)
            label = METHOD_LABELS.get(method, method)
            for ax in (ax_top, ax_bottom):
                ax.plot(
                    xs,
                    center,
                    marker="o",
                    linewidth=2.2,
                    markersize=7.0,
                    label=label,
                    color=color,
                )
                ax.fill_between(xs, low, high, alpha=0.18, color=color)

        ax_top.set_ylim(high_ylim_bottom, high_ylim_top)
        ax_bottom.set_ylim(0.0, low_ylim_top)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()
        ax_bottom.set_xlabel("Number of users")
        fig.supylabel("Stage-II solver calls", fontsize=axis_label_size)
        ax_bottom.set_xticks([10, 15, 20, 25, 30])
        for ax in (ax_top, ax_bottom):
            ax.tick_params(axis="both", labelsize=tick_label_size, pad=7)
            ax.grid(True, alpha=0.25)

        d = 0.012
        kwargs = dict(color="k", clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

        handles, labels = ax_top.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.76, 0.5),
            fontsize=legend_font_size,
            frameon=True,
        )
        fig.subplots_adjust(left=0.18, right=0.74, bottom=0.16, top=0.97)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot fixed Stage-I summary figures from CSV outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--axis-label-size", type=float, default=40.0)
    parser.add_argument("--tick-label-size", type=float, default=30.0)
    parser.add_argument("--legend-font-size", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    rows = _read_csv_rows(run_dir / "stage1_final_grid_ne_gap_vs_users_stats.csv")
    _plot_metric(
        rows=rows,
        out_path=out_dir / "stage1_runtime_vs_users_log.png",
        metric="runtime_sec",
        ylabel="Runtime (s)",
        axis_label_size=float(args.axis_label_size),
        tick_label_size=float(args.tick_label_size),
        legend_font_size=float(args.legend_font_size),
        logy=True,
    )
    _plot_stage2_calls_broken_axis(
        rows=rows,
        out_path=out_dir / "stage1_stage2_calls_vs_users.png",
        axis_label_size=float(args.axis_label_size),
        tick_label_size=float(args.tick_label_size),
        legend_font_size=float(args.legend_font_size),
    )


if __name__ == "__main__":
    main()

"""Replot selected VBBR cost-ratio sensitivity figures from an existing CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
for path in (THIS_DIR, ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import run_vbbr_cost_ratio_sweep as source


def _ratio_tick_values(rows: list[dict[str, object]]) -> list[float]:
    ratios = sorted({float(row["ratio"]) for row in rows})
    if not ratios:
        return []
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    if min_ratio <= 0.0:
        return ratios
    start_exp = int(np.floor(np.log10(min_ratio)))
    end_exp = int(np.ceil(np.log10(max_ratio)))
    return [10.0**exp for exp in range(start_exp, end_exp + 1)]


def _ratio_tick_labels(values: list[float]) -> list[str]:
    labels: list[str] = []
    for value in values:
        exponent = int(round(np.log10(value)))
        if np.isclose(value, 1.0):
            labels.append(r"$10^0$")
        else:
            labels.append(rf"$10^{{{exponent}}}$")
    return labels


def _apply_ratio_axis(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    ratios = sorted({float(row["ratio"]) for row in rows})
    ticks = _ratio_tick_values(rows)
    if ratios:
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        if min_ratio > 0.0:
            ax.set_xlim(min_ratio / 1.35, max_ratio * 1.35)
        else:
            span = max_ratio - min_ratio
            pad = span * 0.06 if span > 0.0 else 1.0
            ax.set_xlim(min_ratio - pad, max_ratio + pad)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(_ratio_tick_labels(ticks))


def _set_fonts(axis_size: float, tick_size: float, legend_size: float) -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = tick_size
    plt.rcParams["axes.labelsize"] = axis_size
    plt.rcParams["xtick.labelsize"] = tick_size
    plt.rcParams["ytick.labelsize"] = tick_size
    plt.rcParams["legend.fontsize"] = legend_size
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1


def _plot_utilization(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    base_ratio: float,
    xscale: str,
    axis_size: float,
    tick_size: float,
    legend_size: float,
) -> None:
    with plt.rc_context():
        _set_fonts(axis_size, tick_size, legend_size)
        fig, ax = plt.subplots(figsize=(9.6, 6.2), dpi=180)
        for y_key, label, color, marker in [
            ("comp_utilization", "Computation utilization", "tab:green", "o"),
            ("band_utilization", "Bandwidth utilization", "tab:red", "s"),
        ]:
            stats = source._series_stats(rows, x_key="ratio", y_key=y_key)
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            if np.all(~np.isfinite(y)):
                continue
            ax.plot(x, y, color=color, marker=marker, linewidth=2.0, markersize=6.0, label=label)
            if np.any(np.isfinite(e) & (e > 0.0)):
                ax.fill_between(x, y - e, y + e, color=color, alpha=0.14)
        source._add_base_ratio_marker(ax, base_ratio)
        ax.set_xscale(xscale)
        _apply_ratio_axis(ax, rows)
        ax.set_xlabel(r"cost ratio $c_E/c_N$", fontsize=axis_size)
        ax.set_ylabel("Utilization", fontsize=axis_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.grid(alpha=0.25)
        ax.legend(
            loc="center",
            bbox_to_anchor=(0.67, 0.48),
            ncol=1,
            frameon=True,
            fontsize=legend_size,
        )
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_welfare_revenue(
    rows: list[dict[str, object]],
    *,
    out_path: Path,
    base_ratio: float,
    xscale: str,
    axis_size: float,
    tick_size: float,
    legend_size: float,
) -> None:
    with plt.rc_context():
        _set_fonts(axis_size, tick_size, legend_size)
        fig, ax_left = plt.subplots(figsize=(9.8, 6.4), dpi=180)
        ax_right = ax_left.twinx()

        left_drawn = source._plot_metric_with_band(
            ax_left,
            rows,
            y_key="social_cost",
            label="User social cost",
            color="tab:blue",
            marker="o",
            linestyle="--",
            zorder=4.0,
            markerfacecolor="white",
            band_mode="sem",
        )
        right_drawn = source._plot_metric_with_band(
            ax_right,
            rows,
            y_key="joint_revenue",
            label="SP joint revenue",
            color="tab:orange",
            marker="s",
            linestyle="-",
            zorder=3.0,
            band_mode="std",
        )

        source._add_base_ratio_marker(ax_left, base_ratio)
        ax_left.set_xscale(xscale)
        _apply_ratio_axis(ax_left, rows)
        ax_left.set_xlabel(r"cost ratio $c_E/c_N$", color="black", fontsize=axis_size)
        ax_left.set_ylabel("User social cost", color="black", fontsize=axis_size)
        ax_right.set_ylabel("SP joint revenue", color="black", fontsize=axis_size)
        ax_left.tick_params(axis="both", labelsize=tick_size, colors="black")
        ax_right.tick_params(axis="y", labelsize=tick_size, colors="black")
        source._style_axis_black(ax_left, include_x=True, include_y=True)
        source._style_axis_black(ax_right, include_x=False, include_y=True)
        source._set_metric_axis_limits(ax_left, source._series_stats(rows, x_key="ratio", y_key="social_cost"))
        source._set_metric_axis_limits(ax_right, source._series_stats(rows, x_key="ratio", y_key="joint_revenue"))
        ax_left.grid(alpha=0.25)

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        if left_drawn or right_drawn:
            ax_left.legend(
                handles_left + handles_right,
                labels_left + labels_right,
                loc="center",
                bbox_to_anchor=(0.66, 0.58),
                ncol=1,
                columnspacing=1.2,
                handlelength=2.8,
                frameon=True,
                fontsize=legend_size,
            )
        fig.subplots_adjust(left=0.17, right=0.84, bottom=0.18, top=0.96)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot selected VBBR cost-ratio figures from CSV.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--axis-size", type=float, default=30.0)
    parser.add_argument("--tick-size", type=float, default=30.0)
    parser.add_argument("--legend-size", type=float, default=24.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = source._load_metric_rows(run_dir / "vbbr_cost_ratio_sweep_metrics.csv")
    meta = source._load_summary_meta(run_dir / "vbbr_cost_ratio_sweep_summary.txt")
    base_ratio = float(meta.get("base_ratio", 1.0))
    xscale = str(meta.get("xscale", "linear"))
    if xscale == "auto":
        ratios = sorted({float(row["ratio"]) for row in rows})
        ratio_span = max(ratios) / min(ratios)
        xscale = "log" if ratio_span >= 20.0 else "linear"

    _plot_welfare_revenue(
        rows,
        out_path=run_dir / "vbbr_cost_ratio_sweep_welfare_revenue.png",
        base_ratio=base_ratio,
        xscale=xscale,
        axis_size=float(args.axis_size),
        tick_size=float(args.tick_size),
        legend_size=float(args.legend_size),
    )
    _plot_utilization(
        rows,
        out_path=run_dir / "vbbr_cost_ratio_sweep_utilization.png",
        base_ratio=base_ratio,
        xscale=xscale,
        axis_size=float(args.axis_size),
        tick_size=float(args.tick_size),
        legend_size=float(args.legend_size),
    )


if __name__ == "__main__":
    main()

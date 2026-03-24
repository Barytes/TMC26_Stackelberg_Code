from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
SRC = ROOT / "src"
for path in (THIS_DIR, ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.config import load_config


METHOD_ORDER = ["Full model", "ME", "SingleSP", "Coop", "Rand"]


def _load_rows(path: Path, *, F_total: float | None = None, B_total: float | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, object] = {
                "method": raw["method"],
                "n_users": int(raw["n_users"]),
                "trial": int(raw["trial"]),
                "social_cost": float(raw["social_cost"]),
                "comp_utilization": float(raw["comp_utilization"]),
                "band_utilization": float(raw["band_utilization"]),
                "final_pE": float(raw["final_pE"]),
                "final_pN": float(raw["final_pN"]),
                "offloading_size": float(raw["offloading_size"]),
            }
            row["offloading_ratio"] = (
                float(row["offloading_size"]) / float(row["n_users"]) if float(row["n_users"]) > 0 else float("nan")
            )
            if F_total is not None and B_total is not None:
                total_payment = (
                    float(row["final_pE"]) * float(row["comp_utilization"]) * float(F_total)
                    + float(row["final_pN"]) * float(row["band_utilization"]) * float(B_total)
                )
                row["total_user_payment"] = float(total_payment)
                row["avg_payment_per_offloading_user"] = (
                    float(total_payment) / float(row["offloading_size"])
                    if float(row["offloading_size"]) > 0.0
                    else float("nan")
                )
            rows.append(row)
    return rows


def _mean_std_by_method(rows: list[dict[str, object]], x_key: str, y_key: str) -> dict[str, list[tuple[float, float, float]]]:
    grouped: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        method = str(row["method"])
        x = float(row[x_key])
        y = float(row[y_key])
        grouped.setdefault(method, {}).setdefault(x, []).append(y)
    out: dict[str, list[tuple[float, float, float]]] = {}
    for method, stats in grouped.items():
        seq: list[tuple[float, float, float]] = []
        for x in sorted(stats):
            vals = np.asarray(stats[x], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                seq.append((x, float("nan"), float("nan")))
            else:
                seq.append((x, float(np.mean(vals)), float(np.std(vals))))
        out[method] = seq
    return out


def _plot_lines(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    grouped = _mean_std_by_method(rows, x_key, y_key)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    for idx, method in enumerate(METHOD_ORDER):
        if method not in grouped:
            continue
        stats = grouped[method]
        x = np.asarray([item[0] for item in stats], dtype=float)
        y = np.asarray([item[1] for item in stats], dtype=float)
        e = np.asarray([item[2] for item in stats], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt="-o",
            capsize=4,
            linewidth=1.8,
            markersize=5.5,
            color=cmap(idx % 10),
            label=method,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_two_panel(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    panels: list[tuple[str, str]],
    xlabel: str,
    title: str,
    out_path: Path,
) -> None:
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    for panel_idx, (y_key, ylabel) in enumerate(panels):
        grouped = _mean_std_by_method(rows, x_key, y_key)
        ax = axes[panel_idx]
        for idx, method in enumerate(METHOD_ORDER):
            if method not in grouped:
                continue
            stats = grouped[method]
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=e,
                fmt="-o",
                capsize=4,
                linewidth=1.7,
                markersize=5.0,
                color=cmap(idx % 10),
                label=method,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if panel_idx == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot strategic-setting metrics from an existing E-series CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to an existing E1/E2/E3/E4 CSV with strategic rows.")
    parser.add_argument("--config", type=str, default=None, help="Config path used to compute derived payment metrics from F and B.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory under outputs/.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    F_total: float | None = None
    B_total: float | None = None
    if args.config:
        cfg = load_config(args.config)
        F_total = float(cfg.system.F)
        B_total = float(cfg.system.B)
    rows = _load_rows(csv_path, F_total=F_total, B_total=B_total)
    out_dir = resolve_out_dir("replot_strategic_metrics_from_csv", args.out_dir)

    _plot_lines(
        rows,
        x_key="n_users",
        y_key="social_cost",
        xlabel="Number of users",
        ylabel="Total user social cost",
        title="User Social Cost Comparison",
        out_path=out_dir / "user_social_cost_compare.png",
    )
    _plot_two_panel(
        rows,
        x_key="n_users",
        panels=[
            ("comp_utilization", "Computation utilization"),
            ("band_utilization", "Bandwidth utilization"),
        ],
        xlabel="Number of users",
        title="Resource Utilization Comparison",
        out_path=out_dir / "resource_utilization_compare.png",
    )
    _plot_two_panel(
        rows,
        x_key="n_users",
        panels=[
            ("offloading_ratio", "Offloading ratio"),
            ("offloading_size", "Number of offloading users"),
        ],
        xlabel="Number of users",
        title="Offloading Outcomes vs. Number of Users",
        out_path=out_dir / "offloading_outcomes_compare.png",
    )
    _plot_two_panel(
        rows,
        x_key="n_users",
        panels=[
            ("final_pE", "Final pE"),
            ("final_pN", "Final pN"),
        ],
        xlabel="Number of users",
        title="Price Outcomes vs. Number of Users",
        out_path=out_dir / "price_outcomes_compare.png",
    )
    if F_total is not None and B_total is not None:
        _plot_lines(
            rows,
            x_key="n_users",
            y_key="total_user_payment",
            xlabel="Number of users",
            ylabel="Total user payment",
            title="Total User Payment vs. Number of Users",
            out_path=out_dir / "total_user_payment_compare.png",
        )
        _plot_lines(
            rows,
            x_key="n_users",
            y_key="avg_payment_per_offloading_user",
            xlabel="Number of users",
            ylabel="Average payment per offloading user",
            title="Average Payment per Offloading User vs. Number of Users",
            out_path=out_dir / "avg_payment_per_offloading_user_compare.png",
        )

    summary_lines = [
        f"source_csv = {csv_path}",
        f"rows = {len(rows)}",
        f"methods = {','.join(sorted({str(row['method']) for row in rows}))}",
        f"n_users = {','.join(str(int(x)) for x in sorted({int(row['n_users']) for row in rows}))}",
    ]
    if args.config:
        summary_lines.extend(
            [
                f"config = {args.config}",
                f"system_F = {F_total}",
                f"system_B = {B_total}",
            ]
        )
    (out_dir / "replot_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

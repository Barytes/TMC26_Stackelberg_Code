from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cache_root = ROOT / "outputs" / "_tmp_cache" / "multi_provider"
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for plotting. Run this script in the project environment, for example with `uv run python`.") from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.multi_provider.multi_provider_core import contrast_text_color, format_pair_cost_label

AXIS_FONT = 35
TICK_FONT = 35
CELL_FONT = 28
BAR_LABEL_FONT = 26


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_summary(out_dir: Path) -> dict[str, object]:
    return json.loads((out_dir / "multi_provider_summary.json").read_text(encoding="utf-8"))


def _plot_assignment(out_dir: Path, summary: dict[str, object]) -> Path:
    matrix = np.asarray(summary["assignment_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_title("Equilibrium assignment")
    ax.set_xlabel("NSP")
    ax.set_ylabel("ESP")
    ax.set_xticks(range(matrix.shape[1]), ["NSP%d" % (i + 1) for i in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), ["ESP%d" % (i + 1) for i in range(matrix.shape[0])])
    for e_idx in range(matrix.shape[0]):
        for n_idx in range(matrix.shape[1]):
            ax.text(n_idx, e_idx, "%d" % int(matrix[e_idx, n_idx]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Offloading users")
    path = out_dir / "multi_provider_assignment_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _average_cost_matrix(out_dir: Path, summary: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(out_dir / "multi_provider_assignment.csv")
    num_esp = int(summary["num_esp"])
    num_nsp = int(summary["num_nsp"])
    totals = np.zeros((num_esp, num_nsp), dtype=float)
    counts = np.zeros((num_esp, num_nsp), dtype=int)
    for row in rows:
        if str(row.get("offloading", "")).strip() != "1":
            continue
        esp = int(row["esp"]) - 1
        nsp = int(row["nsp"]) - 1
        totals[esp, nsp] += float(row["offload_cost"])
        counts[esp, nsp] += 1
    avg = np.full((num_esp, num_nsp), np.nan, dtype=float)
    np.divide(totals, counts, out=avg, where=counts > 0)
    return avg, counts


def _plot_average_offloading_cost(out_dir: Path, summary: dict[str, object]) -> Path:
    avg, counts = _average_cost_matrix(out_dir, summary)
    masked = np.ma.masked_invalid(avg)
    fig, ax = plt.subplots(figsize=(12.0, 9.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(masked, cmap=cmap)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(range(avg.shape[1]), ["NSP%d" % (i + 1) for i in range(avg.shape[1])], fontsize=TICK_FONT)
    ax.set_yticks(range(avg.shape[0]), ["ESP%d" % (i + 1) for i in range(avg.shape[0])], fontsize=TICK_FONT)
    ax.set_xticks(np.arange(-0.5, avg.shape[1], 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, avg.shape[0], 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=3.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for e_idx in range(avg.shape[0]):
        for n_idx in range(avg.shape[1]):
            if np.isfinite(avg[e_idx, n_idx]):
                text = format_pair_cost_label(float(avg[e_idx, n_idx]), int(counts[e_idx, n_idx]))
                color = contrast_text_color(im.cmap(im.norm(float(avg[e_idx, n_idx]))))
            else:
                text = "N/A"
                color = "#555555"
            ax.text(n_idx, e_idx, text, ha="center", va="center", color=color, fontsize=CELL_FONT, fontweight="semibold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean offloading cost", fontsize=AXIS_FONT)
    cbar.ax.tick_params(labelsize=TICK_FONT)
    path = out_dir / "multi_provider_average_offloading_cost_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_convergence(out_dir: Path) -> Path:
    rows = _read_csv(out_dir / "multi_provider_trajectory.csv")
    iters = np.asarray([int(r["iteration"]) for r in rows], dtype=float)
    gap = np.asarray([float(r["restricted_gap"]) for r in rows], dtype=float)
    social = np.asarray([float(r["social_cost"]) for r in rows], dtype=float)
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax1.plot(iters, gap, marker="o", linewidth=2.0, label="Restricted NE gap", color="#2f6f9f")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Restricted NE gap")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(iters, social, marker="s", linewidth=1.8, label="User social cost", color="#c15f3c")
    ax2.set_ylabel("User social cost")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    path = out_dir / "multi_provider_convergence.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_provider_metrics(out_dir: Path) -> Path:
    rows = _read_csv(out_dir / "multi_provider_provider_metrics.csv")
    labels = [r["provider"] for r in rows]
    prices = np.asarray([float(r["price"]) for r in rows], dtype=float)
    revenues = np.asarray([float(r["revenue"]) for r in rows], dtype=float)
    util = np.asarray([float(r["utilization"]) for r in rows], dtype=float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), constrained_layout=True)
    axes[0].bar(x, prices, color="#4f7cac")
    axes[0].set_title("Final prices")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Unit price")
    axes[1].bar(x, revenues, color="#5b9f6e")
    axes[1].set_title("Provider revenues")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Revenue")
    axes[2].bar(x, util, color="#c7813b")
    axes[2].set_title("Resource utilization")
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set_ylim(0.0, max(1.05, float(np.max(util)) * 1.1 if util.size else 1.0))
    axes[2].set_ylabel("Demand / capacity")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.22)
    path = out_dir / "multi_provider_provider_metrics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_provider_revenue(out_dir: Path) -> Path:
    rows = _read_csv(out_dir / "multi_provider_provider_metrics.csv")
    labels = [r["provider"] for r in rows]
    revenues = np.asarray([float(r["revenue"]) for r in rows], dtype=float)
    colors = ["#2f6f9f" if r["kind"] == "ESP" else "#c66b3d" for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14.0, 8.8), constrained_layout=True)
    bars = ax.bar(x, revenues, color=colors, edgecolor="#202020", linewidth=1.2)
    ax.set_xlabel("")
    ax.set_ylabel("Revenue", fontsize=AXIS_FONT)
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=TICK_FONT)
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(True, axis="y", alpha=0.22, linewidth=1.2)
    upper = max(1.0, float(np.max(revenues)) * 1.18 if revenues.size else 1.0)
    ax.set_ylim(0.0, upper)
    for bar, value in zip(bars, revenues):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + upper * 0.018,
            "%.1f" % value,
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONT,
            color="#202020",
        )
    path = out_dir / "multi_provider_provider_revenue.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot figures for a multi-provider solver output directory.")
    parser.add_argument("out_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    summary = _load_summary(out_dir)
    artifacts = [
        _plot_assignment(out_dir, summary),
        _plot_average_offloading_cost(out_dir, summary),
        _plot_convergence(out_dir),
        _plot_provider_revenue(out_dir),
        _plot_provider_metrics(out_dir),
    ]
    plot_summary = ["plot_artifacts = %s" % ",".join(path.name for path in artifacts)]
    (out_dir / "plot_summary.txt").write_text("\n".join(plot_summary) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()

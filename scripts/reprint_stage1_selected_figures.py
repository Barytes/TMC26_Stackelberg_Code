from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_boundary_hypothesis_check import _scan_exact_boundaries
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    pE_vals = sorted({float(row["pE"]) for row in rows})
    pN_vals = sorted({float(row["pN"]) for row in rows})
    pE_grid = np.asarray(pE_vals, dtype=float)
    pN_grid = np.asarray(pN_vals, dtype=float)
    esp_grid = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    nsp_grid = np.full_like(esp_grid, np.nan)

    e_map = {value: idx for idx, value in enumerate(pE_vals)}
    n_map = {value: idx for idx, value in enumerate(pN_vals)}
    for row in rows:
        i = e_map[float(row["pE"])]
        j = n_map[float(row["pN"])]
        esp_grid[j, i] = float(row["esp_revenue"])
        nsp_grid[j, i] = float(row["nsp_revenue"])

    if np.any(~np.isfinite(esp_grid)) or np.any(~np.isfinite(nsp_grid)):
        raise ValueError(f"Incomplete grid in CSV: {path}")
    return pE_grid, pN_grid, esp_grid, nsp_grid


def _load_summary_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_exact_boundaries(path: Path, provider: str) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        values = [float(row["boundary_price"]) for row in reader if row["provider"] == provider]
    return np.asarray(values, dtype=float)


def _nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _configure_fonts(language: str) -> None:
    if language == "zh":
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 19.5
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 21
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["axes.linewidth"] = 1.8
    plt.rcParams["xtick.major.width"] = 1.6
    plt.rcParams["ytick.major.width"] = 1.6
    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["ytick.major.size"] = 8


def _plot_heatmap(
    values: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2), dpi=220)
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[float(pE_grid[0]), float(pE_grid[-1]), float(pN_grid[0]), float(pN_grid[-1])],
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=16, width=1.4, length=6)
    cbar.outline.set_linewidth(1.4)
    ax.tick_params(axis="both", which="major", pad=8)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.13, top=0.90)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _plot_slice(
    *,
    provider: str,
    prices: np.ndarray,
    revenues: np.ndarray,
    switching_prices: np.ndarray,
    fixed_axis_label: str,
    fixed_value: float,
    title: str,
    xlabel: str,
    ylabel: str,
    revenue_label: str,
    switching_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2), dpi=220)
    ax.plot(prices, revenues, color="black", linewidth=3.0, label=revenue_label)
    for idx, value in enumerate(switching_prices):
        ax.axvline(
            float(value),
            color="red",
            linewidth=2.0,
            alpha=0.45,
            linestyle="-",
            label=switching_label if idx == 0 else None,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({fixed_axis_label}={fixed_value:.4g})")
    ax.tick_params(axis="both", which="major", pad=8)
    legend = ax.legend(
        loc="upper right",
        fontsize=16,
        frameon=True,
        framealpha=0.92,
        borderpad=0.5,
        handlelength=1.6,
        labelspacing=0.3,
    )
    legend.get_frame().set_linewidth(1.2)
    ax.margins(x=0.02, y=0.08)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.90)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprint selected stage-1 figures from existing outputs with Chinese and English variants."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/run_stage1_price_heatmaps_20260309_113102/price_grid_metrics.csv",
        help="Path to price_grid_metrics.csv.",
    )
    parser.add_argument(
        "--boundary-dir",
        type=str,
        default="outputs/boundary_price_overlays/_tmp_boundary_hypothesis_check",
        help="Directory containing summary.txt and exact_dg_boundaries.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reprinted_stage1_figures_20260316",
        help="Directory to store the regenerated figures.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML config used for exact slice recomputation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for exact slice recomputation.",
    )
    parser.add_argument(
        "--root-tol",
        type=float,
        default=1e-5,
        help="Bisection tolerance for exact switching-price recomputation.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    boundary_dir = Path(args.boundary_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pE_grid, pN_grid, esp_grid, nsp_grid = _load_grid_csv(csv_path)
    summary = _load_summary_map(boundary_dir / "summary.txt")
    start_pE = float(summary["start_pE"])
    start_pN = float(summary["start_pN"])
    scan_pE_max = float(summary["scan_pE_max"])
    scan_pN_max = float(summary["scan_pN_max"])
    scan_pE_points = int(summary["scan_pE_points"])
    scan_pN_points = int(summary["scan_pN_points"])
    exact_path = boundary_dir / "exact_dg_boundaries.csv"
    esp_switching = _load_exact_boundaries(exact_path, "E")
    nsp_switching = _load_exact_boundaries(exact_path, "N")

    cfg = load_config(args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)
    users = sample_users(cfg, rng)
    system = cfg.system
    stack_cfg = cfg.stackelberg

    esp_prices, esp_slice_evals, _ = _scan_exact_boundaries(
        provider="E",
        fixed_price=start_pN,
        price_max=scan_pE_max,
        scan_points=scan_pE_points,
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    nsp_prices, nsp_slice_evals, _ = _scan_exact_boundaries(
        provider="N",
        fixed_price=start_pE,
        price_max=scan_pN_max,
        scan_points=scan_pN_points,
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    esp_slice = np.asarray([float(item.provider_revenue) for item in esp_slice_evals], dtype=float)
    nsp_slice = np.asarray([float(item.provider_revenue) for item in nsp_slice_evals], dtype=float)

    langs = {
        "en": {
            "heatmaps": [
                ("esp_revenue_heatmap_en.png", esp_grid, "ESP Revenue Heatmap", "ESP Revenue", "viridis"),
                ("nsp_revenue_heatmap_en.png", nsp_grid, "NSP Revenue Heatmap", "NSP Revenue", "plasma"),
            ],
            "slice_esp": {
                "title": "ESP Revenue Slice and Switching Prices",
                "xlabel": r"$p_E$",
                "ylabel": "ESP revenue",
                "revenue_label": "Revenue",
                "switching_label": "switching prices",
                "fixed_axis_label": r"$p_N$",
            },
            "slice_nsp": {
                "title": "NSP Revenue Slice and Switching Prices",
                "xlabel": r"$p_N$",
                "ylabel": "NSP revenue",
                "revenue_label": "Revenue",
                "switching_label": "switching prices",
                "fixed_axis_label": r"$p_E$",
            },
            "x": r"$p_E$",
            "y": r"$p_N$",
        },
        "zh": {
            "heatmaps": [
                ("esp_revenue_heatmap_zh.png", esp_grid, "ESP 收益热图", "ESP 收益", "viridis"),
                ("nsp_revenue_heatmap_zh.png", nsp_grid, "NSP 收益热图", "NSP 收益", "plasma"),
            ],
            "slice_esp": {
                "title": "ESP 收益切片与切换价格",
                "xlabel": r"$p_E$",
                "ylabel": "ESP 收益",
                "revenue_label": "收益",
                "switching_label": "切换价格",
                "fixed_axis_label": r"$p_N$",
            },
            "slice_nsp": {
                "title": "NSP 收益切片与切换价格",
                "xlabel": r"$p_N$",
                "ylabel": "NSP 收益",
                "revenue_label": "收益",
                "switching_label": "切换价格",
                "fixed_axis_label": r"$p_E$",
            },
            "x": r"$p_E$",
            "y": r"$p_N$",
        },
    }

    for language, meta in langs.items():
        _configure_fonts(language)
        for file_name, values, title, cbar_label, cmap in meta["heatmaps"]:
            _plot_heatmap(
                values,
                pE_grid,
                pN_grid,
                title=title,
                xlabel=meta["x"],
                ylabel=meta["y"],
                cbar_label=cbar_label,
                cmap=cmap,
                out_path=out_dir / file_name,
            )
        _plot_slice(
            provider="E",
            prices=esp_prices,
            revenues=esp_slice,
            switching_prices=esp_switching,
            fixed_axis_label=meta["slice_esp"]["fixed_axis_label"],
            fixed_value=start_pN,
            title=meta["slice_esp"]["title"],
            xlabel=meta["slice_esp"]["xlabel"],
            ylabel=meta["slice_esp"]["ylabel"],
            revenue_label=meta["slice_esp"]["revenue_label"],
            switching_label=meta["slice_esp"]["switching_label"],
            out_path=out_dir / f"esp_slice_boundary_comparison_{language}.png",
        )
        _plot_slice(
            provider="N",
            prices=nsp_prices,
            revenues=nsp_slice,
            switching_prices=nsp_switching,
            fixed_axis_label=meta["slice_nsp"]["fixed_axis_label"],
            fixed_value=start_pE,
            title=meta["slice_nsp"]["title"],
            xlabel=meta["slice_nsp"]["xlabel"],
            ylabel=meta["slice_nsp"]["ylabel"],
            revenue_label=meta["slice_nsp"]["revenue_label"],
            switching_label=meta["slice_nsp"]["switching_label"],
            out_path=out_dir / f"nsp_slice_boundary_comparison_{language}.png",
        )

    summary_lines = [
        f"csv = {csv_path}",
        f"boundary_dir = {boundary_dir}",
        f"config = {args.config}",
        f"seed = {seed}",
        f"start_pE = {start_pE:.12g}",
        f"start_pN = {start_pN:.12g}",
        f"scan_pE_max = {scan_pE_max:.12g}",
        f"scan_pN_max = {scan_pN_max:.12g}",
        f"scan_pE_points = {scan_pE_points}",
        f"scan_pN_points = {scan_pN_points}",
        "notes = revenue heatmaps are redrawn without Stackelberg Equilibrium markers; slice figures are recomputed on the exact diagnostic slices and keep only Revenue and switching prices.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

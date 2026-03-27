"""Replot customized C1/C2 convergence figures from existing CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replot customized C1/C2 convergence figures from CSV outputs.")
    parser.add_argument("--c1-dir", type=str, required=True, help="Directory containing C1_restricted_gap_trajectory.csv.")
    parser.add_argument("--c2-dir", type=str, required=True, help="Directory containing C2_best_response_gain_trajectory.csv.")
    parser.add_argument("--font-scale", type=float, default=1.25, help="Global font scale factor.")
    return parser


def _choose_cjk_font() -> str | None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimSun",
    ]
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def _configure_fonts(*, chinese: bool) -> str | None:
    plt.rcParams["axes.unicode_minus"] = False
    if not chinese:
        return None
    font_name = _choose_cjk_font()
    if font_name is not None:
        existing = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["font.sans-serif"] = [font_name] + [name for name in existing if name != font_name]
    return font_name


def _aligned_sequence_matrix(sequences: list[list[float]]) -> np.ndarray:
    if not sequences:
        raise ValueError("Expected at least one sequence.")
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=float)
    for idx, seq in enumerate(sequences):
        seq_arr = np.asarray(seq, dtype=float)
        arr[idx, : seq_arr.size] = seq_arr
        if seq_arr.size < max_len and seq_arr.size > 0:
            arr[idx, seq_arr.size :] = seq_arr[-1]
    return arr


def _aligned_quantile_band(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    return (
        np.nanmedian(arr, axis=0),
        np.nanpercentile(arr, 25, axis=0),
        np.nanpercentile(arr, 75, axis=0),
    )


def _aligned_mean_std(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_series_by_trial(
    rows: list[dict[str, str]],
    *fields: str,
) -> dict[str, list[list[float]]]:
    by_trial: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_trial.setdefault(int(row["trial"]), []).append(row)
    grouped: dict[str, list[list[float]]] = {field: [] for field in fields}
    for trial in sorted(by_trial):
        ordered = sorted(by_trial[trial], key=lambda item: int(item["iteration"]))
        for field in fields:
            grouped[field].append([float(item[field]) for item in ordered])
    return grouped


def _apply_text_sizes(font_scale: float) -> None:
    plt.rcParams["font.size"] = 12.5 * font_scale
    plt.rcParams["axes.titlesize"] = 16.5 * font_scale
    plt.rcParams["axes.labelsize"] = 14.5 * font_scale
    plt.rcParams["xtick.labelsize"] = 12.2 * font_scale
    plt.rcParams["ytick.labelsize"] = 12.2 * font_scale
    plt.rcParams["legend.fontsize"] = 11.0 * font_scale


def _plot_c1(
    *,
    csv_path: Path,
    out_path: Path,
    font_scale: float,
    chinese: bool,
) -> None:
    rows = _read_csv_rows(csv_path)
    grouped = _group_series_by_trial(rows, "restricted_gap", "grid_ne_gap")
    restricted_median, restricted_q25, restricted_q75 = _aligned_quantile_band(grouped["restricted_gap"])
    grid_median, grid_q25, grid_q75 = _aligned_quantile_band(grouped["grid_ne_gap"])
    stopping_tol = 0.0
    if rows:
        finite_vals = [float(row["restricted_gap"]) for row in rows if float(row["restricted_gap"]) >= 0.0]
        if finite_vals:
            stopping_tol = min(finite_vals) * 0.0

    with plt.rc_context():
        font_name = _configure_fonts(chinese=chinese)
        _apply_text_sizes(font_scale)
        fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=150)
        x = np.arange(1, restricted_median.size + 1)
        if chinese:
            labels = {
                "title": "算法5.2的纳什均衡间隙收敛性",
                "xlabel": "迭代次数",
                "ylabel": "纳什均衡间隙值",
                "restricted_median": "受限纳什均衡间隙（中位数）",
                "restricted_band": "受限纳什均衡间隙（25%-75%分位区间）",
                "grid_median": "基准纳什均衡间隙（中位数）",
                "grid_band": "基准纳什均衡间隙（25%-75%分位区间）",
                "tol": "停止容差",
            }
        else:
            labels = {
                "title": "Convergence of NE gap",
                "xlabel": "Iterations",
                "ylabel": "NE gap",
                "restricted_median": "Restricted NE gap (median)",
                "restricted_band": "Restricted NE gap (25%-75%)",
                "grid_median": "Oracle NE gap (median)",
                "grid_band": "Oracle NE gap (25%-75%)",
                "tol": "Stopping tolerance",
            }
        ax.plot(x, restricted_median, marker="o", linewidth=2.2, color="tab:blue", label=labels["restricted_median"])
        ax.fill_between(x, restricted_q25, restricted_q75, alpha=0.18, color="tab:blue", label=labels["restricted_band"])
        ax.plot(
            x,
            grid_median,
            marker="s",
            linewidth=2.1,
            linestyle="--",
            color="tab:orange",
            label=labels["grid_median"],
        )
        ax.fill_between(x, grid_q25, grid_q75, alpha=0.14, color="tab:orange", label=labels["grid_band"])
        ax.axhline(stopping_tol, color="black", linestyle="--", linewidth=1.3, label=labels["tol"])
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.set_title(labels["title"])
        ax.grid(alpha=0.25)
        if chinese and font_name is not None:
            ax.legend(loc="best", prop={"family": font_name, "size": float(plt.rcParams["legend.fontsize"])})
        else:
            ax.legend(loc="best", fontsize=float(plt.rcParams["legend.fontsize"]))
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def _plot_c2(
    *,
    csv_path: Path,
    out_path: Path,
    font_scale: float,
    chinese: bool,
) -> None:
    rows = _read_csv_rows(csv_path)
    grouped = _group_series_by_trial(
        rows,
        "esp_gain",
        "nsp_gain",
        "grid_true_esp_gain",
        "grid_true_nsp_gain",
    )
    mean_esp, std_esp = _aligned_mean_std(grouped["esp_gain"])
    mean_nsp, std_nsp = _aligned_mean_std(grouped["nsp_gain"])
    mean_grid_esp, std_grid_esp = _aligned_mean_std(grouped["grid_true_esp_gain"])
    mean_grid_nsp, std_grid_nsp = _aligned_mean_std(grouped["grid_true_nsp_gain"])

    with plt.rc_context():
        font_name = _configure_fonts(chinese=chinese)
        _apply_text_sizes(font_scale)
        fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=150)
        x = np.arange(1, mean_esp.size + 1)
        if chinese:
            labels = {
                "title": "算法5.2的最优响应增益的收敛性",
                "xlabel": "迭代次数",
                "ylabel": "最优响应增益值",
                "esp": "ESP受限最优响应增益",
                "nsp": "NSP受限最优响应增益",
                "grid_esp": "ESP基准最优响应增益",
                "grid_nsp": "NSP基准最优响应增益",
            }
        else:
            labels = {
                "title": "Convergence of best-response gain",
                "xlabel": "Iterations",
                "ylabel": "Best-response gain",
                "esp": "Restricted ESP gain",
                "nsp": "Restricted NSP gain",
                "grid_esp": "Oracle ESP gain",
                "grid_nsp": "Oracle NSP gain",
            }
        ax.plot(x, mean_esp, marker="o", linewidth=2.2, color="tab:blue", label=labels["esp"])
        ax.fill_between(x, mean_esp - std_esp, mean_esp + std_esp, alpha=0.16, color="tab:blue")
        ax.plot(x, mean_nsp, marker="s", linewidth=2.2, color="tab:orange", label=labels["nsp"])
        ax.fill_between(x, mean_nsp - std_nsp, mean_nsp + std_nsp, alpha=0.16, color="tab:orange")
        ax.plot(x, mean_grid_esp, marker="o", linewidth=2.0, linestyle="--", color="tab:blue", label=labels["grid_esp"])
        ax.fill_between(x, mean_grid_esp - std_grid_esp, mean_grid_esp + std_grid_esp, alpha=0.08, color="tab:blue")
        ax.plot(x, mean_grid_nsp, marker="s", linewidth=2.0, linestyle="--", color="tab:orange", label=labels["grid_nsp"])
        ax.fill_between(x, mean_grid_nsp - std_grid_nsp, mean_grid_nsp + std_grid_nsp, alpha=0.08, color="tab:orange")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.3, label="_nolegend_")
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.set_title(labels["title"])
        ax.grid(alpha=0.25)
        if chinese and font_name is not None:
            ax.legend(loc="upper right", ncol=2, prop={"family": font_name, "size": float(plt.rcParams["legend.fontsize"])})
        else:
            ax.legend(loc="upper right", ncol=2, fontsize=float(plt.rcParams["legend.fontsize"]))
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    c1_dir = Path(args.c1_dir)
    c2_dir = Path(args.c2_dir)
    font_scale = float(args.font_scale)

    _plot_c1(
        csv_path=c1_dir / "C1_restricted_gap_trajectory.csv",
        out_path=c1_dir / "C1_restricted_gap_trajectory.png",
        font_scale=font_scale,
        chinese=False,
    )
    _plot_c1(
        csv_path=c1_dir / "C1_restricted_gap_trajectory.csv",
        out_path=c1_dir / "C1_restricted_gap_trajectory_zh.png",
        font_scale=font_scale,
        chinese=True,
    )
    _plot_c2(
        csv_path=c2_dir / "C2_best_response_gain_trajectory.csv",
        out_path=c2_dir / "C2_best_response_gain_trajectory.png",
        font_scale=font_scale,
        chinese=False,
    )
    _plot_c2(
        csv_path=c2_dir / "C2_best_response_gain_trajectory.csv",
        out_path=c2_dir / "C2_best_response_gain_trajectory_zh.png",
        font_scale=font_scale,
        chinese=True,
    )


if __name__ == "__main__":
    main()

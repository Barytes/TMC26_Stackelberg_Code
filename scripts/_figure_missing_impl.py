from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import time

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", str(ROOT / "outputs" / "_tmp_cache" / "tmc26_cache")))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _figure_output_schema import load_figure_manifest, write_standard_figure_summary
from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
import run_stage2_approximation_ratio as s2_ratio
import run_boundary_hypothesis_check as boundary_diag
from tmc26_exp.baselines import (
    BaselineOutcome,
    _grid_ne_gap_audit,
    _solve_centralized_minlp,
    _stage2_solver,
    baseline_coop,
    baseline_market_equilibrium,
    baseline_random_offloading,
    baseline_single_sp,
    baseline_stage1_bo,
    baseline_stage1_bo_online_grid_ne_gap,
    baseline_stage1_ga,
    baseline_stage1_grid_search_oracle,
    baseline_stage1_marl,
    evaluate_stage1_price_grid,
)
from tmc26_exp.config import DistributionSpec, ExperimentConfig, load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    _boundary_price_for_provider,
    _build_data,
    _candidate_family,
    _provider_revenue_from_stage2_result,
    algorithm_3_gain_approximation,
    solve_stage1_pricing,
    solve_stage2_scm,
)


def _load_cfg(config_path: str, n_users: int | None = None) -> ExperimentConfig:
    cfg = load_config(config_path)
    if n_users is not None:
        cfg = replace(cfg, n_users=int(n_users))
    return cfg


def _sample_users(cfg: ExperimentConfig, n_users: int, seed: int, trial: int = 0):
    cfg_n = replace(cfg, n_users=int(n_users))
    rng = np.random.default_rng(int(seed) + 10007 * int(n_users) + int(trial))
    return sample_users(cfg_n, rng)


def _write_summary(path: Path, lines: list[str]) -> None:
    write_standard_figure_summary(path, lines)


def _mean_std_by_method(rows: list[dict[str, object]], x_key: str, y_key: str) -> dict[str, list[tuple[float, float, float]]]:
    methods = sorted({str(row["method"]) for row in rows})
    out: dict[str, list[tuple[float, float, float]]] = {}
    for method in methods:
        xs = sorted({float(row[x_key]) for row in rows if str(row["method"]) == method})
        stats: list[tuple[float, float, float]] = []
        for x in xs:
            vals = np.asarray(
                [
                    float(row[y_key])
                    for row in rows
                    if str(row["method"]) == method and float(row[x_key]) == float(x) and np.isfinite(float(row[y_key]))
                ],
                dtype=float,
            )
            if vals.size == 0:
                stats.append((float(x), float("nan"), float("nan")))
            else:
                stats.append((float(x), float(np.mean(vals)), float(np.std(vals))))
        out[method] = stats
    return out


def _plot_method_errorbars(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    method_order: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    grouped = _mean_std_by_method(rows, x_key, y_key)
    order = method_order or sorted(grouped.keys())
    cmap = plt.get_cmap("tab10")
    for idx, method in enumerate(order):
        if method not in grouped:
            continue
        stats = grouped[method]
        x = np.asarray([item[0] for item in stats], dtype=float)
        y = np.asarray([item[1] for item in stats], dtype=float)
        e = np.asarray([item[2] for item in stats], dtype=float)
        if np.all(~np.isfinite(y)):
            continue
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


def _plot_three_panel(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    panels: list[tuple[str, str]],
    xlabel: str,
    title: str,
    out_path: Path,
    method_order: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(15.5, 4.8), dpi=150)
    grouped = _mean_std_by_method(rows, x_key, panels[0][0])
    order = method_order or sorted({str(row["method"]) for row in rows})
    cmap = plt.get_cmap("tab10")
    for panel_idx, (y_key, ylabel) in enumerate(panels):
        ax = axes[panel_idx]
        grouped = _mean_std_by_method(rows, x_key, y_key)
        for idx, method in enumerate(order):
            if method not in grouped:
                continue
            stats = grouped[method]
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            if np.all(~np.isfinite(y)):
                continue
            ax.errorbar(
                x,
                y,
                yerr=e,
                fmt="-o",
                capsize=4,
                linewidth=1.6,
                markersize=4.8,
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


def _aligned_sequence_matrix(sequences: list[list[float]]) -> np.ndarray:
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=float)
    for i, seq in enumerate(sequences):
        seq_arr = np.asarray(seq, dtype=float)
        arr[i, : seq_arr.size] = seq_arr
        if seq_arr.size < max_len and seq_arr.size > 0:
            arr[i, seq_arr.size :] = seq_arr[-1]
    return arr


def _aligned_series(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _aligned_quantile_band(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _aligned_sequence_matrix(sequences)
    median = np.nanmedian(arr, axis=0)
    q25 = np.nanquantile(arr, 0.25, axis=0)
    q75 = np.nanquantile(arr, 0.75, axis=0)
    return median, q25, q75


def _c1_grid_gap_audit_grids(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    pE_grid = np.linspace(float(cfg.system.cE), float(cfg.baselines.max_price_E), max(2, int(cfg.baselines.gso_grid_points)))
    pN_grid = np.linspace(float(cfg.system.cN), float(cfg.baselines.max_price_N), max(2, int(cfg.baselines.gso_grid_points)))
    return pE_grid, pN_grid


def _c1_compute_grid_ne_gap(
    users,
    cfg: ExperimentConfig,
    pE: float,
    pN: float,
    stage2_cache: dict[tuple[float, float], BaselineOutcome],
) -> float:
    key = (round(float(pE), 12), round(float(pN), 12))
    if key in stage2_cache:
        current_out = stage2_cache[key]
    else:
        current_out = _stage2_solver(
            cfg.baselines.stage2_solver_for_pricing,
            users,
            float(pE),
            float(pN),
            cfg.system,
            cfg.stackelberg,
            cfg.baselines,
        )
        stage2_cache[key] = current_out
    pE_audit_grid, pN_audit_grid = _c1_grid_gap_audit_grids(cfg)
    return float(
        _grid_ne_gap_audit(
            current_out,
            users,
            cfg.system,
            cfg.stackelberg,
            cfg.baselines,
            stage2_cache,
            pE_audit_grid,
            pN_audit_grid,
        )
    )


def _load_grid_ne_gap_surface(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Grid CSV is empty: {csv_path}")
    header = set(rows[0].keys())
    if "grid_ne_gap" in header:
        gap_key = "grid_ne_gap"
    elif "eps" in header:
        gap_key = "eps"
    else:
        raise ValueError(f"Grid CSV must contain 'grid_ne_gap' or legacy alias 'eps': {csv_path}")

    pE_grid = np.asarray(sorted({float(row["pE"]) for row in rows}), dtype=float)
    pN_grid = np.asarray(sorted({float(row["pN"]) for row in rows}), dtype=float)
    e_map = {float(value): idx for idx, value in enumerate(pE_grid)}
    n_map = {float(value): idx for idx, value in enumerate(pN_grid)}
    grid = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    for row in rows:
        i = e_map[float(row["pE"])]
        j = n_map[float(row["pN"])]
        grid[j, i] = float(row[gap_key])
    if np.any(~np.isfinite(grid)):
        raise ValueError(f"Grid CSV is incomplete: {csv_path}")
    return pE_grid, pN_grid, grid


def _load_revenue_surfaces(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Grid CSV is empty: {csv_path}")
    required = {"pE", "pN", "esp_revenue", "nsp_revenue"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Grid CSV missing required columns {sorted(missing)}: {csv_path}")

    pE_grid = np.asarray(sorted({float(row["pE"]) for row in rows}), dtype=float)
    pN_grid = np.asarray(sorted({float(row["pN"]) for row in rows}), dtype=float)
    e_map = {float(value): idx for idx, value in enumerate(pE_grid)}
    n_map = {float(value): idx for idx, value in enumerate(pN_grid)}
    esp_revenue = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    nsp_revenue = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    for row in rows:
        i = e_map[float(row["pE"])]
        j = n_map[float(row["pN"])]
        esp_revenue[j, i] = float(row["esp_revenue"])
        nsp_revenue[j, i] = float(row["nsp_revenue"])
    if np.any(~np.isfinite(esp_revenue)) or np.any(~np.isfinite(nsp_revenue)):
        raise ValueError(f"Grid CSV is incomplete: {csv_path}")
    return pE_grid, pN_grid, esp_revenue, nsp_revenue


def _nearest_grid_indices(
    pE: float,
    pN: float,
    *,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
) -> tuple[int, int]:
    i = int(np.argmin(np.abs(pE_grid - float(pE))))
    j = int(np.argmin(np.abs(pN_grid - float(pN))))
    return i, j


def _nearest_grid_ne_gap(
    pE: float,
    pN: float,
    *,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    grid_ne_gap: np.ndarray,
) -> float:
    i = int(np.argmin(np.abs(pE_grid - float(pE))))
    j = int(np.argmin(np.abs(pN_grid - float(pN))))
    return float(grid_ne_gap[j, i])


def _grid_true_gains_from_surface(
    pE: float,
    pN: float,
    *,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    esp_revenue: np.ndarray,
    nsp_revenue: np.ndarray,
) -> tuple[float, float]:
    i, j = _nearest_grid_indices(pE, pN, pE_grid=pE_grid, pN_grid=pN_grid)
    current_esp = float(esp_revenue[j, i])
    current_nsp = float(nsp_revenue[j, i])
    best_esp = float(np.nanmax(esp_revenue[j, :]))
    best_nsp = float(np.nanmax(nsp_revenue[:, i]))
    return max(0.0, best_esp - current_esp), max(0.0, best_nsp - current_nsp)


def _c1_trial_rows_from_result(
    *,
    trial: int,
    users,
    cfg: ExperimentConfig,
    result,
    restricted_override: list[float] | None = None,
) -> tuple[list[dict[str, object]], list[float], list[float]]:
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}
    rows: list[dict[str, object]] = []
    restricted_seq: list[float] = []
    grid_seq: list[float] = []

    if result.trajectory:
        step_payloads = [
            (
                float(step.pE),
                float(step.pN),
                float(step.restricted_gap if np.isfinite(step.restricted_gap) else step.epsilon),
            )
            for step in result.trajectory
        ]
    else:
        fallback_gap = float(result.restricted_gap if np.isfinite(result.restricted_gap) else result.epsilon)
        step_payloads = [(float(result.price[0]), float(result.price[1]), fallback_gap)]

    if restricted_override is not None and len(restricted_override) != len(step_payloads):
        raise ValueError(
            f"Trial {trial}: restricted-gap length mismatch between saved CSV ({len(restricted_override)}) "
            f"and replayed trajectory ({len(step_payloads)})."
        )

    for step_idx, (pE, pN, restricted_gap) in enumerate(step_payloads, start=1):
        restricted_value = float(restricted_override[step_idx - 1]) if restricted_override is not None else float(restricted_gap)
        grid_gap = _c1_compute_grid_ne_gap(users, cfg, pE, pN, stage2_cache)
        rows.append(
            {
                "trial": int(trial),
                "iteration": int(step_idx),
                "pE": float(pE),
                "pN": float(pN),
                "restricted_gap": float(restricted_value),
                "grid_ne_gap": float(grid_gap),
            }
        )
        restricted_seq.append(float(restricted_value))
        grid_seq.append(float(grid_gap))
    return rows, restricted_seq, grid_seq


def _plot_c1_dual_gap_trajectories(
    restricted_sequences: list[list[float]],
    grid_sequences: list[list[float]],
    *,
    stopping_tol: float,
    out_path: Path,
) -> None:
    restricted_median, restricted_q25, restricted_q75 = _aligned_quantile_band(restricted_sequences)
    grid_median, grid_q25, grid_q75 = _aligned_quantile_band(grid_sequences)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=150)
    x = np.arange(1, restricted_median.size + 1)
    ax.plot(x, restricted_median, marker="o", linewidth=1.9, color="tab:blue", label="Restricted gap (median)")
    ax.fill_between(x, restricted_q25, restricted_q75, alpha=0.18, color="tab:blue", label="Restricted gap (25%-75%)")
    ax.plot(x, grid_median, marker="s", linewidth=1.8, linestyle="--", color="tab:orange", label="Grid NE gap (median)")
    ax.fill_between(x, grid_q25, grid_q75, alpha=0.14, color="tab:orange", label="Grid NE gap (25%-75%)")
    ax.axhline(stopping_tol, color="black", linestyle="--", linewidth=1.2, label="Stopping tolerance")
    ax.set_xlabel("Stage I iteration")
    ax.set_ylabel("Gap value")
    ax.set_title("Stage-I restricted-gap and grid-NE-gap trajectories")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_c1_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    restricted_sequences: list[list[float]],
    grid_sequences: list[list[float]],
    config_path: str,
    seed: int,
    n_users: int,
    trials: int,
    cfg: ExperimentConfig,
    replay_verified: bool | None = None,
    grid_ne_gap_source: str = "audit_grid",
    grid_ne_gap_source_path: str | None = None,
) -> None:
    stopping_tol = float(cfg.stackelberg.paper_restricted_gap_tol)
    write_csv_rows(
        out_dir / "C1_restricted_gap_trajectory.csv",
        ["trial", "iteration", "pE", "pN", "restricted_gap", "grid_ne_gap"],
        rows,
    )
    _plot_c1_dual_gap_trajectories(
        restricted_sequences,
        grid_sequences,
        stopping_tol=stopping_tol,
        out_path=out_dir / "C1_restricted_gap_trajectory.png",
    )
    summary_lines = [
        f"config = {config_path}",
        f"seed = {seed}",
        f"n_users = {n_users}",
        f"trials = {trials}",
        "center_statistic = median",
        "band_statistic = q25_q75",
        f"stopping_tolerance = {stopping_tol}",
        f"grid_ne_gap_source = {grid_ne_gap_source}",
    ]
    if grid_ne_gap_source == "audit_grid":
        summary_lines.extend(
            [
                f"grid_ne_gap_stage2_method = {cfg.baselines.stage2_solver_for_pricing}",
                f"grid_ne_gap_audit_points = {int(cfg.baselines.gso_grid_points)}",
                "grid_ne_gap_definition = max unilateral revenue gain on the audit price grid with the other provider price fixed",
            ]
        )
    elif grid_ne_gap_source == "heatmap_csv_nearest":
        if grid_ne_gap_source_path is not None:
            summary_lines.append(f"grid_ne_gap_heatmap_csv = {grid_ne_gap_source_path}")
        summary_lines.append(
            "grid_ne_gap_definition = precomputed grid_ne_gap surface value at the nearest heatmap grid point to each trajectory price"
        )
    else:
        summary_lines.append("grid_ne_gap_definition = unspecified")
    if replay_verified is not None:
        summary_lines.append(f"trajectory_replay_verified = {str(bool(replay_verified)).lower()}")
    _write_summary(out_dir / "C1_restricted_gap_trajectory_summary.txt", summary_lines)


def _c2_trial_rows_from_result(
    *,
    trial: int,
    result,
) -> tuple[list[dict[str, object]], list[float], list[float], list[tuple[float, float]], list[float]]:
    rows: list[dict[str, object]] = []
    if result.trajectory:
        step_payloads = [
            (
                float(step.pE),
                float(step.pN),
                float(step.esp_gain if np.isfinite(step.esp_gain) else 0.0),
                float(step.nsp_gain if np.isfinite(step.nsp_gain) else 0.0),
                float(step.restricted_gap if np.isfinite(step.restricted_gap) else step.epsilon),
            )
            for step in result.trajectory
        ]
    else:
        step_payloads = [
            (
                float(result.price[0]),
                float(result.price[1]),
                float(result.gain_E.gain),
                float(result.gain_N.gain),
                float(result.restricted_gap if np.isfinite(result.restricted_gap) else result.epsilon),
            )
        ]
    seq_E: list[float] = []
    seq_N: list[float] = []
    prices: list[tuple[float, float]] = []
    restricted_seq: list[float] = []
    for step_idx, (pE, pN, gE, gN, restricted_gap) in enumerate(step_payloads, start=1):
        rows.append(
            {
                "trial": int(trial),
                "iteration": int(step_idx),
                "pE": float(pE),
                "pN": float(pN),
                "esp_gain": float(gE),
                "nsp_gain": float(gN),
                "restricted_gap": float(restricted_gap),
            }
        )
        seq_E.append(float(gE))
        seq_N.append(float(gN))
        prices.append((float(pE), float(pN)))
        restricted_seq.append(float(restricted_gap))
    return rows, seq_E, seq_N, prices, restricted_seq


def _c2_attach_grid_true_gains(
    rows: list[dict[str, object]],
    *,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    esp_revenue: np.ndarray,
    nsp_revenue: np.ndarray,
) -> tuple[list[dict[str, object]], list[float], list[float]]:
    updated_rows: list[dict[str, object]] = []
    seq_E: list[float] = []
    seq_N: list[float] = []
    for row in rows:
        grid_true_esp_gain, grid_true_nsp_gain = _grid_true_gains_from_surface(
            float(row["pE"]),
            float(row["pN"]),
            pE_grid=pE_grid,
            pN_grid=pN_grid,
            esp_revenue=esp_revenue,
            nsp_revenue=nsp_revenue,
        )
        updated = dict(row)
        updated["grid_true_esp_gain"] = float(grid_true_esp_gain)
        updated["grid_true_nsp_gain"] = float(grid_true_nsp_gain)
        updated["grid_true_gap"] = float(max(grid_true_esp_gain, grid_true_nsp_gain))
        updated_rows.append(updated)
        seq_E.append(float(grid_true_esp_gain))
        seq_N.append(float(grid_true_nsp_gain))
    return updated_rows, seq_E, seq_N


def _plot_c2_gain_panel(
    ax,
    seq_E: list[list[float]],
    seq_N: list[list[float]],
    *,
    title: str,
    label_prefix: str = "",
    linestyle: str = "-",
    band_alpha: float = 0.18,
) -> None:
    mean_E, std_E = _aligned_series(seq_E)
    mean_N, std_N = _aligned_series(seq_N)
    x = np.arange(1, mean_E.size + 1)
    ax.plot(
        x,
        mean_E,
        marker="o",
        linewidth=1.8,
        linestyle=linestyle,
        color="tab:blue",
        label=f"{label_prefix}ESP gain",
    )
    ax.fill_between(x, mean_E - std_E, mean_E + std_E, alpha=band_alpha, color="tab:blue")
    ax.plot(
        x,
        mean_N,
        marker="s",
        linewidth=1.8,
        linestyle=linestyle,
        color="tab:orange",
        label=f"{label_prefix}NSP gain",
    )
    ax.fill_between(x, mean_N - std_N, mean_N + std_N, alpha=band_alpha, color="tab:orange")
    ax.set_xlabel("Stage I iteration")
    ax.set_ylabel("Best-response gain")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _write_c2_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    seq_E: list[list[float]],
    seq_N: list[list[float]],
    config_path: str,
    seed: int,
    n_users: int,
    trials: int,
    grid_seq_E: list[list[float]] | None = None,
    grid_seq_N: list[list[float]] | None = None,
    grid_gain_source_path: str | None = None,
    source_c1_run: str | None = None,
    replay_verified: bool | None = None,
) -> None:
    fieldnames = ["trial", "iteration", "pE", "pN", "esp_gain", "nsp_gain", "restricted_gap"]
    have_grid_true = grid_seq_E is not None and grid_seq_N is not None
    if have_grid_true:
        fieldnames.extend(["grid_true_esp_gain", "grid_true_nsp_gain", "grid_true_gap"])
    write_csv_rows(
        out_dir / "C2_best_response_gain_trajectory.csv",
        fieldnames,
        rows,
    )
    if have_grid_true:
        fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=150)
        _plot_c2_gain_panel(
            ax,
            seq_E,
            seq_N,
            title="Best-response gain trajectories",
            label_prefix="Restricted ",
            linestyle="-",
            band_alpha=0.16,
        )
        _plot_c2_gain_panel(
            ax,
            grid_seq_E,
            grid_seq_N,
            title="Best-response gain trajectories",
            label_prefix="Grid true ",
            linestyle="--",
            band_alpha=0.08,
        )
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
        _plot_c2_gain_panel(ax, seq_E, seq_N, title="Best-response gain trajectories")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
    fig.savefig(out_dir / "C2_best_response_gain_trajectory.png")
    plt.close(fig)

    summary_lines = [
        f"config = {config_path}",
        f"seed = {seed}",
        f"n_users = {n_users}",
        f"trials = {trials}",
        "center_statistic = mean",
        "band_statistic = std",
    ]
    if have_grid_true:
        summary_lines.append("grid_true_gain_source = heatmap_csv_nearest")
        summary_lines.append("plot_layout = single_axes_overlay")
        if grid_gain_source_path is not None:
            summary_lines.append(f"grid_true_gain_heatmap_csv = {grid_gain_source_path}")
        summary_lines.append(
            "grid_true_gain_definition = at the nearest heatmap grid point to each trajectory price, "
            "ESP gain is the rowwise best-response revenue improvement on the heatmap and NSP gain is the columnwise "
            "best-response revenue improvement on the heatmap"
        )
    if source_c1_run is not None:
        summary_lines.append(f"source_c1_run = {source_c1_run}")
    if replay_verified is not None:
        summary_lines.append(f"trajectory_replay_verified = {str(bool(replay_verified)).lower()}")
    _write_summary(out_dir / "C2_best_response_gain_trajectory_summary.txt", summary_lines)


def _load_c1_rows_by_trial(run_dir: Path) -> dict[int, list[dict[str, str]]]:
    rows = load_csv_rows(run_dir / "C1_restricted_gap_trajectory.csv")
    if not rows:
        raise ValueError(f"C1 CSV is empty or missing: {run_dir / 'C1_restricted_gap_trajectory.csv'}")
    by_trial: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        trial = int(row["trial"])
        by_trial.setdefault(trial, []).append(row)
    for trial_rows in by_trial.values():
        trial_rows.sort(key=lambda row: int(row["iteration"]))
    return by_trial


def _load_c2_rows_by_trial(run_dir: Path) -> dict[int, list[dict[str, str]]]:
    rows = load_csv_rows(run_dir / "C2_best_response_gain_trajectory.csv")
    if not rows:
        raise ValueError(f"C2 CSV is empty or missing: {run_dir / 'C2_best_response_gain_trajectory.csv'}")
    by_trial: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        trial = int(row["trial"])
        by_trial.setdefault(trial, []).append(row)
    for trial_rows in by_trial.values():
        trial_rows.sort(key=lambda row: int(row["iteration"]))
    return by_trial


def _resolve_existing_path(path_text: str, *, base_dir: Path | None = None) -> Path:
    raw = Path(path_text)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([raw, ROOT / raw])
        if base_dir is not None:
            candidates.append(base_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _grid_gain_csv_from_c1_manifest(manifest: dict[str, object]) -> str | None:
    primary_metrics = manifest.get("primary_metrics")
    if not isinstance(primary_metrics, dict):
        return None
    value = str(primary_metrics.get("grid_ne_gap_heatmap_csv", "")).strip()
    return value or None


def _run_c2_replot_from_existing_run(c2_run_dir: Path) -> None:
    if not c2_run_dir.is_absolute():
        c2_run_dir = _resolve_existing_path(str(c2_run_dir))
    manifest_path = c2_run_dir / "figure_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing C2 manifest: {manifest_path}")
    manifest = load_figure_manifest(manifest_path)
    if str(manifest.get("figure_id", "")) != "C2":
        raise ValueError(f"Expected a C2 run directory, got figure_id={manifest.get('figure_id', '')!r}")

    config_path = str(manifest.get("config", "")).strip()
    seed = int(manifest.get("seed", "0"))
    n_users = int(manifest.get("n_users", "0"))
    trials = int(manifest.get("n_trials", "0"))
    primary_metrics = manifest.get("primary_metrics", {})
    if not isinstance(primary_metrics, dict):
        primary_metrics = {}

    rows_by_trial = _load_c2_rows_by_trial(c2_run_dir)
    rows: list[dict[str, object]] = []
    seq_E: list[list[float]] = []
    seq_N: list[list[float]] = []
    grid_seq_E: list[list[float]] = []
    grid_seq_N: list[list[float]] = []
    have_grid_true = False

    for trial in sorted(rows_by_trial):
        trial_rows = rows_by_trial[trial]
        rows.extend(
            {
                key: (
                    float(value)
                    if key
                    in {
                        "pE",
                        "pN",
                        "esp_gain",
                        "nsp_gain",
                        "restricted_gap",
                        "grid_true_esp_gain",
                        "grid_true_nsp_gain",
                        "grid_true_gap",
                    }
                    else int(value)
                    if key in {"trial", "iteration"}
                    else value
                )
                for key, value in row.items()
                if key not in {"figure_id", "block"}
            }
            for row in trial_rows
        )
        seq_E.append([float(row["esp_gain"]) for row in trial_rows])
        seq_N.append([float(row["nsp_gain"]) for row in trial_rows])
        if {"grid_true_esp_gain", "grid_true_nsp_gain"} <= set(trial_rows[0].keys()):
            have_grid_true = True
            grid_seq_E.append([float(row["grid_true_esp_gain"]) for row in trial_rows])
            grid_seq_N.append([float(row["grid_true_nsp_gain"]) for row in trial_rows])

    _write_c2_outputs(
        out_dir=c2_run_dir,
        rows=rows,
        seq_E=seq_E,
        seq_N=seq_N,
        config_path=config_path,
        seed=seed,
        n_users=n_users,
        trials=trials,
        grid_seq_E=(grid_seq_E if have_grid_true else None),
        grid_seq_N=(grid_seq_N if have_grid_true else None),
        grid_gain_source_path=(
            str(primary_metrics.get("grid_true_gain_heatmap_csv", "")).strip() or None
        ),
        source_c1_run=(str(primary_metrics.get("source_c1_run", "")).strip() or None),
        replay_verified=(
            str(primary_metrics.get("trajectory_replay_verified", "")).strip().lower() == "true"
            if "trajectory_replay_verified" in primary_metrics
            else None
        ),
    )


def _run_c2_from_c1_run(c1_run_dir: Path, out_dir: Path, *, grid_gain_csv: str | None = None) -> None:
    if not c1_run_dir.is_absolute():
        c1_run_dir = _resolve_existing_path(str(c1_run_dir))
    manifest_path = c1_run_dir / "figure_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing C1 manifest: {manifest_path}")
    manifest = load_figure_manifest(manifest_path)
    if str(manifest.get("figure_id", "")) != "C1":
        raise ValueError(f"Expected a C1 run directory, got figure_id={manifest.get('figure_id', '')!r}")

    config_path = str(manifest.get("config", "")).strip()
    if not config_path:
        raise ValueError(f"C1 manifest does not record a config path: {manifest_path}")
    seed = int(manifest.get("seed", "0"))
    n_users = int(manifest.get("n_users", "0"))
    trials = int(manifest.get("n_trials", "0"))
    if n_users <= 0 or trials <= 0:
        raise ValueError(f"Invalid n_users/trials in C1 manifest: {manifest_path}")

    cfg = _load_cfg(config_path, n_users=n_users)
    saved_by_trial = _load_c1_rows_by_trial(c1_run_dir)
    grid_gain_csv_text = grid_gain_csv or _grid_gain_csv_from_c1_manifest(manifest)
    if grid_gain_csv_text:
        grid_gain_path = _resolve_existing_path(grid_gain_csv_text, base_dir=c1_run_dir)
        heatmap_pE_grid, heatmap_pN_grid, heatmap_esp_rev, heatmap_nsp_rev = _load_revenue_surfaces(grid_gain_path)
    else:
        grid_gain_path = None
        heatmap_pE_grid = np.asarray([], dtype=float)
        heatmap_pN_grid = np.asarray([], dtype=float)
        heatmap_esp_rev = np.asarray([[]], dtype=float)
        heatmap_nsp_rev = np.asarray([[]], dtype=float)

    rows: list[dict[str, object]] = []
    seq_E: list[list[float]] = []
    seq_N: list[list[float]] = []
    grid_seq_E: list[list[float]] = []
    grid_seq_N: list[list[float]] = []
    tol = 1e-9

    for trial in range(1, trials + 1):
        if trial not in saved_by_trial:
            raise ValueError(f"Missing saved C1 rows for trial {trial}.")
        saved_rows = saved_by_trial[trial]
        users = _sample_users(cfg, n_users, seed, trial)
        result = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        trial_rows, sE, sN, _, _ = _c2_trial_rows_from_result(trial=trial, result=result)
        if len(trial_rows) != len(saved_rows):
            raise ValueError(
                f"Trial {trial}: saved C1 trajectory length {len(saved_rows)} "
                f"does not match replayed trajectory length {len(trial_rows)}."
            )
        for step_idx, (saved, replay) in enumerate(zip(saved_rows, trial_rows), start=1):
            for key in ("pE", "pN", "restricted_gap"):
                saved_val = float(saved[key])
                replay_val = float(replay[key])
                if abs(saved_val - replay_val) > tol:
                    raise ValueError(
                        f"Trial {trial}, iteration {step_idx}: saved C1 {key}={saved_val:.12g} "
                        f"does not match replayed value {replay_val:.12g}."
                    )
        if grid_gain_path is not None:
            trial_rows, grid_sE, grid_sN = _c2_attach_grid_true_gains(
                trial_rows,
                pE_grid=heatmap_pE_grid,
                pN_grid=heatmap_pN_grid,
                esp_revenue=heatmap_esp_rev,
                nsp_revenue=heatmap_nsp_rev,
            )
            grid_seq_E.append(grid_sE)
            grid_seq_N.append(grid_sN)
        rows.extend(trial_rows)
        seq_E.append(sE)
        seq_N.append(sN)

    _write_c2_outputs(
        out_dir=out_dir,
        rows=rows,
        seq_E=seq_E,
        seq_N=seq_N,
        config_path=config_path,
        seed=seed,
        n_users=n_users,
        trials=trials,
        grid_seq_E=(grid_seq_E if grid_gain_path is not None else None),
        grid_seq_N=(grid_seq_N if grid_gain_path is not None else None),
        grid_gain_source_path=(str(grid_gain_path) if grid_gain_path is not None else None),
        source_c1_run=str(c1_run_dir),
        replay_verified=True,
    )


def _compute_restricted_gap(users, system, stack_cfg, price: tuple[float, float], offloading_set: tuple[int, ...]) -> float:
    gain_E = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "E", system, estimator_variant=stack_cfg.gain_estimator_variant)
    gain_N = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "N", system, estimator_variant=stack_cfg.gain_estimator_variant)
    return float(max(gain_E.gain, gain_N.gain))


def _run_stage1_method(users, system, stack_cfg, base_cfg, method: str) -> tuple[tuple[float, float], tuple[int, ...], float, float, float, dict[str, object]]:
    method_key = method.lower()
    t0 = time.perf_counter()
    if method_key == "proposed":
        res = solve_stage1_pricing(users, system, stack_cfg)
        runtime = time.perf_counter() - t0
        return (
            (float(res.price[0]), float(res.price[1])),
            tuple(int(x) for x in res.offloading_set),
            float(res.restricted_gap),
            float(res.esp_revenue),
            float(res.nsp_revenue),
            {
                "runtime_sec": float(runtime),
                "stage2_calls": int(res.stage2_oracle_calls),
                "social_cost": float(res.social_cost),
                "offloading_size": int(len(res.offloading_set)),
            },
        )
    if method_key == "gso":
        out = baseline_stage1_grid_search_oracle(users, system, stack_cfg, base_cfg)
    elif method_key == "ga":
        out = baseline_stage1_ga(users, system, stack_cfg, base_cfg)
    elif method_key == "bo":
        out = baseline_stage1_bo(users, system, stack_cfg, base_cfg)
    elif method_key in {"bo-online", "bo_online"}:
        out = baseline_stage1_bo_online_grid_ne_gap(users, system, stack_cfg, base_cfg)
    elif method_key == "marl":
        out = baseline_stage1_marl(users, system, stack_cfg, base_cfg)
    elif method_key == "me":
        out = baseline_market_equilibrium(users, system, stack_cfg, base_cfg)
    elif method_key == "singlesp":
        out = baseline_single_sp(users, system, stack_cfg, base_cfg)
    elif method_key == "rand":
        out = baseline_random_offloading(users, system, stack_cfg, base_cfg)
    elif method_key == "coop":
        out = baseline_coop(users, system, stack_cfg, base_cfg)
    else:
        raise ValueError(f"Unsupported method={method}")
    runtime = time.perf_counter() - t0
    gap = _compute_restricted_gap(users, system, stack_cfg, out.price, out.offloading_set)
    return (
        (float(out.price[0]), float(out.price[1])),
        tuple(int(x) for x in out.offloading_set),
        float(gap),
        float(out.esp_revenue),
        float(out.nsp_revenue),
        {
            "runtime_sec": float(runtime),
            "stage2_calls": int(out.meta.get("stage2_unique_prices", 0)),
            "social_cost": float(out.social_cost),
            "offloading_size": int(len(out.offloading_set)),
        },
    )


def _normalize_stage1_runtime_method(method: str) -> str:
    method_key = method.strip().lower()
    if method_key in {"proposed", "vbbr"}:
        return "Proposed"
    if method_key == "marl":
        return "MARL"
    return method


def _apply_d2_baseline_caps(
    base_cfg,
    *,
    bo_iters: int | None,
    ga_generations: int | None,
    marl_episodes: int | None,
    marl_steps_per_episode: int | None,
):
    updates: dict[str, int] = {}
    if bo_iters is not None:
        updates["bo_iters"] = max(0, int(bo_iters))
    if ga_generations is not None:
        updates["ga_generations"] = max(0, int(ga_generations))
    if marl_episodes is not None:
        updates["marl_episodes"] = max(1, int(marl_episodes))
    if marl_steps_per_episode is not None:
        updates["marl_steps_per_episode"] = max(1, int(marl_steps_per_episode))
    return replace(base_cfg, **updates) if updates else base_cfg


def _run_d2_point(
    config_path: str,
    seed: int,
    n_users: int,
    trial: int,
    method: str,
    bo_iters: int | None = None,
    ga_generations: int | None = None,
    marl_episodes: int | None = None,
    marl_steps_per_episode: int | None = None,
) -> dict[str, object]:
    cfg = _load_cfg(config_path)
    row_method = str(method)
    users = _sample_users(cfg, int(n_users), int(seed), int(trial))
    normalized_method = _normalize_stage1_runtime_method(row_method)
    base_cfg = cfg.baselines
    if normalized_method != "Proposed":
        base_cfg = _apply_d2_baseline_caps(
            base_cfg,
            bo_iters=bo_iters,
            ga_generations=ga_generations,
            marl_episodes=marl_episodes,
            marl_steps_per_episode=marl_steps_per_episode,
        )
    price, offloading_set, gap, _, _, meta = _run_stage1_method(
        users,
        cfg.system,
        cfg.stackelberg,
        base_cfg,
        normalized_method,
    )
    return {
        "method": row_method,
        "n_users": int(n_users),
        "trial": int(trial),
        "runtime_sec": float(meta["runtime_sec"]),
        "restricted_gap": float(gap),
        "offloading_size": int(len(offloading_set)),
        "final_pE": float(price[0]),
        "final_pN": float(price[1]),
    }


def _budgeted_cfg(stack_cfg, base_cfg, method: str, budget: int):
    method_key = method.lower()
    budget = max(int(budget), 2)
    if method_key == "proposed":
        return replace(stack_cfg, search_max_iters=budget), base_cfg
    if method_key == "bo":
        init_points = min(4, budget)
        return stack_cfg, replace(base_cfg, bo_init_points=init_points, bo_iters=max(0, budget - init_points))
    if method_key == "ga":
        pop = min(8, budget)
        pop = max(pop, 2)
        gens = max(0, budget // pop - 1)
        return stack_cfg, replace(base_cfg, ga_population_size=pop, ga_generations=gens)
    if method_key == "marl":
        steps = min(20, max(5, budget // 4))
        episodes = max(1, budget // steps)
        return stack_cfg, replace(base_cfg, marl_episodes=episodes, marl_steps_per_episode=steps)
    if method_key == "gso":
        grid_points = max(2, int(round(math.sqrt(budget))))
        return stack_cfg, replace(base_cfg, gso_grid_points=grid_points)
    return stack_cfg, base_cfg


def _plot_simple_heatmap(matrix: np.ndarray, x_vals: list[float], y_vals: list[float], *, title: str, xlabel: str, ylabel: str, cbar_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=150)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels([f"{x:.2g}" for x in x_vals], rotation=45)
    ax.set_yticks(np.arange(len(y_vals)))
    ax.set_yticklabels([f"{y:.2g}" for y in y_vals])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main_A4() -> None:
    parser = argparse.ArgumentParser(description="Figure A4: Stage II runtime versus number of users.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16,20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--centralized-max-n", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir(Path(__file__).stem.replace("_figure_missing_impl", "run_figure_A4_stage2_runtime_vs_users"), args.out_dir)
    cfg = _load_cfg(args.config)
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for n in n_list:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            t0 = time.perf_counter()
            out = solve_stage2_scm(users, args.pE, args.pN, cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
            runtime = time.perf_counter() - t0
            rows.append({"method": "Proposed", "n_users": n, "runtime_sec": float(runtime)})
            if n <= int(args.centralized_max_n):
                t0 = time.perf_counter()
                cs_out, success = s2_ratio._run_centralized(users, args.pE, args.pN, cfg.system, cfg, "enum")
                c_runtime = time.perf_counter() - t0
                if success:
                    rows.append({"method": "Centralized-exact", "n_users": n, "runtime_sec": float(c_runtime)})
    write_csv_rows(out_dir / "A4_stage2_runtime_vs_users.csv", ["method", "n_users", "runtime_sec"], rows)
    _plot_method_errorbars(
        rows,
        x_key="n_users",
        y_key="runtime_sec",
        xlabel="Number of users",
        ylabel="Runtime (sec)",
        title="Stage II runtime versus number of users",
        out_path=out_dir / "A4_stage2_runtime_vs_users.png",
        method_order=["Proposed", "Centralized-exact"],
    )
    _write_summary(out_dir / "A4_stage2_runtime_vs_users_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_A5() -> None:
    parser = argparse.ArgumentParser(description="Figure A5: Stage II rollback diagnostics.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_A5_stage2_rollback_diagnostics", args.out_dir)
    cfg = _load_cfg(args.config)
    rows: list[dict[str, object]] = []
    for n in [int(x) for x in args.n_users_list.split(",") if x.strip()]:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            out = solve_stage2_scm(users, args.pE, args.pN, cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
            rows.append(
                {
                    "method": "Proposed",
                    "n_users": n,
                    "rollback_count": int(out.rollback_count),
                    "accepted_admissions": int(out.accepted_admissions),
                    "offloading_size": int(len(out.offloading_set)),
                }
            )
    write_csv_rows(out_dir / "A5_stage2_rollback_diagnostics.csv", ["method", "n_users", "rollback_count", "accepted_admissions", "offloading_size"], rows)
    _plot_three_panel(
        rows,
        x_key="n_users",
        panels=[
            ("rollback_count", "Rollback count"),
            ("accepted_admissions", "Accepted admissions"),
            ("offloading_size", "Final offloading size"),
        ],
        xlabel="Number of users",
        title="Stage II rollback diagnostics",
        out_path=out_dir / "A5_stage2_rollback_diagnostics.png",
        method_order=["Proposed"],
    )
    _write_summary(out_dir / "A5_stage2_rollback_diagnostics_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_B6() -> None:
    parser = argparse.ArgumentParser(description="Figure B6: candidate-family diagnostics.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B6_candidate_family_diagnostics", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    users = _sample_users(cfg, args.n_users, args.seed, 0)
    current_stage2 = solve_stage2_scm(users, args.pE, args.pN, cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
    current_set = tuple(int(x) for x in current_stage2.offloading_set)
    data = _build_data(users)
    family = _candidate_family(data, current_set, args.pE, args.pN, cfg.system)
    current_joint = float(
        _provider_revenue_from_stage2_result(current_stage2, args.pE, args.pN, "E", cfg.system)
        + _provider_revenue_from_stage2_result(current_stage2, args.pE, args.pN, "N", cfg.system)
    )
    rows: list[dict[str, object]] = []
    type_counts = {"drop": 0, "add": 0, "swap": 0, "mixed": 0, "same": 0}
    for candidate_set in family:
        diff_drop = set(current_set) - set(candidate_set)
        diff_add = set(candidate_set) - set(current_set)
        if diff_drop and diff_add:
            op_type = "swap" if len(diff_drop) == len(diff_add) == 1 else "mixed"
        elif diff_drop:
            op_type = "drop"
        elif diff_add:
            op_type = "add"
        else:
            op_type = "same"
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
        best_joint = current_joint
        for provider in ("E", "N"):
            fixed_price = args.pN if provider == "E" else args.pE
            boundary_price = _boundary_price_for_provider(data, candidate_set, fixed_price, provider, cfg.system)
            if boundary_price is None:
                continue
            pE_eval, pN_eval = (float(boundary_price), float(args.pN)) if provider == "E" else (float(args.pE), float(boundary_price))
            s2 = solve_stage2_scm(users, pE_eval, pN_eval, cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
            joint = float(
                _provider_revenue_from_stage2_result(s2, pE_eval, pN_eval, "E", cfg.system)
                + _provider_revenue_from_stage2_result(s2, pE_eval, pN_eval, "N", cfg.system)
            )
            best_joint = max(best_joint, joint)
        rows.append({"candidate_set": ";".join(str(x) for x in candidate_set), "operation_type": op_type, "joint_revenue_score": best_joint})
    rows_sorted = sorted(rows, key=lambda row: float(row["joint_revenue_score"]), reverse=True)
    write_csv_rows(out_dir / "B6_candidate_family_diagnostics.csv", ["candidate_set", "operation_type", "joint_revenue_score"], rows_sorted)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=150)
    top_rows = rows_sorted[: min(20, len(rows_sorted))]
    axes[0].plot(np.arange(1, len(top_rows) + 1), [float(row["joint_revenue_score"]) for row in top_rows], marker="o", linewidth=1.7)
    axes[0].set_xlabel("Candidate rank")
    axes[0].set_ylabel("Best joint revenue on candidate boundary")
    axes[0].set_title("Top candidate-family scores")
    axes[0].grid(alpha=0.25)
    op_names = ["drop", "add", "swap", "mixed", "same"]
    axes[1].bar(op_names, [type_counts.get(name, 0) for name in op_names], color="tab:blue")
    axes[1].set_xlabel("Operation type")
    axes[1].set_ylabel("Candidate count")
    axes[1].set_title("Candidate-family composition")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "B6_candidate_family_diagnostics.png")
    plt.close(fig)
    _write_summary(out_dir / "B6_candidate_family_diagnostics_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"candidate_family_size = {len(rows_sorted)}"])


def _legacy_main_C1() -> None:
    parser = argparse.ArgumentParser(description="Figure C1: restricted-gap trajectory.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C1_restricted_gap_trajectory", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    rows: list[dict[str, object]] = []
    sequences: list[list[float]] = []
    for trial in range(1, args.trials + 1):
        users = _sample_users(cfg, args.n_users, args.seed, trial)
        res = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        seq = [float(step.restricted_gap if np.isfinite(step.restricted_gap) else step.epsilon) for step in res.trajectory]
        if not seq:
            seq = [float(res.restricted_gap)]
        sequences.append(seq)
        for step_idx, value in enumerate(seq, start=1):
            rows.append({"trial": trial, "iteration": step_idx, "restricted_gap": float(value)})
    median, q25, q75 = _aligned_quantile_band(sequences)
    stopping_tol = float(cfg.stackelberg.paper_restricted_gap_tol)
    write_csv_rows(out_dir / "C1_restricted_gap_trajectory.csv", ["trial", "iteration", "restricted_gap"], rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    x = np.arange(1, median.size + 1)
    ax.plot(x, median, marker="o", linewidth=1.8, label="Median")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="Mean ± std")
    ax.fill_between(x, q25, q75, alpha=0.2, label="25%-75% quantile")
    ax.axhline(stopping_tol, color="black", linestyle="--", linewidth=1.2, label="Stopping tolerance")
    ax.set_xlabel("Stage I iteration")
    ax.set_ylabel("Restricted gap")
    ax.set_title("Restricted-gap trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C1_restricted_gap_trajectory.png")
    plt.close(fig)
    _write_summary(out_dir / "C1_restricted_gap_trajectory_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"trials = {args.trials}"])


def main_C1() -> None:
    parser = argparse.ArgumentParser(description="Figure C1: restricted-gap trajectory.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--grid-ne-gap-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C1_restricted_gap_trajectory", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    if args.grid_ne_gap_csv is not None:
        csv_path = _resolve_existing_path(str(args.grid_ne_gap_csv))
        heatmap_pE_grid, heatmap_pN_grid, heatmap_grid_ne_gap = _load_grid_ne_gap_surface(csv_path)
        grid_source = "heatmap_csv_nearest"
        grid_source_path = str(csv_path)
    else:
        heatmap_pE_grid = np.asarray([], dtype=float)
        heatmap_pN_grid = np.asarray([], dtype=float)
        heatmap_grid_ne_gap = np.asarray([[]], dtype=float)
        grid_source = "audit_grid"
        grid_source_path = None
    rows: list[dict[str, object]] = []
    restricted_sequences: list[list[float]] = []
    grid_sequences: list[list[float]] = []
    for trial in range(1, args.trials + 1):
        users = _sample_users(cfg, args.n_users, args.seed, trial)
        res = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        if grid_source == "heatmap_csv_nearest":
            if res.trajectory:
                step_payloads = [
                    (
                        float(step.pE),
                        float(step.pN),
                        float(step.restricted_gap if np.isfinite(step.restricted_gap) else step.epsilon),
                    )
                    for step in res.trajectory
                ]
            else:
                step_payloads = [
                    (
                        float(res.price[0]),
                        float(res.price[1]),
                        float(res.restricted_gap if np.isfinite(res.restricted_gap) else res.epsilon),
                    )
                ]
            trial_rows = []
            restricted_seq = []
            grid_seq = []
            for step_idx, (pE, pN, restricted_gap) in enumerate(step_payloads, start=1):
                grid_gap = _nearest_grid_ne_gap(
                    pE,
                    pN,
                    pE_grid=heatmap_pE_grid,
                    pN_grid=heatmap_pN_grid,
                    grid_ne_gap=heatmap_grid_ne_gap,
                )
                trial_rows.append(
                    {
                        "trial": int(trial),
                        "iteration": int(step_idx),
                        "pE": float(pE),
                        "pN": float(pN),
                        "restricted_gap": float(restricted_gap),
                        "grid_ne_gap": float(grid_gap),
                    }
                )
                restricted_seq.append(float(restricted_gap))
                grid_seq.append(float(grid_gap))
        else:
            trial_rows, restricted_seq, grid_seq = _c1_trial_rows_from_result(
                trial=trial,
                users=users,
                cfg=cfg,
                result=res,
            )
        rows.extend(trial_rows)
        restricted_sequences.append(restricted_seq)
        grid_sequences.append(grid_seq)

    _write_c1_outputs(
        out_dir=out_dir,
        rows=rows,
        restricted_sequences=restricted_sequences,
        grid_sequences=grid_sequences,
        config_path=args.config,
        seed=int(args.seed),
        n_users=int(args.n_users),
        trials=int(args.trials),
        cfg=cfg,
        grid_ne_gap_source=grid_source,
        grid_ne_gap_source_path=grid_source_path,
    )


def main_C1_backfill_grid_ne_gap() -> None:
    parser = argparse.ArgumentParser(description="Backfill grid-NE-gap trajectory into an existing C1 output directory.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--grid-ne-gap-csv", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = load_figure_manifest(run_dir / "figure_manifest.json")
    config_raw = str(manifest.get("config", "")).strip()
    if not config_raw:
        raise ValueError("figure_manifest.json does not contain a config path.")
    config_path = Path(config_raw)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _load_cfg(str(config_path), n_users=int(manifest["n_users"]))
    seed = int(manifest["seed"])
    n_users = int(manifest["n_users"])
    trials = int(manifest["n_trials"])
    if args.grid_ne_gap_csv is not None:
        csv_path = Path(args.grid_ne_gap_csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        heatmap_pE_grid, heatmap_pN_grid, heatmap_grid_ne_gap = _load_grid_ne_gap_surface(csv_path)
        grid_source = "heatmap_csv_nearest"
        grid_source_path = str(args.grid_ne_gap_csv)
    else:
        csv_path = None
        heatmap_pE_grid = np.asarray([], dtype=float)
        heatmap_pN_grid = np.asarray([], dtype=float)
        heatmap_grid_ne_gap = np.asarray([[]], dtype=float)
        grid_source = "audit_grid"
        grid_source_path = None

    existing_rows = load_csv_rows(run_dir / "C1_restricted_gap_trajectory.csv")
    existing_by_trial: dict[int, list[dict[str, str]]] = {}
    for row in existing_rows:
        trial = int(row["trial"])
        existing_by_trial.setdefault(trial, []).append(row)
    for trial_rows in existing_by_trial.values():
        trial_rows.sort(key=lambda row: int(row["iteration"]))

    updated_rows: list[dict[str, object]] = []
    restricted_sequences: list[list[float]] = []
    grid_sequences: list[list[float]] = []

    for trial in range(1, trials + 1):
        if trial not in existing_by_trial:
            raise ValueError(f"Missing saved restricted-gap rows for trial {trial}.")
        restricted_saved = [float(row["restricted_gap"]) for row in existing_by_trial[trial]]
        users = _sample_users(cfg, n_users, seed, trial)
        result = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        if result.trajectory:
            replay_restricted = [
                float(step.restricted_gap if np.isfinite(step.restricted_gap) else step.epsilon)
                for step in result.trajectory
            ]
            replay_prices = [(float(step.pE), float(step.pN)) for step in result.trajectory]
        else:
            replay_restricted = [float(result.restricted_gap if np.isfinite(result.restricted_gap) else result.epsilon)]
            replay_prices = [(float(result.price[0]), float(result.price[1]))]
        if len(replay_restricted) != len(restricted_saved):
            raise ValueError(
                f"Trial {trial}: saved restricted-gap length {len(restricted_saved)} "
                f"does not match replayed trajectory length {len(replay_restricted)}."
            )
        for step_idx, (saved_val, replay_val) in enumerate(zip(restricted_saved, replay_restricted), start=1):
            if abs(float(saved_val) - float(replay_val)) > 1e-9:
                raise ValueError(
                    f"Trial {trial}, iteration {step_idx}: saved restricted_gap={saved_val:.12g} "
                    f"does not match replayed value {replay_val:.12g}."
                )
        if grid_source == "heatmap_csv_nearest":
            trial_rows = []
            restricted_seq = [float(value) for value in restricted_saved]
            grid_seq = []
            for step_idx, ((pE, pN), restricted_value) in enumerate(zip(replay_prices, restricted_saved), start=1):
                grid_gap = _nearest_grid_ne_gap(
                    pE,
                    pN,
                    pE_grid=heatmap_pE_grid,
                    pN_grid=heatmap_pN_grid,
                    grid_ne_gap=heatmap_grid_ne_gap,
                )
                trial_rows.append(
                    {
                        "trial": int(trial),
                        "iteration": int(step_idx),
                        "pE": float(pE),
                        "pN": float(pN),
                        "restricted_gap": float(restricted_value),
                        "grid_ne_gap": float(grid_gap),
                    }
                )
                grid_seq.append(float(grid_gap))
        else:
            trial_rows, restricted_seq, grid_seq = _c1_trial_rows_from_result(
                trial=trial,
                users=users,
                cfg=cfg,
                result=result,
                restricted_override=restricted_saved,
            )
        updated_rows.extend(trial_rows)
        restricted_sequences.append(restricted_seq)
        grid_sequences.append(grid_seq)

    _write_c1_outputs(
        out_dir=run_dir,
        rows=updated_rows,
        restricted_sequences=restricted_sequences,
        grid_sequences=grid_sequences,
        config_path=config_raw,
        seed=seed,
        n_users=n_users,
        trials=trials,
        cfg=cfg,
        replay_verified=True,
        grid_ne_gap_source=grid_source,
        grid_ne_gap_source_path=grid_source_path,
    )


def main_C2() -> None:
    parser = argparse.ArgumentParser(description="Figure C2: best-response gain trajectories.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--c1-run-dir", type=str, default=None)
    parser.add_argument("--c2-run-dir", type=str, default=None)
    parser.add_argument("--grid-gain-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.c2_run_dir:
        _run_c2_replot_from_existing_run(Path(args.c2_run_dir))
        return

    out_dir = resolve_out_dir("run_figure_C2_best_response_gain_trajectory", args.out_dir)
    if args.c1_run_dir:
        _run_c2_from_c1_run(Path(args.c1_run_dir), out_dir, grid_gain_csv=args.grid_gain_csv)
        return

    cfg = _load_cfg(args.config, n_users=args.n_users)
    if args.grid_gain_csv:
        grid_gain_path = _resolve_existing_path(args.grid_gain_csv)
        heatmap_pE_grid, heatmap_pN_grid, heatmap_esp_rev, heatmap_nsp_rev = _load_revenue_surfaces(grid_gain_path)
    else:
        grid_gain_path = None
        heatmap_pE_grid = np.asarray([], dtype=float)
        heatmap_pN_grid = np.asarray([], dtype=float)
        heatmap_esp_rev = np.asarray([[]], dtype=float)
        heatmap_nsp_rev = np.asarray([[]], dtype=float)
    seq_E: list[list[float]] = []
    seq_N: list[list[float]] = []
    grid_seq_E: list[list[float]] = []
    grid_seq_N: list[list[float]] = []
    rows: list[dict[str, object]] = []
    for trial in range(1, args.trials + 1):
        users = _sample_users(cfg, args.n_users, args.seed, trial)
        res = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        trial_rows, sE, sN, _, _ = _c2_trial_rows_from_result(trial=trial, result=res)
        if grid_gain_path is not None:
            trial_rows, grid_sE, grid_sN = _c2_attach_grid_true_gains(
                trial_rows,
                pE_grid=heatmap_pE_grid,
                pN_grid=heatmap_pN_grid,
                esp_revenue=heatmap_esp_rev,
                nsp_revenue=heatmap_nsp_rev,
            )
            grid_seq_E.append(grid_sE)
            grid_seq_N.append(grid_sN)
        seq_E.append(sE)
        seq_N.append(sN)
        rows.extend(trial_rows)
    _write_c2_outputs(
        out_dir=out_dir,
        rows=rows,
        seq_E=seq_E,
        seq_N=seq_N,
        config_path=args.config,
        seed=args.seed,
        n_users=args.n_users,
        trials=args.trials,
        grid_seq_E=(grid_seq_E if grid_gain_path is not None else None),
        grid_seq_N=(grid_seq_N if grid_gain_path is not None else None),
        grid_gain_source_path=(str(grid_gain_path) if grid_gain_path is not None else None),
    )


def main_C4() -> None:
    parser = argparse.ArgumentParser(description="Figure C4: final restricted gap versus evaluation budget.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16")
    parser.add_argument("--budget-list", type=str, default="16,32,64,96,128")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C4_final_gap_vs_budget", args.out_dir)
    cfg = _load_cfg(args.config)
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]
    budget_list = [int(x) for x in args.budget_list.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    methods = ["Proposed", "GSO", "GA", "BO", "MARL"]
    for n in n_list:
        for budget in budget_list:
            for trial in range(1, args.trials + 1):
                users = _sample_users(cfg, n, args.seed, trial)
                for method in methods:
                    if method == "GSO" and n > 16:
                        continue
                    stack_cfg, base_cfg = _budgeted_cfg(cfg.stackelberg, cfg.baselines, method if method != "MARL" else "marl", budget)
                    internal_method = method
                    if method == "MARL":
                        internal_method = "MARL"
                    price, offloading_set, gap, _, _, meta = _run_stage1_method(users, cfg.system, stack_cfg, base_cfg, internal_method)
                    rows.append(
                        {
                            "method": method,
                            "n_users": n,
                            "budget": budget,
                            "trial": trial,
                            "restricted_gap": float(gap),
                            "runtime_sec": float(meta["runtime_sec"]),
                            "final_pE": float(price[0]),
                            "final_pN": float(price[1]),
                            "offloading_size": int(len(offloading_set)),
                        }
                    )
    write_csv_rows(out_dir / "C4_final_gap_vs_budget.csv", ["method", "n_users", "budget", "trial", "restricted_gap", "runtime_sec", "final_pE", "final_pN", "offloading_size"], rows)
    plot_rows = [row for row in rows if int(row["n_users"]) == n_list[0]]
    _plot_method_errorbars(
        plot_rows,
        x_key="budget",
        y_key="restricted_gap",
        xlabel="Evaluation budget",
        ylabel="Final restricted gap",
        title=f"Final restricted gap versus budget (n={n_list[0]})",
        out_path=out_dir / "C4_final_gap_vs_budget.png",
        method_order=methods,
    )
    _write_summary(out_dir / "C4_final_gap_vs_budget_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users_list = {args.n_users_list}", f"budget_list = {args.budget_list}", f"trials = {args.trials}", "plot_note = main image uses the smallest n in n_users_list"])


def _stage2_runtime_rows(cfg: ExperimentConfig, n_list: list[int], trials: int, seed: int, pE: float, pN: float, *, include_centralized: bool = True, centralized_max_n: int = 16) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in n_list:
        for trial in range(1, trials + 1):
            users = _sample_users(cfg, n, seed, trial)
            t0 = time.perf_counter()
            out = solve_stage2_scm(users, pE, pN, cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
            runtime = time.perf_counter() - t0
            rows.append(
                {
                    "method": "Proposed",
                    "n_users": n,
                    "trial": trial,
                    "runtime_sec": float(runtime),
                    "rollback_count": int(out.rollback_count),
                    "accepted_admissions": int(out.accepted_admissions),
                    "offloading_size": int(len(out.offloading_set)),
                    "social_cost": float(out.social_cost),
                }
            )
            if include_centralized and n <= int(centralized_max_n):
                t0 = time.perf_counter()
                cs_out, success = s2_ratio._run_centralized(users, pE, pN, cfg.system, cfg, "enum")
                c_runtime = time.perf_counter() - t0
                if success:
                    rows.append(
                        {
                            "method": "Centralized-exact",
                            "n_users": n,
                            "trial": trial,
                            "runtime_sec": float(c_runtime),
                            "rollback_count": float("nan"),
                            "accepted_admissions": float("nan"),
                            "offloading_size": int(len(cs_out.offloading_set)),
                            "social_cost": float(cs_out.social_cost),
                        }
                    )
    return rows


def main_D1() -> None:
    parser = argparse.ArgumentParser(description="Figure D1: Stage II runtime versus users.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16,20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--centralized-max-n", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_D1_stage2_runtime_vs_users", args.out_dir)
    cfg = _load_cfg(args.config)
    rows = _stage2_runtime_rows(cfg, [int(x) for x in args.n_users_list.split(",") if x.strip()], args.trials, args.seed, args.pE, args.pN, include_centralized=True, centralized_max_n=args.centralized_max_n)
    write_csv_rows(out_dir / "D1_stage2_runtime_vs_users.csv", ["method", "n_users", "trial", "runtime_sec", "rollback_count", "accepted_admissions", "offloading_size", "social_cost"], rows)
    _plot_method_errorbars(rows, x_key="n_users", y_key="runtime_sec", xlabel="Number of users", ylabel="Runtime (sec)", title="Stage II runtime and scalability", out_path=out_dir / "D1_stage2_runtime_vs_users.png", method_order=["Proposed", "Centralized-exact"])
    _write_summary(out_dir / "D1_stage2_runtime_vs_users_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_D2() -> None:
    parser = argparse.ArgumentParser(description="Figure D2: Stage I runtime versus users.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16,20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--methods", type=str, default="VBBR,GA,BO,MARL")
    parser.add_argument("--baseline-bo-iters", type=int, default=None)
    parser.add_argument("--baseline-ga-generations", type=int, default=None)
    parser.add_argument("--baseline-marl-episodes", type=int, default=None)
    parser.add_argument("--baseline-marl-steps-per-episode", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_D2_stage1_runtime_vs_users", args.out_dir)
    methods = [str(x).strip() for x in args.methods.split(",") if str(x).strip()]
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]
    csv_path = out_dir / "D2_stage1_runtime_vs_users.csv"
    rows: list[dict[str, object]] = list(load_csv_rows(csv_path)) if csv_path.exists() else []
    completed_points = {
        (str(row["method"]), int(row["n_users"]), int(row["trial"]))
        for row in rows
    }
    total_points = sum(1 for n in n_list for method in methods if not (method.upper() == "GSO" and n > 16)) * int(args.trials)
    method_order_idx = {method: idx for idx, method in enumerate(methods)}

    def _flush_d2_progress(*, completed: bool) -> None:
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                int(row["n_users"]),
                int(row["trial"]),
                method_order_idx.get(str(row["method"]), len(method_order_idx)),
                str(row["method"]),
            ),
        )
        write_csv_rows(
            csv_path,
            ["method", "n_users", "trial", "runtime_sec", "restricted_gap", "offloading_size", "final_pE", "final_pN"],
            ordered_rows,
        )
        if ordered_rows:
            _plot_method_errorbars(
                ordered_rows,
                x_key="n_users",
                y_key="runtime_sec",
                xlabel="Number of users",
                ylabel="Runtime (sec)",
                title="Stage I runtime versus users",
                out_path=out_dir / "D2_stage1_runtime_vs_users.png",
                method_order=methods,
            )
        _write_summary(
            out_dir / "D2_stage1_runtime_vs_users_summary.txt",
            [
                f"config = {args.config}",
                f"seed = {args.seed}",
                f"trials = {args.trials}",
                f"n_users_list = {args.n_users_list}",
                f"methods = {','.join(methods)}",
                f"baseline_bo_iters = {'' if args.baseline_bo_iters is None else int(args.baseline_bo_iters)}",
                f"baseline_ga_generations = {'' if args.baseline_ga_generations is None else int(args.baseline_ga_generations)}",
                f"baseline_marl_episodes = {'' if args.baseline_marl_episodes is None else int(args.baseline_marl_episodes)}",
                f"baseline_marl_steps_per_episode = {'' if args.baseline_marl_steps_per_episode is None else int(args.baseline_marl_steps_per_episode)}",
                f"progress_completed_points = {len(completed_points)}",
                f"progress_total_points = {total_points}",
                f"progress_complete = {'true' if completed else 'false'}",
            ],
        )

    _flush_d2_progress(completed=(len(completed_points) >= total_points and total_points > 0))

    pending_points: list[tuple[int, int, str]] = []
    for n in n_list:
        for trial in range(1, args.trials + 1):
            for method in methods:
                if method.upper() == "GSO" and n > 16:
                    continue
                point_key = (method, int(n), int(trial))
                if point_key not in completed_points:
                    pending_points.append((int(n), int(trial), method))

    if int(args.jobs) <= 1:
        for n, trial, method in pending_points:
            rows.append(
                _run_d2_point(
                    str(args.config),
                    int(args.seed),
                    int(n),
                    int(trial),
                    method,
                    bo_iters=args.baseline_bo_iters,
                    ga_generations=args.baseline_ga_generations,
                    marl_episodes=args.baseline_marl_episodes,
                    marl_steps_per_episode=args.baseline_marl_steps_per_episode,
                )
            )
            completed_points.add((str(method), int(n), int(trial)))
            _flush_d2_progress(completed=False)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
            future_to_point = {
                ex.submit(
                    _run_d2_point,
                    str(args.config),
                    int(args.seed),
                    int(n),
                    int(trial),
                    method,
                    args.baseline_bo_iters,
                    args.baseline_ga_generations,
                    args.baseline_marl_episodes,
                    args.baseline_marl_steps_per_episode,
                ): (n, trial, method)
                for n, trial, method in pending_points
            }
            for future in concurrent.futures.as_completed(future_to_point):
                n, trial, method = future_to_point[future]
                rows.append(future.result())
                completed_points.add((str(method), int(n), int(trial)))
                _flush_d2_progress(completed=False)
    _flush_d2_progress(completed=True)


def main_D3() -> None:
    parser = argparse.ArgumentParser(description="Figure D3: number of Stage II solves inside Stage I.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_D3_stage2_calls_inside_stage1", args.out_dir)
    cfg = _load_cfg(args.config)
    rows: list[dict[str, object]] = []
    for n in [int(x) for x in args.n_users_list.split(",") if x.strip()]:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            res = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
            rows.append({"method": "Proposed", "n_users": n, "trial": trial, "stage2_calls": int(res.stage2_oracle_calls)})
    write_csv_rows(out_dir / "D3_stage2_calls_inside_stage1.csv", ["method", "n_users", "trial", "stage2_calls"], rows)
    _plot_method_errorbars(rows, x_key="n_users", y_key="stage2_calls", xlabel="Number of users", ylabel="Stage II calls", title="Number of Stage II solves inside Stage I", out_path=out_dir / "D3_stage2_calls_inside_stage1.png", method_order=["Proposed"])
    _write_summary(out_dir / "D3_stage2_calls_inside_stage1_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_D4() -> None:
    parser = argparse.ArgumentParser(description="Figure D4: exact reference runtime and feasibility.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16,20,24")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_D4_exact_runtime_feasibility", args.out_dir)
    cfg = _load_cfg(args.config)
    rows: list[dict[str, object]] = []
    for n in [int(x) for x in args.n_users_list.split(",") if x.strip()]:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            t0 = time.perf_counter()
            try:
                out = _solve_centralized_minlp(users, args.pE, args.pN, cfg.system, cfg.stackelberg, cfg.baselines)
                success = bool(out.meta.get("success", True))
            except Exception:
                success = False
            runtime = time.perf_counter() - t0
            rows.append({"method": "Centralized-MINLP", "n_users": n, "trial": trial, "runtime_sec": float(runtime), "success": int(success)})
    write_csv_rows(out_dir / "D4_exact_runtime_feasibility.csv", ["method", "n_users", "trial", "runtime_sec", "success"], rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=150)
    runtime_rows = [{"method": "Centralized-MINLP", "n_users": row["n_users"], "runtime_sec": row["runtime_sec"]} for row in rows if int(row["success"]) == 1]
    grouped = _mean_std_by_method(runtime_rows, "n_users", "runtime_sec")
    stats = grouped.get("Centralized-MINLP", [])
    x = np.asarray([item[0] for item in stats], dtype=float)
    y = np.asarray([item[1] for item in stats], dtype=float)
    e = np.asarray([item[2] for item in stats], dtype=float)
    axes[0].errorbar(x, y, yerr=e, fmt="-o", capsize=4)
    axes[0].set_xlabel("Number of users")
    axes[0].set_ylabel("Runtime (sec)")
    axes[0].set_title("Exact runtime")
    axes[0].grid(alpha=0.25)
    success_rates = []
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]
    for n in n_list:
        vals = [int(row["success"]) for row in rows if int(row["n_users"]) == n]
        success_rates.append(float(np.mean(vals)) if vals else float("nan"))
    axes[1].plot(n_list, success_rates, marker="o", linewidth=1.8)
    axes[1].set_xlabel("Number of users")
    axes[1].set_ylabel("Success rate")
    axes[1].set_title("Exact solver feasibility")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "D4_exact_runtime_feasibility.png")
    plt.close(fig)
    _write_summary(out_dir / "D4_exact_runtime_feasibility_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def _collect_strategic_rows(cfg: ExperimentConfig, n_list: list[int], trials: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = ["Full model", "ME", "SingleSP", "Coop", "Rand"]
    for n in n_list:
        for trial in range(1, trials + 1):
            users = _sample_users(cfg, n, seed, trial)
            for method in methods:
                internal = {"Full model": "Proposed", "ME": "ME", "SingleSP": "SingleSP", "Coop": "Coop", "Rand": "Rand"}[method]
                price, offloading_set, gap, esp_rev, nsp_rev, meta = _run_stage1_method(users, cfg.system, cfg.stackelberg, cfg.baselines, internal)
                rows.append(
                    {
                        "method": method,
                        "n_users": n,
                        "trial": trial,
                        "social_cost": float(meta["social_cost"]),
                        "esp_revenue": float(esp_rev),
                        "nsp_revenue": float(nsp_rev),
                        "joint_revenue": float(esp_rev + nsp_rev),
                        "comp_utilization": float("nan"),
                        "band_utilization": float("nan"),
                        "final_pE": float(price[0]),
                        "final_pN": float(price[1]),
                        "offloading_size": int(len(offloading_set)),
                        "restricted_gap": float(gap),
                    }
                )
                stage2 = solve_stage2_scm(users, price[0], price[1], cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
                rows[-1]["comp_utilization"] = float(np.sum(stage2.inner_result.f) / cfg.system.F)
                rows[-1]["band_utilization"] = float(np.sum(stage2.inner_result.b) / cfg.system.B)
    return rows


_STRATEGIC_FIELDNAMES = [
    "method",
    "n_users",
    "trial",
    "social_cost",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "comp_utilization",
    "band_utilization",
    "final_pE",
    "final_pN",
    "offloading_size",
    "restricted_gap",
]


def _strategic_rows_from_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in load_csv_rows(path):
        rows.append(
            {
                "method": str(row["method"]),
                "n_users": int(row["n_users"]),
                "trial": int(row["trial"]),
                "social_cost": float(row["social_cost"]),
                "esp_revenue": float(row["esp_revenue"]),
                "nsp_revenue": float(row["nsp_revenue"]),
                "joint_revenue": float(row["joint_revenue"]),
                "comp_utilization": float(row["comp_utilization"]),
                "band_utilization": float(row["band_utilization"]),
                "final_pE": float(row["final_pE"]),
                "final_pN": float(row["final_pN"]),
                "offloading_size": float(row["offloading_size"]),
                "restricted_gap": float(row["restricted_gap"]),
            }
        )
    return rows


def _flush_E2_progress(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    config_path: str,
    seed: int,
    n_users_list: list[int],
    trials: int,
    completed_points: int,
    total_points: int,
) -> None:
    csv_path = out_dir / "E2_provider_revenue_compare.csv"
    png_path = out_dir / "E2_provider_revenue_compare.png"
    summary_path = out_dir / "E2_provider_revenue_compare_summary.txt"
    write_csv_rows(csv_path, list(_STRATEGIC_FIELDNAMES), rows)
    _plot_three_panel(
        rows,
        x_key="n_users",
        panels=[("esp_revenue", "ESP revenue"), ("nsp_revenue", "NSP revenue"), ("joint_revenue", "Joint revenue")],
        xlabel="Number of users",
        title="Strategic-setting comparison: provider revenues",
        out_path=png_path,
        method_order=["Full model", "ME", "SingleSP", "Coop", "Rand"],
    )
    _write_summary(
        summary_path,
        [
            f"config = {config_path}",
            f"seed = {seed}",
            f"trials = {trials}",
            f"n_users_list = {','.join(str(int(x)) for x in n_users_list)}",
            f"progress_completed_points = {completed_points}",
            f"progress_total_points = {total_points}",
            f"progress_status = {'completed' if completed_points >= total_points else 'running'}",
        ],
    )


def main_E1() -> None:
    parser = argparse.ArgumentParser(description="Figure E1: user social cost comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_E1_user_social_cost_compare", args.out_dir)
    cfg = _load_cfg(args.config)
    rows = _collect_strategic_rows(cfg, [int(x) for x in args.n_users_list.split(",") if x.strip()], args.trials, args.seed)
    write_csv_rows(out_dir / "E1_user_social_cost_compare.csv", ["method", "n_users", "trial", "social_cost", "esp_revenue", "nsp_revenue", "joint_revenue", "comp_utilization", "band_utilization", "final_pE", "final_pN", "offloading_size", "restricted_gap"], rows)
    _plot_method_errorbars(rows, x_key="n_users", y_key="social_cost", xlabel="Number of users", ylabel="Total user social cost", title="Strategic-setting comparison: user social cost", out_path=out_dir / "E1_user_social_cost_compare.png", method_order=["Full model", "ME", "SingleSP", "Coop", "Rand"])
    _write_summary(out_dir / "E1_user_social_cost_compare_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_E2() -> None:
    parser = argparse.ArgumentParser(description="Figure E2: provider revenue comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_E2_provider_revenue_compare", args.out_dir)
    cfg = _load_cfg(args.config)
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]
    methods = ["Full model", "ME", "SingleSP", "Coop", "Rand"]
    csv_path = out_dir / "E2_provider_revenue_compare.csv"
    rows = _strategic_rows_from_csv(csv_path)
    completed = {(str(row["method"]), int(row["n_users"]), int(row["trial"])) for row in rows}
    total_points = len(n_list) * int(args.trials) * len(methods)

    if rows:
        _flush_E2_progress(
            out_dir=out_dir,
            rows=rows,
            config_path=args.config,
            seed=args.seed,
            n_users_list=n_list,
            trials=args.trials,
            completed_points=len(completed),
            total_points=total_points,
        )

    for n in n_list:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            for method in methods:
                key = (method, int(n), int(trial))
                if key in completed:
                    continue
                internal = {"Full model": "Proposed", "ME": "ME", "SingleSP": "SingleSP", "Coop": "Coop", "Rand": "Rand"}[method]
                price, offloading_set, gap, esp_rev, nsp_rev, meta = _run_stage1_method(users, cfg.system, cfg.stackelberg, cfg.baselines, internal)
                stage2 = solve_stage2_scm(users, price[0], price[1], cfg.system, cfg.stackelberg, inner_solver_mode="primal_dual")
                rows.append(
                    {
                        "method": method,
                        "n_users": int(n),
                        "trial": int(trial),
                        "social_cost": float(meta["social_cost"]),
                        "esp_revenue": float(esp_rev),
                        "nsp_revenue": float(nsp_rev),
                        "joint_revenue": float(esp_rev + nsp_rev),
                        "comp_utilization": float(np.sum(stage2.inner_result.f) / cfg.system.F),
                        "band_utilization": float(np.sum(stage2.inner_result.b) / cfg.system.B),
                        "final_pE": float(price[0]),
                        "final_pN": float(price[1]),
                        "offloading_size": int(len(offloading_set)),
                        "restricted_gap": float(gap),
                    }
                )
                completed.add(key)
                _flush_E2_progress(
                    out_dir=out_dir,
                    rows=rows,
                    config_path=args.config,
                    seed=args.seed,
                    n_users_list=n_list,
                    trials=args.trials,
                    completed_points=len(completed),
                    total_points=total_points,
                )


def main_E3() -> None:
    parser = argparse.ArgumentParser(description="Figure E3: resource utilization comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_E3_resource_utilization_compare", args.out_dir)
    cfg = _load_cfg(args.config)
    rows = _collect_strategic_rows(cfg, [int(x) for x in args.n_users_list.split(",") if x.strip()], args.trials, args.seed)
    write_csv_rows(out_dir / "E3_resource_utilization_compare.csv", ["method", "n_users", "trial", "social_cost", "esp_revenue", "nsp_revenue", "joint_revenue", "comp_utilization", "band_utilization", "final_pE", "final_pN", "offloading_size", "restricted_gap"], rows)
    _plot_three_panel(
        rows,
        x_key="n_users",
        panels=[("comp_utilization", "Computation utilization"), ("band_utilization", "Bandwidth utilization"), ("restricted_gap", "Restricted gap")],
        xlabel="Number of users",
        title="Strategic-setting comparison: utilization",
        out_path=out_dir / "E3_resource_utilization_compare.png",
        method_order=["Full model", "ME", "SingleSP", "Coop", "Rand"],
    )
    _write_summary(out_dir / "E3_resource_utilization_compare_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def main_E4() -> None:
    parser = argparse.ArgumentParser(description="Figure E4: final price and offloading-size comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_E4_price_and_offloading_compare", args.out_dir)
    cfg = _load_cfg(args.config)
    rows = _collect_strategic_rows(cfg, [int(x) for x in args.n_users_list.split(",") if x.strip()], args.trials, args.seed)
    write_csv_rows(out_dir / "E4_price_and_offloading_compare.csv", ["method", "n_users", "trial", "social_cost", "esp_revenue", "nsp_revenue", "joint_revenue", "comp_utilization", "band_utilization", "final_pE", "final_pN", "offloading_size", "restricted_gap"], rows)
    _plot_three_panel(
        rows,
        x_key="n_users",
        panels=[("final_pE", "Final pE"), ("final_pN", "Final pN"), ("offloading_size", "Offloading-set size")],
        xlabel="Number of users",
        title="Strategic-setting comparison: prices and offloading size",
        out_path=out_dir / "E4_price_and_offloading_compare.png",
        method_order=["Full model", "ME", "SingleSP", "Coop", "Rand"],
    )
    _write_summary(out_dir / "E4_price_and_offloading_compare_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


def _f1_rows_from_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    def _float_value(row: dict[str, str], key: str) -> float:
        value = str(row.get(key, "")).strip()
        return float(value) if value else float("nan")

    def _int_value(row: dict[str, str], key: str) -> int:
        value = str(row.get(key, "")).strip()
        return int(float(value)) if value else 0

    rows: list[dict[str, object]] = []
    for row in load_csv_rows(path):
        rows.append(
            {
                "method": str(row.get("method", "Proposed")),
                "Q": _int_value(row, "Q"),
                "trial": _int_value(row, "trial"),
                "candidate_family_size": _float_value(row, "candidate_family_size"),
                "restricted_gap": _float_value(row, "restricted_gap"),
                "grid_ne_gap": _float_value(row, "grid_ne_gap"),
                "runtime_sec": _float_value(row, "runtime_sec"),
                "stage2_calls": _int_value(row, "stage2_calls"),
                "final_pE": _float_value(row, "final_pE"),
                "final_pN": _float_value(row, "final_pN"),
            }
        )
    return rows


def _write_f1_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    cfg: ExperimentConfig,
    config_path: str,
    seed: int,
    n_users: int,
    q_list: list[int],
    trials: int,
    plot_mode: str,
    grid_csv_path: Path | None,
    completed_points: int,
    total_points: int,
) -> None:
    csv_path = out_dir / "F1_q_sensitivity.csv"
    stats_path = out_dir / "F1_q_sensitivity_stats.csv"
    fig_path = out_dir / "F1_q_sensitivity.png"
    summary_path = out_dir / "F1_q_sensitivity_summary.txt"

    fieldnames = [
        "method",
        "Q",
        "trial",
        "candidate_family_size",
        "restricted_gap",
        "grid_ne_gap",
        "runtime_sec",
        "stage2_calls",
        "final_pE",
        "final_pN",
    ]
    write_csv_rows(csv_path, fieldnames, rows)

    stats_rows: list[dict[str, object]] = []
    stats_fieldnames = [
        "Q",
        "trials",
        "candidate_family_size_mean",
        "candidate_family_size_std",
        "candidate_family_size_median",
        "restricted_gap_mean",
        "restricted_gap_std",
        "restricted_gap_median",
        "grid_ne_gap_mean",
        "grid_ne_gap_std",
        "grid_ne_gap_median",
        "runtime_sec_mean",
        "runtime_sec_std",
        "runtime_sec_median",
        "stage2_calls_mean",
        "stage2_calls_std",
        "stage2_calls_median",
        "final_pE_mean",
        "final_pE_std",
        "final_pE_median",
        "final_pN_mean",
        "final_pN_std",
        "final_pN_median",
    ]
    metric_keys = [
        "candidate_family_size",
        "restricted_gap",
        "grid_ne_gap",
        "runtime_sec",
        "stage2_calls",
        "final_pE",
        "final_pN",
    ]
    for q in sorted({int(row["Q"]) for row in rows}):
        subset = [row for row in rows if int(row["Q"]) == q]
        stat_row: dict[str, object] = {"Q": int(q), "trials": len(subset)}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in subset if np.isfinite(float(row[key]))], dtype=float)
            if values.size == 0:
                stat_row[f"{key}_mean"] = float("nan")
                stat_row[f"{key}_std"] = float("nan")
                stat_row[f"{key}_median"] = float("nan")
            else:
                stat_row[f"{key}_mean"] = float(np.mean(values))
                stat_row[f"{key}_std"] = float(np.std(values))
                stat_row[f"{key}_median"] = float(np.median(values))
        stats_rows.append(stat_row)
    write_csv_rows(stats_path, stats_fieldnames, stats_rows)

    if plot_mode == "quality_runtime_calls":
        panels = [
            ("grid_ne_gap", "Final grid-evaluated NE gap"),
            ("runtime_sec", "Runtime (sec)"),
            ("stage2_calls", "Stage-II solver calls"),
        ]
        title = "Sensitivity to Q: final grid NE gap, runtime, and Stage-II calls"
    else:
        panels = [
            ("candidate_family_size", "Candidate-family size"),
            ("restricted_gap", "Final restricted gap"),
            ("runtime_sec", "Runtime (sec)"),
        ]
        title = "Sensitivity to local candidate-family size Q"
    _plot_three_panel(rows, x_key="Q", panels=panels, xlabel="Q", title=title, out_path=fig_path, method_order=["Proposed"])

    solver_variant = str(cfg.stackelberg.stage1_solver_variant)
    if solver_variant == "paper_iterative_pricing":
        q_mapping = "paper_local_Q = Q"
    elif solver_variant == "vbbr_brd":
        q_mapping = f"vbbr_local_R = Q, vbbr_local_S = Q, vbbr_local_budget fixed at {int(cfg.stackelberg.vbbr_local_budget)}"
    else:
        q_mapping = "solver-specific Q override"

    summary_lines = [
        f"config = {config_path}",
        f"seed = {seed}",
        f"n_users = {n_users}",
        f"q_list = {','.join(str(int(q)) for q in q_list)}",
        f"trials = {trials}",
        f"stage1_solver_variant = {solver_variant}",
        f"q_mapping = {q_mapping}",
        f"plot_mode = {plot_mode}",
        f"progress_completed_points = {completed_points}",
        f"progress_total_points = {total_points}",
        f"progress_status = {'completed' if completed_points >= total_points else 'running'}",
        "recorded_metrics = candidate_family_size,restricted_gap,grid_ne_gap,runtime_sec,stage2_calls,final_pE,final_pN",
    ]
    if grid_csv_path is not None:
        summary_lines.extend(
            [
                "grid_ne_gap_source = heatmap_csv_nearest",
                f"grid_ne_gap_heatmap_csv = {grid_csv_path}",
                "grid_ne_gap_definition = precomputed grid_ne_gap surface value at the nearest heatmap grid point to each returned final price",
            ]
        )
    else:
        summary_lines.append("grid_ne_gap_source = none")
    _write_summary(summary_path, summary_lines)


def main_F1() -> None:
    parser = argparse.ArgumentParser(description="Figure F1: Q sensitivity.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--q-list", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--plot-mode", type=str, choices=["default", "quality_runtime_calls"], default="default")
    parser.add_argument("--grid-ne-gap-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_F1_q_sensitivity", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)

    grid_csv_path: Path | None = None
    if args.grid_ne_gap_csv is not None:
        grid_csv_path = _resolve_existing_path(str(args.grid_ne_gap_csv))
        heatmap_pE_grid, heatmap_pN_grid, heatmap_grid_ne_gap = _load_grid_ne_gap_surface(grid_csv_path)
    else:
        heatmap_pE_grid = np.asarray([], dtype=float)
        heatmap_pN_grid = np.asarray([], dtype=float)
        heatmap_grid_ne_gap = np.asarray([[]], dtype=float)

    q_list = [int(x) for x in args.q_list.split(",") if x.strip()]
    csv_path = out_dir / "F1_q_sensitivity.csv"
    rows = _f1_rows_from_csv(csv_path)
    completed = {(int(row["Q"]), int(row["trial"])) for row in rows}
    total_points = len(q_list) * int(args.trials)
    if rows:
        _write_f1_outputs(
            out_dir=out_dir,
            rows=rows,
            cfg=cfg,
            config_path=args.config,
            seed=args.seed,
            n_users=args.n_users,
            q_list=q_list,
            trials=args.trials,
            plot_mode=args.plot_mode,
            grid_csv_path=grid_csv_path,
            completed_points=len(completed),
            total_points=total_points,
        )

    for q in q_list:
        if str(cfg.stackelberg.stage1_solver_variant) == "paper_iterative_pricing":
            stack_cfg = replace(cfg.stackelberg, paper_local_Q=q)
        else:
            stack_cfg = replace(cfg.stackelberg, vbbr_local_R=q, vbbr_local_S=q)
        for trial in range(1, args.trials + 1):
            key = (int(q), int(trial))
            if key in completed:
                continue
            users = _sample_users(cfg, args.n_users, args.seed, trial)
            t0 = time.perf_counter()
            res = solve_stage1_pricing(users, cfg.system, stack_cfg)
            runtime = time.perf_counter() - t0
            candidate_sizes = [int(step.candidate_family_size) for step in res.trajectory if int(step.candidate_family_size) > 0]
            if grid_csv_path is not None:
                grid_gap = _nearest_grid_ne_gap(
                    res.price[0],
                    res.price[1],
                    pE_grid=heatmap_pE_grid,
                    pN_grid=heatmap_pN_grid,
                    grid_ne_gap=heatmap_grid_ne_gap,
                )
            else:
                grid_gap = float("nan")
            rows.append(
                {
                    "method": "Proposed",
                    "Q": int(q),
                    "trial": int(trial),
                    "candidate_family_size": float(np.mean(candidate_sizes) if candidate_sizes else 0.0),
                    "restricted_gap": float(res.restricted_gap),
                    "grid_ne_gap": float(grid_gap),
                    "runtime_sec": float(runtime),
                    "stage2_calls": int(res.stage2_oracle_calls),
                    "final_pE": float(res.price[0]),
                    "final_pN": float(res.price[1]),
                }
            )
            completed.add(key)
            _write_f1_outputs(
                out_dir=out_dir,
                rows=rows,
                cfg=cfg,
                config_path=args.config,
                seed=args.seed,
                n_users=args.n_users,
                q_list=q_list,
                trials=args.trials,
                plot_mode=args.plot_mode,
                grid_csv_path=grid_csv_path,
                completed_points=len(completed),
                total_points=total_points,
            )


def main_F2() -> None:
    parser = argparse.ArgumentParser(description="Figure F2: resource asymmetry sensitivity.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--ratio-list", type=str, default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_F2_resource_asymmetry_sensitivity", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    product = float(cfg.system.F * cfg.system.B)
    rows: list[dict[str, object]] = []
    for ratio in [float(x) for x in args.ratio_list.split(",") if x.strip()]:
        F_val = math.sqrt(product * ratio)
        B_val = math.sqrt(product / ratio)
        system = replace(cfg.system, F=float(F_val), B=float(B_val))
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, args.n_users, args.seed, trial)
            res = solve_stage1_pricing(users, system, cfg.stackelberg)
            rows.append({"method": "Proposed", "ratio": float(ratio), "trial": trial, "social_cost": float(res.social_cost), "restricted_gap": float(res.restricted_gap), "offloading_size": int(len(res.offloading_set))})
    write_csv_rows(out_dir / "F2_resource_asymmetry_sensitivity.csv", ["method", "ratio", "trial", "social_cost", "restricted_gap", "offloading_size"], rows)
    _plot_three_panel(rows, x_key="ratio", panels=[("social_cost", "Final social cost"), ("restricted_gap", "Final restricted gap"), ("offloading_size", "Offloading-set size")], xlabel="F/B", title="Sensitivity to resource asymmetry", out_path=out_dir / "F2_resource_asymmetry_sensitivity.png", method_order=["Proposed"])
    _write_summary(out_dir / "F2_resource_asymmetry_sensitivity_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"ratio_list = {args.ratio_list}", f"trials = {args.trials}"])


def main_F3() -> None:
    parser = argparse.ArgumentParser(description="Figure F3: provider cost sensitivity.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--cost-list", type=str, default="0.05,0.1,0.15,0.2,0.25")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_F3_provider_cost_sensitivity", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    costs = [float(x) for x in args.cost_list.split(",") if x.strip()]
    social = np.full((len(costs), len(costs)), np.nan, dtype=float)
    gap = np.full_like(social, np.nan)
    rows: list[dict[str, object]] = []
    for i, cN in enumerate(costs):
        for j, cE in enumerate(costs):
            system = replace(cfg.system, cE=float(cE), cN=float(cN))
            social_vals = []
            gap_vals = []
            for trial in range(1, args.trials + 1):
                users = _sample_users(cfg, args.n_users, args.seed, trial)
                res = solve_stage1_pricing(users, system, cfg.stackelberg)
                social_vals.append(float(res.social_cost))
                gap_vals.append(float(res.restricted_gap))
                rows.append({"cE": float(cE), "cN": float(cN), "trial": trial, "social_cost": float(res.social_cost), "restricted_gap": float(res.restricted_gap)})
            social[i, j] = float(np.mean(social_vals))
            gap[i, j] = float(np.mean(gap_vals))
    write_csv_rows(out_dir / "F3_provider_cost_sensitivity.csv", ["cE", "cN", "trial", "social_cost", "restricted_gap"], rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=150)
    for ax, matrix, label, title in [
        (axes[0], social, "Final social cost", "Social-cost sensitivity"),
        (axes[1], gap, "Final restricted gap", "Restricted-gap sensitivity"),
    ]:
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(costs)))
        ax.set_xticklabels([f"{x:.2g}" for x in costs], rotation=45)
        ax.set_yticks(np.arange(len(costs)))
        ax.set_yticklabels([f"{x:.2g}" for x in costs])
        ax.set_xlabel("cE")
        ax.set_ylabel("cN")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(out_dir / "F3_provider_cost_sensitivity.png")
    plt.close(fig)
    _write_summary(out_dir / "F3_provider_cost_sensitivity_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"cost_list = {args.cost_list}", f"trials = {args.trials}"])


def _scenario_cfg(cfg: ExperimentConfig, scenario: str) -> ExperimentConfig:
    users_cfg = cfg.users
    if scenario == "paper_default":
        return cfg
    if scenario == "heavy_workload":
        return replace(cfg, users=replace(users_cfg, w=DistributionSpec(kind="uniform", params={"low": 0.6, "high": 1.5})))
    if scenario == "bandwidth_poor":
        return replace(
            cfg,
            users=replace(
                users_cfg,
                d=DistributionSpec(kind="uniform", params={"low": 3.0, "high": 7.0}),
                sigma=DistributionSpec(kind="constant", params={"value": 0.75}),
            ),
        )
    if scenario == "high_delay_sensitivity":
        return replace(cfg, users=replace(users_cfg, alpha=DistributionSpec(kind="uniform", params={"low": 1.6, "high": 2.6})))
    if scenario == "high_energy_sensitivity":
        return replace(cfg, users=replace(users_cfg, beta=DistributionSpec(kind="uniform", params={"low": 0.3, "high": 0.7})))
    raise ValueError(f"Unknown scenario={scenario}")


def main_F4() -> None:
    parser = argparse.ArgumentParser(description="Figure F4: user-distribution sensitivity.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_F4_user_distribution_sensitivity", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    scenarios = ["paper_default", "heavy_workload", "bandwidth_poor", "high_delay_sensitivity", "high_energy_sensitivity"]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        cfg_s = _scenario_cfg(cfg, scenario)
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg_s, args.n_users, args.seed, trial)
            t0 = time.perf_counter()
            res = solve_stage1_pricing(users, cfg_s.system, cfg_s.stackelberg)
            runtime = time.perf_counter() - t0
            rows.append({"scenario": scenario, "trial": trial, "social_cost": float(res.social_cost), "restricted_gap": float(res.restricted_gap), "runtime_sec": float(runtime)})
    write_csv_rows(out_dir / "F4_user_distribution_sensitivity.csv", ["scenario", "trial", "social_cost", "restricted_gap", "runtime_sec"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), dpi=150)
    for ax, y_key, ylabel in [
        (axes[0], "social_cost", "Final social cost"),
        (axes[1], "restricted_gap", "Final restricted gap"),
        (axes[2], "runtime_sec", "Runtime (sec)"),
    ]:
        groups = [np.asarray([float(row[y_key]) for row in rows if row["scenario"] == scenario], dtype=float) for scenario in scenarios]
        ax.boxplot(groups, labels=[s.replace("_", "\n") for s in scenarios], patch_artist=True)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "F4_user_distribution_sensitivity.png")
    plt.close(fig)
    _write_summary(out_dir / "F4_user_distribution_sensitivity_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"trials = {args.trials}", f"scenarios = {','.join(scenarios)}"])

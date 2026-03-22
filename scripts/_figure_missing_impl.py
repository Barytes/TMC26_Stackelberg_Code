from __future__ import annotations

import argparse
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

from _figure_output_schema import write_standard_figure_summary
from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
import run_stage2_approximation_ratio as s2_ratio
import run_boundary_hypothesis_check as boundary_diag
from tmc26_exp.baselines import (
    BaselineOutcome,
    _solve_centralized_minlp,
    baseline_coop,
    baseline_market_equilibrium,
    baseline_random_offloading,
    baseline_single_sp,
    baseline_stage1_bo,
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


def _aligned_series(sequences: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=float)
    for i, seq in enumerate(sequences):
        seq_arr = np.asarray(seq, dtype=float)
        arr[i, : seq_arr.size] = seq_arr
        if seq_arr.size < max_len and seq_arr.size > 0:
            arr[i, seq_arr.size :] = seq_arr[-1]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


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


def main_C1() -> None:
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
    mean, std = _aligned_series(sequences)
    write_csv_rows(out_dir / "C1_restricted_gap_trajectory.csv", ["trial", "iteration", "restricted_gap"], rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    x = np.arange(1, mean.size + 1)
    ax.plot(x, mean, marker="o", linewidth=1.8, label="Mean")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="Mean ± std")
    ax.set_xlabel("Stage I iteration")
    ax.set_ylabel("Restricted gap")
    ax.set_title("Restricted-gap trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C1_restricted_gap_trajectory.png")
    plt.close(fig)
    _write_summary(out_dir / "C1_restricted_gap_trajectory_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"trials = {args.trials}"])


def main_C2() -> None:
    parser = argparse.ArgumentParser(description="Figure C2: best-response gain trajectories.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C2_best_response_gain_trajectory", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    seq_E: list[list[float]] = []
    seq_N: list[list[float]] = []
    rows: list[dict[str, object]] = []
    for trial in range(1, args.trials + 1):
        users = _sample_users(cfg, args.n_users, args.seed, trial)
        res = solve_stage1_pricing(users, cfg.system, cfg.stackelberg)
        sE = [float(step.esp_gain if np.isfinite(step.esp_gain) else 0.0) for step in res.trajectory]
        sN = [float(step.nsp_gain if np.isfinite(step.nsp_gain) else 0.0) for step in res.trajectory]
        if not sE:
            sE = [float(res.gain_E.gain)]
        if not sN:
            sN = [float(res.gain_N.gain)]
        seq_E.append(sE)
        seq_N.append(sN)
        for step_idx, (gE, gN) in enumerate(zip(sE, sN), start=1):
            rows.append({"trial": trial, "iteration": step_idx, "esp_gain": gE, "nsp_gain": gN})
    mean_E, std_E = _aligned_series(seq_E)
    mean_N, std_N = _aligned_series(seq_N)
    write_csv_rows(out_dir / "C2_best_response_gain_trajectory.csv", ["trial", "iteration", "esp_gain", "nsp_gain"], rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    x = np.arange(1, mean_E.size + 1)
    ax.plot(x, mean_E, marker="o", linewidth=1.8, label="ESP gain")
    ax.fill_between(x, mean_E - std_E, mean_E + std_E, alpha=0.18)
    ax.plot(x, mean_N, marker="s", linewidth=1.8, label="NSP gain")
    ax.fill_between(x, mean_N - std_N, mean_N + std_N, alpha=0.18)
    ax.set_xlabel("Stage I iteration")
    ax.set_ylabel("Best-response gain")
    ax.set_title("Best-response gain trajectories")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "C2_best_response_gain_trajectory.png")
    plt.close(fig)
    _write_summary(out_dir / "C2_best_response_gain_trajectory_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"trials = {args.trials}"])


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
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_D2_stage1_runtime_vs_users", args.out_dir)
    cfg = _load_cfg(args.config)
    rows: list[dict[str, object]] = []
    methods = ["Proposed", "GA", "BO", "MARL", "GSO"]
    for n in [int(x) for x in args.n_users_list.split(",") if x.strip()]:
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, n, args.seed, trial)
            for method in methods:
                if method == "GSO" and n > 16:
                    continue
                internal_method = method if method != "MARL" else "MARL"
                price, offloading_set, gap, _, _, meta = _run_stage1_method(users, cfg.system, cfg.stackelberg, cfg.baselines, internal_method)
                rows.append({"method": method, "n_users": n, "trial": trial, "runtime_sec": float(meta["runtime_sec"]), "restricted_gap": float(gap), "offloading_size": int(len(offloading_set)), "final_pE": float(price[0]), "final_pN": float(price[1])})
    write_csv_rows(out_dir / "D2_stage1_runtime_vs_users.csv", ["method", "n_users", "trial", "runtime_sec", "restricted_gap", "offloading_size", "final_pE", "final_pN"], rows)
    _plot_method_errorbars(rows, x_key="n_users", y_key="runtime_sec", xlabel="Number of users", ylabel="Runtime (sec)", title="Stage I runtime versus users", out_path=out_dir / "D2_stage1_runtime_vs_users.png", method_order=methods)
    _write_summary(out_dir / "D2_stage1_runtime_vs_users_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


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
    rows = _collect_strategic_rows(cfg, [int(x) for x in args.n_users_list.split(",") if x.strip()], args.trials, args.seed)
    write_csv_rows(out_dir / "E2_provider_revenue_compare.csv", ["method", "n_users", "trial", "social_cost", "esp_revenue", "nsp_revenue", "joint_revenue", "comp_utilization", "band_utilization", "final_pE", "final_pN", "offloading_size", "restricted_gap"], rows)
    _plot_three_panel(
        rows,
        x_key="n_users",
        panels=[("esp_revenue", "ESP revenue"), ("nsp_revenue", "NSP revenue"), ("joint_revenue", "Joint revenue")],
        xlabel="Number of users",
        title="Strategic-setting comparison: provider revenues",
        out_path=out_dir / "E2_provider_revenue_compare.png",
        method_order=["Full model", "ME", "SingleSP", "Coop", "Rand"],
    )
    _write_summary(out_dir / "E2_provider_revenue_compare_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"trials = {args.trials}", f"n_users_list = {args.n_users_list}"])


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


def main_F1() -> None:
    parser = argparse.ArgumentParser(description="Figure F1: Q sensitivity.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--q-list", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = resolve_out_dir("run_figure_F1_q_sensitivity", args.out_dir)
    cfg = _load_cfg(args.config, n_users=args.n_users)
    rows: list[dict[str, object]] = []
    for q in [int(x) for x in args.q_list.split(",") if x.strip()]:
        stack_cfg = replace(cfg.stackelberg, vbbr_local_R=q, vbbr_local_S=q)
        for trial in range(1, args.trials + 1):
            users = _sample_users(cfg, args.n_users, args.seed, trial)
            t0 = time.perf_counter()
            res = solve_stage1_pricing(users, cfg.system, stack_cfg)
            runtime = time.perf_counter() - t0
            candidate_sizes = [int(step.candidate_family_size) for step in res.trajectory if int(step.candidate_family_size) > 0]
            rows.append({"method": "Proposed", "Q": q, "trial": trial, "candidate_family_size": float(np.mean(candidate_sizes) if candidate_sizes else 0.0), "restricted_gap": float(res.restricted_gap), "runtime_sec": float(runtime)})
    write_csv_rows(out_dir / "F1_q_sensitivity.csv", ["method", "Q", "trial", "candidate_family_size", "restricted_gap", "runtime_sec"], rows)
    _plot_three_panel(rows, x_key="Q", panels=[("candidate_family_size", "Candidate-family size"), ("restricted_gap", "Final restricted gap"), ("runtime_sec", "Runtime (sec)")], xlabel="Q", title="Sensitivity to local candidate-family size Q", out_path=out_dir / "F1_q_sensitivity.png", method_order=["Proposed"])
    _write_summary(out_dir / "F1_q_sensitivity_summary.txt", [f"config = {args.config}", f"seed = {args.seed}", f"n_users = {args.n_users}", f"q_list = {args.q_list}", f"trials = {args.trials}"])


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

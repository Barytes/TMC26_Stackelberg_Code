#!/usr/bin/env python3
"""Ablation: Sampling density L (rne_directions) sensitivity.

This script analyzes how the number of sampling directions L affects:
- Final epsilon (accuracy)
- Runtime
- Number of Stage-II calls

Output: Line plots showing accuracy-complexity tradeoff.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.config import load_config, StackelbergConfig
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search


# Global counter for Stage-II calls
_stage2_call_count = 0


def _make_counting_algorithm_2(original_func):
    """Wrap algorithm_2 to count calls."""
    def wrapper(users, pE, pN, system, cfg):
        global _stage2_call_count
        _stage2_call_count += 1
        return original_func(users, pE, pN, system, cfg)
    return wrapper


def run_stage1_with_counting(users, system, stackelberg_cfg):
    """Run Algorithm 5 with Stage-II call counting."""
    global _stage2_call_count
    _stage2_call_count = 0

    # Monkey-patch to count calls
    import tmc26_exp.stackelberg as stackelberg_module
    original_alg2 = stackelberg_module.algorithm_2_heuristic_user_selection
    stackelberg_module.algorithm_2_heuristic_user_selection = _make_counting_algorithm_2(original_alg2)

    try:
        t0 = time.perf_counter()
        result = algorithm_5_stackelberg_guided_search(users, system, stackelberg_cfg)
        runtime = time.perf_counter() - t0
        return result, runtime, _stage2_call_count
    finally:
        stackelberg_module.algorithm_2_heuristic_user_selection = original_alg2


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "L", "epsilon", "runtime_sec", "stage2_calls", "search_iterations"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict]:
    """Aggregate by (n_users, L)."""
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), int(row["L"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    fields = [
        "n_users", "L",
        "epsilon_mean", "epsilon_std",
        "runtime_mean", "runtime_std",
        "stage2_calls_mean", "stage2_calls_std",
        "count"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, L) in sorted(grouped.keys()):
            bucket = grouped[(n_users, L)]
            eps = [float(r["epsilon"]) for r in bucket]
            rt = [float(r["runtime_sec"]) for r in bucket]
            calls = [int(r["stage2_calls"]) for r in bucket]

            eps_mean, eps_std = _mean_std(eps)
            rt_mean, rt_std = _mean_std(rt)
            calls_mean, calls_std = _mean_std([float(c) for c in calls])

            row = {
                "n_users": n_users,
                "L": L,
                "epsilon_mean": eps_mean,
                "epsilon_std": eps_std,
                "runtime_mean": rt_mean,
                "runtime_std": rt_std,
                "stage2_calls_mean": calls_mean,
                "stage2_calls_std": calls_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)

    return summary_rows


def _plot_epsilon_vs_L(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot epsilon vs L for different user counts."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    by_users: dict[int, list[dict]] = {}
    for row in summary_rows:
        n = int(row["n_users"])
        by_users.setdefault(n, []).append(row)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_users_list)))

    for idx, n_users in enumerate(n_users_list):
        if n_users not in by_users:
            continue
        rows = sorted(by_users[n_users], key=lambda r: int(r["L"]))
        x = [int(r["L"]) for r in rows]
        y = [float(r["epsilon_mean"]) for r in rows]
        yerr = [float(r["epsilon_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o",
            color=colors[idx],
            linewidth=2.0,
            capsize=3,
            label=f"n={n_users}",
            markersize=8,
        )

    ax.set_title("Ablation: Epsilon vs Sampling Directions L")
    ax.set_xlabel("Number of Directions L")
    ax.set_ylabel("Final Epsilon (Deviation Gap)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Users", loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_runtime_vs_L(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot runtime vs L."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    by_users: dict[int, list[dict]] = {}
    for row in summary_rows:
        n = int(row["n_users"])
        by_users.setdefault(n, []).append(row)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_users_list)))

    for idx, n_users in enumerate(n_users_list):
        if n_users not in by_users:
            continue
        rows = sorted(by_users[n_users], key=lambda r: int(r["L"]))
        x = [int(r["L"]) for r in rows]
        y = [float(r["runtime_mean"]) for r in rows]
        yerr = [float(r["runtime_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o",
            color=colors[idx],
            linewidth=2.0,
            capsize=3,
            label=f"n={n_users}",
            markersize=8,
        )

    ax.set_title("Ablation: Runtime vs Sampling Directions L")
    ax.set_xlabel("Number of Directions L")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Users", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_stage2_calls_vs_L(summary_rows: list[dict], out_path: Path, n_users_list: list[int]) -> None:
    """Plot Stage-II calls vs L."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    by_users: dict[int, list[dict]] = {}
    for row in summary_rows:
        n = int(row["n_users"])
        by_users.setdefault(n, []).append(row)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(n_users_list)))

    for idx, n_users in enumerate(n_users_list):
        if n_users not in by_users:
            continue
        rows = sorted(by_users[n_users], key=lambda r: int(r["L"]))
        x = [int(r["L"]) for r in rows]
        y = [float(r["stage2_calls_mean"]) for r in rows]
        yerr = [float(r["stage2_calls_std"]) for r in rows]

        ax.errorbar(
            x, y, yerr=yerr,
            marker="o",
            color=colors[idx],
            linewidth=2.0,
            capsize=3,
            label=f"n={n_users}",
            markersize=8,
        )

    ax.set_title("Ablation: Stage-II Oracle Calls vs Sampling Directions L")
    ax.set_xlabel("Number of Directions L")
    ax.set_ylabel("Stage-II Oracle Calls")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(title="Users", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_tradeoff(summary_rows: list[dict], out_path: Path, n_users: int = 100) -> None:
    """Plot accuracy vs complexity tradeoff."""
    rows = [r for r in summary_rows if int(r["n_users"]) == n_users]
    if not rows:
        return

    rows = sorted(rows, key=lambda r: int(r["L"]))
    x = [float(r["stage2_calls_mean"]) for r in rows]
    y = [float(r["epsilon_mean"]) for r in rows]
    labels = [f"L={int(r['L'])}" for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot(x, y, 'o-', color='#1b5e20', linewidth=2.0, markersize=10)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                   xytext=(5, 5), fontsize=9)

    ax.set_title(f"Ablation: Accuracy-Complexity Tradeoff (n={n_users})")
    ax.set_xlabel("Stage-II Oracle Calls (Complexity)")
    ax.set_ylabel("Final Epsilon (Accuracy)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation: L sensitivity analysis")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--L-values", type=str, default="4,8,12,20,32,48")
    parser.add_argument("--n-users", type=str, default="50,100,200")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20281001)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    L_values = [int(x.strip()) for x in args.L_values.split(",") if x.strip()]
    n_users_list = [int(x.strip()) for x in args.n_users.split(",") if x.strip()]

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ablation_L_sensitivity_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []

    for n_users in n_users_list:
        trial_cfg = replace(cfg, n_users=n_users)
        for L in L_values:
            # Create modified Stackelberg config with L
            stackelberg_cfg = replace(cfg.stackelberg, rne_directions=L)

            for trial in range(args.trials):
                seed = args.seed + 1000 * (n_users * 100 + L) + trial
                rng = np.random.default_rng(seed)
                users = sample_users(trial_cfg, rng)

                result, runtime, stage2_calls = run_stage1_with_counting(
                    users, cfg.system, stackelberg_cfg
                )

                raw_rows.append({
                    "trial": trial,
                    "seed": seed,
                    "n_users": n_users,
                    "L": L,
                    "epsilon": result.epsilon,
                    "runtime_sec": runtime,
                    "stage2_calls": stage2_calls,
                    "search_iterations": len(result.trajectory),
                })

    _write_raw_csv(raw_rows, run_dir / "raw_L_sensitivity.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_L_sensitivity.csv")
    _plot_epsilon_vs_L(summary_rows, run_dir / "L_vs_epsilon.png", n_users_list)
    _plot_runtime_vs_L(summary_rows, run_dir / "L_vs_runtime.png", n_users_list)
    _plot_stage2_calls_vs_L(summary_rows, run_dir / "L_vs_stage2_calls.png", n_users_list)

    # Tradeoff plots for each n_users
    for n in n_users_list:
        _plot_tradeoff(summary_rows, run_dir / f"tradeoff_n{n}.png", n)

    # Write metadata
    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"L_values = {L_values}",
        f"n_users = {n_users_list}",
        f"trials = {args.trials}",
        f"seed = {args.seed}",
        "parameter = rne_directions (number of sampling directions)",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()
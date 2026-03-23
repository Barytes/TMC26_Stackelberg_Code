"""Block A primary script: Stage II SCM social-cost trace vs centralized reference."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.baselines import (
    _solve_centralized_minlp,
    _solve_centralized_pyomo_scip,
    baseline_stage2_centralized_solver,
)
from tmc26_exp.config import ExperimentConfig, StackelbergConfig, SystemConfig, load_config
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    solve_stage2_scm,
)


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _plot_trace(trace: list[float], centralized_social: float | None, out_path: Path) -> None:
    x = np.arange(1, len(trace) + 1, dtype=int)
    y = np.asarray(trace, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(x, y, marker="o", linewidth=1.8, label="Algorithm 2 Heuristic")
    if centralized_social is not None:
        ax.axhline(
            y=float(centralized_social),
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
            label="Centralized Baseline",
        )
    ax.set_xlabel("Iteration (Algorithm 2)")
    ax.set_ylabel("Social Cost")
    ax.set_title("Stage-II Social Cost Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_trace_csv(out_path: Path, trace: list[float], centralized_social: float | None) -> None:
    rows = ["iteration,social_cost,centralized_social_cost"]
    for i, value in enumerate(trace, start=1):
        central_str = "" if centralized_social is None else f"{float(centralized_social):.10g}"
        rows.append(f"{i},{float(value):.10g},{central_str}")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _default_output_dir(base_output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / f"run_stage2_social_cost_compare_{timestamp}"


def _parse_n_users_list(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n-users-list cannot be empty.")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Each n in n-users-list must be > 0.")
    return values


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _sample_users_for_n(cfg: ExperimentConfig, n_users: int, seed: int) -> UserBatch:
    cfg_n = replace(cfg, n_users=int(n_users))
    rng = np.random.default_rng(int(seed))
    return sample_users(cfg_n, rng)


def _plot_multi_n_trace(
    traces_by_n: dict[int, list[float]],
    centralized_by_n: dict[int, float | None],
    out_path: Path,
    max_iter_plot: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    cmap = plt.get_cmap("tab10")
    ns = sorted(traces_by_n.keys())
    for idx, n in enumerate(ns):
        color = cmap(idx % 10)
        trace = traces_by_n[n]
        if max_iter_plot is not None:
            trace = trace[:max_iter_plot]
        x = np.arange(1, len(trace) + 1, dtype=int)
        ax.plot(x, np.asarray(trace, dtype=float), marker="o", linewidth=1.8, color=color, label=f"Alg2 n={n}")
        cs_val = centralized_by_n.get(n)
        if cs_val is not None:
            ax.axhline(
                y=float(cs_val),
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                label=f"CS n={n}",
            )
    ax.set_xlabel("Iteration (Algorithm 2)")
    ax.set_ylabel("Social Cost")
    ax.set_title("Stage-II Social Cost: Algorithm 2 vs CS Across User Scales")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Algorithm-2 iterative social cost against baseline_stage2_centralized_solver "
            "for a fixed price pair."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--pE", type=_positive_float, required=True, help="Fixed pE used in Stage II.")
    parser.add_argument("--pN", type=_positive_float, required=True, help="Fixed pN used in Stage II.")
    parser.add_argument("--seed", type=int, default=None, help="Optional user sampling seed override.")
    parser.add_argument(
        "--n-users-list",
        type=_parse_n_users_list,
        default=None,
        help="Comma-separated user sizes, e.g. 30,50,100. When set, run all sizes and draw one combined figure.",
    )
    parser.add_argument(
        "--max-iter-plot",
        type=int,
        default=None,
        help="Optional maximum iteration count to show on x-axis (helps when trajectory converges early).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_stage2_social_cost_compare_<timestamp>).",
    )
    parser.add_argument(
        "--force-enum",
        action="store_true",
        help="Force exhaustive enumeration for the centralized reference solver.",
    )
    parser.add_argument(
        "--centralized-solver",
        type=str,
        default="pyomo_scip",
        choices=["enum", "gekko", "pyomo_scip", "skip"],
        help="Centralized reference solver used for the horizontal line.",
    )
    parser.add_argument(
        "--inner-solver",
        type=str,
        default="hybrid",
        choices=["hybrid", "exact", "primal_dual"],
        help="Inner solver used in Algorithm 2.",
    )
    parser.add_argument(
        "--skip-centralized",
        action="store_true",
        help="Run only Algorithm 2 (skip centralized baseline, useful for large n).",
    )
    parser.add_argument(
        "--centralized-max-n",
        type=_positive_int,
        default=None,
        help="Optional maximum n for running the centralized reference.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)

    out_dir = resolve_out_dir("run_stage2_social_cost_compare", args.out_dir)
    n_list = args.n_users_list if args.n_users_list is not None else [cfg.n_users]

    traces_by_n: dict[int, list[float]] = {}
    centralized_by_n: dict[int, float | None] = {}
    rows = ["n_users,iteration,social_cost,centralized_social_cost"]
    summary_lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"pE = {float(args.pE):.10g}",
        f"pN = {float(args.pN):.10g}",
        f"algorithm2_inner_solver = {args.inner_solver}",
        f"centralized_mode = {args.centralized_solver}",
        f"n_users_list = {','.join(str(n) for n in n_list)}",
    ]

    for n in n_list:
        users = _sample_users_for_n(cfg, n_users=n, seed=seed + n)
        greedy_result = solve_stage2_scm(
            users=users,
            pE=float(args.pE),
            pN=float(args.pN),
            system=cfg.system,
            cfg=cfg.stackelberg,
            inner_solver_mode=args.inner_solver,
        )
        social_trace = list(greedy_result.social_cost_trace)
        centralized_social: float | None = None
        centralized_solver = "skipped"
        run_centralized = (
            (not args.skip_centralized)
            and (args.centralized_solver != "skip")
            and (args.centralized_max_n is None or int(n) <= int(args.centralized_max_n))
        )
        if run_centralized:
            if args.centralized_solver == "gekko":
                centralized = _solve_centralized_minlp(
                    users=users,
                    pE=float(args.pE),
                    pN=float(args.pN),
                    system=cfg.system,
                    stack_cfg=cfg.stackelberg,
                    base_cfg=cfg.baselines,
                )
            elif args.centralized_solver == "pyomo_scip":
                centralized = _solve_centralized_pyomo_scip(
                    users=users,
                    pE=float(args.pE),
                    pN=float(args.pN),
                    system=cfg.system,
                    stack_cfg=cfg.stackelberg,
                    base_cfg=cfg.baselines,
                )
            else:
                base_cfg = replace(cfg.baselines, cs_use_minlp=False) if args.force_enum else cfg.baselines
                centralized = baseline_stage2_centralized_solver(
                    users=users,
                    pE=float(args.pE),
                    pN=float(args.pN),
                    system=cfg.system,
                    stack_cfg=cfg.stackelberg,
                    base_cfg=base_cfg,
                )
            centralized_solver = str(centralized.meta.get("solver", "unknown"))
            success_flag = bool(centralized.meta.get("success", True))
            if success_flag:
                centralized_social = float(centralized.social_cost)
            else:
                centralized_social = None
                centralized_solver = f"{centralized_solver}_failed"

        traces_by_n[n] = social_trace
        centralized_by_n[n] = centralized_social

        if centralized_social is None:
            gap_str = "nan"
            central_str = "nan"
        else:
            gap_str = f"{float(greedy_result.social_cost - centralized_social):.10g}"
            central_str = f"{centralized_social:.10g}"
        summary_lines.extend(
            [
                f"--- n={n} ---",
                f"algorithm2_iterations = {greedy_result.iterations}",
                f"algorithm2_final_social_cost = {float(greedy_result.social_cost):.10g}",
                f"algorithm2_runtime_sec = {float(greedy_result.runtime_sec):.10g}",
                f"algorithm2_rollbacks = {greedy_result.rollback_count}",
                f"algorithm2_accepted_admissions = {greedy_result.accepted_admissions}",
                f"algorithm2_inner_calls = {greedy_result.inner_call_count}",
                f"algorithm2_final_offloading_size = {len(greedy_result.offloading_set)}",
                f"algorithm2_used_exact_inner = {int(greedy_result.used_exact_inner)}",
                f"centralized_social_cost = {central_str}",
                f"gap_algorithm2_minus_centralized = {gap_str}",
                f"centralized_solver = {centralized_solver}",
            ]
        )
        for i, value in enumerate(social_trace, start=1):
            central_csv = "" if centralized_social is None else f"{centralized_social:.10g}"
            rows.append(f"{n},{i},{float(value):.10g},{central_csv}")

    if len(n_list) == 1:
        n0 = n_list[0]
        _plot_trace(
            trace=traces_by_n[n0],
            centralized_social=centralized_by_n[n0],
            out_path=out_dir / "social_cost_trace_vs_centralized.png",
        )
    else:
        _plot_multi_n_trace(
            traces_by_n=traces_by_n,
            centralized_by_n=centralized_by_n,
            out_path=out_dir / "social_cost_trace_vs_centralized_multi_n.png",
            max_iter_plot=args.max_iter_plot,
        )

    (out_dir / "social_cost_trace.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()

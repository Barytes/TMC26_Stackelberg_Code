from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.baselines import (
    _solve_centralized_minlp,
    _solve_centralized_pyomo_scip,
    baseline_stage2_centralized_solver,
)
from tmc26_exp.config import ExperimentConfig, StackelbergConfig, SystemConfig, load_config
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    GreedySelectionResult,
    InnerSolveResult,
    _build_data,
    _heuristic_score_with_t,
    _solve_fixed_set_inner_exact,
    algorithm_1_distributed_primal_dual,
)


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _solve_inner(
    users: UserBatch,
    offloading_set: set[int],
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    inner_solver: str,
) -> InnerSolveResult:
    if inner_solver == "exact":
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is None:
            raise RuntimeError("Exact inner solver failed for a fixed set; cannot continue in exact mode.")
        return inner
    if inner_solver == "primal_dual":
        return algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
    if inner_solver == "hybrid":
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is None:
            inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
        return inner
    raise ValueError(f"Unknown inner_solver={inner_solver}")


def _social_cost_from_inner(local_costs: np.ndarray, n_users: int, inner: InnerSolveResult) -> float:
    outside = set(range(n_users)) - set(inner.offloading_set)
    return inner.offloading_objective + (float(np.sum(local_costs[list(outside)])) if outside else 0.0)


def algorithm_2_with_social_cost_trace(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    inner_solver: str = "hybrid",
) -> tuple[GreedySelectionResult, list[float]]:
    data = _build_data(users)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()

    previous_ve = 0.0
    last_added: int | None = None
    iterations = 0
    social_trace: list[float] = []

    for t in range(cfg.greedy_max_iters):
        inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
        ve = inner.offloading_objective

        if t >= 1 and last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
                ve = inner.offloading_objective

        social_trace.append(float(_social_cost_from_inner(data.cl, users.n, inner)))

        candidates = sorted(active_users - offloading_set)
        if not candidates:
            iterations = t + 1
            break

        if offloading_set:
            lambda_F, lambda_B = inner.lambda_F, inner.lambda_B
        else:
            lambda_F, lambda_B = 0.0, 0.0

        best_user = min(
            candidates,
            key=lambda j: _heuristic_score_with_t(data, j, pE + lambda_F, pN + lambda_B, system),
        )
        best_score = _heuristic_score_with_t(data, best_user, pE + lambda_F, pN + lambda_B, system)

        if best_score < 0.0:
            previous_ve = ve
            offloading_set.add(best_user)
            last_added = best_user
            iterations = t + 1
            continue

        iterations = t + 1
        break

    final_inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
    final_set = final_inner.offloading_set
    outside = set(range(users.n)) - set(final_set)
    social_cost = (
        final_inner.offloading_objective + float(np.sum(data.cl[list(outside)]))
        if outside
        else final_inner.offloading_objective
    )

    if social_trace:
        social_trace[-1] = float(social_cost)
    else:
        social_trace.append(float(social_cost))

    result = GreedySelectionResult(
        offloading_set=final_set,
        inner_result=final_inner,
        social_cost=float(social_cost),
        iterations=iterations,
    )
    return result, social_trace


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
        help="No-op compatibility flag; centralized solver is already exhaustive enumeration.",
    )
    parser.add_argument(
        "--centralized-solver",
        type=str,
        default="enum",
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else _default_output_dir(cfg.output_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
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
        greedy_result, social_trace = algorithm_2_with_social_cost_trace(
            users=users,
            pE=float(args.pE),
            pN=float(args.pN),
            system=cfg.system,
            cfg=cfg.stackelberg,
            inner_solver=args.inner_solver,
        )
        centralized_social: float | None = None
        centralized_solver = "skipped"
        run_centralized = (not args.skip_centralized) and (args.centralized_solver != "skip")
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

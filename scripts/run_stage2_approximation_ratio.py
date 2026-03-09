from __future__ import annotations

import argparse
import csv
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
from tmc26_exp.model import UserBatch, local_cost, theta
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    InnerSolveResult,
    _build_data,
    _heuristic_score_with_t,
    _solve_fixed_set_inner_exact,
    algorithm_1_distributed_primal_dual,
    algorithm_2_heuristic_user_selection,
)


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _parse_n_users_list(raw: str) -> list[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n-users-list cannot be empty.")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Each n in n-users-list must be > 0.")
    return values


def _default_output_dir(base_output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / f"run_stage2_approximation_ratio_{timestamp}"


def _axis_min_with_cap(a: np.ndarray, price: float, cap: float) -> np.ndarray:
    eps = 1e-12
    p = max(float(price), eps)
    u = max(float(cap), eps)
    a_pos = np.maximum(np.asarray(a, dtype=float), 0.0)
    x_star = np.sqrt(a_pos / p)
    unclipped = 2.0 * np.sqrt(a_pos * p)
    capped = a_pos / u + p * u
    return np.where(x_star <= u, unclipped, capped)


def _compute_delta_hat_star(users: UserBatch, pE: float, pN: float, system: SystemConfig) -> float:
    cl = local_cost(users)
    aw = users.alpha * users.w
    th = theta(users)
    min_f_term = _axis_min_with_cap(aw, pE, system.F)
    min_b_term = _axis_min_with_cap(th, pN, system.B)
    delta_hat = cl - min_f_term - min_b_term
    return float(np.max(delta_hat))


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


def _algorithm2_social_cost(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    inner_solver: str,
) -> float:
    if inner_solver == "hybrid":
        out = algorithm_2_heuristic_user_selection(users, pE, pN, system, cfg)
        return float(out.social_cost)

    data = _build_data(users)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()
    previous_ve = 0.0
    last_added: int | None = None

    for _t in range(cfg.greedy_max_iters):
        inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
        ve = inner.offloading_objective

        if last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
                ve = inner.offloading_objective

        candidates = sorted(active_users - offloading_set)
        if not candidates:
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
            continue
        break

    final_inner = _solve_inner(users, offloading_set, pE, pN, system, cfg, inner_solver)
    return float(_social_cost_from_inner(data.cl, users.n, final_inner))


def _run_centralized(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: ExperimentConfig,
    centralized_solver: str,
):
    if centralized_solver == "gekko":
        out = _solve_centralized_minlp(users, pE, pN, system, cfg.stackelberg, cfg.baselines)
        success = bool(out.meta.get("success", True))
        return out, success
    if centralized_solver == "pyomo_scip":
        out = _solve_centralized_pyomo_scip(users, pE, pN, system, cfg.stackelberg, cfg.baselines)
        success = bool(out.meta.get("success", True))
        return out, success
    if centralized_solver == "enum":
        out = baseline_stage2_centralized_solver(users, pE, pN, system, cfg.stackelberg, cfg.baselines)
        return out, True
    raise ValueError(f"Unknown centralized_solver={centralized_solver}")


def _write_points_csv(out_path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    fields = [
        "n_users",
        "trial",
        "V0",
        "V_DG",
        "V_Xstar",
        "Xstar_size",
        "delta_hat_star",
        "bound",
        "ratio",
        "slack",
        "violated",
        "valid",
        "centralized_success",
        "centralized_solver",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_points_csv(csv_path: Path) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "n_users": int(r["n_users"]),
                    "trial": int(r["trial"]),
                    "V0": float(r["V0"]) if r["V0"] else float("nan"),
                    "V_DG": float(r["V_DG"]) if r["V_DG"] else float("nan"),
                    "V_Xstar": float(r["V_Xstar"]) if r["V_Xstar"] else float("nan"),
                    "Xstar_size": int(r["Xstar_size"]),
                    "delta_hat_star": float(r["delta_hat_star"]) if r["delta_hat_star"] else float("nan"),
                    "bound": float(r["bound"]) if r["bound"] else float("nan"),
                    "ratio": float(r["ratio"]) if r["ratio"] else float("nan"),
                    "slack": float(r["slack"]) if r["slack"] else float("nan"),
                    "violated": str(r["violated"]).strip().lower() == "true",
                    "valid": str(r["valid"]).strip().lower() == "true",
                    "centralized_success": str(r["centralized_success"]).strip().lower() == "true",
                    "centralized_solver": str(r["centralized_solver"]),
                }
            )
    return rows


def _default_plot_filename(transform: str) -> str:
    if transform == "linear":
        return "approx_ratio_scatter.png"
    if transform == "logy":
        return "approx_ratio_scatter_logy.png"
    if transform == "loglog":
        return "approx_ratio_scatter_loglog.png"
    if transform == "normalized":
        return "approx_ratio_scatter_normalized.png"
    if transform == "margin":
        return "approx_ratio_scatter_margin.png"
    raise ValueError(f"Unknown transform={transform}")


def _violation_rate_str(num_violations: int, num_valid: int) -> str:
    if num_valid <= 0:
        return "nan"
    return f"{(100.0 * num_violations / num_valid):.4f}%"


def _plot_scatter(rows: list[dict[str, float | int | str | bool]], out_path: Path, transform: str = "linear") -> dict[str, int]:
    if transform not in {"linear", "logy", "loglog", "normalized", "margin"}:
        raise ValueError(f"Unknown transform={transform}")

    valid_rows = [
        r
        for r in rows
        if bool(r["valid"])
        and np.isfinite(float(r["bound"]))
        and np.isfinite(float(r["ratio"]))
    ]
    dropped_for_log = 0
    if transform in {"logy", "loglog", "normalized", "margin"}:
        filtered: list[dict[str, float | int | str | bool]] = []
        for r in valid_rows:
            if float(r["bound"]) > 0.0 and float(r["ratio"]) > 0.0:
                filtered.append(r)
            else:
                dropped_for_log += 1
        valid_rows = filtered

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=150)
    cmap = plt.get_cmap("tab10")

    ns = sorted({int(r["n_users"]) for r in rows})
    for idx, n in enumerate(ns):
        n_rows = [r for r in valid_rows if int(r["n_users"]) == n]
        if not n_rows:
            continue
        bounds_raw = np.asarray([float(r["bound"]) for r in n_rows], dtype=float)
        ratios_raw = np.asarray([float(r["ratio"]) for r in n_rows], dtype=float)
        if transform == "linear":
            x_plot = bounds_raw
            y_plot = ratios_raw
        elif transform == "logy":
            x_plot = bounds_raw
            y_plot = np.log(ratios_raw)
        elif transform == "loglog":
            x_plot = np.log(bounds_raw)
            y_plot = np.log(ratios_raw)
        elif transform == "normalized":
            x_plot = bounds_raw
            y_plot = ratios_raw / bounds_raw
        else:  # margin
            x_plot = bounds_raw
            y_plot = np.log(bounds_raw / ratios_raw)
        color = cmap(idx % 10)
        ax.scatter(x_plot, y_plot, s=34, alpha=0.9, color=color, label=f"n={n}")

    violation_rows = [r for r in valid_rows if bool(r["violated"])]
    if violation_rows:
        x_v_raw = np.asarray([float(r["bound"]) for r in violation_rows], dtype=float)
        y_v_raw = np.asarray([float(r["ratio"]) for r in violation_rows], dtype=float)
        if transform == "linear":
            x_v = x_v_raw
            y_v = y_v_raw
        elif transform == "logy":
            x_v = x_v_raw
            y_v = np.log(y_v_raw)
        elif transform == "loglog":
            x_v = np.log(x_v_raw)
            y_v = np.log(y_v_raw)
        elif transform == "normalized":
            x_v = x_v_raw
            y_v = y_v_raw / x_v_raw
        else:  # margin
            x_v = x_v_raw
            y_v = np.log(x_v_raw / y_v_raw)
        ax.scatter(x_v, y_v, s=58, marker="x", color="red", linewidths=1.1, label="Violation (ratio > bound)")

    if valid_rows:
        x_all_raw = np.asarray([float(r["bound"]) for r in valid_rows], dtype=float)
        y_all_raw = np.asarray([float(r["ratio"]) for r in valid_rows], dtype=float)
        if transform == "linear":
            x_all = x_all_raw
            y_all = y_all_raw
            line_y = lambda x: x
            line_label = "y = x"
        elif transform == "logy":
            x_all = x_all_raw
            y_all = np.log(y_all_raw)
            line_y = lambda x: np.log(np.maximum(x, 1e-12))
            line_label = "y = log(x)"
        elif transform == "loglog":
            x_all = np.log(x_all_raw)
            y_all = np.log(y_all_raw)
            line_y = lambda x: x
            line_label = "y = x"
        elif transform == "normalized":
            x_all = x_all_raw
            y_all = y_all_raw / x_all_raw
            line_y = lambda x: np.ones_like(x)
            line_label = "y = 1"
        else:  # margin
            x_all = x_all_raw
            y_all = np.log(x_all_raw / y_all_raw)
            line_y = lambda x: np.zeros_like(x)
            line_label = "y = 0"

        x_low = float(np.min(x_all))
        x_high = float(np.max(x_all))
        y_low = float(np.min(y_all))
        y_high = float(np.max(y_all))
        x_pad = 0.03 * max(x_high - x_low, 1e-6)
        y_pad = 0.08 * max(y_high - y_low, 1e-6)
        x_low -= x_pad
        x_high += x_pad
        y_low -= y_pad
        y_high += y_pad
        if transform == "normalized":
            y_low = min(y_low, 1.0)
            y_high = max(y_high, 1.0)
        if transform == "margin":
            y_low = min(y_low, 0.0)
            y_high = max(y_high, 0.0)
        xs = np.linspace(x_low, x_high, 300)
        ys = line_y(xs)
        ax.plot(xs, ys, linestyle="--", color="black", linewidth=1.3, label=line_label)
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

    if transform == "linear":
        ax.set_xlabel("Theoretical Upper Bound (RHS)")
        ax.set_ylabel("Empirical Ratio V(X_DG)/V(X*)")
        ax.set_title("Empirical Approximation Ratio vs Theoretical Bound (Stage II)")
    elif transform == "logy":
        ax.set_xlabel("Theoretical Upper Bound (RHS)")
        ax.set_ylabel("log Empirical Ratio log(V(X_DG)/V(X*))")
        ax.set_title("Empirical Approximation Ratio vs Theoretical Bound (log-y)")
    elif transform == "loglog":
        ax.set_xlabel("log Theoretical Upper Bound log(RHS)")
        ax.set_ylabel("log Empirical Ratio log(V(X_DG)/V(X*))")
        ax.set_title("Empirical Approximation Ratio vs Theoretical Bound (log-log)")
    else:
        ax.set_xlabel("Theoretical Upper Bound (RHS)")
        if transform == "normalized":
            ax.set_ylabel("Normalized Ratio V(X_DG)/(V(X*)·RHS)")
            ax.set_title("Empirical/Theoretical Ratio Normalization (Stage II)")
        else:
            ax.set_ylabel("Margin log(RHS / Empirical Ratio)")
            ax.set_title("Bound Margin log(bound/ratio) (Stage II)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"valid_rows_used": len(valid_rows), "dropped_for_log": dropped_for_log}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical verification of Stage-II approximation bound: ratio V(X_DG)/V(X*) vs theorem RHS."
    )
    parser.add_argument("--config", type=str, default="configs/final_touchup_fast.toml", help="Path to TOML config.")
    parser.add_argument("--pE", type=_positive_float, required=True, help="Fixed pE used in Stage II.")
    parser.add_argument("--pN", type=_positive_float, required=True, help="Fixed pN used in Stage II.")
    parser.add_argument(
        "--n-users-list",
        type=_parse_n_users_list,
        default=_parse_n_users_list("20,30,40,50"),
        help="Comma-separated user sizes, e.g. 20,30,40,50.",
    )
    parser.add_argument("--trials", type=_positive_int, default=20, help="Number of random user draws per n.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--inner-solver",
        type=str,
        default="primal_dual",
        choices=["hybrid", "exact", "primal_dual"],
        help="Inner solver used in Algorithm 2.",
    )
    parser.add_argument(
        "--centralized-solver",
        type=str,
        default="gekko",
        choices=["gekko", "enum", "pyomo_scip"],
        help="Centralized solver used as X* proxy.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/run_stage2_approximation_ratio_<timestamp>).",
    )
    parser.add_argument(
        "--plot-transform",
        type=str,
        default="linear",
        choices=["linear", "logy", "loglog", "normalized", "margin"],
        help="Scatter plot transform mode.",
    )
    parser.add_argument(
        "--plot-output-name",
        type=str,
        default=None,
        help="Optional output filename for scatter plot.",
    )
    parser.add_argument(
        "--replot-csv",
        type=str,
        default=None,
        help="If provided, skip optimization and replot from an existing approx_ratio_points.csv.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)
    plot_name = args.plot_output_name if args.plot_output_name is not None else _default_plot_filename(args.plot_transform)

    if args.replot_csv is not None:
        csv_path = Path(args.replot_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _read_points_csv(csv_path)
        plot_info = _plot_scatter(rows, out_dir / plot_name, transform=args.plot_transform)
        print(
            f"Done replot. Files written to: {out_dir}; "
            f"valid_rows_used={plot_info['valid_rows_used']}; dropped_for_log={plot_info['dropped_for_log']}"
        )
        return

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_output_dir(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | bool]] = []
    pE = float(args.pE)
    pN = float(args.pN)
    n_list = list(args.n_users_list)

    for n in n_list:
        cfg_n = replace(cfg, n_users=int(n))
        rng = np.random.default_rng(seed + 1009 * int(n))
        for trial in range(1, args.trials + 1):
            users = sample_users(cfg_n, rng)
            V0 = float(np.sum(local_cost(users)))
            V_DG = float(_algorithm2_social_cost(users, pE, pN, cfg.system, cfg.stackelberg, args.inner_solver))
            cs_out, cs_success = _run_centralized(users, pE, pN, cfg.system, cfg, args.centralized_solver)

            V_Xstar = float(cs_out.social_cost) if cs_success else float("nan")
            Xstar_size = int(len(cs_out.offloading_set)) if cs_success else -1
            delta_hat_star = float(_compute_delta_hat_star(users, pE, pN, cfg.system))

            bound = float("nan")
            ratio = float("nan")
            slack = float("nan")
            violated = False
            valid = False

            if cs_success and np.isfinite(V_Xstar) and V_Xstar > 0.0:
                denom = V0 - Xstar_size * delta_hat_star
                if denom > 0.0 and np.isfinite(denom):
                    bound = float((V0 - delta_hat_star) / denom)
                    ratio = float(V_DG / V_Xstar)
                    slack = float(bound - ratio)
                    if np.isfinite(bound) and np.isfinite(ratio):
                        valid = True
                        violated = bool(ratio > bound)

            rows.append(
                {
                    "n_users": int(n),
                    "trial": int(trial),
                    "V0": float(V0),
                    "V_DG": float(V_DG),
                    "V_Xstar": float(V_Xstar),
                    "Xstar_size": int(Xstar_size),
                    "delta_hat_star": float(delta_hat_star),
                    "bound": float(bound),
                    "ratio": float(ratio),
                    "slack": float(slack),
                    "violated": bool(violated),
                    "valid": bool(valid),
                    "centralized_success": bool(cs_success),
                    "centralized_solver": str(cs_out.meta.get("solver", args.centralized_solver)),
                }
            )

    csv_path = out_dir / "approx_ratio_points.csv"
    fig_path = out_dir / "approx_ratio_scatter.png"
    summary_path = out_dir / "approx_ratio_summary.txt"

    _write_points_csv(csv_path, rows)
    plot_info = _plot_scatter(rows, fig_path, transform=args.plot_transform)

    total = len(rows)
    success_count = sum(1 for r in rows if bool(r["centralized_success"]))
    valid_rows = [r for r in rows if bool(r["valid"])]
    valid_count = len(valid_rows)
    violation_count = sum(1 for r in valid_rows if bool(r["violated"]))

    summary_lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"pE = {pE:.10g}",
        f"pN = {pN:.10g}",
        f"inner_solver = {args.inner_solver}",
        f"centralized_solver = {args.centralized_solver}",
        f"plot_transform = {args.plot_transform}",
        f"plot_output_name = {plot_name}",
        f"n_users_list = {','.join(str(n) for n in n_list)}",
        f"trials_per_n = {args.trials}",
        f"total_instances = {total}",
        f"centralized_success_instances = {success_count}",
        f"valid_instances = {valid_count}",
        f"plot_valid_instances = {plot_info['valid_rows_used']}",
        f"plot_dropped_for_log = {plot_info['dropped_for_log']}",
        f"violations = {violation_count}",
        f"overall_violation_rate = {_violation_rate_str(violation_count, valid_count)}",
    ]

    for n in n_list:
        n_rows = [r for r in rows if int(r["n_users"]) == int(n)]
        n_success = sum(1 for r in n_rows if bool(r["centralized_success"]))
        n_valid = [r for r in n_rows if bool(r["valid"])]
        n_viol = sum(1 for r in n_valid if bool(r["violated"]))
        summary_lines.extend(
            [
                f"--- n={n} ---",
                f"total = {len(n_rows)}",
                f"centralized_success = {n_success}",
                f"valid = {len(n_valid)}",
                f"violations = {n_viol}",
                f"violation_rate = {_violation_rate_str(n_viol, len(n_valid))}",
            ]
        )

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()

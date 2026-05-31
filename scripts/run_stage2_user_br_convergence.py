"""Supplementary Stage-II diagnostic: sequential user best-response convergence."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figure_wrapper_utils import resolve_out_dir, write_csv_rows
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import solve_stage2_user_best_response_dynamics


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _parse_n_users_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("n-users-list must contain positive integers.")
    return values


def _summarize_by_n(rows: list[dict[str, object]], n_list: list[int]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for n in n_list:
        group = [r for r in rows if int(r["n_users"]) == int(n)]
        if not group:
            summary.append(
                {
                    "n_users": int(n),
                    "count": 0,
                    "convergence_rate": float("nan"),
                    "cycle_rate": float("nan"),
                    "mean_rounds": float("nan"),
                    "mean_updates": float("nan"),
                    "mean_final_max_gain": float("nan"),
                    "mean_social_cost_reduction": float("nan"),
                }
            )
            continue
        converged = np.asarray([int(r["converged"]) for r in group], dtype=float)
        cycles = np.asarray([int(r["cycle_detected"]) for r in group], dtype=float)
        rounds = np.asarray([float(r["rounds"]) for r in group], dtype=float)
        updates = np.asarray([float(r["updates"]) for r in group], dtype=float)
        final_gain = np.asarray([float(r["final_max_gain"]) for r in group], dtype=float)
        reduction = np.asarray([float(r["social_cost_reduction"]) for r in group], dtype=float)
        summary.append(
            {
                "n_users": int(n),
                "count": int(len(group)),
                "convergence_rate": float(np.mean(converged)),
                "cycle_rate": float(np.mean(cycles)),
                "mean_rounds": float(np.mean(rounds)),
                "mean_updates": float(np.mean(updates)),
                "mean_final_max_gain": float(np.mean(final_gain)),
                "mean_social_cost_reduction": float(np.mean(reduction)),
            }
        )
    return summary


def _plot_summary(summary: list[dict[str, object]], out_path: Path, title: str) -> None:
    valid = [r for r in summary if int(r["count"]) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=140)
    if valid:
        x = np.asarray([int(r["n_users"]) for r in valid], dtype=float)
        conv = np.asarray([float(r["convergence_rate"]) for r in valid], dtype=float)
        gain = np.asarray([float(r["mean_final_max_gain"]) for r in valid], dtype=float)
        rounds = np.asarray([float(r["mean_rounds"]) for r in valid], dtype=float)

        axes[0].plot(x, conv, "-o", linewidth=1.8, markersize=5, label="Convergence rate")
        axes[0].plot(x, rounds / max(float(np.max(rounds)), 1.0), "--s", linewidth=1.4, markersize=4, label="Rounds, normalized")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].legend(loc="lower left", fontsize=8)

        axes[1].plot(x, gain, "-o", linewidth=1.8, markersize=5, color="tab:red")
        axes[1].set_yscale("symlog", linthresh=1e-9)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No successful trials", transform=ax.transAxes, ha="center", va="center")

    axes[0].set_xlabel("Number of Users")
    axes[0].set_ylabel("Rate / Normalized Rounds")
    axes[0].set_title("Convergence")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Number of Users")
    axes[1].set_ylabel("Mean Final Max BR Gain")
    axes[1].set_title("Residual User Incentive")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run sequential Stage-II user best-response dynamics at fixed prices. "
            "This is a supplementary diagnostic and does not replace the SCM/SOE solver."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--pE", type=_positive_float, default=0.5, help="Fixed ESP price.")
    parser.add_argument("--pN", type=_positive_float, default=0.5, help="Fixed NSP price.")
    parser.add_argument(
        "--n-users-list",
        type=_parse_n_users_list,
        default=_parse_n_users_list("20,40,60,80,100"),
        help="Comma-separated user sizes.",
    )
    parser.add_argument("--trials", type=_positive_int, default=20, help="Number of random draws per n.")
    parser.add_argument("--seed", type=int, default=None, help="Optional base seed override.")
    parser.add_argument("--max-rounds", type=_positive_int, default=100, help="Maximum BR rounds per trial.")
    parser.add_argument("--tol", type=_positive_float, default=1e-8, help="Convergence tolerance for max BR gain.")
    parser.add_argument("--randomize-order", action="store_true", help="Shuffle user order independently per round.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory under outputs/.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    n_list = list(args.n_users_list)
    out_dir = resolve_out_dir("run_stage2_user_br_convergence", args.out_dir)

    trial_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    for n in n_list:
        cfg_n = replace(cfg, n_users=int(n))
        for trial in range(1, int(args.trials) + 1):
            trial_seed = int(seed + 100000 * int(n) + trial)
            try:
                users = sample_users(cfg_n, np.random.default_rng(trial_seed))
                result = solve_stage2_user_best_response_dynamics(
                    users=users,
                    pE=float(args.pE),
                    pN=float(args.pN),
                    system=cfg.system,
                    cfg=cfg.stackelberg,
                    max_rounds=int(args.max_rounds),
                    improvement_tol=float(args.tol),
                    randomize_order=bool(args.randomize_order),
                    seed=trial_seed,
                )
                initial_social = float(result.trajectory[0].social_cost)
                final_social = float(result.social_cost)
                final_step = result.trajectory[-1]
                trial_rows.append(
                    {
                        "n_users": int(n),
                        "trial": int(trial),
                        "seed": int(trial_seed),
                        "converged": int(result.converged),
                        "cycle_detected": int(result.cycle_detected),
                        "rounds": int(result.rounds),
                        "updates": int(result.updates),
                        "initial_social_cost": initial_social,
                        "final_social_cost": final_social,
                        "social_cost_reduction": float(initial_social - final_social),
                        "final_avg_gain": float(final_step.avg_gain),
                        "final_max_gain": float(final_step.max_gain),
                        "offloading_size": int(len(result.offloading_set)),
                        "total_f": float(final_step.total_f),
                        "total_b": float(final_step.total_b),
                        "runtime_sec": float(result.runtime_sec),
                        "stopping_reason": result.stopping_reason,
                        "pE": float(args.pE),
                        "pN": float(args.pN),
                        "randomize_order": int(bool(args.randomize_order)),
                    }
                )
                for step in result.trajectory:
                    trajectory_rows.append(
                        {
                            "n_users": int(n),
                            "trial": int(trial),
                            "seed": int(trial_seed),
                            "round": int(step.round),
                            "updates": int(step.updates),
                            "social_cost": float(step.social_cost),
                            "avg_gain": float(step.avg_gain),
                            "max_gain": float(step.max_gain),
                            "offloading_size": int(step.offloading_size),
                            "total_f": float(step.total_f),
                            "total_b": float(step.total_b),
                        }
                    )
            except Exception as exc:
                failed_rows.append(
                    {
                        "n_users": int(n),
                        "trial": int(trial),
                        "seed": int(trial_seed),
                        "error": str(exc).replace("\n", " "),
                    }
                )

    summary_rows = _summarize_by_n(trial_rows, n_list)

    write_csv_rows(
        out_dir / "user_br_convergence_trials.csv",
        [
            "n_users",
            "trial",
            "seed",
            "converged",
            "cycle_detected",
            "rounds",
            "updates",
            "initial_social_cost",
            "final_social_cost",
            "social_cost_reduction",
            "final_avg_gain",
            "final_max_gain",
            "offloading_size",
            "total_f",
            "total_b",
            "runtime_sec",
            "stopping_reason",
            "pE",
            "pN",
            "randomize_order",
        ],
        trial_rows,
    )
    write_csv_rows(
        out_dir / "user_br_convergence_trajectory.csv",
        [
            "n_users",
            "trial",
            "seed",
            "round",
            "updates",
            "social_cost",
            "avg_gain",
            "max_gain",
            "offloading_size",
            "total_f",
            "total_b",
        ],
        trajectory_rows,
    )
    write_csv_rows(
        out_dir / "user_br_convergence_summary.csv",
        [
            "n_users",
            "count",
            "convergence_rate",
            "cycle_rate",
            "mean_rounds",
            "mean_updates",
            "mean_final_max_gain",
            "mean_social_cost_reduction",
        ],
        summary_rows,
    )
    if failed_rows:
        write_csv_rows(out_dir / "user_br_convergence_failed_trials.csv", ["n_users", "trial", "seed", "error"], failed_rows)

    title = f"Sequential User Best-Response Dynamics (pE={float(args.pE):.3g}, pN={float(args.pN):.3g})"
    _plot_summary(summary_rows, out_dir / "user_br_convergence_summary.png", title)

    lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"pE = {float(args.pE):.10g}",
        f"pN = {float(args.pN):.10g}",
        f"n_users_list = {','.join(str(n) for n in n_list)}",
        f"trials_per_n = {int(args.trials)}",
        f"max_rounds = {int(args.max_rounds)}",
        f"tol = {float(args.tol):.10g}",
        f"randomize_order = {int(bool(args.randomize_order))}",
        f"successful_trials = {len(trial_rows)}",
        f"failed_trials = {len(failed_rows)}",
        "status = supplementary Stage-II user best-response diagnostic; not the paper-facing SCM/SOE solver",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"--- n={int(row['n_users'])} ---",
                f"count = {int(row['count'])}",
                f"convergence_rate = {float(row['convergence_rate']):.10g}",
                f"cycle_rate = {float(row['cycle_rate']):.10g}",
                f"mean_rounds = {float(row['mean_rounds']):.10g}",
                f"mean_final_max_gain = {float(row['mean_final_max_gain']):.10g}",
                f"mean_social_cost_reduction = {float(row['mean_social_cost_reduction']):.10g}",
            ]
        )
    if failed_rows:
        lines.append("--- failed_trial_details ---")
        for row in failed_rows:
            lines.append(f"n={row['n_users']}, trial={row['trial']}, seed={row['seed']}, error={row['error']}")
    (out_dir / "user_br_convergence_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()

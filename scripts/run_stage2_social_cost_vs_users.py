#!/usr/bin/env python3
"""Small Stage-II experiment: social cost vs number of users.

Compares:
- Algorithm 2 (DG)
- UBRD
- VI
- PEN

Default setup is intentionally small:
- user counts: 20, 35, 50
- trials per point: 3

Outputs are written to an independent run directory under outputs/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.baselines import (
    baseline_stage2_centralized_solver,
    baseline_stage2_penalty,
    baseline_stage2_ubrd,
    baseline_stage2_vi,
)
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import algorithm_2_heuristic_user_selection


def _parse_user_counts(raw: str) -> list[int]:
    counts = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not counts:
        raise ValueError("No user counts provided.")
    if any(x <= 0 for x in counts):
        raise ValueError("All user counts must be positive.")
    return counts


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "social_cost", "offloading_size"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["social_cost"] is None:
            continue
        key = (int(row["n_users"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    fields = [
        "n_users",
        "method",
        "social_cost_mean",
        "social_cost_std",
        "offloading_size_mean",
        "offloading_size_std",
        "count",
    ]
    summary_rows: list[dict[str, float | int | str]] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (n_users, method) in sorted(grouped.keys()):
            bucket = grouped[(n_users, method)]
            social = [float(x["social_cost"]) for x in bucket]
            off = [float(x["offloading_size"]) for x in bucket]
            social_m, social_s = _mean_std(social)
            off_m, off_s = _mean_std(off)
            row = {
                "n_users": n_users,
                "method": method,
                "social_cost_mean": social_m,
                "social_cost_std": social_s,
                "offloading_size_mean": off_m,
                "offloading_size_std": off_s,
                "count": len(bucket),
            }
            summary_rows.append(row)
            w.writerow(row)
    return summary_rows


def _plot(summary_rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    methods = ["Algorithm2", "UBRD", "VI", "PEN", "CS"]
    style = {
        "Algorithm2": {"color": "#1b5e20", "marker": "o", "label": "Algorithm 2 (DG)"},
        "UBRD": {"color": "#b71c1c", "marker": "s", "label": "UBRD"},
        "VI": {"color": "#0d47a1", "marker": "^", "label": "VI (shared multiplier)"},
        "PEN": {"color": "#e65100", "marker": "v", "label": "Penalty BRD"},
        "CS": {"color": "#4a148c", "marker": "D", "label": "CS (solver)"},
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda r: int(r["n_users"]))
        x = [int(r["n_users"]) for r in rows]
        y = [float(r["social_cost_mean"]) for r in rows]
        yerr = [float(r["social_cost_std"]) for r in rows]
        st = style[method]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=st["marker"],
            color=st["color"],
            linewidth=2.0,
            capsize=4,
            label=st["label"],
        )

    ax.set_title("Stage-II Social Cost vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Social Cost")
    ax.set_xticks(sorted({int(r["n_users"]) for r in summary_rows}))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-II social cost vs number of users")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--user-counts", type=str, default="20,35,50")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261000)
    parser.add_argument("--pE", type=float, default=None, help="Fixed pE. Defaults to stackelberg.initial_pE")
    parser.add_argument("--pN", type=float, default=None, help="Fixed pN. Defaults to stackelberg.initial_pN")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--include-cs", action="store_true", help="Include CS baseline when n_users <= exact_max_users")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")

    cfg = load_config(args.config)
    user_counts = _parse_user_counts(args.user_counts)
    pE = float(args.pE if args.pE is not None else cfg.stackelberg.initial_pE)
    pN = float(args.pN if args.pN is not None else cfg.stackelberg.initial_pN)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage2_social_cost_vs_users_small_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    skipped_cs_counts: list[int] = []
    for n_users in user_counts:
        trial_cfg = replace(cfg, n_users=n_users)
        for t in range(args.trials):
            rng = np.random.default_rng(args.seed + 1000 * n_users + t)
            users = sample_users(trial_cfg, rng)

            alg2 = algorithm_2_heuristic_user_selection(users, pE, pN, cfg.system, cfg.stackelberg)
            ubrd = baseline_stage2_ubrd(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            vi = baseline_stage2_vi(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            pen = baseline_stage2_penalty(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            cs = None
            if args.include_cs and n_users <= cfg.baselines.exact_max_users:
                cs = baseline_stage2_centralized_solver(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
            elif args.include_cs and n_users > cfg.baselines.exact_max_users and n_users not in skipped_cs_counts:
                skipped_cs_counts.append(n_users)

            raw_rows.append(
                {
                    "trial": t,
                    "seed": args.seed + 1000 * n_users + t,
                    "n_users": n_users,
                    "method": "Algorithm2",
                    "social_cost": alg2.social_cost,
                    "offloading_size": len(alg2.offloading_set),
                }
            )
            raw_rows.append(
                {
                    "trial": t,
                    "seed": args.seed + 1000 * n_users + t,
                    "n_users": n_users,
                    "method": "UBRD",
                    "social_cost": ubrd.social_cost,
                    "offloading_size": len(ubrd.offloading_set),
                }
            )
            raw_rows.append(
                {
                    "trial": t,
                    "seed": args.seed + 1000 * n_users + t,
                    "n_users": n_users,
                    "method": "VI",
                    "social_cost": vi.social_cost,
                    "offloading_size": len(vi.offloading_set),
                }
            )
            raw_rows.append(
                {
                    "trial": t,
                    "seed": args.seed + 1000 * n_users + t,
                    "n_users": n_users,
                    "method": "PEN",
                    "social_cost": pen.social_cost,
                    "offloading_size": len(pen.offloading_set),
                }
            )
            if args.include_cs:
                raw_rows.append(
                    {
                        "trial": t,
                        "seed": args.seed + 1000 * n_users + t,
                        "n_users": n_users,
                        "method": "CS",
                        "social_cost": None if cs is None else cs.social_cost,
                        "offloading_size": None if cs is None else len(cs.offloading_set),
                    }
                )

    _write_raw_csv(raw_rows, run_dir / "raw_stage2_social_cost_vs_users.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage2_social_cost_vs_users.csv")
    _plot(summary_rows, run_dir / "stage2_social_cost_vs_users.png")

    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"user_counts = {user_counts}",
        f"trials = {args.trials}",
        f"methods = {'Algorithm2, UBRD, VI, PEN, CS' if args.include_cs else 'Algorithm2, UBRD, VI, PEN'}",
        f"skipped_cs_counts = {skipped_cs_counts}",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

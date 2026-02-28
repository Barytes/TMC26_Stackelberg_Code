#!/usr/bin/env python3
"""Stage-II offloading-set size vs number of users.

Compares the final number of offloading users returned by:
- Algorithm 2
- UBRD
- VI
- PEN
- CS

`CS` is skipped automatically when `n_users > exact_max_users`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np

from _stage2_metric_helpers import METHODS, STYLE, mean_std, parse_user_counts, run_stage2_methods

ROOT = SCRIPT_DIR.parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = ["trial", "seed", "n_users", "method", "offloading_size"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["n_users"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    fields = ["n_users", "method", "offloading_size_mean", "offloading_size_std", "count"]
    summary_rows: list[dict[str, float | int | str]] = []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (n_users, method) in sorted(grouped):
            bucket = grouped[(n_users, method)]
            off = [float(x["offloading_size"]) for x in bucket]
            off_mean, off_std = mean_std(off)
            row = {
                "n_users": n_users,
                "method": method,
                "offloading_size_mean": off_mean,
                "offloading_size_std": off_std,
                "count": len(bucket),
            }
            summary_rows.append(row)
            writer.writerow(row)
    return summary_rows


def _plot(summary_rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for method in METHODS:
        rows = [row for row in summary_rows if row["method"] == method]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["n_users"]))
        x = [int(row["n_users"]) for row in rows]
        y = [float(row["offloading_size_mean"]) for row in rows]
        yerr = [float(row["offloading_size_std"]) for row in rows]
        style = STYLE[method]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            capsize=4,
            label=style["label"],
        )

    ax.set_title("Stage-II Offloading Users vs Number of Users")
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Final Offloading-set Size")
    ax.set_xticks(sorted({int(row["n_users"]) for row in summary_rows}))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-II offloading-set size vs number of users")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--user-counts", type=str, default="10,20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20265000)
    parser.add_argument("--pE", type=float, default=None)
    parser.add_argument("--pN", type=float, default=None)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")

    cfg = load_config(args.config)
    user_counts = parse_user_counts(args.user_counts)
    pE = float(args.pE if args.pE is not None else cfg.stackelberg.initial_pE)
    pN = float(args.pN if args.pN is not None else cfg.stackelberg.initial_pN)

    if not args.run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"stage2_offloading_size_vs_users_{timestamp}"
    else:
        run_name = args.run_name
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, object]] = []
    skipped_cs_counts: list[int] = []
    for n_users in user_counts:
        trial_cfg = replace(cfg, n_users=n_users)
        for trial in range(args.trials):
            seed = args.seed + 1000 * n_users + trial
            rng = np.random.default_rng(seed)
            users = sample_users(trial_cfg, rng)
            runs = run_stage2_methods(users, pE, pN, cfg)

            for method in METHODS:
                run = runs.get(method)
                if run is None:
                    if method == "CS" and n_users not in skipped_cs_counts:
                        skipped_cs_counts.append(n_users)
                    continue
                raw_rows.append(
                    {
                        "trial": trial,
                        "seed": seed,
                        "n_users": n_users,
                        "method": method,
                        "offloading_size": run.offloading_size,
                    }
                )

    _write_raw_csv(raw_rows, run_dir / "raw_stage2_offloading_size_vs_users.csv")
    summary_rows = _write_summary_csv(raw_rows, run_dir / "summary_stage2_offloading_size_vs_users.csv")
    _plot(summary_rows, run_dir / "stage2_offloading_size_vs_users.png")

    meta_lines = [
        f"timestamp_utc = {datetime.now(timezone.utc).isoformat()}",
        f"config = {args.config}",
        f"pE = {pE}",
        f"pN = {pN}",
        f"user_counts = {user_counts}",
        f"trials = {args.trials}",
        f"methods = {', '.join(METHODS)}",
        f"skipped_cs_counts = {skipped_cs_counts}",
        "metric = final offloading-set cardinality",
    ]
    (run_dir / "run_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {run_dir}")


if __name__ == "__main__":
    main()

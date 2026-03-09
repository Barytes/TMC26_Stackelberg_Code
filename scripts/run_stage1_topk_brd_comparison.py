#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.baselines import baseline_stage1_grid_search_oracle
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import algorithm_5_stackelberg_guided_search, algorithm_topk_brd_stage1


def _run_with_timing(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    dt = time.perf_counter() - t0
    return out, dt


def _dist(p, q):
    return float(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage-I topk_brd vs Algorithm 5 on same instance")
    parser.add_argument("--config", type=str, default="configs/stage1_fast_diag.toml")
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--out-subdir", type=str, default="stage1_topk_brd_comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)
    users = sample_users(cfg, np.random.default_rng(args.seed))

    topk_cfg = replace(cfg.stackelberg, gain_topk_k=4, stage1_solver_variant="topk_brd")
    alg5_cfg = replace(cfg.stackelberg, stage1_solver_variant="algorithm5")

    se = baseline_stage1_grid_search_oracle(users, cfg.system, cfg.stackelberg, cfg.baselines).price

    res_topk, dt_topk = _run_with_timing(algorithm_topk_brd_stage1, users, cfg.system, topk_cfg)
    res_alg5, dt_alg5 = _run_with_timing(algorithm_5_stackelberg_guided_search, users, cfg.system, alg5_cfg)

    out_dir = Path(cfg.output_dir) / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "steps", "final_epsilon", "dist_to_se_proxy", "runtime_sec", "stop_reason", "final_pE", "final_pN"])
        w.writerow([
            "algorithm5",
            len(res_alg5.trajectory),
            res_alg5.epsilon,
            _dist(res_alg5.price, se),
            dt_alg5,
            res_alg5.stopping_reason,
            res_alg5.price[0],
            res_alg5.price[1],
        ])
        w.writerow([
            "topk_brd",
            len(res_topk.trajectory),
            res_topk.epsilon,
            _dist(res_topk.price, se),
            dt_topk,
            res_topk.stopping_reason,
            res_topk.price[0],
            res_topk.price[1],
        ])

    # Trajectory overlay
    plt.figure(figsize=(6, 5))
    if res_alg5.trajectory:
        plt.plot([s.pE for s in res_alg5.trajectory], [s.pN for s in res_alg5.trajectory], "-o", ms=3, label="Algorithm5")
    if res_topk.trajectory:
        plt.plot([s.pE for s in res_topk.trajectory], [s.pN for s in res_topk.trajectory], "-s", ms=3, label="topk_brd")
    plt.scatter([se[0]], [se[1]], marker="*", s=120, label="SE proxy")
    plt.xlabel("pE")
    plt.ylabel("pN")
    plt.title("Stage-I price trajectory overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_overlay.png", dpi=180)
    plt.close()

    # epsilon vs iteration
    plt.figure(figsize=(6, 4))
    if res_alg5.trajectory:
        plt.plot([s.iteration for s in res_alg5.trajectory], [s.epsilon for s in res_alg5.trajectory], "-o", ms=3, label="Algorithm5")
    if res_topk.trajectory:
        plt.plot([s.iteration for s in res_topk.trajectory], [s.epsilon for s in res_topk.trajectory], "-s", ms=3, label="topk_brd")
    plt.xlabel("iteration")
    plt.ylabel("epsilon")
    plt.title("Epsilon vs iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "epsilon_vs_iteration.png", dpi=180)
    plt.close()

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()

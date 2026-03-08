#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_5_stackelberg_guided_search,
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    _candidate_family,
    _build_data,
    _refine_price_for_fixed_set,
)
from tmc26_exp.baselines import baseline_stage1_grid_search_oracle


def _eps(users, offloading_set, price, system, cfg):
    gE = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "E", system, estimator_variant=cfg.gain_estimator_variant)
    gN = algorithm_3_gain_approximation(users, offloading_set, price[0], price[1], "N", system, estimator_variant=cfg.gain_estimator_variant)
    return float(max(gE.gain, gN.gain))


def _dist(price, true_se):
    return float(((price[0] - true_se[0]) ** 2 + (price[1] - true_se[1]) ** 2) ** 0.5)


def _run_no_prio(users, system, cfg, true_se):
    p = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))
    s2 = algorithm_2_heuristic_user_selection(users, p[0], p[1], system, cfg)
    x = s2.offloading_set
    traj = []
    for t in range(cfg.search_max_iters):
        p = _refine_price_for_fixed_set(users, x, p, system)
        eps = _eps(users, x, p, system, cfg)
        traj.append((t, p[0], p[1], eps, _dist(p, true_se)))

        data = _build_data(users)
        neighborhood = [c for c in _candidate_family(data, x, p[0], p[1], system) if c != x]
        if not neighborhood:
            break
        rng = np.random.default_rng(2026 + t)
        neighborhood = list(neighborhood)
        rng.shuffle(neighborhood)
        improved = False
        for cand in neighborhood[: cfg.stage1_neighborhood_max_candidates]:
            cp = _refine_price_for_fixed_set(users, cand, p, system)
            ce = _eps(users, cand, cp, system, cfg)
            if ce + cfg.search_improvement_tol < eps:
                x, p = cand, cp
                improved = True
                break
        if not improved:
            break
    final_eps = _eps(users, x, p, system, cfg)
    return {"price": p, "epsilon": final_eps, "steps": len(traj), "trajectory": traj, "stopping_reason": "no_improving_candidate"}


def _run_neighborhood_only(users, system, cfg, true_se):
    p = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))
    s2 = algorithm_2_heuristic_user_selection(users, p[0], p[1], system, cfg)
    x = s2.offloading_set
    traj = []
    for t in range(cfg.search_max_iters):
        p = _refine_price_for_fixed_set(users, x, p, system)
        eps = _eps(users, x, p, system, cfg)
        traj.append((t, p[0], p[1], eps, _dist(p, true_se)))
        data = _build_data(users)
        neighborhood = [c for c in _candidate_family(data, x, p[0], p[1], system) if c != x]
        if not neighborhood:
            break
        best = None
        best_eps = eps
        for cand in neighborhood[: cfg.stage1_neighborhood_max_candidates]:
            cp = _refine_price_for_fixed_set(users, cand, p, system)
            ce = _eps(users, cand, cp, system, cfg)
            if ce + cfg.search_improvement_tol < best_eps:
                best = (cand, cp, ce)
                best_eps = ce
        if best is None:
            break
        x, p, _ = best
    final_eps = _eps(users, x, p, system, cfg)
    return {"price": p, "epsilon": final_eps, "steps": len(traj), "trajectory": traj, "stopping_reason": "no_improving_candidate"}


def _run_guided(users, system, cfg, true_se):
    res = algorithm_5_stackelberg_guided_search(users, system, cfg)
    traj = [(s.iteration, s.pE, s.pN, s.epsilon, _dist((s.pE, s.pN), true_se)) for s in res.trajectory]
    if not traj or (traj[-1][1], traj[-1][2]) != res.price:
        traj.append((len(traj), res.price[0], res.price[1], res.epsilon, _dist(res.price, true_se)))
    return {
        "price": res.price,
        "epsilon": float(res.epsilon),
        "steps": len(res.trajectory),
        "trajectory": traj,
        "stopping_reason": getattr(res, "stopping_reason", "unknown"),
    }


def _write_traj(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "pE", "pN", "epsilon", "dist_to_se"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_fast_diag.toml")
    ap.add_argument("--n-users", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260002)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfgu = replace(cfg, n_users=args.n_users)
    users = sample_users(cfgu, np.random.default_rng(args.seed))

    true_se = baseline_stage1_grid_search_oracle(users, cfg.system, cfg.stackelberg, cfg.baselines).price
    if not (np.isfinite(true_se[0]) and np.isfinite(true_se[1])):
        raise RuntimeError("distance metric unavailable: true SE proxy is not finite")

    run_name = args.run_name or f"stage1_alignment_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("guided", _run_guided),
        ("no_prio", _run_no_prio),
        ("neighborhood_only", _run_neighborhood_only),
    ]

    summary = []
    all_traj = {}
    for name, fn in variants:
        t0 = time.perf_counter()
        out = fn(users, cfg.system, cfg.stackelberg, true_se)
        rt = time.perf_counter() - t0
        if any(not np.isfinite(r[4]) for r in out["trajectory"]):
            raise RuntimeError(f"{name}: dist_to_se has NaN/Inf")
        _write_traj(out_dir / f"trajectory_{name}.csv", out["trajectory"])
        all_traj[name] = out["trajectory"]
        summary.append({
            "variant": name,
            "steps": out["steps"],
            "final_epsilon": out["epsilon"],
            "dist_to_se": _dist(out["price"], true_se),
            "runtime_sec": rt,
            "final_pE": out["price"][0],
            "final_pN": out["price"][1],
            "stopping_reason": out["stopping_reason"],
        })

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "steps", "final_epsilon", "dist_to_se", "runtime_sec", "final_pE", "final_pN", "stopping_reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    colors = {"guided": "#1976d2", "no_prio": "#d32f2f", "neighborhood_only": "#388e3c"}
    for name, traj in all_traj.items():
        xs = [r[1] for r in traj]
        ys = [r[2] for r in traj]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, label=name, color=colors.get(name))
    ax.scatter([true_se[0]], [true_se[1]], marker="X", s=120, color="#ff1744", edgecolors="black", label="true_se")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    ax.set_title("Stage1 convergence trajectories (single instance)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure6_alignment_overlay.png")
    plt.close(fig)

    lines = [
        f"config={args.config}",
        f"seed={args.seed}",
        f"n_users={args.n_users}",
        f"true_se=({true_se[0]:.10g},{true_se[1]:.10g})",
        "strict_mode=single_config_single_seed_single_instance",
    ]
    (out_dir / "run_meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

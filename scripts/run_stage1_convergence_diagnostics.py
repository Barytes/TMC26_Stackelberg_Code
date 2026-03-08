#!/usr/bin/env python3
from __future__ import annotations
import csv
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    algorithm_5_stackelberg_guided_search,
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    _refine_price_for_fixed_set,
)
from tmc26_exp.baselines import baseline_stage1_grid_search_oracle


def run_variant_sync(users, system, cfg):
    p = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))
    s2 = algorithm_2_heuristic_user_selection(users, p[0], p[1], system, cfg)
    X = s2.offloading_set
    traj = []
    stop = 'max_iters'
    for t in range(cfg.search_max_iters):
        p = _refine_price_for_fixed_set(users, X, p, system)
        s2 = algorithm_2_heuristic_user_selection(users, p[0], p[1], system, cfg)
        X = s2.offloading_set
        gE = algorithm_3_gain_approximation(users, X, p[0], p[1], 'E', system)
        gN = algorithm_3_gain_approximation(users, X, p[0], p[1], 'N', system)
        eps = max(gE.gain, gN.gain)
        traj.append((t, p[0], p[1], eps))
        if t > 0 and abs(traj[-1][3] - traj[-2][3]) < 1e-8:
            stop = 'epsilon_plateau'
            break
    return traj, stop


def main():
    cfg = load_config('configs/stage1_fast_diag.toml')
    users = sample_users(cfg, np.random.default_rng(20260002))
    se = baseline_stage1_grid_search_oracle(users, cfg.system, cfg.stackelberg, cfg.baselines).price

    a = algorithm_5_stackelberg_guided_search(users, cfg.system, cfg.stackelberg)
    b_traj, b_stop = run_variant_sync(users, cfg.system, cfg.stackelberg)

    out_dir = Path(cfg.output_dir) / 'stage1_convergence_diag_ab'
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / 'variant_a_minimal.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['iter','pE','pN','epsilon'])
        for s in a.trajectory:
            w.writerow([s.iteration, s.pE, s.pN, s.epsilon])

    with (out_dir / 'variant_b_sync.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['iter','pE','pN','epsilon'])
        w.writerows(b_traj)

    def dist(p):
        return float(((p[0]-se[0])**2 + (p[1]-se[1])**2) ** 0.5)

    a_dist = dist(a.price)
    b_final = (b_traj[-1][1], b_traj[-1][2]) if b_traj else (np.nan, np.nan)
    b_eps = b_traj[-1][3] if b_traj else float('nan')
    b_dist = dist(b_final) if b_traj else float('nan')

    lines = [
        f'true_se_price = {se}',
        f'variant_a_steps = {len(a.trajectory)}',
        f'variant_a_final_price = {a.price}',
        f'variant_a_final_epsilon = {a.epsilon}',
        f'variant_a_dist_to_true_se = {a_dist}',
        f'variant_a_stopping_reason = {a.stopping_reason}',
        f'variant_b_steps = {len(b_traj)}',
        f'variant_b_final_price = {b_final}',
        f'variant_b_final_epsilon = {b_eps}',
        f'variant_b_dist_to_true_se = {b_dist}',
        f'variant_b_stopping_reason = {b_stop}',
    ]
    (out_dir / 'ab_summary.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', out_dir)


if __name__ == '__main__':
    main()

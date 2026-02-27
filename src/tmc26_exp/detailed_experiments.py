from __future__ import annotations

from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .baselines import BaselineOutcome, proposed_gsse, run_all_baselines
from .config import load_config
from .experiment_plan import build_detailed_plan
from .simulator import sample_users
from .stackelberg import algorithm_3_gain_approximation, algorithm_5_stackelberg_guided_search


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row[k]) for k in keys))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_row(seed: int, out: BaselineOutcome, runtime_sec: float) -> dict[str, object]:
    return {
        "seed": seed,
        "method": out.name,
        "pE": out.price[0],
        "pN": out.price[1],
        "offloading_size": len(out.offloading_set),
        "social_cost": out.social_cost,
        "esp_revenue": out.esp_revenue,
        "nsp_revenue": out.nsp_revenue,
        "epsilon_proxy": out.epsilon_proxy,
        "runtime_sec": runtime_sec,
        "meta": json.dumps(out.meta, ensure_ascii=True),
    }


def run_detailed_experiments(config_path: str, out_dir: Path) -> None:
    cfg = load_config(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    plan = build_detailed_plan(cfg)
    (out_dir / "plan_snapshot.json").write_text(json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")

    # A1: core comparison
    rows_a1: list[dict[str, object]] = []
    for trial in range(cfg.detailed_experiment.suggested_trials):
        users = sample_users(cfg, rng)
        t0 = time.perf_counter()
        outcomes = run_all_baselines(users, cfg.system, cfg.stackelberg, cfg.baselines)
        dt = time.perf_counter() - t0
        per = max(dt / max(len(outcomes), 1), 1e-9)
        for out in outcomes:
            rows_a1.append(_to_row(cfg.seed + trial, out, per))
    _write_csv(rows_a1, out_dir / "A1_core_comparison.csv")

    # A2: approximation fidelity on small instances (proxy exact via exhaustive price-deviation grid)
    rows_a2: list[dict[str, object]] = []
    if cfg.n_users <= cfg.baselines.exact_max_users:
        users = sample_users(cfg, rng)
        gsse = algorithm_5_stackelberg_guided_search(users, cfg.system, cfg.stackelberg)
        pE, pN = gsse.price
        approx_E = algorithm_3_gain_approximation(users, gsse.offloading_set, pE, pN, "E", cfg.system).gain
        approx_N = algorithm_3_gain_approximation(users, gsse.offloading_set, pE, pN, "N", cfg.system).gain

        grid_E = np.linspace(cfg.system.cE, cfg.baselines.max_price_E, cfg.baselines.gso_grid_points)
        grid_N = np.linspace(cfg.system.cN, cfg.baselines.max_price_N, cfg.baselines.gso_grid_points)
        # Proxy exact gain by full grid over one-dimensional unilateral deviations.
        exact_E = 0.0
        exact_N = 0.0
        from .baselines import _stage2_solver  # local import to avoid public API noise

        cur = _stage2_solver(
            cfg.baselines.stage2_solver_for_pricing,
            users,
            pE,
            pN,
            cfg.system,
            cfg.stackelberg,
            cfg.baselines,
        )
        for p in grid_E:
            cand = _stage2_solver(
                cfg.baselines.stage2_solver_for_pricing,
                users,
                float(p),
                pN,
                cfg.system,
                cfg.stackelberg,
                cfg.baselines,
            )
            exact_E = max(exact_E, cand.esp_revenue - cur.esp_revenue)
        for p in grid_N:
            cand = _stage2_solver(
                cfg.baselines.stage2_solver_for_pricing,
                users,
                pE,
                float(p),
                cfg.system,
                cfg.stackelberg,
                cfg.baselines,
            )
            exact_N = max(exact_N, cand.nsp_revenue - cur.nsp_revenue)
        rows_a2.append(
            {
                "approx_gain_E": approx_E,
                "exact_gain_E_proxy": exact_E,
                "abs_err_E": abs(approx_E - exact_E),
                "approx_gain_N": approx_N,
                "exact_gain_N_proxy": exact_N,
                "abs_err_N": abs(approx_N - exact_N),
            }
        )
    _write_csv(rows_a2, out_dir / "A2_gain_fidelity.csv")

    # A3: L sensitivity for Algorithm 4/5
    rows_a3: list[dict[str, object]] = []
    users = sample_users(cfg, rng)
    for L in [4, 8, 12, 20, 32, 48]:
        local_stack = cfg.stackelberg.__class__(**{**cfg.stackelberg.__dict__, "rne_directions": L})
        t0 = time.perf_counter()
        res = algorithm_5_stackelberg_guided_search(users, cfg.system, local_stack)
        dt = time.perf_counter() - t0
        rows_a3.append(
            {
                "L": L,
                "epsilon_proxy": res.epsilon,
                "revenue_sum": res.gain_E.current_revenue + res.gain_N.current_revenue,
                "social_cost": res.social_cost,
                "runtime_sec": dt,
            }
        )
    _write_csv(rows_a3, out_dir / "A3_L_sensitivity.csv")

    # A4: candidate family ablation proxy by varying stage-II solver in pricing routines
    rows_a4: list[dict[str, object]] = []
    for solver in ["DG", "UBRD", "URA"]:
        local_base = cfg.baselines.__class__(**{**cfg.baselines.__dict__, "stage2_solver_for_pricing": solver})
        users = sample_users(cfg, rng)
        t0 = time.perf_counter()
        outs = run_all_baselines(users, cfg.system, cfg.stackelberg, local_base)
        dt = time.perf_counter() - t0
        g = [o for o in outs if o.name == "GSSE"][0]
        rows_a4.append(
            {
                "pricing_stage2_solver": solver,
                "epsilon_proxy": g.epsilon_proxy,
                "social_cost": g.social_cost,
                "runtime_sec": dt,
            }
        )
    _write_csv(rows_a4, out_dir / "A4_candidate_family_ablation.csv")

    # A5: scalability
    rows_a5: list[dict[str, object]] = []
    base_n = cfg.n_users
    for n in [20, 40, 80, 120, 200]:
        tmp_cfg = cfg.__class__(**{**cfg.__dict__, "n_users": n})
        users = sample_users(tmp_cfg, rng)
        t0 = time.perf_counter()
        out = proposed_gsse(users, cfg.system, cfg.stackelberg)
        dt = time.perf_counter() - t0
        rows_a5.append(
            {
                "n_users": n,
                "runtime_sec": dt,
                "epsilon_proxy": out.epsilon_proxy,
                "social_cost": out.social_cost,
                "offloading_size": len(out.offloading_set),
            }
        )
    _write_csv(rows_a5, out_dir / "A5_scalability.csv")

    # A6: robustness sweeps
    rows_a6: list[dict[str, object]] = []
    for F_scale, B_scale in itertools.product([0.5, 1.0, 1.5], [0.5, 1.0, 1.5]):
        system = cfg.system.__class__(
            F=cfg.system.F * F_scale,
            B=cfg.system.B * B_scale,
            cE=cfg.system.cE,
            cN=cfg.system.cN,
        )
        users = sample_users(cfg, rng)
        out = proposed_gsse(users, system, cfg.stackelberg)
        rows_a6.append(
            {
                "F_scale": F_scale,
                "B_scale": B_scale,
                "social_cost": out.social_cost,
                "revenue_sum": out.esp_revenue + out.nsp_revenue,
                "offloading_size": len(out.offloading_set),
            }
        )
    _write_csv(rows_a6, out_dir / "A6_robustness.csv")

    # A7: convergence traces
    rows_a7: list[dict[str, object]] = []
    users = sample_users(cfg, rng)
    res = algorithm_5_stackelberg_guided_search(users, cfg.system, cfg.stackelberg)
    for step in res.trajectory:
        rows_a7.append(
            {
                "iter": step.iteration,
                "pE": step.pE,
                "pN": step.pN,
                "epsilon_proxy": step.epsilon,
                "offloading_size": len(step.offloading_set),
            }
        )
    _write_csv(rows_a7, out_dir / "A7_convergence_trace.csv")


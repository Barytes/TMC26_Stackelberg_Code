from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", "/tmp/tmc26_cache"))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")
import numpy as np

import run_stage1_final_grid_ne_gap_vs_users as quality
from _figure_missing_impl import _load_cfg, _sample_users, _write_summary
from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
from tmc26_exp.baselines import (
    BaselineOutcome,
    _gp_predict,
    _grid_ne_gap_audit,
    _joint_action_q_greedy_action,
    _objective_is_better,
    _price_cache_key,
    _stage2_solver,
)

TRIAL_FIELDS = [
    "method",
    "n_users",
    "trial",
    "source",
    "success",
    "budget_stage2_calls",
    "search_budget_exhausted",
    "budget_stop_mode",
    "final_pE",
    "final_pN",
    "offloading_size",
    "final_grid_ne_gap",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "runtime_sec",
    "stage2_solver_calls",
    "audit_stage2_solver_calls",
    "total_stage2_solver_calls",
    "error",
]


class Stage2BudgetReached(RuntimeError):
    pass


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _parse_n_users_list(raw: str) -> list[int]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n-users-list cannot be empty.")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Each n in n-users-list must be > 0.")
    return values


def _parse_methods(raw: str) -> list[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("methods cannot be empty.")
    allowed = {
        "bo-online": "BO-online",
        "bo_online": "BO-online",
        "ga": "GA",
        "marl": "MARL",
    }
    methods: list[str] = []
    for item in items:
        key = item.lower()
        if key not in allowed:
            raise argparse.ArgumentTypeError("Allowed methods: BO-online, GA, MARL.")
        methods.append(allowed[key])
    return methods


def _load_reference_budgets(reference_csv: Path, n_list: list[int], trials: int) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    rows = list(load_csv_rows(reference_csv))
    proposed_rows = [row for row in rows if str(row["method"]) == "Proposed"]
    budget_map: dict[tuple[int, int], int] = {}
    for row in proposed_rows:
        budget_map[(int(row["n_users"]), int(row["trial"]))] = int(row["stage2_solver_calls"])
    missing = [
        (n, trial)
        for n in n_list
        for trial in range(1, trials + 1)
        if (int(n), int(trial)) not in budget_map
    ]
    if missing:
        raise ValueError(f"Reference proposed CSV missing budgets for: {missing[:5]}")
    return proposed_rows, budget_map


def _make_budgeted_stage2_accessor(users, system, stack_cfg, base_cfg, budget: int):
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}

    def cached_stage2(pE: float, pN: float) -> BaselineOutcome:
        key = _price_cache_key(pE, pN)
        if key in stage2_cache:
            return stage2_cache[key]
        if len(stage2_cache) >= int(budget):
            raise Stage2BudgetReached
        stage2_cache[key] = _stage2_solver(
            base_cfg.stage2_solver_for_pricing,
            users,
            float(pE),
            float(pN),
            system,
            stack_cfg,
            base_cfg,
        )
        return stage2_cache[key]

    return stage2_cache, cached_stage2


def _budgeted_grid_gap_candidate(
    out: BaselineOutcome,
    *,
    users,
    system,
    stack_cfg,
    base_cfg,
    cached_stage2,
    pE_audit_grid: np.ndarray,
    pN_audit_grid: np.ndarray,
) -> tuple[float, BaselineOutcome]:
    best_esp_rev = float(out.esp_revenue)
    best_nsp_rev = float(out.nsp_revenue)
    pE_cur, pN_cur = float(out.price[0]), float(out.price[1])

    for cand_pE in pE_audit_grid:
        cand_out = cached_stage2(float(cand_pE), pN_cur)
        if cand_out.esp_revenue > best_esp_rev:
            best_esp_rev = float(cand_out.esp_revenue)

    for cand_pN in pN_audit_grid:
        cand_out = cached_stage2(pE_cur, float(cand_pN))
        if cand_out.nsp_revenue > best_nsp_rev:
            best_nsp_rev = float(cand_out.nsp_revenue)

    gap = float(max(best_esp_rev - float(out.esp_revenue), best_nsp_rev - float(out.nsp_revenue)))
    return gap, replace(out, grid_ne_gap=gap)


def _run_budgeted_bo_online(users, system, stack_cfg, base_cfg, budget: int) -> tuple[BaselineOutcome, dict[str, object]]:
    t0 = time.perf_counter()
    stage2_cache, cached_stage2 = _make_budgeted_stage2_accessor(users, system, stack_cfg, base_cfg, budget)
    rng = np.random.default_rng(base_cfg.random_seed + 151)
    pE_min, pE_max = float(system.cE), float(base_cfg.max_price_E)
    pN_min, pN_max = float(system.cN), float(base_cfg.max_price_N)
    pE_audit_grid = np.linspace(pE_min, pE_max, max(2, int(base_cfg.gso_grid_points)))
    pN_audit_grid = np.linspace(pN_min, pN_max, max(2, int(base_cfg.gso_grid_points)))
    candidate_pool = max(1, int(base_cfg.bo_candidate_pool))
    n_iters = max(0, int(base_cfg.bo_iters))

    x: list[tuple[float, float]] = []
    y: list[float] = []
    best: BaselineOutcome | None = None
    best_score: float | None = None
    last_direct: BaselineOutcome | None = None
    exhausted = False
    stop_mode = "completed"

    def evaluate(pE: float, pN: float) -> bool:
        nonlocal best, best_score, last_direct, exhausted, stop_mode
        try:
            out = cached_stage2(float(pE), float(pN))
            last_direct = out
            score, candidate = _budgeted_grid_gap_candidate(
                out,
                users=users,
                system=system,
                stack_cfg=stack_cfg,
                base_cfg=base_cfg,
                cached_stage2=cached_stage2,
                pE_audit_grid=pE_audit_grid,
                pN_audit_grid=pN_audit_grid,
            )
        except Stage2BudgetReached:
            exhausted = True
            stop_mode = "budget_during_objective"
            return False
        x.append((float(pE), float(pN)))
        y.append(float(score))
        if _objective_is_better(score, candidate, best_score, best):
            best = candidate
            best_score = float(score)
        if len(stage2_cache) >= int(budget):
            exhausted = True
            stop_mode = "budget_after_objective"
            return False
        return True

    start_pE = min(max(float(stack_cfg.initial_pE), pE_min), pE_max)
    start_pN = min(max(float(stack_cfg.initial_pN), pN_min), pN_max)
    if evaluate(start_pE, start_pN):
        for t in range(n_iters):
            cand = np.column_stack(
                [
                    rng.uniform(pE_min, pE_max, size=candidate_pool),
                    rng.uniform(pN_min, pN_max, size=candidate_pool),
                ]
            )
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mu, sigma = _gp_predict(x_arr, y_arr, cand, length_scale=base_cfg.bo_kernel_bandwidth)
            beta = max(float(base_cfg.bo_ucb_beta), 0.0) * np.sqrt(2.0 * np.log(t + 2.0))
            acquisition = mu - beta * sigma
            order = np.argsort(acquisition)
            pick = int(order[0])
            for idx in order:
                d2 = np.sum((x_arr - cand[int(idx)]) ** 2, axis=1)
                if float(np.min(d2)) > 1e-12:
                    pick = int(idx)
                    break
            if not evaluate(float(cand[pick, 0]), float(cand[pick, 1])):
                break

    out = best if best is not None else last_direct
    if out is None:
        raise RuntimeError("BO-online budgeted run failed to evaluate any direct price.")
    meta = {
        "runtime_sec": float(time.perf_counter() - t0),
        "stage2_calls": int(len(stage2_cache)),
        "budget_stage2_calls": int(budget),
        "budget_exhausted": bool(exhausted),
        "budget_stop_mode": str(stop_mode),
        "direct_evals_completed": int(len(x)),
    }
    return out, meta


def _run_budgeted_ga(users, system, stack_cfg, base_cfg, budget: int) -> tuple[BaselineOutcome, dict[str, object]]:
    t0 = time.perf_counter()
    stage2_cache, cached_stage2 = _make_budgeted_stage2_accessor(users, system, stack_cfg, base_cfg, budget)
    rng = np.random.default_rng(base_cfg.random_seed + 401)
    n_pop = max(2, int(base_cfg.ga_population_size))
    n_gen = max(0, int(base_cfg.ga_generations))
    elite_size = min(max(1, int(base_cfg.ga_elite_size)), n_pop)
    tournament_size = min(max(1, int(base_cfg.ga_tournament_size)), n_pop)
    crossover_rate = min(max(float(base_cfg.ga_crossover_rate), 0.0), 1.0)
    mutation_rate = min(max(float(base_cfg.ga_mutation_rate), 0.0), 1.0)
    mutation_std = max(float(base_cfg.ga_mutation_std), 1e-9)
    pE_min, pE_max = float(system.cE), float(base_cfg.max_price_E)
    pN_min, pN_max = float(system.cN), float(base_cfg.max_price_N)
    span_E = max(pE_max - pE_min, 1e-9)
    span_N = max(pN_max - pN_min, 1e-9)
    pE_audit_grid = np.linspace(pE_min, pE_max, max(2, int(base_cfg.gso_grid_points)))
    pN_audit_grid = np.linspace(pN_min, pN_max, max(2, int(base_cfg.gso_grid_points)))

    best: BaselineOutcome | None = None
    best_score: float | None = None
    best_price: tuple[float, float] | None = None
    last_direct: BaselineOutcome | None = None
    evals = 0
    exhausted = False
    stop_mode = "completed"

    def _clip_price(pE: float, pN: float) -> tuple[float, float]:
        return (
            round(min(max(float(pE), pE_min), pE_max), 6),
            round(min(max(float(pN), pN_min), pN_max), 6),
        )

    def _evaluate_individual(pE: float, pN: float) -> tuple[tuple[float, float], float, BaselineOutcome] | None:
        nonlocal best, best_score, best_price, last_direct, evals, exhausted, stop_mode
        pE, pN = _clip_price(pE, pN)
        try:
            out = cached_stage2(pE, pN)
            last_direct = out
            evals += 1
            score, candidate = _budgeted_grid_gap_candidate(
                out,
                users=users,
                system=system,
                stack_cfg=stack_cfg,
                base_cfg=base_cfg,
                cached_stage2=cached_stage2,
                pE_audit_grid=pE_audit_grid,
                pN_audit_grid=pN_audit_grid,
            )
        except Stage2BudgetReached:
            exhausted = True
            stop_mode = "budget_during_objective"
            return None
        if _objective_is_better(score, candidate, best_score, best):
            best = candidate
            best_score = float(score)
            best_price = (pE, pN)
        if len(stage2_cache) >= int(budget):
            exhausted = True
            stop_mode = "budget_after_objective"
        return (pE, pN), float(score), candidate

    def _tournament_select(population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, population.shape[0], size=tournament_size)
        best_idx = int(idx[int(np.argmin(scores[idx]))])
        return population[best_idx].copy()

    def _crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if float(rng.uniform()) >= crossover_rate:
            return parent_a.copy(), parent_b.copy()
        lam = float(rng.uniform())
        child_a = lam * parent_a + (1.0 - lam) * parent_b
        child_b = lam * parent_b + (1.0 - lam) * parent_a
        return child_a, child_b

    def _mutate(child: np.ndarray) -> np.ndarray:
        out = child.copy()
        if float(rng.uniform()) < mutation_rate:
            out[0] += float(rng.normal(0.0, mutation_std * span_E))
        if float(rng.uniform()) < mutation_rate:
            out[1] += float(rng.normal(0.0, mutation_std * span_N))
        out[0], out[1] = _clip_price(out[0], out[1])
        return out

    population = np.column_stack(
        [
            rng.uniform(pE_min, pE_max, size=n_pop),
            rng.uniform(pN_min, pN_max, size=n_pop),
        ]
    )
    population[0, 0] = min(max(float(stack_cfg.initial_pE), pE_min), pE_max)
    population[0, 1] = min(max(float(stack_cfg.initial_pN), pN_min), pN_max)

    def _evaluate_population(pop: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        scored_prices: list[tuple[float, float]] = []
        score_values: list[float] = []
        for pE, pN in pop:
            result = _evaluate_individual(float(pE), float(pN))
            if result is None:
                return None
            price, score, _ = result
            scored_prices.append(price)
            score_values.append(score)
            if exhausted:
                return np.asarray(scored_prices, dtype=float), np.asarray(score_values, dtype=float)
        return np.asarray(scored_prices, dtype=float), np.asarray(score_values, dtype=float)

    initial = _evaluate_population(population)
    if initial is not None:
        population, scores = initial
        if not exhausted:
            for _ in range(n_gen):
                order = np.argsort(scores)
                next_population: list[np.ndarray] = [population[int(i)].copy() for i in order[: min(elite_size, population.shape[0])]]
                while len(next_population) < n_pop:
                    parent_a = _tournament_select(population, scores)
                    parent_b = _tournament_select(population, scores)
                    child_a, child_b = _crossover(parent_a, parent_b)
                    next_population.append(_mutate(child_a))
                    if len(next_population) < n_pop:
                        next_population.append(_mutate(child_b))
                pop_eval = _evaluate_population(np.asarray(next_population, dtype=float))
                if pop_eval is None:
                    break
                population, scores = pop_eval
                if exhausted:
                    break

    out = best if best is not None else last_direct
    if out is None:
        raise RuntimeError("GA budgeted run failed to evaluate any direct price.")
    meta = {
        "runtime_sec": float(time.perf_counter() - t0),
        "stage2_calls": int(len(stage2_cache)),
        "budget_stage2_calls": int(budget),
        "budget_exhausted": bool(exhausted),
        "budget_stop_mode": str(stop_mode),
        "direct_evals_completed": int(evals),
        "best_price_found": "" if best_price is None else f"{best_price[0]:.6g},{best_price[1]:.6g}",
    }
    return out, meta


def _run_budgeted_marl(users, system, stack_cfg, base_cfg, budget: int) -> tuple[BaselineOutcome, dict[str, object]]:
    t0 = time.perf_counter()
    grid_e = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.marl_price_levels)
    grid_n = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.marl_price_levels)
    q_esp = np.zeros((grid_e.size, grid_n.size), dtype=float)
    q_nsp = np.zeros((grid_e.size, grid_n.size), dtype=float)
    rng = np.random.default_rng(base_cfg.random_seed + 303)
    stage2_cache: dict[tuple[int, int], BaselineOutcome] = {}
    total_updates = 0
    exhausted = False
    stop_mode = "completed"

    def joint_action_outcome(e_idx: int, n_idx: int) -> BaselineOutcome:
        key = (int(e_idx), int(n_idx))
        if key not in stage2_cache:
            if len(stage2_cache) >= int(budget):
                raise Stage2BudgetReached
            stage2_cache[key] = _stage2_solver(
                base_cfg.stage2_solver_for_pricing,
                users,
                float(grid_e[key[0]]),
                float(grid_n[key[1]]),
                system,
                stack_cfg,
                base_cfg,
            )
        return stage2_cache[key]

    for _ in range(base_cfg.marl_episodes):
        for _step in range(base_cfg.marl_steps_per_episode):
            greedy_e, greedy_n, _ = _joint_action_q_greedy_action(q_esp, q_nsp)
            if rng.uniform() < base_cfg.marl_epsilon:
                a_esp = int(rng.integers(0, grid_e.size))
            else:
                a_esp = int(greedy_e)
            if rng.uniform() < base_cfg.marl_epsilon:
                a_nsp = int(rng.integers(0, grid_n.size))
            else:
                a_nsp = int(greedy_n)

            try:
                out = joint_action_outcome(a_esp, a_nsp)
            except Stage2BudgetReached:
                exhausted = True
                stop_mode = "budget_before_new_action"
                break

            reward_esp = float(out.esp_revenue)
            reward_nsp = float(out.nsp_revenue)
            next_a_esp, next_a_nsp, _ = _joint_action_q_greedy_action(q_esp, q_nsp)
            td_target_esp = reward_esp + base_cfg.marl_gamma * float(q_esp[next_a_esp, next_a_nsp])
            td_target_nsp = reward_nsp + base_cfg.marl_gamma * float(q_nsp[next_a_esp, next_a_nsp])
            q_esp[a_esp, a_nsp] += base_cfg.marl_alpha * (td_target_esp - q_esp[a_esp, a_nsp])
            q_nsp[a_esp, a_nsp] += base_cfg.marl_alpha * (td_target_nsp - q_nsp[a_esp, a_nsp])
            total_updates += 1

            if len(stage2_cache) >= int(budget):
                exhausted = True
                stop_mode = "budget_after_new_action"
                break
        if exhausted:
            break

    final_e, final_n, final_has_pure_nash = _joint_action_q_greedy_action(q_esp, q_nsp)
    final_key = (int(final_e), int(final_n))
    if final_key in stage2_cache:
        out = stage2_cache[final_key]
    elif stage2_cache:
        cached_keys = list(stage2_cache.keys())
        scores = np.asarray([q_esp[i, j] + q_nsp[i, j] for i, j in cached_keys], dtype=float)
        pick = cached_keys[int(np.argmax(scores))]
        out = stage2_cache[pick]
    else:
        raise RuntimeError("MARL budgeted run did not evaluate any action before budget stop.")

    meta = {
        "runtime_sec": float(time.perf_counter() - t0),
        "stage2_calls": int(len(stage2_cache)),
        "budget_stage2_calls": int(budget),
        "budget_exhausted": bool(exhausted),
        "budget_stop_mode": str(stop_mode),
        "training_updates": int(total_updates),
        "final_policy_has_pure_nash": int(final_has_pure_nash),
    }
    return out, meta


def _posthoc_final_gap(out: BaselineOutcome, *, users, system, stack_cfg, base_cfg, audit_points: int) -> tuple[float, int]:
    pE_grid = np.linspace(float(system.cE), float(base_cfg.max_price_E), max(2, int(audit_points)))
    pN_grid = np.linspace(float(system.cN), float(base_cfg.max_price_N), max(2, int(audit_points)))
    audit_cache = {_price_cache_key(float(out.price[0]), float(out.price[1])): out}
    gap = _grid_ne_gap_audit(out, users, system, stack_cfg, base_cfg, audit_cache, pE_grid, pN_grid)
    return float(gap), int(max(0, len(audit_cache) - 1))


def _budgeted_baseline_outcome(method: str, *, users, system, stack_cfg, base_cfg, budget: int) -> tuple[BaselineOutcome, dict[str, object]]:
    if method == "BO-online":
        return _run_budgeted_bo_online(users, system, stack_cfg, base_cfg, budget)
    if method == "GA":
        return _run_budgeted_ga(users, system, stack_cfg, base_cfg, budget)
    if method == "MARL":
        return _run_budgeted_marl(users, system, stack_cfg, base_cfg, budget)
    raise ValueError(f"Unsupported method={method}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse an existing Proposed run as the Stage-II-call budget source, rerun BO-online/GA/MARL "
            "with per-trial matched budgets, and record the posthoc final gap and revenue at cutoff."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=_parse_n_users_list, default="10,15,20,25,30")
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument("--methods", type=_parse_methods, default="BO-online,GA,MARL")
    parser.add_argument("--reference-run-dir", type=str, required=True)
    parser.add_argument("--final-audit-grid-points", type=_positive_int, default=120)
    parser.add_argument("--bo-candidate-pool", type=_positive_int, default=48)
    parser.add_argument("--bo-iters", type=int, default=20)
    parser.add_argument("--ga-population-size", type=_positive_int, default=12)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--marl-price-levels", type=_positive_int, default=11)
    parser.add_argument("--marl-episodes", type=_positive_int, default=60)
    parser.add_argument("--marl-steps-per-episode", type=_positive_int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    n_list = list(args.n_users_list) if not isinstance(args.n_users_list, str) else _parse_n_users_list(args.n_users_list)
    methods = list(args.methods) if not isinstance(args.methods, str) else _parse_methods(args.methods)
    reference_dir = Path(args.reference_run_dir)
    reference_csv = reference_dir / "stage1_final_grid_ne_gap_vs_users.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    proposed_rows, budget_map = _load_reference_budgets(reference_csv, n_list, int(args.trials))
    cfg = _load_cfg(str(args.config))
    out_dir = resolve_out_dir("run_stage1_budget_matched_quality_vs_users", args.out_dir)

    rows: list[dict[str, object]] = []
    proposed_subset = [
        row
        for row in proposed_rows
        if int(row["n_users"]) in n_list and 1 <= int(row["trial"]) <= int(args.trials)
    ]
    for row in proposed_subset:
        rows.append(
            {
                "method": "Proposed",
                "n_users": int(row["n_users"]),
                "trial": int(row["trial"]),
                "source": "reused_reference",
                "success": int(row["success"]),
                "budget_stage2_calls": int(row["stage2_solver_calls"]),
                "search_budget_exhausted": 0,
                "budget_stop_mode": "reference_reuse",
                "final_pE": float(row["final_pE"]),
                "final_pN": float(row["final_pN"]),
                "offloading_size": int(row["offloading_size"]),
                "final_grid_ne_gap": float(row["final_grid_ne_gap"]),
                "esp_revenue": float(row["esp_revenue"]),
                "nsp_revenue": float(row["nsp_revenue"]),
                "joint_revenue": float(row["joint_revenue"]),
                "runtime_sec": float(row["runtime_sec"]),
                "stage2_solver_calls": int(row["stage2_solver_calls"]),
                "audit_stage2_solver_calls": int(row["audit_stage2_solver_calls"]),
                "total_stage2_solver_calls": int(row["total_stage2_solver_calls"]),
                "error": str(row.get("error", "")),
            }
        )

    failures = 0
    for n in n_list:
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users(cfg, int(n), int(args.seed), int(trial))
            budget = int(budget_map[(int(n), int(trial))])
            base_cfg = quality._apply_baseline_overrides(
                cfg.baselines,
                bo_candidate_pool=args.bo_candidate_pool,
                bo_iters=args.bo_iters,
                ga_population_size=args.ga_population_size,
                ga_generations=args.ga_generations,
                marl_price_levels=args.marl_price_levels,
                marl_episodes=args.marl_episodes,
                marl_steps_per_episode=args.marl_steps_per_episode,
            )
            for method in methods:
                try:
                    out, meta = _budgeted_baseline_outcome(
                        method,
                        users=users,
                        system=cfg.system,
                        stack_cfg=cfg.stackelberg,
                        base_cfg=base_cfg,
                        budget=budget,
                    )
                    final_gap, audit_calls = _posthoc_final_gap(
                        out,
                        users=users,
                        system=cfg.system,
                        stack_cfg=cfg.stackelberg,
                        base_cfg=base_cfg,
                        audit_points=int(args.final_audit_grid_points),
                    )
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "source": "budget_matched_rerun",
                            "success": 1,
                            "budget_stage2_calls": int(budget),
                            "search_budget_exhausted": int(bool(meta["budget_exhausted"])),
                            "budget_stop_mode": str(meta["budget_stop_mode"]),
                            "final_pE": float(out.price[0]),
                            "final_pN": float(out.price[1]),
                            "offloading_size": int(len(out.offloading_set)),
                            "final_grid_ne_gap": float(final_gap),
                            "esp_revenue": float(out.esp_revenue),
                            "nsp_revenue": float(out.nsp_revenue),
                            "joint_revenue": float(out.esp_revenue + out.nsp_revenue),
                            "runtime_sec": float(meta["runtime_sec"]),
                            "stage2_solver_calls": int(meta["stage2_calls"]),
                            "audit_stage2_solver_calls": int(audit_calls),
                            "total_stage2_solver_calls": int(meta["stage2_calls"] + audit_calls),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    failures += 1
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "source": "budget_matched_rerun",
                            "success": 0,
                            "budget_stage2_calls": int(budget),
                            "search_budget_exhausted": 0,
                            "budget_stop_mode": "error",
                            "final_pE": float("nan"),
                            "final_pN": float("nan"),
                            "offloading_size": -1,
                            "final_grid_ne_gap": float("nan"),
                            "esp_revenue": float("nan"),
                            "nsp_revenue": float("nan"),
                            "joint_revenue": float("nan"),
                            "runtime_sec": float("nan"),
                            "stage2_solver_calls": float("nan"),
                            "audit_stage2_solver_calls": float("nan"),
                            "total_stage2_solver_calls": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    method_order = ["Proposed", *methods]
    summary_rows = quality.summarize_trials(rows, method_order, n_list)
    write_csv_rows(out_dir / "stage1_budget_matched_quality_vs_users.csv", TRIAL_FIELDS, rows)
    write_csv_rows(out_dir / "stage1_budget_matched_quality_vs_users_stats.csv", quality._summary_fieldnames(), summary_rows)
    quality.plot_metric_summary(
        summary_rows,
        out_dir / "stage1_budget_matched_gap_vs_users.png",
        methods=method_order,
        metric="final_grid_ne_gap",
        statistic="median_iqr",
        ylabel="Final grid-evaluated NE gap",
        title="Budget-matched final NE gap vs. number of users",
    )
    quality.plot_metric_summary(
        summary_rows,
        out_dir / "stage1_budget_matched_joint_revenue_vs_users.png",
        methods=method_order,
        metric="joint_revenue",
        statistic="median_iqr",
        ylabel="Joint revenue",
        title="Budget-matched joint revenue vs. number of users",
    )
    quality.plot_stage2_calls_broken_axis(
        summary_rows,
        out_dir / "stage1_budget_matched_stage2_calls_vs_users.png",
        methods=method_order,
        statistic="median_iqr",
    )
    _write_summary(
        out_dir / "stage1_budget_matched_quality_vs_users_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"trials = {args.trials}",
            f"n_users_list = {','.join(str(x) for x in n_list)}",
            f"methods = {','.join(method_order)}",
            f"reference_run_dir = {reference_dir}",
            "budget_source_metric = stage2_solver_calls",
            f"final_audit_grid_points = {int(args.final_audit_grid_points)}",
            f"bo_candidate_pool = {int(args.bo_candidate_pool)}",
            f"bo_iters = {int(args.bo_iters)}",
            f"ga_population_size = {int(args.ga_population_size)}",
            f"ga_generations = {int(args.ga_generations)}",
            f"marl_price_levels = {int(args.marl_price_levels)}",
            f"marl_episodes = {int(args.marl_episodes)}",
            f"marl_steps_per_episode = {int(args.marl_steps_per_episode)}",
            "budget_rule = for each (n,trial), rerun baseline until a new Stage-II solve would exceed the reused Proposed search-call budget",
            "final_grid_ne_gap_definition = posthoc audit at the returned cutoff price on the final audit grid",
            "proposed_rows = reused unchanged from reference_run_dir",
            f"failed_runs = {int(failures)}",
        ],
    )


if __name__ == "__main__":
    main()

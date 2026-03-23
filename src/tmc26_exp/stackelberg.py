from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize

from .config import StackelbergConfig, SystemConfig
from .model import UserBatch, local_cost, theta

Provider = Literal["E", "N"]
InnerSolverMode = Literal["primal_dual", "exact", "hybrid"]
_EPS = 1e-12


@dataclass(frozen=True)
class InnerSolveResult:
    offloading_set: tuple[int, ...]
    f: np.ndarray
    b: np.ndarray
    lambda_F: float
    lambda_B: float
    mu: np.ndarray
    offloading_objective: float
    converged: bool
    iterations: int


@dataclass(frozen=True)
class GreedySelectionResult:
    offloading_set: tuple[int, ...]
    inner_result: InnerSolveResult
    social_cost: float
    iterations: int
    rollback_count: int = 0
    accepted_admissions: int = 0
    inner_call_count: int = 0
    runtime_sec: float = 0.0
    stage2_method: str = "algorithm_2_plus_algorithm_1"
    inner_solver_mode: str = "primal_dual"
    used_exact_inner: bool = False
    social_cost_trace: tuple[float, ...] = ()


@dataclass(frozen=True)
class GainApproxResult:
    provider: Provider
    gain: float
    best_set: tuple[int, ...]
    current_revenue: float
    candidate_count: int


@dataclass(frozen=True)
class VBBROracleResult:
    provider: Provider
    best_price: float
    best_trigger_set: tuple[int, ...]
    best_set: tuple[int, ...]
    best_stage2_result: GreedySelectionResult
    current_revenue: float
    best_revenue: float
    gain: float
    stage2_calls: int
    evaluated_candidates: int
    evaluated_boundary_points: int
    max_candidate_family_size: int = 0


@dataclass(frozen=True)
class BoundaryPriceBROracleResult:
    provider: Provider
    best_price: float
    best_set: tuple[int, ...]
    best_stage2_result: GreedySelectionResult
    current_revenue: float
    best_revenue: float
    gain: float
    stage2_calls: int
    candidate_family_size: int
    evaluated_boundary_points: int


@dataclass(frozen=True)
class SearchStep:
    iteration: int
    offloading_set: tuple[int, ...]
    pE: float
    pN: float
    epsilon: float
    dist_to_se: float = float("nan")
    epsilon_delta: float = float("nan")
    esp_best_set_size: int = 0
    nsp_best_set_size: int = 0
    esp_gain: float = float("nan")
    nsp_gain: float = float("nan")
    restricted_gap: float = float("nan")
    restricted_gap_delta: float = float("nan")
    candidate_family_size: int = 0
    esp_candidate_family_size: int = 0
    nsp_candidate_family_size: int = 0
    stage2_offloading_size: int = 0
    stage2_social_cost: float = float("nan")


@dataclass(frozen=True)
class StackelbergResult:
    price: tuple[float, float]
    offloading_set: tuple[int, ...]
    epsilon: float
    gain_E: GainApproxResult
    gain_N: GainApproxResult
    inner_result: InnerSolveResult
    social_cost: float
    trajectory: tuple[SearchStep, ...]
    outer_iterations: int = 0
    stage2_oracle_calls: int = 0
    evaluated_candidates: int = 0
    evaluated_boundary_points: int = 0
    esp_revenue: float = 0.0
    nsp_revenue: float = 0.0
    stopping_reason: str = "unknown"
    restricted_gap: float = float("nan")
    final_stage2_result: GreedySelectionResult | None = None
    stage1_method: str = "unknown"


@dataclass(frozen=True)
class _ProblemData:
    aw: np.ndarray
    th: np.ndarray
    cl: np.ndarray


def _build_data(users: UserBatch) -> _ProblemData:
    return _ProblemData(
        aw=users.alpha * users.w,
        th=theta(users),
        cl=local_cost(users),
    )


def _sorted_tuple(indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(i) for i in indices}))


def _bar_prices(data: _ProblemData, offloading_set: tuple[int, ...], system: SystemConfig) -> tuple[float, float]:
    if not offloading_set:
        return 0.0, 0.0
    idx = np.asarray(offloading_set, dtype=int)
    sE = float(np.sum(np.sqrt(data.aw[idx])))
    sN = float(np.sum(np.sqrt(data.th[idx])))
    bar_pE = (sE / system.F) ** 2 if sE > 0 else 0.0
    bar_pN = (sN / system.B) ** 2 if sN > 0 else 0.0
    return bar_pE, bar_pN


def _tilde_prices(
    data: _ProblemData,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> tuple[float, float]:
    bar_pE, bar_pN = _bar_prices(data, offloading_set, system)
    return max(pE, bar_pE), max(pN, bar_pN)


def _resource_demands(
    data: _ProblemData,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> tuple[float, float]:
    if not offloading_set:
        return 0.0, 0.0
    idx = np.asarray(offloading_set, dtype=int)
    sE = float(np.sum(np.sqrt(data.aw[idx])))
    sN = float(np.sum(np.sqrt(data.th[idx])))
    tE, tN = _tilde_prices(data, offloading_set, pE, pN, system)
    sum_f = sE / math.sqrt(max(tE, _EPS))
    sum_b = sN / math.sqrt(max(tN, _EPS))
    return sum_f, sum_b


def _provider_revenue(
    data: _ProblemData,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> float:
    sum_f, sum_b = _resource_demands(data, offloading_set, pE, pN, system)
    if provider == "E":
        return (pE - system.cE) * sum_f
    return (pN - system.cN) * sum_b


def _provider_revenue_from_stage2_result(
    stage2_result: GreedySelectionResult,
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> float:
    if not stage2_result.offloading_set:
        return 0.0
    idx = np.asarray(stage2_result.offloading_set, dtype=int)
    if provider == "E":
        demand = float(np.sum(stage2_result.inner_result.f[idx]))
        return (pE - system.cE) * demand
    demand = float(np.sum(stage2_result.inner_result.b[idx]))
    return (pN - system.cN) * demand


def _social_cost_from_inner_result(local_costs: np.ndarray, n_users: int, inner: InnerSolveResult) -> float:
    outside = set(range(n_users)) - set(inner.offloading_set)
    return inner.offloading_objective + (float(np.sum(local_costs[list(outside)])) if outside else 0.0)


def _normalize_stage2_inner_solver_mode(
    allow_exact_inner: bool | None,
    inner_solver_mode: str | None,
) -> InnerSolverMode:
    if inner_solver_mode is None:
        return "hybrid" if (allow_exact_inner is None or allow_exact_inner) else "primal_dual"
    mode = str(inner_solver_mode).strip().lower()
    if mode not in {"primal_dual", "exact", "hybrid"}:
        raise ValueError("inner_solver_mode must be 'primal_dual', 'exact', or 'hybrid'.")
    return mode


def _offload_cost_for_user(
    data: _ProblemData,
    user_idx: int,
    f_i: float,
    b_i: float,
    pE: float,
    pN: float,
) -> float:
    return data.aw[user_idx] / f_i + data.th[user_idx] / b_i + pE * f_i + pN * b_i


def _fixed_set_cost_component(weight_sqrt: float, price: float, bar_price: float) -> float:
    bar_eff = max(bar_price, _EPS)
    if price <= bar_price + 1e-12:
        bar_root = math.sqrt(bar_eff)
        return weight_sqrt * (bar_root + price / bar_root)
    return 2.0 * weight_sqrt * math.sqrt(max(price, _EPS))


def _fixed_set_margin_exact(
    data: _ProblemData,
    candidate_set: tuple[int, ...],
    user_idx: int,
    pE: float,
    pN: float,
    system: SystemConfig,
) -> float:
    bar_pE, bar_pN = _bar_prices(data, candidate_set, system)
    a_i = math.sqrt(max(float(data.aw[user_idx]), 0.0))
    s_i = math.sqrt(max(float(data.th[user_idx]), 0.0))
    compute_cost = _fixed_set_cost_component(a_i, pE, bar_pE)
    bandwidth_cost = _fixed_set_cost_component(s_i, pN, bar_pN)
    return float(data.cl[user_idx] - compute_cost - bandwidth_cost)


def _realized_margin_from_stage2(
    data: _ProblemData,
    stage2_result: GreedySelectionResult,
    user_idx: int,
    pE: float,
    pN: float,
) -> float:
    f_i = float(stage2_result.inner_result.f[user_idx])
    b_i = float(stage2_result.inner_result.b[user_idx])
    if f_i <= 0.0 or b_i <= 0.0:
        return float("-inf")
    return float(data.cl[user_idx] - _offload_cost_for_user(data, user_idx, f_i, b_i, pE, pN))


def _bounded_1d_min(a: float, t: float, upper: float) -> float:
    t_eff = max(t, _EPS)
    a_eff = max(a, _EPS)
    x_star = math.sqrt(a_eff / t_eff)
    if x_star <= upper:
        return 2.0 * math.sqrt(a_eff * t_eff)
    return a_eff / upper + t_eff * upper


def _heuristic_score_with_t(
    data: _ProblemData,
    user_idx: int,
    tE: float,
    tN: float,
    system: SystemConfig,
) -> float:
    f_term = _bounded_1d_min(float(data.aw[user_idx]), tE, system.F)
    b_term = _bounded_1d_min(float(data.th[user_idx]), tN, system.B)
    return f_term + b_term - float(data.cl[user_idx])


def _margin_for_user(
    data: _ProblemData,
    offloading_set: tuple[int, ...],
    user_idx: int,
    pE: float,
    pN: float,
    system: SystemConfig,
) -> float:
    return _fixed_set_margin_exact(data, offloading_set, user_idx, pE, pN, system)


def _boundary_price_for_provider(
    data: _ProblemData,
    candidate_set: tuple[int, ...],
    opponent_price: float,
    provider: Provider,
    system: SystemConfig,
) -> float | None:
    if not candidate_set:
        return None
    idx = np.asarray(candidate_set, dtype=int)
    bar_pE, bar_pN = _bar_prices(data, candidate_set, system)
    feasible: list[float] = []

    if provider == "E":
        floor_price = float(system.cE)
        bar_self = float(bar_pE)
        bar_opp = float(bar_pN)
        sum_self = float(np.sum(np.sqrt(data.aw[idx])))

        if min(_fixed_set_margin_exact(data, candidate_set, int(i), floor_price, opponent_price, system) for i in idx) < -1e-9:
            return None

        for i in idx:
            a_i = math.sqrt(max(float(data.aw[i]), 0.0))
            s_i = math.sqrt(max(float(data.th[i]), 0.0))
            chi_opp = _fixed_set_cost_component(s_i, opponent_price, bar_opp)
            residual = float(data.cl[i] - chi_opp)
            if residual <= 0.0 or a_i <= 0.0:
                continue
            threshold = 2.0 * a_i * math.sqrt(max(bar_self, 0.0))
            if residual <= threshold + 1e-12:
                tau_i = (sum_self / system.F) * (residual / a_i) - bar_self
            else:
                tau_i = (residual * residual) / (4.0 * a_i * a_i)
            feasible.append(float(max(tau_i, floor_price)))
    else:
        floor_price = float(system.cN)
        bar_self = float(bar_pN)
        bar_opp = float(bar_pE)
        sum_self = float(np.sum(np.sqrt(data.th[idx])))

        if min(_fixed_set_margin_exact(data, candidate_set, int(i), opponent_price, floor_price, system) for i in idx) < -1e-9:
            return None

        for i in idx:
            a_i = math.sqrt(max(float(data.aw[i]), 0.0))
            s_i = math.sqrt(max(float(data.th[i]), 0.0))
            chi_opp = _fixed_set_cost_component(a_i, opponent_price, bar_opp)
            residual = float(data.cl[i] - chi_opp)
            if residual <= 0.0 or s_i <= 0.0:
                continue
            threshold = 2.0 * s_i * math.sqrt(max(bar_self, 0.0))
            if residual <= threshold + 1e-12:
                tau_i = (sum_self / system.B) * (residual / s_i) - bar_self
            else:
                tau_i = (residual * residual) / (4.0 * s_i * s_i)
            feasible.append(float(max(tau_i, floor_price)))

    if not feasible:
        return None
    return min(feasible)


def _boundary_revenue_for_provider(
    data: _ProblemData,
    candidate_set: tuple[int, ...],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> float | None:
    if not candidate_set:
        return 0.0
    idx = np.asarray(candidate_set, dtype=int)
    bar_pE, bar_pN = _bar_prices(data, candidate_set, system)
    sE = float(np.sum(np.sqrt(data.aw[idx])))
    sN = float(np.sum(np.sqrt(data.th[idx])))

    if provider == "E":
        p_bar = _boundary_price_for_provider(data, candidate_set, pN, "E", system)
        if p_bar is None:
            return None
        if p_bar <= bar_pE:
            demand = system.F
        else:
            demand = sE / math.sqrt(max(p_bar, _EPS))
        return (p_bar - system.cE) * demand

    p_bar = _boundary_price_for_provider(data, candidate_set, pE, "N", system)
    if p_bar is None:
        return None
    if p_bar <= bar_pN:
        demand = system.B
    else:
        demand = sN / math.sqrt(max(p_bar, _EPS))
    return (p_bar - system.cN) * demand


def _candidate_family(
    data: _ProblemData,
    current_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> list[tuple[int, ...]]:
    current = list(current_set)
    current_set_lookup = set(current_set)
    outsiders = [u for u in range(data.cl.size) if u not in current_set_lookup]

    tE, tN = _tilde_prices(data, current_set, pE, pN, system)
    margins = {
        i: data.cl[i] - 2.0 * math.sqrt(data.aw[i] * tE) - 2.0 * math.sqrt(data.th[i] * tN)
        for i in current
    }
    scores = {j: _heuristic_score_with_t(data, j, tE, tN, system) for j in outsiders}

    offloading_order = sorted(current, key=lambda i: margins[i])
    outsider_order = sorted(outsiders, key=lambda j: scores[j])

    seen: set[tuple[int, ...]] = set()
    family: list[tuple[int, ...]] = []
    for r in range(len(offloading_order) + 1):
        remained = offloading_order[r:]
        for s in range(len(outsider_order) + 1):
            cand = _sorted_tuple([*remained, *outsider_order[:s]])
            if cand not in seen:
                seen.add(cand)
                family.append(cand)
    return family


def _paper_local_candidate_family(
    data: _ProblemData,
    current_stage2_result: GreedySelectionResult,
    pE: float,
    pN: float,
    system: SystemConfig,
    Q: int,
) -> list[tuple[int, ...]]:
    current_set = current_stage2_result.offloading_set
    current = list(current_set)
    current_set_lookup = set(current_set)
    outsiders = [u for u in range(data.cl.size) if u not in current_set_lookup]

    if current_set:
        lambda_F = float(current_stage2_result.inner_result.lambda_F)
        lambda_B = float(current_stage2_result.inner_result.lambda_B)
    else:
        lambda_F = 0.0
        lambda_B = 0.0

    margins = {
        i: _realized_margin_from_stage2(data, current_stage2_result, i, pE, pN)
        for i in current
    }
    scores = {
        j: _heuristic_score_with_t(data, j, pE + lambda_F, pN + lambda_B, system)
        for j in outsiders
    }

    offloading_order = sorted(current, key=lambda i: margins[i])
    outsider_order = sorted(outsiders, key=lambda j: scores[j])

    max_q = max(0, int(Q))
    max_r = min(max_q, len(offloading_order))
    max_s = min(max_q, len(outsider_order))

    seen: set[tuple[int, ...]] = set()
    family: list[tuple[int, ...]] = []
    for r in range(max_r + 1):
        remained = offloading_order[r:]
        for s in range(max_s + 1):
            cand = _sorted_tuple([*remained, *outsider_order[:s]])
            if not cand:
                continue
            if cand in seen:
                continue
            seen.add(cand)
            family.append(cand)
    return family


def _vbbr_local_candidate_family(
    data: _ProblemData,
    anchor_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> list[tuple[int, ...]]:
    anchor = list(anchor_set)
    anchor_lookup = set(anchor_set)
    outsiders = [u for u in range(data.cl.size) if u not in anchor_lookup]

    tE, tN = _tilde_prices(data, anchor_set, pE, pN, system)
    margins = {i: _margin_for_user(data, anchor_set, i, pE, pN, system) for i in anchor}
    scores = {j: _heuristic_score_with_t(data, j, tE, tN, system) for j in outsiders}

    offloading_order = sorted(anchor, key=lambda i: margins[i])
    outsider_order = sorted(outsiders, key=lambda j: scores[j])

    max_r = min(cfg.vbbr_local_R, len(offloading_order))
    max_s = min(cfg.vbbr_local_S, len(outsider_order))
    budget = max(0, cfg.vbbr_local_budget)

    seen: set[tuple[int, ...]] = set()
    family: list[tuple[int, ...]] = []
    for r in range(max_r + 1):
        remained = offloading_order[r:]
        for s in range(max_s + 1):
            if r + s > budget:
                continue
            cand = _sorted_tuple([*remained, *outsider_order[:s]])
            if cand in seen:
                continue
            seen.add(cand)
            family.append(cand)
    return family


def _vbbr_verified_br_oracle(
    users: UserBatch,
    data: _ProblemData,
    provider: Provider,
    pE: float,
    pN: float,
    current_stage2_result: GreedySelectionResult,
    current_revenue: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    allow_exact_inner: bool,
) -> VBBROracleResult:
    stage2_calls = 0
    anchor_set = current_stage2_result.offloading_set

    if provider == "E":
        current_price = float(pE)
    else:
        current_price = float(pN)

    best_revenue = float(current_revenue)
    best_price = current_price
    best_trigger_set = anchor_set
    best_set = anchor_set
    best_stage2_result = current_stage2_result

    visited_sets: set[tuple[int, ...]] = {anchor_set}
    evaluated_candidates = 0
    evaluated_boundary_points = 0
    no_improve_rounds = 0
    max_candidate_family_size = 0

    for _ in range(cfg.vbbr_oracle_max_rounds):
        family = _vbbr_local_candidate_family(data, anchor_set, pE, pN, system, cfg)
        max_candidate_family_size = max(max_candidate_family_size, len(family))
        surrogate_pool: list[tuple[float, tuple[int, ...], float]] = []

        for candidate_set in family:
            if not candidate_set:
                continue
            if provider == "E":
                boundary_price = _boundary_price_for_provider(data, candidate_set, pN, "E", system)
            else:
                boundary_price = _boundary_price_for_provider(data, candidate_set, pE, "N", system)
            if boundary_price is None:
                continue
            candidate_price = max(boundary_price, system.cE if provider == "E" else system.cN)
            evaluated_boundary_points += 1

            if provider == "E":
                surrogate = _provider_revenue(data, candidate_set, candidate_price, pN, provider, system)
            else:
                surrogate = _provider_revenue(data, candidate_set, pE, candidate_price, provider, system)
            surrogate_pool.append((float(surrogate), candidate_set, float(candidate_price)))

        if not surrogate_pool:
            break

        surrogate_pool.sort(key=lambda item: item[0], reverse=True)
        top_m = min(cfg.vbbr_top_m, len(surrogate_pool))
        verified: list[tuple[float, float, tuple[int, ...], tuple[int, ...], GreedySelectionResult]] = []

        for _, trigger_set, candidate_price in surrogate_pool[:top_m]:
            if provider == "E":
                eval_pE = max(system.cE, float(candidate_price))
                eval_pN = pN
            else:
                eval_pE = pE
                eval_pN = max(system.cN, float(candidate_price))
            stage2_eval = algorithm_2_heuristic_user_selection(
                users,
                eval_pE,
                eval_pN,
                system,
                cfg,
                allow_exact_inner=allow_exact_inner,
            )
            stage2_calls += 1
            evaluated_candidates += 1
            realized_set = stage2_eval.offloading_set
            realized_revenue = _provider_revenue_from_stage2_result(stage2_eval, eval_pE, eval_pN, provider, system)
            verified.append((float(realized_revenue), float(candidate_price), trigger_set, realized_set, stage2_eval))

        if not verified:
            break

        verified.sort(key=lambda item: item[0], reverse=True)
        selected_revenue, selected_price, selected_trigger_set, selected_set, selected_stage2 = verified[0]

        previous_best = best_revenue
        improvement = float(selected_revenue - previous_best)
        improved = improvement > cfg.vbbr_oracle_improve_tol
        if improved:
            best_revenue = float(selected_revenue)
            best_price = float(selected_price)
            best_trigger_set = selected_trigger_set
            best_set = selected_set
            best_stage2_result = selected_stage2
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        realized_seen = selected_set in visited_sets
        if realized_seen and not improved:
            break
        if no_improve_rounds >= cfg.vbbr_no_improve_patience:
            break

        visited_sets.add(selected_set)
        anchor_set = selected_set

    gain = max(0.0, float(best_revenue - current_revenue))
    return VBBROracleResult(
        provider=provider,
        best_price=float(best_price),
        best_trigger_set=best_trigger_set,
        best_set=best_set,
        best_stage2_result=best_stage2_result,
        current_revenue=float(current_revenue),
        best_revenue=float(best_revenue),
        gain=gain,
        stage2_calls=stage2_calls,
        evaluated_candidates=evaluated_candidates,
        evaluated_boundary_points=evaluated_boundary_points,
        max_candidate_family_size=max_candidate_family_size,
    )


def algorithm_1_distributed_primal_dual(
    users: UserBatch,
    offloading_set: Iterable[int],
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> InnerSolveResult:
    data = _build_data(users)
    chosen = _sorted_tuple(offloading_set)
    n_users = users.n
    f = np.zeros(n_users, dtype=float)
    b = np.zeros(n_users, dtype=float)
    mu = np.zeros(n_users, dtype=float)

    if not chosen:
        return InnerSolveResult(
            offloading_set=chosen,
            f=f,
            b=b,
            lambda_F=0.0,
            lambda_B=0.0,
            mu=mu,
            offloading_objective=0.0,
            converged=True,
            iterations=0,
        )

    idx = np.asarray(chosen, dtype=int)
    lambda_F = 0.0
    lambda_B = 0.0
    converged = False
    total_iters = cfg.inner_max_iters

    for t in range(cfg.inner_max_iters):
        eta_F = cfg.inner_eta_F0 / math.sqrt(t + 1.0)
        eta_B = cfg.inner_eta_B0 / math.sqrt(t + 1.0)
        eta_mu = cfg.inner_eta_mu0 / math.sqrt(t + 1.0)

        for i in idx:
            coeff = 1.0 + mu[i]
            denom_f = coeff * pE + lambda_F
            denom_b = coeff * pN + lambda_B
            f[i] = math.sqrt(coeff * data.aw[i] / max(denom_f, _EPS))
            b[i] = math.sqrt(coeff * data.th[i] / max(denom_b, _EPS))

        ce = data.aw[idx] / np.maximum(f[idx], _EPS) + data.th[idx] / np.maximum(b[idx], _EPS) + pE * f[idx] + pN * b[idx]
        sum_f = float(np.sum(f[idx]))
        sum_b = float(np.sum(b[idx]))

        lambda_F = max(0.0, lambda_F + eta_F * (sum_f - system.F))
        lambda_B = max(0.0, lambda_B + eta_B * (sum_b - system.B))
        mu[idx] = np.maximum(0.0, mu[idx] + eta_mu * (ce - data.cl[idx]))

        cap_violation = max(0.0, sum_f - system.F, sum_b - system.B)
        ir_violation = float(np.max(np.maximum(ce - data.cl[idx], 0.0)))
        if max(cap_violation, ir_violation) <= cfg.inner_tol:
            converged = True
            total_iters = t + 1
            break

    ce_final = data.aw[idx] / np.maximum(f[idx], _EPS) + data.th[idx] / np.maximum(b[idx], _EPS) + pE * f[idx] + pN * b[idx]
    objective = float(np.sum(ce_final))

    return InnerSolveResult(
        offloading_set=chosen,
        f=f,
        b=b,
        lambda_F=float(lambda_F),
        lambda_B=float(lambda_B),
        mu=mu,
        offloading_objective=objective,
        converged=converged,
        iterations=total_iters,
    )


def algorithm_2_heuristic_user_selection(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    allow_exact_inner: bool = True,
    inner_solver_mode: str | None = None,
) -> GreedySelectionResult:
    data = _build_data(users)
    solver_mode = _normalize_stage2_inner_solver_mode(allow_exact_inner, inner_solver_mode)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()

    previous_ve = 0.0
    last_added: int | None = None
    iterations = 0
    rollback_count = 0
    accepted_admissions = 0
    inner_call_count = 0
    used_exact_inner = False
    social_trace: list[float] = []
    start_time = time.perf_counter()

    for t in range(cfg.greedy_max_iters):
        inner, used_exact = _solve_stage2_inner(users, offloading_set, pE, pN, system, cfg, solver_mode)
        inner_call_count += 1
        used_exact_inner = used_exact_inner or used_exact
        ve = inner.offloading_objective

        if t >= 1 and last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                rollback_count += 1
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner, used_exact = _solve_stage2_inner(users, offloading_set, pE, pN, system, cfg, solver_mode)
                inner_call_count += 1
                used_exact_inner = used_exact_inner or used_exact
                ve = inner.offloading_objective

        social_trace.append(float(_social_cost_from_inner_result(data.cl, users.n, inner)))

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
            key=lambda j: _heuristic_score_with_t(
                data,
                j,
                pE + lambda_F,
                pN + lambda_B,
                system,
            ),
        )
        best_score = _heuristic_score_with_t(data, best_user, pE + lambda_F, pN + lambda_B, system)

        if best_score < 0.0:
            previous_ve = ve
            offloading_set.add(best_user)
            last_added = best_user
            accepted_admissions += 1
            iterations = t + 1
            continue

        iterations = t + 1
        break

    final_inner, used_exact = _solve_stage2_inner(users, offloading_set, pE, pN, system, cfg, solver_mode)
    inner_call_count += 1
    used_exact_inner = used_exact_inner or used_exact
    final_set = final_inner.offloading_set
    social_cost = float(_social_cost_from_inner_result(data.cl, users.n, final_inner))
    if social_trace:
        social_trace[-1] = social_cost
    else:
        social_trace.append(social_cost)

    return GreedySelectionResult(
        offloading_set=final_set,
        inner_result=final_inner,
        social_cost=float(social_cost),
        iterations=iterations,
        rollback_count=rollback_count,
        accepted_admissions=accepted_admissions,
        inner_call_count=inner_call_count,
        runtime_sec=float(time.perf_counter() - start_time),
        stage2_method="algorithm_2_plus_algorithm_1",
        inner_solver_mode=solver_mode,
        used_exact_inner=used_exact_inner,
        social_cost_trace=tuple(float(x) for x in social_trace),
    )


def solve_stage2_scm(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    *,
    inner_solver_mode: InnerSolverMode = "primal_dual",
) -> GreedySelectionResult:
    return algorithm_2_heuristic_user_selection(
        users,
        pE,
        pN,
        system,
        cfg,
        allow_exact_inner=False,
        inner_solver_mode=inner_solver_mode,
    )


def _candidate_revenue_estimate(
    data: _ProblemData,
    users: UserBatch,
    candidate_set: tuple[int, ...],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
    estimator_variant: str,
) -> float | None:
    if estimator_variant == "boundary":
        return _boundary_revenue_for_provider(data, candidate_set, pE, pN, provider, system)

    if estimator_variant == "refined_price":
        # Calibration variant: refine prices on candidate set, then evaluate revenue there.
        cand_price = _refine_price_for_fixed_set(users, candidate_set, (pE, pN), system)
        return _provider_revenue(data, candidate_set, cand_price[0], cand_price[1], provider, system)

    raise ValueError(f"Unknown estimator_variant={estimator_variant}")


def algorithm_3_gain_approximation(
    users: UserBatch,
    current_set: Iterable[int],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
    estimator_variant: str = "boundary",
    top_k: int = 4,
) -> GainApproxResult:
    data = _build_data(users)
    X = _sorted_tuple(current_set)
    current_revenue = _provider_revenue(data, X, pE, pN, provider, system)
    family = _candidate_family(data, X, pE, pN, system)

    best_gain = 0.0
    best_set = X

    if estimator_variant == "topk_real_reval":
        # Two-stage estimator: rank by fast boundary estimate, then exact re-eval on Top-K.
        scored: list[tuple[float, tuple[int, ...]]] = []
        for Y in family:
            s = _candidate_revenue_estimate(
                data,
                users,
                Y,
                pE,
                pN,
                provider,
                system,
                estimator_variant="boundary",
            )
            if s is None:
                continue
            scored.append((float(s), Y))
        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = max(1, min(int(top_k), len(scored)))
        for _, Y in scored[:top_k]:
            candidate_revenue = _provider_revenue(data, Y, pE, pN, provider, system)
            gain = candidate_revenue - current_revenue
            if gain > best_gain:
                best_gain = float(gain)
                best_set = Y
    else:
        for Y in family:
            candidate_revenue = _candidate_revenue_estimate(
                data,
                users,
                Y,
                pE,
                pN,
                provider,
                system,
                estimator_variant=estimator_variant,
            )
            if candidate_revenue is None:
                continue
            gain = candidate_revenue - current_revenue
            if gain > best_gain:
                best_gain = float(gain)
                best_set = Y

    return GainApproxResult(
        provider=provider,
        gain=float(best_gain),
        best_set=best_set,
        current_revenue=float(current_revenue),
        candidate_count=len(family),
    )


def _solve_fixed_set_inner_exact(
    users: UserBatch,
    offloading_set: Iterable[int],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> InnerSolveResult | None:
    data = _build_data(users)
    chosen = _sorted_tuple(offloading_set)
    n_users = users.n
    f = np.zeros(n_users, dtype=float)
    b = np.zeros(n_users, dtype=float)
    mu = np.zeros(n_users, dtype=float)

    if not chosen:
        return InnerSolveResult(
            offloading_set=chosen,
            f=f,
            b=b,
            lambda_F=0.0,
            lambda_B=0.0,
            mu=mu,
            offloading_objective=0.0,
            converged=True,
            iterations=0,
        )

    idx = np.asarray(chosen, dtype=int)
    m = idx.size
    eps = 1e-8

    tE, tN = _tilde_prices(data, chosen, pE, pN, system)
    x0 = np.concatenate(
        [
            np.clip(np.sqrt(data.aw[idx] / max(tE, eps)), eps, system.F),
            np.clip(np.sqrt(data.th[idx] / max(tN, eps)), eps, system.B),
        ]
    )

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:m], x[m:]

    def objective(x: np.ndarray) -> float:
        f_loc, b_loc = unpack(x)
        return float(np.sum(data.aw[idx] / f_loc + data.th[idx] / b_loc + pE * f_loc + pN * b_loc))

    constraints: list[dict[str, object]] = [
        {"type": "ineq", "fun": lambda x: system.F - float(np.sum(x[:m]))},
        {"type": "ineq", "fun": lambda x: system.B - float(np.sum(x[m:]))},
    ]
    for k, i in enumerate(idx):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, kk=k, ii=i: float(
                    data.cl[ii]
                    - (data.aw[ii] / x[kk] + data.th[ii] / x[m + kk] + pE * x[kk] + pN * x[m + kk])
                ),
            }
        )

    bounds = [(eps, system.F)] * m + [(eps, system.B)] * m
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )
    if not res.success:
        return None

    f_sol, b_sol = unpack(np.asarray(res.x, dtype=float))
    ce = data.aw[idx] / f_sol + data.th[idx] / b_sol + pE * f_sol + pN * b_sol
    if np.sum(f_sol) > system.F + 1e-6 or np.sum(b_sol) > system.B + 1e-6:
        return None
    if np.any(ce > data.cl[idx] + 1e-6):
        return None

    f[idx] = f_sol
    b[idx] = b_sol

    raw_multipliers = np.atleast_1d(np.asarray(getattr(res, "multipliers", np.zeros(2 + m)), dtype=float))
    multipliers = np.zeros(2 + m, dtype=float)
    used = min(multipliers.size, raw_multipliers.size)
    multipliers[:used] = raw_multipliers[:used]

    lambda_F = float(max(multipliers[0], 0.0))
    lambda_B = float(max(multipliers[1], 0.0))
    mu[idx] = np.maximum(multipliers[2 : 2 + m], 0.0)

    return InnerSolveResult(
        offloading_set=chosen,
        f=f,
        b=b,
        lambda_F=lambda_F,
        lambda_B=lambda_B,
        mu=mu,
        offloading_objective=float(np.sum(ce)),
        converged=True,
        iterations=int(getattr(res, "nit", 0)),
    )


def _solve_stage2_inner(
    users: UserBatch,
    offloading_set: Iterable[int],
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    inner_solver_mode: InnerSolverMode,
) -> tuple[InnerSolveResult, bool]:
    if inner_solver_mode == "exact":
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is None:
            raise RuntimeError("Exact inner solver failed for a fixed set; cannot continue in exact mode.")
        return inner, True

    if inner_solver_mode == "hybrid":
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is not None:
            return inner, True

    inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
    return inner, False


def _paper_boundary_price_br_oracle(
    users: UserBatch,
    data: _ProblemData,
    provider: Provider,
    pE: float,
    pN: float,
    system: SystemConfig,
    cfg: StackelbergConfig,
    *,
    current_stage2_result: GreedySelectionResult | None = None,
    current_revenue: float | None = None,
) -> BoundaryPriceBROracleResult:
    stage2_calls = 0
    if current_stage2_result is None:
        current_stage2_result = solve_stage2_scm(
            users,
            pE,
            pN,
            system,
            cfg,
            inner_solver_mode="primal_dual",
        )
        stage2_calls += 1

    if provider == "E":
        current_price = float(pE)
    else:
        current_price = float(pN)

    if current_revenue is None:
        current_revenue = _provider_revenue_from_stage2_result(
            current_stage2_result,
            pE,
            pN,
            provider,
            system,
        )

    family = _paper_local_candidate_family(
        data,
        current_stage2_result,
        pE,
        pN,
        system,
        cfg.paper_local_Q,
    )

    boundary_prices: dict[float, tuple[int, ...]] = {}
    for candidate_set in family:
        if not candidate_set:
            continue
        if provider == "E":
            boundary_price = _boundary_price_for_provider(data, candidate_set, pN, "E", system)
        else:
            boundary_price = _boundary_price_for_provider(data, candidate_set, pE, "N", system)
        if boundary_price is None:
            continue
        candidate_price = max(
            system.cE if provider == "E" else system.cN,
            float(boundary_price),
        )
        boundary_prices.setdefault(round(candidate_price, 12), candidate_set)

    best_price = current_price
    best_set = current_stage2_result.offloading_set
    best_stage2_result = current_stage2_result
    best_revenue = float(current_revenue)

    for price_key in sorted(boundary_prices):
        candidate_price = float(price_key)
        if abs(candidate_price - current_price) <= 1e-12:
            continue
        if provider == "E":
            eval_pE, eval_pN = candidate_price, pN
        else:
            eval_pE, eval_pN = pE, candidate_price
        stage2_eval = solve_stage2_scm(
            users,
            eval_pE,
            eval_pN,
            system,
            cfg,
            inner_solver_mode="primal_dual",
        )
        stage2_calls += 1
        realized_revenue = _provider_revenue_from_stage2_result(stage2_eval, eval_pE, eval_pN, provider, system)
        if realized_revenue > best_revenue + cfg.search_improvement_tol:
            best_price = candidate_price
            best_set = stage2_eval.offloading_set
            best_stage2_result = stage2_eval
            best_revenue = float(realized_revenue)

    gain = max(0.0, float(best_revenue - current_revenue))
    return BoundaryPriceBROracleResult(
        provider=provider,
        best_price=float(best_price),
        best_set=best_set,
        best_stage2_result=best_stage2_result,
        current_revenue=float(current_revenue),
        best_revenue=float(best_revenue),
        gain=float(gain),
        stage2_calls=stage2_calls,
        candidate_family_size=len(family),
        evaluated_boundary_points=len(boundary_prices),
    )


def algorithm_3_boundary_price_br_estimation(
    users: UserBatch,
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
    cfg: StackelbergConfig,
    *,
    current_stage2_result: GreedySelectionResult | None = None,
    current_revenue: float | None = None,
) -> BoundaryPriceBROracleResult:
    data = _build_data(users)
    return _paper_boundary_price_br_oracle(
        users,
        data,
        provider,
        pE,
        pN,
        system,
        cfg,
        current_stage2_result=current_stage2_result,
        current_revenue=current_revenue,
    )


def _refine_price_for_fixed_set(
    users: UserBatch,
    offloading_set: tuple[int, ...],
    price: tuple[float, float],
    system: SystemConfig,
    iters: int = 6,
    damping: float = 0.7,
) -> tuple[float, float]:
    if not offloading_set:
        return price
    data = _build_data(users)
    pE, pN = price
    for _ in range(max(1, iters)):
        brE = _boundary_price_for_provider(data, offloading_set, pN, "E", system)
        brN = _boundary_price_for_provider(data, offloading_set, pE, "N", system)
        tgtE = float(brE) if brE is not None else pE
        tgtN = float(brN) if brN is not None else pN
        newE = max(system.cE, (1.0 - damping) * pE + damping * tgtE)
        newN = max(system.cN, (1.0 - damping) * pN + damping * tgtN)
        if abs(newE - pE) + abs(newN - pN) < 1e-6:
            pE, pN = newE, newN
            break
        pE, pN = newE, newN
    return float(pE), float(pN)


def algorithm_paper_iterative_pricing_stage1(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    """Paper-aligned Stage I pipeline: boundary-price BR estimation + iterative pricing."""
    update_mode = cfg.paper_outer_update_mode
    next_update_esp = (update_mode == "esp_first")
    pE = max(cfg.initial_pE, system.cE)
    pN = max(cfg.initial_pN, system.cN)
    stopping_reason = "max_iters"

    stage2_oracle_calls = 0
    evaluated_candidates = 0
    evaluated_boundary_points = 0
    trajectory: list[SearchStep] = []
    prev_gap: float | None = None

    final_stage2: GreedySelectionResult | None = None
    final_br_E: BoundaryPriceBROracleResult | None = None
    final_br_N: BoundaryPriceBROracleResult | None = None

    for t in range(cfg.search_max_iters):
        stage2_cur = solve_stage2_scm(
            users,
            pE,
            pN,
            system,
            cfg,
            inner_solver_mode="primal_dual",
        )
        stage2_oracle_calls += 1
        current_revenue_E = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "E", system)
        current_revenue_N = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "N", system)

        br_E = algorithm_3_boundary_price_br_estimation(
            users,
            pE,
            pN,
            "E",
            system,
            cfg,
            current_stage2_result=stage2_cur,
            current_revenue=current_revenue_E,
        )
        br_N = algorithm_3_boundary_price_br_estimation(
            users,
            pE,
            pN,
            "N",
            system,
            cfg,
            current_stage2_result=stage2_cur,
            current_revenue=current_revenue_N,
        )
        stage2_oracle_calls += br_E.stage2_calls + br_N.stage2_calls
        evaluated_candidates += br_E.candidate_family_size + br_N.candidate_family_size
        evaluated_boundary_points += br_E.evaluated_boundary_points + br_N.evaluated_boundary_points

        restricted_gap = max(br_E.gain, br_N.gain)
        gap_delta = float("nan") if prev_gap is None else (prev_gap - restricted_gap)
        prev_gap = restricted_gap

        trajectory.append(
            SearchStep(
                iteration=t,
                offloading_set=stage2_cur.offloading_set,
                pE=float(pE),
                pN=float(pN),
                epsilon=float(restricted_gap),
                epsilon_delta=float(gap_delta),
                esp_best_set_size=len(br_E.best_set),
                nsp_best_set_size=len(br_N.best_set),
                esp_gain=float(br_E.gain),
                nsp_gain=float(br_N.gain),
                restricted_gap=float(restricted_gap),
                restricted_gap_delta=float(gap_delta),
                candidate_family_size=max(br_E.candidate_family_size, br_N.candidate_family_size),
                esp_candidate_family_size=br_E.candidate_family_size,
                nsp_candidate_family_size=br_N.candidate_family_size,
                stage2_offloading_size=len(stage2_cur.offloading_set),
                stage2_social_cost=float(stage2_cur.social_cost),
            )
        )

        if restricted_gap <= cfg.paper_restricted_gap_tol:
            stopping_reason = "restricted_gap_tolerance"
            final_stage2 = stage2_cur
            final_br_E = br_E
            final_br_N = br_N
            break

        if update_mode == "gain_max":
            update_esp = br_E.gain >= br_N.gain
        elif update_mode == "gain_min":
            update_esp = br_E.gain <= br_N.gain
        else:
            update_esp = next_update_esp
            next_update_esp = not next_update_esp

        if update_esp:
            pE = max(system.cE, float(br_E.best_price))
        else:
            pN = max(system.cN, float(br_N.best_price))
    else:
        final_stage2 = None

    if final_stage2 is None:
        final_stage2 = solve_stage2_scm(
            users,
            pE,
            pN,
            system,
            cfg,
            inner_solver_mode="primal_dual",
        )
        stage2_oracle_calls += 1

    current_revenue_E = _provider_revenue_from_stage2_result(final_stage2, pE, pN, "E", system)
    current_revenue_N = _provider_revenue_from_stage2_result(final_stage2, pE, pN, "N", system)

    if final_br_E is None or final_br_N is None:
        final_br_E = algorithm_3_boundary_price_br_estimation(
            users,
            pE,
            pN,
            "E",
            system,
            cfg,
            current_stage2_result=final_stage2,
            current_revenue=current_revenue_E,
        )
        final_br_N = algorithm_3_boundary_price_br_estimation(
            users,
            pE,
            pN,
            "N",
            system,
            cfg,
            current_stage2_result=final_stage2,
            current_revenue=current_revenue_N,
        )
        stage2_oracle_calls += final_br_E.stage2_calls + final_br_N.stage2_calls
        evaluated_candidates += final_br_E.candidate_family_size + final_br_N.candidate_family_size
        evaluated_boundary_points += final_br_E.evaluated_boundary_points + final_br_N.evaluated_boundary_points

    final_gain_E = GainApproxResult(
        provider="E",
        gain=float(final_br_E.gain),
        best_set=final_br_E.best_set,
        current_revenue=float(current_revenue_E),
        candidate_count=int(final_br_E.candidate_family_size),
    )
    final_gain_N = GainApproxResult(
        provider="N",
        gain=float(final_br_N.gain),
        best_set=final_br_N.best_set,
        current_revenue=float(current_revenue_N),
        candidate_count=int(final_br_N.candidate_family_size),
    )
    final_gap = max(final_gain_E.gain, final_gain_N.gain)
    final_inner = final_stage2.inner_result
    social_cost = float(final_stage2.social_cost)

    return StackelbergResult(
        price=(float(pE), float(pN)),
        offloading_set=final_stage2.offloading_set,
        epsilon=float(final_gap),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_inner,
        social_cost=float(social_cost),
        trajectory=tuple(trajectory),
        outer_iterations=len(trajectory),
        stage2_oracle_calls=stage2_oracle_calls,
        evaluated_candidates=evaluated_candidates,
        evaluated_boundary_points=evaluated_boundary_points,
        esp_revenue=float(current_revenue_E),
        nsp_revenue=float(current_revenue_N),
        stopping_reason=stopping_reason,
        restricted_gap=float(final_gap),
        final_stage2_result=final_stage2,
        stage1_method="paper_iterative_pricing",
    )


def algorithm_vbbr_brd_stage1(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    """Verified Boundary Best Response Dynamics (VBBR-BRD)."""
    data = _build_data(users)
    allow_exact_inner = not cfg.vbbr_disable_exact_inner
    alpha = float(cfg.vbbr_damping_alpha)
    update_mode = cfg.vbbr_outer_update_mode
    next_update_esp = (update_mode == "esp_first")
    pE = max(cfg.initial_pE, system.cE)
    pN = max(cfg.initial_pN, system.cN)
    stopping_reason = "max_iters"

    stage2_oracle_calls = 0
    evaluated_candidates = 0
    evaluated_boundary_points = 0
    seen_keys: dict[tuple[float, float, tuple[int, ...]], int] = {}
    cycle_hits = 0

    trajectory: list[SearchStep] = []
    prev_eps: float | None = None
    stage2_cur = algorithm_2_heuristic_user_selection(
        users,
        pE,
        pN,
        system,
        cfg,
        allow_exact_inner=allow_exact_inner,
    )
    stage2_oracle_calls += 1
    current_revenue_E = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "E", system)
    current_revenue_N = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "N", system)

    for t in range(cfg.search_max_iters):
        current_set = stage2_cur.offloading_set
        br_E: VBBROracleResult | None = None
        br_N: VBBROracleResult | None = None
        gain_E = float("nan")
        gain_N = float("nan")

        if update_mode in {"gain_max", "gain_min"}:
            br_E = _vbbr_verified_br_oracle(
                users,
                data,
                "E",
                pE,
                pN,
                current_stage2_result=stage2_cur,
                current_revenue=current_revenue_E,
                system=system,
                cfg=cfg,
                allow_exact_inner=allow_exact_inner,
            )
            br_N = _vbbr_verified_br_oracle(
                users,
                data,
                "N",
                pE,
                pN,
                current_stage2_result=stage2_cur,
                current_revenue=current_revenue_N,
                system=system,
                cfg=cfg,
                allow_exact_inner=allow_exact_inner,
            )
            stage2_oracle_calls += br_E.stage2_calls + br_N.stage2_calls
            evaluated_candidates += br_E.evaluated_candidates + br_N.evaluated_candidates
            evaluated_boundary_points += br_E.evaluated_boundary_points + br_N.evaluated_boundary_points

            gain_E = max(0.0, float(br_E.best_revenue - current_revenue_E))
            gain_N = max(0.0, float(br_N.best_revenue - current_revenue_N))
            eps = max(gain_E, gain_N)
            have_bilateral_check = True
            if update_mode == "gain_max":
                update_esp = gain_E >= gain_N
            else:
                update_esp = gain_E <= gain_N
        else:
            update_esp = next_update_esp
            next_update_esp = not next_update_esp
            have_bilateral_check = False

            if update_esp:
                br_E = _vbbr_verified_br_oracle(
                    users,
                    data,
                    "E",
                    pE,
                    pN,
                    current_stage2_result=stage2_cur,
                    current_revenue=current_revenue_E,
                    system=system,
                    cfg=cfg,
                    allow_exact_inner=allow_exact_inner,
                )
                stage2_oracle_calls += br_E.stage2_calls
                evaluated_candidates += br_E.evaluated_candidates
                evaluated_boundary_points += br_E.evaluated_boundary_points
                gain_E = max(0.0, float(br_E.best_revenue - current_revenue_E))
                eps = gain_E
                if gain_E < cfg.vbbr_outer_gain_tol:
                    br_N = _vbbr_verified_br_oracle(
                        users,
                        data,
                        "N",
                        pE,
                        pN,
                        current_stage2_result=stage2_cur,
                        current_revenue=current_revenue_N,
                        system=system,
                        cfg=cfg,
                        allow_exact_inner=allow_exact_inner,
                    )
                    stage2_oracle_calls += br_N.stage2_calls
                    evaluated_candidates += br_N.evaluated_candidates
                    evaluated_boundary_points += br_N.evaluated_boundary_points
                    gain_N = max(0.0, float(br_N.best_revenue - current_revenue_N))
                    eps = max(gain_E, gain_N)
                    have_bilateral_check = True
            else:
                br_N = _vbbr_verified_br_oracle(
                    users,
                    data,
                    "N",
                    pE,
                    pN,
                    current_stage2_result=stage2_cur,
                    current_revenue=current_revenue_N,
                    system=system,
                    cfg=cfg,
                    allow_exact_inner=allow_exact_inner,
                )
                stage2_oracle_calls += br_N.stage2_calls
                evaluated_candidates += br_N.evaluated_candidates
                evaluated_boundary_points += br_N.evaluated_boundary_points
                gain_N = max(0.0, float(br_N.best_revenue - current_revenue_N))
                eps = gain_N
                if gain_N < cfg.vbbr_outer_gain_tol:
                    br_E = _vbbr_verified_br_oracle(
                        users,
                        data,
                        "E",
                        pE,
                        pN,
                        current_stage2_result=stage2_cur,
                        current_revenue=current_revenue_E,
                        system=system,
                        cfg=cfg,
                        allow_exact_inner=allow_exact_inner,
                    )
                    stage2_oracle_calls += br_E.stage2_calls
                    evaluated_candidates += br_E.evaluated_candidates
                    evaluated_boundary_points += br_E.evaluated_boundary_points
                    gain_E = max(0.0, float(br_E.best_revenue - current_revenue_E))
                    eps = max(gain_E, gain_N)
                    have_bilateral_check = True

        eps_delta = float("nan") if prev_eps is None else (prev_eps - eps)
        prev_eps = eps

        trajectory.append(
            SearchStep(
                iteration=t,
                offloading_set=current_set,
                pE=float(pE),
                pN=float(pN),
                epsilon=float(eps),
                epsilon_delta=float(eps_delta),
                esp_best_set_size=(len(br_E.best_set) if br_E is not None else 0),
                nsp_best_set_size=(len(br_N.best_set) if br_N is not None else 0),
                esp_gain=float(gain_E),
                nsp_gain=float(gain_N),
                restricted_gap=float(eps),
                restricted_gap_delta=float(eps_delta),
                candidate_family_size=max(
                    (br_E.max_candidate_family_size if br_E is not None else 0),
                    (br_N.max_candidate_family_size if br_N is not None else 0),
                ),
                esp_candidate_family_size=(br_E.max_candidate_family_size if br_E is not None else 0),
                nsp_candidate_family_size=(br_N.max_candidate_family_size if br_N is not None else 0),
                stage2_offloading_size=len(current_set),
                stage2_social_cost=float(stage2_cur.social_cost),
            )
        )

        key = (round(pE, 10), round(pN, 10), current_set)
        if key in seen_keys:
            cycle_hits += 1
        seen_keys[key] = t

        if cycle_hits >= cfg.vbbr_cycle_window:
            stopping_reason = "cycle_safeguard"
            break
        if have_bilateral_check and gain_E < cfg.vbbr_outer_gain_tol and gain_N < cfg.vbbr_outer_gain_tol:
            stopping_reason = "verified_gain_tolerance"
            break

        selected_oracle = br_E if update_esp else br_N
        if selected_oracle is None:
            continue

        prev_pE, prev_pN = pE, pN

        if update_esp:
            target_pE = max(system.cE, float(selected_oracle.best_price))
            pE = max(system.cE, float((1.0 - alpha) * pE + alpha * target_pE))
        else:
            target_pN = max(system.cN, float(selected_oracle.best_price))
            pN = max(system.cN, float((1.0 - alpha) * pN + alpha * target_pN))

        if abs(pE - prev_pE) + abs(pN - prev_pN) <= 1e-12:
            continue

        reuse_stage2 = False
        if abs(alpha - 1.0) <= 1e-12:
            if update_esp and abs(pE - max(system.cE, float(selected_oracle.best_price))) <= 1e-12:
                stage2_cur = selected_oracle.best_stage2_result
                reuse_stage2 = True
            if (not update_esp) and abs(pN - max(system.cN, float(selected_oracle.best_price))) <= 1e-12:
                stage2_cur = selected_oracle.best_stage2_result
                reuse_stage2 = True

        if not reuse_stage2:
            stage2_cur = algorithm_2_heuristic_user_selection(
                users,
                pE,
                pN,
                system,
                cfg,
                allow_exact_inner=allow_exact_inner,
            )
            stage2_oracle_calls += 1

        current_revenue_E = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "E", system)
        current_revenue_N = _provider_revenue_from_stage2_result(stage2_cur, pE, pN, "N", system)

    stage2_final = stage2_cur
    final_set = stage2_final.offloading_set
    final_inner = stage2_final.inner_result
    social_cost = float(stage2_final.social_cost)
    esp_rev = _provider_revenue_from_stage2_result(stage2_final, pE, pN, "E", system)
    nsp_rev = _provider_revenue_from_stage2_result(stage2_final, pE, pN, "N", system)

    final_br_E = _vbbr_verified_br_oracle(
        users,
        data,
        "E",
        pE,
        pN,
        current_stage2_result=stage2_final,
        current_revenue=esp_rev,
        system=system,
        cfg=cfg,
        allow_exact_inner=allow_exact_inner,
    )
    final_br_N = _vbbr_verified_br_oracle(
        users,
        data,
        "N",
        pE,
        pN,
        current_stage2_result=stage2_final,
        current_revenue=nsp_rev,
        system=system,
        cfg=cfg,
        allow_exact_inner=allow_exact_inner,
    )
    stage2_oracle_calls += final_br_E.stage2_calls + final_br_N.stage2_calls
    evaluated_candidates += final_br_E.evaluated_candidates + final_br_N.evaluated_candidates
    evaluated_boundary_points += final_br_E.evaluated_boundary_points + final_br_N.evaluated_boundary_points

    final_gain_E_val = max(0.0, float(final_br_E.best_revenue - esp_rev))
    final_gain_N_val = max(0.0, float(final_br_N.best_revenue - nsp_rev))
    final_eps = max(final_gain_E_val, final_gain_N_val)

    final_gain_E = GainApproxResult(
        provider="E",
        gain=float(final_gain_E_val),
        best_set=final_br_E.best_set,
        current_revenue=float(esp_rev),
        candidate_count=int(final_br_E.evaluated_candidates),
    )
    final_gain_N = GainApproxResult(
        provider="N",
        gain=float(final_gain_N_val),
        best_set=final_br_N.best_set,
        current_revenue=float(nsp_rev),
        candidate_count=int(final_br_N.evaluated_candidates),
    )

    return StackelbergResult(
        price=(float(pE), float(pN)),
        offloading_set=final_set,
        epsilon=float(final_eps),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_inner,
        social_cost=float(social_cost),
        trajectory=tuple(trajectory),
        outer_iterations=len(trajectory),
        stage2_oracle_calls=stage2_oracle_calls,
        evaluated_candidates=evaluated_candidates,
        evaluated_boundary_points=evaluated_boundary_points,
        esp_revenue=float(esp_rev),
        nsp_revenue=float(nsp_rev),
        stopping_reason=stopping_reason,
        restricted_gap=float(final_eps),
        final_stage2_result=stage2_final,
        stage1_method="vbbr_brd",
    )


def algorithm_topk_brd_stage1(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    """Alternating top-k best-response dynamics over ESP/NSP prices."""
    data = _build_data(users)
    pE = max(cfg.initial_pE, system.cE)
    pN = max(cfg.initial_pN, system.cN)
    stopping_reason = "max_iters"
    seen_keys: dict[tuple[float, float, tuple[int, ...]], int] = {}
    cycle_hits = 0

    trajectory: list[SearchStep] = []
    prev_eps: float | None = None
    current_set: tuple[int, ...] = tuple()

    for t in range(cfg.search_max_iters):
        gE = algorithm_3_gain_approximation(
            users, current_set, pE, pN, "E", system,
            estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
        )
        next_set_E = gE.best_set if gE.gain > cfg.search_improvement_tol else current_set
        brE = _boundary_price_for_provider(data, next_set_E, pN, "E", system)
        pE_next = max(system.cE, float(brE)) if brE is not None else pE

        gN = algorithm_3_gain_approximation(
            users, next_set_E, pE_next, pN, "N", system,
            estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
        )
        next_set_N = gN.best_set if gN.gain > cfg.search_improvement_tol else next_set_E
        brN = _boundary_price_for_provider(data, next_set_N, pE_next, "N", system)
        pN_next = max(system.cN, float(brN)) if brN is not None else pN

        gE_eval = algorithm_3_gain_approximation(
            users, next_set_N, pE_next, pN_next, "E", system,
            estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
        )
        gN_eval = algorithm_3_gain_approximation(
            users, next_set_N, pE_next, pN_next, "N", system,
            estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
        )
        eps = max(gE_eval.gain, gN_eval.gain)
        eps_delta = float("nan") if prev_eps is None else (prev_eps - eps)
        prev_eps = eps

        trajectory.append(SearchStep(
            iteration=t,
            offloading_set=next_set_N,
            pE=float(pE_next),
            pN=float(pN_next),
            epsilon=float(eps),
            epsilon_delta=eps_delta,
            esp_best_set_size=len(gE.best_set),
            nsp_best_set_size=len(gN.best_set),
            esp_gain=float(gE.gain),
            nsp_gain=float(gN.gain),
            restricted_gap=float(eps),
            restricted_gap_delta=float(eps_delta),
            stage2_offloading_size=len(next_set_N),
        ))

        price_move = abs(pE_next - pE) + abs(pN_next - pN)
        pE, pN = float(pE_next), float(pN_next)
        current_set = next_set_N

        key = (round(pE, 10), round(pN, 10), current_set)
        if key in seen_keys:
            cycle_hits += 1
        seen_keys[key] = t

        if price_move <= cfg.topk_brd_price_tol and abs(float(eps_delta)) <= cfg.topk_brd_epsilon_tol:
            stopping_reason = "price_and_epsilon_converged"
            break
        if eps <= cfg.topk_brd_epsilon_tol:
            stopping_reason = "epsilon_tolerance"
            break
        if cycle_hits >= cfg.topk_brd_cycle_window:
            stopping_reason = "cycle_safeguard"
            break

    final_stage2 = solve_stage2_scm(
        users,
        pE,
        pN,
        system,
        cfg,
        inner_solver_mode="primal_dual",
    )
    final_set = final_stage2.offloading_set
    final_gain_E = algorithm_3_gain_approximation(
        users, final_set, pE, pN, "E", system,
        estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
    )
    final_gain_N = algorithm_3_gain_approximation(
        users, final_set, pE, pN, "N", system,
        estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
    )
    final_eps = max(final_gain_E.gain, final_gain_N.gain)
    final_inner = final_stage2.inner_result
    social_cost = float(final_stage2.social_cost)

    esp_rev = _provider_revenue_from_stage2_result(final_stage2, pE, pN, "E", system)
    nsp_rev = _provider_revenue_from_stage2_result(final_stage2, pE, pN, "N", system)

    return StackelbergResult(
        price=(float(pE), float(pN)),
        offloading_set=final_set,
        epsilon=float(final_eps),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_inner,
        social_cost=float(social_cost),
        trajectory=tuple(trajectory),
        outer_iterations=len(trajectory),
        stage2_oracle_calls=1,
        evaluated_candidates=len(trajectory) * 2,
        evaluated_boundary_points=0,
        esp_revenue=float(esp_rev),
        nsp_revenue=float(nsp_rev),
        stopping_reason=stopping_reason,
        restricted_gap=float(final_eps),
        final_stage2_result=final_stage2,
        stage1_method="topk_brd",
    )


def run_stage1_solver(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    if cfg.stage1_solver_variant == "paper_iterative_pricing":
        return algorithm_paper_iterative_pricing_stage1(users, system, cfg)
    if cfg.stage1_solver_variant == "topk_brd":
        return algorithm_topk_brd_stage1(users, system, cfg)
    if cfg.stage1_solver_variant == "vbbr_brd":
        return algorithm_vbbr_brd_stage1(users, system, cfg)
    return algorithm_paper_iterative_pricing_stage1(users, system, cfg)


def solve_stage1_pricing(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    return run_stage1_solver(users, system, cfg)


def summarize_stackelberg_result(
    users: UserBatch,
    result: StackelbergResult,
    system: SystemConfig,
) -> str:
    pE, pN = result.price
    ratio = (len(result.offloading_set) / users.n) if users.n else 0.0
    rev_E = float(result.esp_revenue)
    rev_N = float(result.nsp_revenue)
    restricted_gap = float(result.restricted_gap if np.isfinite(result.restricted_gap) else result.epsilon)
    lines = [
        "Stackelberg Stage-I result",
        f"price = ({pE:.10g}, {pN:.10g})",
        f"epsilon = {result.epsilon:.10g}",
        f"restricted_gap = {restricted_gap:.10g}",
        f"offloading_users = {len(result.offloading_set)}/{users.n} ({ratio:.2%})",
        f"social_cost = {result.social_cost:.10g}",
        f"esp_revenue = {rev_E:.10g}",
        f"nsp_revenue = {rev_N:.10g}",
        f"inner_converged = {result.inner_result.converged}",
        f"inner_iterations = {result.inner_result.iterations}",
        f"search_steps = {len(result.trajectory)}",
        f"outer_iterations = {result.outer_iterations}",
        f"stage2_oracle_calls = {result.stage2_oracle_calls}",
        f"evaluated_candidates = {result.evaluated_candidates}",
        f"evaluated_boundary_points = {result.evaluated_boundary_points}",
        f"stopping_reason = {result.stopping_reason}",
        f"stage1_method = {result.stage1_method}",
    ]
    if result.final_stage2_result is not None:
        lines.extend(
            [
                f"final_stage2_runtime_sec = {float(result.final_stage2_result.runtime_sec):.10g}",
                f"final_stage2_offloading_size = {len(result.final_stage2_result.offloading_set)}",
                f"final_stage2_inner_solver_mode = {result.final_stage2_result.inner_solver_mode}",
            ]
        )
    return "\n".join(lines) + "\n"

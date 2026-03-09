from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize

from .config import StackelbergConfig, SystemConfig
from .model import UserBatch, local_cost, theta

Provider = Literal["E", "N"]
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


@dataclass(frozen=True)
class RNEResult:
    offloading_set: tuple[int, ...]
    price: tuple[float, float]
    epsilon: float
    gain_E: GainApproxResult
    gain_N: GainApproxResult


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


def _offload_cost_for_user(
    data: _ProblemData,
    user_idx: int,
    f_i: float,
    b_i: float,
    pE: float,
    pN: float,
) -> float:
    return data.aw[user_idx] / f_i + data.th[user_idx] / b_i + pE * f_i + pN * b_i


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
    tE, tN = _tilde_prices(data, offloading_set, pE, pN, system)
    ce_star = 2.0 * math.sqrt(data.aw[user_idx] * tE) + 2.0 * math.sqrt(data.th[user_idx] * tN)
    return float(data.cl[user_idx] - ce_star)


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
        tilde_pN = max(opponent_price, bar_pN)
        for i in idx:
            numerator = data.cl[i] - 2.0 * math.sqrt(data.th[i] * tilde_pN)
            if numerator <= 0:
                continue
            p_i = (numerator / (2.0 * math.sqrt(data.aw[i]))) ** 2
            if p_i > bar_pE + _EPS and p_i >= system.cE:
                feasible.append(float(p_i))
    else:
        tilde_pE = max(opponent_price, bar_pE)
        for i in idx:
            numerator = data.cl[i] - 2.0 * math.sqrt(data.aw[i] * tilde_pE)
            if numerator <= 0:
                continue
            p_i = (numerator / (2.0 * math.sqrt(data.th[i]))) ** 2
            if p_i > bar_pN + _EPS and p_i >= system.cN:
                feasible.append(float(p_i))

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

    for _ in range(cfg.vbbr_oracle_max_rounds):
        family = _vbbr_local_candidate_family(data, anchor_set, pE, pN, system, cfg)
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
) -> GreedySelectionResult:
    data = _build_data(users)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()

    previous_ve = 0.0
    last_added: int | None = None
    iterations = 0

    for t in range(cfg.greedy_max_iters):
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system) if allow_exact_inner else None
        if inner is None:
            inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
        ve = inner.offloading_objective

        if t >= 1 and last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system) if allow_exact_inner else None
                if inner is None:
                    inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
                ve = inner.offloading_objective

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
            iterations = t + 1
            continue

        iterations = t + 1
        break

    final_inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system) if allow_exact_inner else None
    if final_inner is None:
        final_inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
    final_set = final_inner.offloading_set
    outside = set(range(users.n)) - set(final_set)
    social_cost = final_inner.offloading_objective + float(np.sum(data.cl[list(outside)])) if outside else final_inner.offloading_objective
    return GreedySelectionResult(
        offloading_set=final_set,
        inner_result=final_inner,
        social_cost=float(social_cost),
        iterations=iterations,
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


def _find_boundary_step_for_user(
    data: _ProblemData,
    offloading_set: tuple[int, ...],
    user_idx: int,
    p_in: tuple[float, float],
    direction: tuple[float, float],
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> float | None:
    dE, dN = direction
    if dE < 0 or dN < 0:
        return None
    if dE == 0 and dN == 0:
        return None

    def margin_at(t: float) -> float:
        pE = p_in[0] + t * dE
        pN = p_in[1] + t * dN
        return _margin_for_user(data, offloading_set, user_idx, pE, pN, system)

    m0 = margin_at(0.0)
    if m0 <= 0.0:
        return 0.0

    low = 0.0
    high = 1.0
    found = False
    for _ in range(cfg.rne_max_expand_steps):
        if margin_at(high) <= 0.0:
            found = True
            break
        high *= 2.0
    if not found:
        return None

    for _ in range(64):
        mid = 0.5 * (low + high)
        if margin_at(mid) > 0.0:
            low = mid
        else:
            high = mid
        if high - low <= cfg.rne_root_tol:
            break
    return high


def _sample_directions(count: int) -> list[tuple[float, float]]:
    # Midpoint sampling avoids exactly axis-aligned duplicates.
    angles = (np.arange(count, dtype=float) + 0.5) * (0.5 * math.pi / count)
    return [(float(math.cos(a)), float(math.sin(a))) for a in angles]


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


def algorithm_4_optimal_rne_sampling(
    users: UserBatch,
    offloading_set: Iterable[int],
    initial_price: tuple[float, float],
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> RNEResult:
    data = _build_data(users)
    X = _sorted_tuple(offloading_set)
    p_in = (max(initial_price[0], system.cE), max(initial_price[1], system.cN))

    if not X:
        gain_E = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "E", system, estimator_variant=cfg.gain_estimator_variant)
        gain_N = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "N", system, estimator_variant=cfg.gain_estimator_variant)
        eps = max(gain_E.gain, gain_N.gain)
        return RNEResult(offloading_set=X, price=p_in, epsilon=eps, gain_E=gain_E, gain_N=gain_N)

    best_price = p_in
    best_eps = float("inf")
    best_gain_E: GainApproxResult | None = None
    best_gain_N: GainApproxResult | None = None

    directions = _sample_directions(cfg.rne_directions)
    for direction in directions:
        t_values: list[float] = []
        for i in X:
            t_i = _find_boundary_step_for_user(data, X, i, p_in, direction, system, cfg)
            if t_i is not None:
                t_values.append(t_i)
        if not t_values:
            continue

        t_min = min(t_values)
        p_eval = (p_in[0] + t_min * direction[0], p_in[1] + t_min * direction[1])
        gain_E = algorithm_3_gain_approximation(users, X, p_eval[0], p_eval[1], "E", system, estimator_variant=cfg.gain_estimator_variant)
        gain_N = algorithm_3_gain_approximation(users, X, p_eval[0], p_eval[1], "N", system, estimator_variant=cfg.gain_estimator_variant)
        eps = max(gain_E.gain, gain_N.gain)

        if eps < best_eps:
            best_eps = eps
            best_price = p_eval
            best_gain_E = gain_E
            best_gain_N = gain_N

    if best_gain_E is None or best_gain_N is None:
        best_gain_E = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "E", system, estimator_variant=cfg.gain_estimator_variant)
        best_gain_N = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "N", system, estimator_variant=cfg.gain_estimator_variant)
        best_eps = max(best_gain_E.gain, best_gain_N.gain)

    return RNEResult(
        offloading_set=X,
        price=best_price,
        epsilon=float(best_eps),
        gain_E=best_gain_E,
        gain_N=best_gain_N,
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


def _best_local_price_on_fixed_set(
    users: UserBatch,
    offloading_set: tuple[int, ...],
    price: tuple[float, float],
    system: SystemConfig,
) -> tuple[tuple[float, float], float]:
    """Try local boundary-guided price updates while keeping the offloading set fixed."""
    data = _build_data(users)
    pE, pN = float(price[0]), float(price[1])
    brE = _boundary_price_for_provider(data, offloading_set, pN, "E", system)
    brN = _boundary_price_for_provider(data, offloading_set, pE, "N", system)
    tgtE = float(brE) if brE is not None else pE
    tgtN = float(brN) if brN is not None else pN

    candidates: list[tuple[float, float]] = [(pE, pN)]
    for alpha in (0.35, 0.65, 1.0):
        cE = max(system.cE, (1.0 - alpha) * pE + alpha * tgtE)
        cN = max(system.cN, (1.0 - alpha) * pN + alpha * tgtN)
        candidates.append((float(cE), float(cN)))

    best_price = candidates[0]
    gE = algorithm_3_gain_approximation(users, offloading_set, best_price[0], best_price[1], "E", system)
    gN = algorithm_3_gain_approximation(users, offloading_set, best_price[0], best_price[1], "N", system)
    best_eps = max(gE.gain, gN.gain)

    for cand in candidates[1:]:
        cE_gain = algorithm_3_gain_approximation(users, offloading_set, cand[0], cand[1], "E", system)
        cN_gain = algorithm_3_gain_approximation(users, offloading_set, cand[0], cand[1], "N", system)
        cand_eps = max(cE_gain.gain, cN_gain.gain)
        if cand_eps < best_eps:
            best_eps = cand_eps
            best_price = cand

    return best_price, float(best_eps)


def _perturb_restart_candidate(
    users: UserBatch,
    offloading_set: tuple[int, ...],
    current_price: tuple[float, float],
    system: SystemConfig,
    cfg: StackelbergConfig,
    step_idx: int,
) -> tuple[tuple[int, ...], tuple[float, float], float, int] | None:
    """Try deterministic perturb-restart around current price to escape local plateaus."""
    base_pE, base_pN = float(current_price[0]), float(current_price[1])
    base_gE = algorithm_3_gain_approximation(users, offloading_set, base_pE, base_pN, "E", system, estimator_variant=cfg.gain_estimator_variant)
    base_gN = algorithm_3_gain_approximation(users, offloading_set, base_pE, base_pN, "N", system, estimator_variant=cfg.gain_estimator_variant)
    best_eps = max(base_gE.gain, base_gN.gain)
    best: tuple[tuple[int, ...], tuple[float, float], float] | None = None

    rng = np.random.default_rng(2026 + 97 * step_idx + 13 * len(offloading_set))
    scales = (0.08, 0.2, 0.4)
    probes: list[tuple[float, float]] = []
    for s in scales:
        for _ in range(4):
            dE = float(rng.uniform(-s, s))
            dN = float(rng.uniform(-s, s))
            probes.append((max(system.cE, base_pE * (1.0 + dE)), max(system.cN, base_pN * (1.0 + dN))))

    stage2_calls = 0
    for pE, pN in probes:
        s2 = algorithm_2_heuristic_user_selection(users, pE, pN, system, cfg)
        stage2_calls += 1
        cand_set = s2.offloading_set
        cand_price = _refine_price_for_fixed_set(users, cand_set, (pE, pN), system)
        gE = algorithm_3_gain_approximation(users, cand_set, cand_price[0], cand_price[1], "E", system, estimator_variant=cfg.gain_estimator_variant)
        gN = algorithm_3_gain_approximation(users, cand_set, cand_price[0], cand_price[1], "N", system, estimator_variant=cfg.gain_estimator_variant)
        eps = max(gE.gain, gN.gain)
        if eps + cfg.search_improvement_tol < best_eps:
            best_eps = eps
            best = (cand_set, cand_price, float(eps))

    if best is None:
        return None
    return best[0], best[1], best[2], stage2_calls


def _compute_se_proxy_price(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> tuple[float, float]:
    """Proxy for true Stackelberg equilibrium via dense grid-search baseline."""
    from .baselines import baseline_stage1_grid_search_oracle
    from .config import BaselineConfig

    base_cfg = BaselineConfig(
        enabled=False,
        random_seed=2026,
        exact_max_users=16,
        stage2_solver_for_pricing="DG",
        max_price_E=max(cfg.initial_pE * 2.0, system.cE + 1.0),
        max_price_N=max(cfg.initial_pN * 2.0, system.cN + 1.0),
        gso_grid_points=41,
        pbdr_grid_points=31,
        pbdr_max_iters=30,
        pbdr_tol=1e-4,
        bo_init_points=10,
        bo_iters=10,
        bo_candidate_pool=32,
        bo_kernel_bandwidth=0.25,
        bo_ucb_beta=2.0,
        drl_price_levels=9,
        drl_episodes=10,
        drl_steps_per_episode=10,
        drl_alpha=0.1,
        drl_gamma=0.95,
        drl_epsilon=0.2,
        market_max_iters=20,
        market_step_size=0.2,
        market_tol=1e-4,
        single_sp_max_iters=50,
        random_offloading_trials=16,
        random_offloading_prob=0.5,
        ubrd_max_iters=50,
        vi_max_iters=50,
        vi_step_size=0.5,
        vi_tol=1e-5,
        penalty_outer_iters=4,
        penalty_inner_iters=20,
        penalty_init_rho=0.1,
        penalty_rho_scale=4.0,
        penalty_tol=1e-4,
        cs_use_minlp=False,
        cs_fallback_to_enum=True,
        gekko_time_limit=30,
        gekko_max_iter=200,
        gekko_mip_gap=1e-4,
    )
    out = baseline_stage1_grid_search_oracle(users, system, cfg, base_cfg)
    return float(out.price[0]), float(out.price[1])


def algorithm_5_stackelberg_guided_search(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    """Algorithm 5 with additional Stage-I price refinement for stable convergence."""
    stage2_oracle_calls = 0
    evaluated_candidates = 0
    evaluated_boundary_points = 0
    stopping_reason = "max_iters"

    initial_price = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))
    init_stage2 = algorithm_2_heuristic_user_selection(users, initial_price[0], initial_price[1], system, cfg)
    stage2_oracle_calls += 1
    current_set = init_stage2.offloading_set
    current_price = _refine_price_for_fixed_set(users, current_set, initial_price, system)

    se_pE, se_pN = float("nan"), float("nan")
    if users.n <= 16:
        se_pE, se_pN = _compute_se_proxy_price(users, system, cfg)
    trajectory: list[SearchStep] = []
    outer_iterations = 0
    prev_eps: float | None = None
    no_improve_rounds = 0

    for t in range(cfg.search_max_iters):
        outer_iterations = t + 1
        current_price = _refine_price_for_fixed_set(users, current_set, current_price, system)

        gain_E = algorithm_3_gain_approximation(
            users,
            current_set,
            current_price[0],
            current_price[1],
            "E",
            system,
            estimator_variant=cfg.gain_estimator_variant,
        )
        gain_N = algorithm_3_gain_approximation(
            users,
            current_set,
            current_price[0],
            current_price[1],
            "N",
            system,
            estimator_variant=cfg.gain_estimator_variant,
        )
        current_eps = max(gain_E.gain, gain_N.gain)
        dist_to_se = math.sqrt((current_price[0] - se_pE) ** 2 + (current_price[1] - se_pN) ** 2)
        eps_delta = float("nan") if prev_eps is None else (prev_eps - current_eps)
        prev_eps = current_eps

        trajectory.append(SearchStep(
            iteration=t,
            offloading_set=current_set,
            pE=current_price[0],
            pN=current_price[1],
            epsilon=current_eps,
            dist_to_se=dist_to_se,
            epsilon_delta=eps_delta,
        ))

        exact_targets: list[tuple[int, ...]] = []
        for cand in (gain_E.best_set, gain_N.best_set):
            if cand not in exact_targets and cand != current_set:
                exact_targets.append(cand)

        best_candidate: tuple[int, ...] | None = None
        best_eps = current_eps

        data = _build_data(users)
        neighborhood = _candidate_family(data, current_set, current_price[0], current_price[1], system)

        if cfg.stage1_neighborhood_mode == "full_search":
            eval_pool = [c for c in neighborhood if c != current_set]
        else:
            tail_pool = [c for c in neighborhood if c != current_set and c not in exact_targets]
            eval_pool = [*exact_targets, *tail_pool]

        for candidate_set in eval_pool[: cfg.stage1_neighborhood_max_candidates]:
            evaluated_candidates += 1
            candidate_price = _refine_price_for_fixed_set(users, candidate_set, current_price, system)
            cand_gE = algorithm_3_gain_approximation(
                users,
                candidate_set,
                candidate_price[0],
                candidate_price[1],
                "E",
                system,
                estimator_variant=cfg.gain_estimator_variant,
            )
            cand_gN = algorithm_3_gain_approximation(
                users,
                candidate_set,
                candidate_price[0],
                candidate_price[1],
                "N",
                system,
                estimator_variant=cfg.gain_estimator_variant,
            )
            cand_eps = max(cand_gE.gain, cand_gN.gain)
            if cand_eps + cfg.search_improvement_tol < best_eps:
                best_eps = cand_eps
                best_candidate = candidate_set

        if best_candidate is None:
            local_price, local_eps = _best_local_price_on_fixed_set(users, current_set, current_price, system)
            if local_eps + cfg.search_improvement_tol < current_eps:
                current_price = local_price
                no_improve_rounds = 0
                continue

            restart = _perturb_restart_candidate(users, current_set, current_price, system, cfg, t)
            if restart is not None:
                restart_set, restart_price, restart_eps, restart_calls = restart
                stage2_oracle_calls += restart_calls
                if restart_eps + cfg.search_improvement_tol < current_eps:
                    current_set = restart_set
                    current_price = restart_price
                    no_improve_rounds = 0
                    continue

            no_improve_rounds += 1
            if no_improve_rounds >= 3:
                stopping_reason = "no_improving_candidate"
                break
            continue

        if best_candidate == current_set:
            stopping_reason = "fixed_point_reached"
            break

        current_set = best_candidate
        current_price = _refine_price_for_fixed_set(users, current_set, current_price, system)
        no_improve_rounds = 0

    final_set = current_set
    final_gain_E = algorithm_3_gain_approximation(
        users,
        final_set,
        current_price[0],
        current_price[1],
        "E",
        system,
        estimator_variant=cfg.gain_estimator_variant,
    )
    final_gain_N = algorithm_3_gain_approximation(
        users,
        final_set,
        current_price[0],
        current_price[1],
        "N",
        system,
        estimator_variant=cfg.gain_estimator_variant,
    )
    final_eps = max(final_gain_E.gain, final_gain_N.gain)

    final_inner = _solve_fixed_set_inner_exact(users, final_set, current_price[0], current_price[1], system)
    if final_inner is None:
        final_inner = algorithm_1_distributed_primal_dual(users, final_set, current_price[0], current_price[1], system, cfg)
    data = _build_data(users)
    outside = set(range(users.n)) - set(final_set)
    social_cost = final_inner.offloading_objective + (float(np.sum(data.cl[list(outside)])) if outside else 0.0)

    esp_rev = _provider_revenue(data, final_set, current_price[0], current_price[1], "E", system)
    nsp_rev = _provider_revenue(data, final_set, current_price[0], current_price[1], "N", system)

    return StackelbergResult(
        price=current_price,
        offloading_set=final_set,
        epsilon=float(final_eps),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_inner,
        social_cost=float(social_cost),
        trajectory=tuple(trajectory),
        outer_iterations=outer_iterations,
        stage2_oracle_calls=stage2_oracle_calls,
        evaluated_candidates=evaluated_candidates,
        evaluated_boundary_points=evaluated_boundary_points,
        esp_revenue=float(esp_rev),
        nsp_revenue=float(nsp_rev),
        stopping_reason=stopping_reason,
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

    final_gain_E = algorithm_3_gain_approximation(
        users, current_set, pE, pN, "E", system,
        estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
    )
    final_gain_N = algorithm_3_gain_approximation(
        users, current_set, pE, pN, "N", system,
        estimator_variant="topk_real_reval", top_k=cfg.gain_topk_k,
    )
    final_eps = max(final_gain_E.gain, final_gain_N.gain)
    final_inner = _solve_fixed_set_inner_exact(users, current_set, pE, pN, system)
    if final_inner is None:
        final_inner = algorithm_1_distributed_primal_dual(users, current_set, pE, pN, system, cfg)
    outside = set(range(users.n)) - set(current_set)
    social_cost = final_inner.offloading_objective + (float(np.sum(data.cl[list(outside)])) if outside else 0.0)

    esp_rev = _provider_revenue(data, current_set, pE, pN, "E", system)
    nsp_rev = _provider_revenue(data, current_set, pE, pN, "N", system)

    return StackelbergResult(
        price=(float(pE), float(pN)),
        offloading_set=current_set,
        epsilon=float(final_eps),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_inner,
        social_cost=float(social_cost),
        trajectory=tuple(trajectory),
        outer_iterations=len(trajectory),
        stage2_oracle_calls=0,
        evaluated_candidates=len(trajectory) * 2,
        evaluated_boundary_points=0,
        esp_revenue=float(esp_rev),
        nsp_revenue=float(nsp_rev),
        stopping_reason=stopping_reason,
    )


def run_stage1_solver(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    if cfg.stage1_solver_variant == "topk_brd":
        return algorithm_topk_brd_stage1(users, system, cfg)
    if cfg.stage1_solver_variant == "algorithm5":
        return algorithm_5_stackelberg_guided_search(users, system, cfg)
    if cfg.stage1_solver_variant == "vbbr_brd":
        return algorithm_vbbr_brd_stage1(users, system, cfg)
    return algorithm_5_stackelberg_guided_search(users, system, cfg)


def summarize_stackelberg_result(
    users: UserBatch,
    result: StackelbergResult,
    system: SystemConfig,
) -> str:
    pE, pN = result.price
    ratio = (len(result.offloading_set) / users.n) if users.n else 0.0
    rev_E = float(result.esp_revenue)
    rev_N = float(result.nsp_revenue)
    lines = [
        "Stackelberg Stage-I result",
        f"price = ({pE:.10g}, {pN:.10g})",
        f"epsilon = {result.epsilon:.10g}",
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
    ]
    return "\n".join(lines) + "\n"

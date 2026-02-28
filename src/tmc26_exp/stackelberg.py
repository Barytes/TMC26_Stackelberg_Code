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
) -> GreedySelectionResult:
    data = _build_data(users)
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()

    previous_ve = 0.0
    last_added: int | None = None
    iterations = 0

    for t in range(cfg.greedy_max_iters):
        inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
        if inner is None:
            inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, cfg)
        ve = inner.offloading_objective

        if t >= 1 and last_added is not None:
            delta_true = ve - previous_ve - data.cl[last_added]
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
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

    final_inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
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


def algorithm_3_gain_approximation(
    users: UserBatch,
    current_set: Iterable[int],
    pE: float,
    pN: float,
    provider: Provider,
    system: SystemConfig,
) -> GainApproxResult:
    data = _build_data(users)
    X = _sorted_tuple(current_set)
    current_revenue = _provider_revenue(data, X, pE, pN, provider, system)
    family = _candidate_family(data, X, pE, pN, system)

    best_gain = 0.0
    best_set = X
    for Y in family:
        candidate_revenue = _boundary_revenue_for_provider(data, Y, pE, pN, provider, system)
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
        gain_E = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "E", system)
        gain_N = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "N", system)
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
        gain_E = algorithm_3_gain_approximation(users, X, p_eval[0], p_eval[1], "E", system)
        gain_N = algorithm_3_gain_approximation(users, X, p_eval[0], p_eval[1], "N", system)
        eps = max(gain_E.gain, gain_N.gain)

        if eps < best_eps:
            best_eps = eps
            best_price = p_eval
            best_gain_E = gain_E
            best_gain_N = gain_N

    if best_gain_E is None or best_gain_N is None:
        best_gain_E = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "E", system)
        best_gain_N = algorithm_3_gain_approximation(users, X, p_in[0], p_in[1], "N", system)
        best_eps = max(best_gain_E.gain, best_gain_N.gain)

    return RNEResult(
        offloading_set=X,
        price=best_price,
        epsilon=float(best_eps),
        gain_E=best_gain_E,
        gain_N=best_gain_N,
    )


def algorithm_5_stackelberg_guided_search(
    users: UserBatch,
    system: SystemConfig,
    cfg: StackelbergConfig,
) -> StackelbergResult:
    initial_price = (max(cfg.initial_pE, system.cE), max(cfg.initial_pN, system.cN))

    init_stage2 = algorithm_2_heuristic_user_selection(users, initial_price[0], initial_price[1], system, cfg)
    current_set = init_stage2.offloading_set
    current_rne = algorithm_4_optimal_rne_sampling(users, current_set, initial_price, system, cfg)
    current_price = current_rne.price

    visited: set[tuple[int, ...]] = {current_set}
    trajectory: list[SearchStep] = []

    for t in range(cfg.search_max_iters):
        gain_E = algorithm_3_gain_approximation(users, current_set, current_price[0], current_price[1], "E", system)
        gain_N = algorithm_3_gain_approximation(users, current_set, current_price[0], current_price[1], "N", system)
        current_eps = max(gain_E.gain, gain_N.gain)
        trajectory.append(
            SearchStep(
                iteration=t,
                offloading_set=current_set,
                pE=current_price[0],
                pN=current_price[1],
                epsilon=current_eps,
            )
        )

        candidate_sets: list[tuple[int, ...]] = []
        for cand in (gain_E.best_set, gain_N.best_set):
            if cand not in candidate_sets and cand != current_set:
                candidate_sets.append(cand)

        best_candidate: tuple[int, ...] | None = None
        best_rne: RNEResult | None = None
        best_eps = current_eps

        for candidate_set in candidate_sets:
            candidate_rne = algorithm_4_optimal_rne_sampling(users, candidate_set, current_price, system, cfg)
            if candidate_rne.epsilon + cfg.search_improvement_tol < best_eps:
                best_eps = candidate_rne.epsilon
                best_candidate = candidate_set
                best_rne = candidate_rne

        if best_candidate is None or best_rne is None:
            break
        if best_candidate in visited:
            break

        visited.add(best_candidate)
        current_set = best_candidate
        current_price = best_rne.price

    final_stage2 = algorithm_2_heuristic_user_selection(users, current_price[0], current_price[1], system, cfg)
    final_set = final_stage2.offloading_set
    final_gain_E = algorithm_3_gain_approximation(users, final_set, current_price[0], current_price[1], "E", system)
    final_gain_N = algorithm_3_gain_approximation(users, final_set, current_price[0], current_price[1], "N", system)
    final_eps = max(final_gain_E.gain, final_gain_N.gain)

    return StackelbergResult(
        price=current_price,
        offloading_set=final_set,
        epsilon=float(final_eps),
        gain_E=final_gain_E,
        gain_N=final_gain_N,
        inner_result=final_stage2.inner_result,
        social_cost=float(final_stage2.social_cost),
        trajectory=tuple(trajectory),
    )


def summarize_stackelberg_result(
    users: UserBatch,
    result: StackelbergResult,
    system: SystemConfig,
) -> str:
    data = _build_data(users)
    pE, pN = result.price
    ratio = (len(result.offloading_set) / users.n) if users.n else 0.0
    rev_E = _provider_revenue(data, result.offloading_set, pE, pN, "E", system)
    rev_N = _provider_revenue(data, result.offloading_set, pE, pN, "N", system)
    lines = [
        "Stackelberg guided search result (Algorithm 5)",
        f"price = ({pE:.10g}, {pN:.10g})",
        f"epsilon = {result.epsilon:.10g}",
        f"offloading_users = {len(result.offloading_set)}/{users.n} ({ratio:.2%})",
        f"social_cost = {result.social_cost:.10g}",
        f"esp_revenue = {rev_E:.10g}",
        f"nsp_revenue = {rev_N:.10g}",
        f"inner_converged = {result.inner_result.converged}",
        f"inner_iterations = {result.inner_result.iterations}",
        f"search_steps = {len(result.trajectory)}",
    ]
    return "\n".join(lines) + "\n"

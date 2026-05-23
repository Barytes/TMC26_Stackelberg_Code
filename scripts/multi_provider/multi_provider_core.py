from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12


def contrast_text_color(rgba: Sequence[float]) -> str:
    red = float(rgba[0])
    green = float(rgba[1])
    blue = float(rgba[2])
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.52 else "white"


def format_pair_cost_label(mean_cost: float, count: int) -> str:
    return "%.2f\n$|\\mathcal{X}|=%d$" % (mean_cost, count)


@dataclass(frozen=True)
class MultiProviderProblem:
    aw: np.ndarray
    theta: np.ndarray
    local_cost: np.ndarray
    F: np.ndarray
    B: np.ndarray
    cE: np.ndarray
    cN: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "aw", np.asarray(self.aw, dtype=float))
        object.__setattr__(self, "theta", np.asarray(self.theta, dtype=float))
        object.__setattr__(self, "local_cost", np.asarray(self.local_cost, dtype=float))
        object.__setattr__(self, "F", np.asarray(self.F, dtype=float))
        object.__setattr__(self, "B", np.asarray(self.B, dtype=float))
        object.__setattr__(self, "cE", np.asarray(self.cE, dtype=float))
        object.__setattr__(self, "cN", np.asarray(self.cN, dtype=float))
        if self.aw.ndim != 1 or self.local_cost.ndim != 1:
            raise ValueError("aw and local_cost must be one-dimensional arrays.")
        if self.theta.ndim != 2:
            raise ValueError("theta must be a two-dimensional users-by-NSP array.")
        if self.theta.shape[0] != self.aw.size or self.local_cost.size != self.aw.size:
            raise ValueError("User arrays must have consistent lengths.")
        if self.F.ndim != 1 or self.cE.ndim != 1 or self.F.size != self.cE.size:
            raise ValueError("F and cE must be one-dimensional arrays with the same length.")
        if self.B.ndim != 1 or self.cN.ndim != 1 or self.B.size != self.cN.size:
            raise ValueError("B and cN must be one-dimensional arrays with the same length.")
        if self.theta.shape[1] != self.B.size:
            raise ValueError("theta must have one column per NSP.")
        if np.any(self.aw <= 0.0) or np.any(self.theta <= 0.0) or np.any(self.local_cost <= 0.0):
            raise ValueError("aw, theta, and local_cost entries must be positive.")
        if np.any(self.F <= 0.0) or np.any(self.B <= 0.0):
            raise ValueError("Provider capacities must be positive.")
        if np.any(self.cE < 0.0) or np.any(self.cN < 0.0):
            raise ValueError("Provider unit costs must be non-negative.")

    @property
    def num_users(self) -> int:
        return int(self.aw.size)

    @property
    def num_esp(self) -> int:
        return int(self.F.size)

    @property
    def num_nsp(self) -> int:
        return int(self.B.size)


@dataclass(frozen=True)
class FixedAssignmentResponse:
    assignment_e: np.ndarray
    assignment_n: np.ndarray
    f: np.ndarray
    b: np.ndarray
    lambda_E: np.ndarray
    lambda_N: np.ndarray
    margins: np.ndarray
    offload_cost: np.ndarray
    social_cost: float


@dataclass(frozen=True)
class Stage2Result:
    assignment_e: np.ndarray
    assignment_n: np.ndarray
    response: FixedAssignmentResponse
    iterations: int
    rollback_count: int
    accepted_admissions: int
    inner_call_count: int
    runtime_sec: float
    social_cost_trace: Tuple[float, ...]

    @property
    def offloading_count(self) -> int:
        return int(np.sum(self.assignment_e >= 0))

    @property
    def social_cost(self) -> float:
        return float(self.response.social_cost)

    @property
    def margins(self) -> np.ndarray:
        return self.response.margins


@dataclass(frozen=True)
class BestResponseResult:
    provider_kind: str
    provider_index: int
    current_price: float
    best_price: float
    current_revenue: float
    best_revenue: float
    gain: float
    candidate_count: int
    stage2_calls: int


@dataclass(frozen=True)
class Stage1Step:
    iteration: int
    updated_provider: str
    pE: Tuple[float, ...]
    pN: Tuple[float, ...]
    esp_revenue: Tuple[float, ...]
    nsp_revenue: Tuple[float, ...]
    provider_gains: Tuple[float, ...]
    restricted_gap: float
    social_cost: float
    offloading_count: int
    stage2_calls: int


@dataclass(frozen=True)
class Stage1Result:
    pE: np.ndarray
    pN: np.ndarray
    stage2_result: Stage2Result
    trajectory: Tuple[Stage1Step, ...]
    restricted_gap: float
    esp_revenue: np.ndarray
    nsp_revenue: np.ndarray
    stage2_calls: int
    runtime_sec: float


def _as_price_vector(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (expected,):
        raise ValueError("%s must have shape (%d,)." % (name, expected))
    return arr


def _offloading_mask(assignment_e: np.ndarray, assignment_n: np.ndarray) -> np.ndarray:
    return (assignment_e >= 0) & (assignment_n >= 0)


def _bounded_1d_min(a: float, t: float, upper: float) -> float:
    t_eff = max(float(t), _EPS)
    a_eff = max(float(a), _EPS)
    x_star = math.sqrt(a_eff / t_eff)
    if x_star <= upper:
        return 2.0 * math.sqrt(a_eff * t_eff)
    return a_eff / upper + t_eff * upper


def _score_pair(problem: MultiProviderProblem, user_idx: int, e_idx: int, n_idx: int, tE: float, tN: float) -> float:
    return (
        _bounded_1d_min(float(problem.aw[user_idx]), tE, float(problem.F[e_idx]))
        + _bounded_1d_min(float(problem.theta[user_idx, n_idx]), tN, float(problem.B[n_idx]))
        - float(problem.local_cost[user_idx])
    )


def fixed_assignment_response(
    problem: MultiProviderProblem,
    assignment_e: Sequence[int],
    assignment_n: Sequence[int],
    pE: Sequence[float],
    pN: Sequence[float],
) -> FixedAssignmentResponse:
    pE_vec = _as_price_vector(pE, problem.num_esp, "pE")
    pN_vec = _as_price_vector(pN, problem.num_nsp, "pN")
    ae = np.asarray(assignment_e, dtype=int).copy()
    an = np.asarray(assignment_n, dtype=int).copy()
    if ae.shape != (problem.num_users,) or an.shape != (problem.num_users,):
        raise ValueError("Assignments must have one entry per user.")

    off = _offloading_mask(ae, an)
    tE = pE_vec.copy()
    tN = pN_vec.copy()
    for e_idx in range(problem.num_esp):
        idx = np.where(off & (ae == e_idx))[0]
        if idx.size:
            sE = float(np.sum(np.sqrt(problem.aw[idx])))
            tE[e_idx] = max(float(pE_vec[e_idx]), (sE / float(problem.F[e_idx])) ** 2)
    for n_idx in range(problem.num_nsp):
        idx = np.where(off & (an == n_idx))[0]
        if idx.size:
            sN = float(np.sum(np.sqrt(problem.theta[idx, n_idx])))
            tN[n_idx] = max(float(pN_vec[n_idx]), (sN / float(problem.B[n_idx])) ** 2)

    f = np.zeros(problem.num_users, dtype=float)
    b = np.zeros(problem.num_users, dtype=float)
    offload_cost = np.zeros(problem.num_users, dtype=float)
    for i in np.where(off)[0]:
        e_idx = int(ae[i])
        n_idx = int(an[i])
        f[i] = math.sqrt(float(problem.aw[i]) / max(float(tE[e_idx]), _EPS))
        b[i] = math.sqrt(float(problem.theta[i, n_idx]) / max(float(tN[n_idx]), _EPS))
        offload_cost[i] = (
            float(problem.aw[i]) / max(float(f[i]), _EPS)
            + float(problem.theta[i, n_idx]) / max(float(b[i]), _EPS)
            + float(pE_vec[e_idx]) * float(f[i])
            + float(pN_vec[n_idx]) * float(b[i])
        )

    margins = np.zeros(problem.num_users, dtype=float)
    margins[off] = problem.local_cost[off] - offload_cost[off]
    social_cost = float(np.sum(np.where(off, offload_cost, problem.local_cost)))
    return FixedAssignmentResponse(
        assignment_e=ae,
        assignment_n=an,
        f=f,
        b=b,
        lambda_E=np.maximum(tE - pE_vec, 0.0),
        lambda_N=np.maximum(tN - pN_vec, 0.0),
        margins=margins,
        offload_cost=offload_cost,
        social_cost=social_cost,
    )


def provider_revenues(
    problem: MultiProviderProblem,
    response: FixedAssignmentResponse,
    pE: Sequence[float],
    pN: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    pE_vec = _as_price_vector(pE, problem.num_esp, "pE")
    pN_vec = _as_price_vector(pN, problem.num_nsp, "pN")
    esp = np.zeros(problem.num_esp, dtype=float)
    nsp = np.zeros(problem.num_nsp, dtype=float)
    for e_idx in range(problem.num_esp):
        demand = float(np.sum(response.f[response.assignment_e == e_idx]))
        esp[e_idx] = max(float(pE_vec[e_idx]) - float(problem.cE[e_idx]), 0.0) * demand
    for n_idx in range(problem.num_nsp):
        demand = float(np.sum(response.b[response.assignment_n == n_idx]))
        nsp[n_idx] = max(float(pN_vec[n_idx]) - float(problem.cN[n_idx]), 0.0) * demand
    return esp, nsp


def _best_pair_for_user(
    problem: MultiProviderProblem,
    user_idx: int,
    pE: np.ndarray,
    pN: np.ndarray,
    response: FixedAssignmentResponse,
) -> Tuple[float, int, int]:
    best = (float("inf"), -1, -1)
    for e_idx in range(problem.num_esp):
        tE = float(pE[e_idx] + response.lambda_E[e_idx])
        for n_idx in range(problem.num_nsp):
            tN = float(pN[n_idx] + response.lambda_N[n_idx])
            score = _score_pair(problem, user_idx, e_idx, n_idx, tE, tN)
            if score < best[0]:
                best = (float(score), e_idx, n_idx)
    return best


def solve_multi_provider_stage2(
    problem: MultiProviderProblem,
    pE: Sequence[float],
    pN: Sequence[float],
    max_iters: int = 64,
    tol: float = 1e-9,
) -> Stage2Result:
    start = time.perf_counter()
    pE_vec = np.maximum(_as_price_vector(pE, problem.num_esp, "pE"), problem.cE)
    pN_vec = np.maximum(_as_price_vector(pN, problem.num_nsp, "pN"), problem.cN)
    assignment_e = np.full(problem.num_users, -1, dtype=int)
    assignment_n = np.full(problem.num_users, -1, dtype=int)
    active = np.ones(problem.num_users, dtype=bool)
    previous_social = float(np.sum(problem.local_cost))
    last_added: Optional[int] = None
    accepted = 0
    rollbacks = 0
    inner_calls = 0
    trace: List[float] = []
    response = fixed_assignment_response(problem, assignment_e, assignment_n, pE_vec, pN_vec)

    for iteration in range(max(1, int(max_iters))):
        response = fixed_assignment_response(problem, assignment_e, assignment_n, pE_vec, pN_vec)
        inner_calls += 1
        off = _offloading_mask(assignment_e, assignment_n)
        invalid = np.where(off & (response.margins < -abs(tol)))[0]
        if invalid.size:
            remove_idx = int(invalid[np.argmin(response.margins[invalid])])
            assignment_e[remove_idx] = -1
            assignment_n[remove_idx] = -1
            active[remove_idx] = False
            last_added = None
            rollbacks += 1
            continue
        if last_added is not None and response.social_cost >= previous_social - abs(tol):
            assignment_e[last_added] = -1
            assignment_n[last_added] = -1
            active[last_added] = False
            last_added = None
            rollbacks += 1
            continue

        trace.append(float(response.social_cost))
        candidates = [i for i in range(problem.num_users) if active[i] and assignment_e[i] < 0]
        if not candidates:
            break

        best = (0.0, -1, -1, -1)
        for user_idx in candidates:
            score, e_idx, n_idx = _best_pair_for_user(problem, user_idx, pE_vec, pN_vec, response)
            if score < best[0]:
                best = (float(score), int(user_idx), int(e_idx), int(n_idx))
        if best[1] < 0:
            break
        previous_social = float(response.social_cost)
        assignment_e[best[1]] = best[2]
        assignment_n[best[1]] = best[3]
        last_added = best[1]
        accepted += 1
    else:
        iteration = int(max_iters) - 1

    while True:
        response = fixed_assignment_response(problem, assignment_e, assignment_n, pE_vec, pN_vec)
        inner_calls += 1
        off = _offloading_mask(assignment_e, assignment_n)
        invalid = np.where(off & (response.margins < -abs(tol)))[0]
        if not invalid.size:
            break
        remove_idx = int(invalid[np.argmin(response.margins[invalid])])
        assignment_e[remove_idx] = -1
        assignment_n[remove_idx] = -1
        rollbacks += 1

    if trace:
        trace[-1] = float(response.social_cost)
    else:
        trace.append(float(response.social_cost))

    return Stage2Result(
        assignment_e=assignment_e.copy(),
        assignment_n=assignment_n.copy(),
        response=response,
        iterations=int(iteration) + 1,
        rollback_count=rollbacks,
        accepted_admissions=accepted,
        inner_call_count=inner_calls,
        runtime_sec=float(time.perf_counter() - start),
        social_cost_trace=tuple(trace),
    )


def _assignment_key(ae: np.ndarray, an: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return tuple(int(x) for x in ae), tuple(int(x) for x in an)


def _candidate_assignments(problem: MultiProviderProblem, stage2: Stage2Result, pE: np.ndarray, pN: np.ndarray, q: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    base_e = stage2.assignment_e
    base_n = stage2.assignment_n
    response = stage2.response
    off = np.where(base_e >= 0)[0].tolist()
    out = np.where(base_e < 0)[0].tolist()
    off_sorted = sorted(off, key=lambda idx: float(response.margins[idx]))
    outsider_rank: List[Tuple[float, int, int, int]] = []
    for user_idx in out:
        score, e_idx, n_idx = _best_pair_for_user(problem, user_idx, pE, pN, response)
        outsider_rank.append((float(score), int(user_idx), int(e_idx), int(n_idx)))
    outsider_rank.sort(key=lambda item: item[0])

    max_q = max(0, int(q))
    seen = set()
    candidates: List[Tuple[np.ndarray, np.ndarray]] = []
    for r_count in range(min(max_q, len(off_sorted)) + 1):
        for s_count in range(min(max_q, len(outsider_rank)) + 1):
            ae = base_e.copy()
            an = base_n.copy()
            for user_idx in off_sorted[:r_count]:
                ae[user_idx] = -1
                an[user_idx] = -1
            for _, user_idx, e_idx, n_idx in outsider_rank[:s_count]:
                ae[user_idx] = e_idx
                an[user_idx] = n_idx
            if not np.any(ae >= 0):
                continue
            key = _assignment_key(ae, an)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((ae, an))
    return candidates


def _min_margin_for_provider(
    problem: MultiProviderProblem,
    assignment_e: np.ndarray,
    assignment_n: np.ndarray,
    pE: np.ndarray,
    pN: np.ndarray,
    provider_kind: str,
    provider_index: int,
) -> Optional[float]:
    response = fixed_assignment_response(problem, assignment_e, assignment_n, pE, pN)
    if provider_kind == "E":
        served = np.where(response.assignment_e == provider_index)[0]
    else:
        served = np.where(response.assignment_n == provider_index)[0]
    if not served.size:
        return None
    return float(np.min(response.margins[served]))


def _boundary_price_for_provider(
    problem: MultiProviderProblem,
    assignment_e: np.ndarray,
    assignment_n: np.ndarray,
    pE: np.ndarray,
    pN: np.ndarray,
    provider_kind: str,
    provider_index: int,
    max_price: float,
) -> Optional[float]:
    floor = float(problem.cE[provider_index] if provider_kind == "E" else problem.cN[provider_index])
    prices_e = pE.copy()
    prices_n = pN.copy()

    def set_price(value: float) -> None:
        if provider_kind == "E":
            prices_e[provider_index] = float(value)
        else:
            prices_n[provider_index] = float(value)

    def margin_at(value: float) -> Optional[float]:
        set_price(value)
        return _min_margin_for_provider(problem, assignment_e, assignment_n, prices_e, prices_n, provider_kind, provider_index)

    low = max(floor, _EPS)
    low_margin = margin_at(low)
    if low_margin is None or low_margin < -1e-8:
        return None
    current = float(pE[provider_index] if provider_kind == "E" else pN[provider_index])
    high = min(float(max_price), max(current * 1.5 + 0.05, low + 0.05))
    high_margin = margin_at(high)
    while high < max_price and high_margin is not None and high_margin > 0.0:
        high = min(float(max_price), max(high * 1.6, high + 0.1))
        high_margin = margin_at(high)
    if high_margin is None:
        return None
    if high_margin > 0.0:
        return float(high)

    lo = low
    hi = high
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        mid_margin = margin_at(mid)
        if mid_margin is None:
            return None
        if mid_margin > 0.0:
            lo = mid
        else:
            hi = mid
    return float(max(floor, hi))


def estimate_provider_best_response(
    problem: MultiProviderProblem,
    stage2: Stage2Result,
    pE: Sequence[float],
    pN: Sequence[float],
    provider_kind: str,
    provider_index: int,
    q: int = 2,
    max_price: float = 4.0,
    stage2_max_iters: int = 64,
) -> BestResponseResult:
    pE_vec = np.maximum(_as_price_vector(pE, problem.num_esp, "pE"), problem.cE)
    pN_vec = np.maximum(_as_price_vector(pN, problem.num_nsp, "pN"), problem.cN)
    if provider_kind not in ("E", "N"):
        raise ValueError("provider_kind must be 'E' or 'N'.")
    if provider_kind == "E" and not (0 <= provider_index < problem.num_esp):
        raise ValueError("ESP provider_index out of range.")
    if provider_kind == "N" and not (0 <= provider_index < problem.num_nsp):
        raise ValueError("NSP provider_index out of range.")

    esp_rev, nsp_rev = provider_revenues(problem, stage2.response, pE_vec, pN_vec)
    current_revenue = float(esp_rev[provider_index] if provider_kind == "E" else nsp_rev[provider_index])
    current_price = float(pE_vec[provider_index] if provider_kind == "E" else pN_vec[provider_index])
    price_candidates = {current_price}
    for ae, an in _candidate_assignments(problem, stage2, pE_vec, pN_vec, q):
        boundary = _boundary_price_for_provider(
            problem,
            ae,
            an,
            pE_vec.copy(),
            pN_vec.copy(),
            provider_kind,
            provider_index,
            max_price=max_price,
        )
        if boundary is not None and math.isfinite(boundary):
            floor = float(problem.cE[provider_index] if provider_kind == "E" else problem.cN[provider_index])
            price_candidates.add(float(min(max(float(boundary), floor), float(max_price))))
    price_candidates.add(float(min(max_price, max(current_price * 1.1, current_price + 0.03))))

    best_price = current_price
    best_revenue = current_revenue
    stage2_calls = 0
    for candidate_price in sorted(price_candidates):
        eval_pE = pE_vec.copy()
        eval_pN = pN_vec.copy()
        if provider_kind == "E":
            eval_pE[provider_index] = candidate_price
        else:
            eval_pN[provider_index] = candidate_price
        evaluated_stage2 = solve_multi_provider_stage2(
            problem,
            eval_pE,
            eval_pN,
            max_iters=stage2_max_iters,
        )
        stage2_calls += 1
        eval_esp, eval_nsp = provider_revenues(problem, evaluated_stage2.response, eval_pE, eval_pN)
        revenue = float(eval_esp[provider_index] if provider_kind == "E" else eval_nsp[provider_index])
        if revenue > best_revenue + 1e-12:
            best_revenue = revenue
            best_price = float(candidate_price)

    return BestResponseResult(
        provider_kind=provider_kind,
        provider_index=int(provider_index),
        current_price=float(current_price),
        best_price=float(best_price),
        current_revenue=float(current_revenue),
        best_revenue=float(best_revenue),
        gain=float(max(best_revenue - current_revenue, 0.0)),
        candidate_count=len(price_candidates),
        stage2_calls=stage2_calls,
    )


def solve_multi_provider_stage1(
    problem: MultiProviderProblem,
    initial_pE: Optional[Sequence[float]] = None,
    initial_pN: Optional[Sequence[float]] = None,
    max_iters: int = 12,
    q: int = 2,
    tol: float = 1e-6,
    max_price_E: float = 4.0,
    max_price_N: float = 4.0,
    stage2_max_iters: int = 64,
) -> Stage1Result:
    start = time.perf_counter()
    if initial_pE is None:
        pE = problem.cE + 0.25
    else:
        pE = np.maximum(_as_price_vector(initial_pE, problem.num_esp, "initial_pE"), problem.cE)
    if initial_pN is None:
        pN = problem.cN + 0.25
    else:
        pN = np.maximum(_as_price_vector(initial_pN, problem.num_nsp, "initial_pN"), problem.cN)

    trajectory: List[Stage1Step] = []
    total_stage2_calls = 0
    final_gap = float("inf")

    for iteration in range(max(1, int(max_iters))):
        stage2 = solve_multi_provider_stage2(problem, pE, pN, max_iters=stage2_max_iters)
        total_stage2_calls += 1
        brs: List[BestResponseResult] = []
        for e_idx in range(problem.num_esp):
            br = estimate_provider_best_response(
                problem,
                stage2,
                pE,
                pN,
                "E",
                e_idx,
                q=q,
                max_price=max_price_E,
                stage2_max_iters=stage2_max_iters,
            )
            total_stage2_calls += br.stage2_calls
            brs.append(br)
        for n_idx in range(problem.num_nsp):
            br = estimate_provider_best_response(
                problem,
                stage2,
                pE,
                pN,
                "N",
                n_idx,
                q=q,
                max_price=max_price_N,
                stage2_max_iters=stage2_max_iters,
            )
            total_stage2_calls += br.stage2_calls
            brs.append(br)
        final_gap = max((br.gain for br in brs), default=0.0)
        esp_rev, nsp_rev = provider_revenues(problem, stage2.response, pE, pN)
        best = max(brs, key=lambda item: item.gain)
        updated = ""
        if final_gap > tol:
            updated = "%s%d" % (best.provider_kind, best.provider_index + 1)
            if best.provider_kind == "E":
                pE[best.provider_index] = max(float(problem.cE[best.provider_index]), float(best.best_price))
            else:
                pN[best.provider_index] = max(float(problem.cN[best.provider_index]), float(best.best_price))
        trajectory.append(
            Stage1Step(
                iteration=int(iteration),
                updated_provider=updated,
                pE=tuple(float(x) for x in pE),
                pN=tuple(float(x) for x in pN),
                esp_revenue=tuple(float(x) for x in esp_rev),
                nsp_revenue=tuple(float(x) for x in nsp_rev),
                provider_gains=tuple(float(br.gain) for br in brs),
                restricted_gap=float(final_gap),
                social_cost=float(stage2.social_cost),
                offloading_count=int(stage2.offloading_count),
                stage2_calls=int(total_stage2_calls),
            )
        )
        if final_gap <= tol:
            break

    final_stage2 = solve_multi_provider_stage2(problem, pE, pN, max_iters=stage2_max_iters)
    total_stage2_calls += 1
    esp_rev, nsp_rev = provider_revenues(problem, final_stage2.response, pE, pN)
    return Stage1Result(
        pE=pE.copy(),
        pN=pN.copy(),
        stage2_result=final_stage2,
        trajectory=tuple(trajectory),
        restricted_gap=float(final_gap),
        esp_revenue=esp_rev,
        nsp_revenue=nsp_rev,
        stage2_calls=int(total_stage2_calls),
        runtime_sec=float(time.perf_counter() - start),
    )


def generate_random_problem(
    n_users: int = 60,
    num_esp: int = 3,
    num_nsp: int = 3,
    seed: int = 7,
    capacity_mode: str = "total_equal",
    setup: str = "code_default_heterogeneous",
    nsp_total_bandwidth: Optional[float] = None,
) -> MultiProviderProblem:
    rng = np.random.default_rng(int(seed))
    setup_name = str(setup).strip().lower()
    if setup_name == "code_default_heterogeneous":
        w = rng.uniform(0.5, 2.5, int(n_users))
        d = rng.uniform(1.0, 10.0, int(n_users))
        fl = rng.uniform(0.5, 1.2, int(n_users))
        alpha = rng.uniform(10.0, 15.0, int(n_users))
        beta = rng.uniform(0.1, 0.5, int(n_users))
        rho = rng.uniform(0.5, 2.0, int(n_users))
        varpi = rng.uniform(0.8, 1.2, int(n_users))
        kappa = rng.uniform(0.01, 0.05, int(n_users))
        base_sigma = rng.uniform(0.5, 3.0, (int(n_users), int(num_nsp)))
        if int(num_nsp) == 3:
            sigma_bias = np.array([1.15, 1.0, 0.85], dtype=float)
        else:
            sigma_bias = np.linspace(1.15, 0.85, int(num_nsp), dtype=float)
        sigma = np.clip(base_sigma * sigma_bias[None, :], 0.5, 3.0)
        aw = alpha * w
        local_cost = alpha * w / fl + beta * kappa * w * (fl ** 2)
        theta = d[:, None] * (alpha[:, None] + beta[:, None] * rho[:, None] * varpi[:, None]) / sigma
        if int(num_esp) == 3:
            F = np.array([45.0, 35.0, 25.0], dtype=float)
            cE = np.array([0.008, 0.010, 0.013], dtype=float)
        else:
            weights = np.linspace(1.25, 0.75, int(num_esp), dtype=float)
            F = 105.0 * weights / float(np.sum(weights))
            cE = np.linspace(0.008, 0.013, int(num_esp), dtype=float)
        if int(num_nsp) == 3:
            B = np.array([20.0, 14.0, 10.0], dtype=float)
            cN = np.array([0.007, 0.010, 0.014], dtype=float)
        else:
            weights = np.linspace(1.25, 0.75, int(num_nsp), dtype=float)
            B = 44.0 * weights / float(np.sum(weights))
            cN = np.linspace(0.007, 0.014, int(num_nsp), dtype=float)
        if nsp_total_bandwidth is not None:
            total_B = float(nsp_total_bandwidth)
            if total_B <= 0.0:
                raise ValueError("nsp_total_bandwidth must be positive.")
            B = B * (total_B / float(np.sum(B)))
    elif setup_name == "lightweight_demo":
        w = rng.uniform(0.1, 1.0, int(n_users))
        d = rng.uniform(0.8, 4.0, int(n_users))
        fl = rng.choice(np.array([0.5, 0.8, 1.0, 1.2]), int(n_users))
        alpha = rng.uniform(1.0, 2.0, int(n_users))
        beta = rng.uniform(0.1, 0.5, int(n_users))
        rho = rng.uniform(0.1, 2.0, int(n_users))
        varpi = np.full(int(n_users), 1.0 / 0.35)
        aw = alpha * w
        local_delay = alpha * w / fl
        local_energy = beta * 0.35 * w * (fl ** 2)
        local_cost = local_delay + local_energy + rng.uniform(0.8, 1.8, int(n_users))
        base_sigma = rng.uniform(0.8, 4.0, (int(n_users), int(num_nsp)))
        theta = d[:, None] * (alpha[:, None] + beta[:, None] * rho[:, None] * varpi[:, None]) / base_sigma
        total_F = 20.0
        total_B = 20.0
        if capacity_mode == "per_provider_paper":
            F = np.full(int(num_esp), total_F)
            B = np.full(int(num_nsp), total_B)
        elif capacity_mode == "total_equal":
            F = np.full(int(num_esp), total_F / float(num_esp))
            B = np.full(int(num_nsp), total_B / float(num_nsp))
        else:
            raise ValueError("capacity_mode must be 'total_equal' or 'per_provider_paper'.")
        cE = np.full(int(num_esp), 0.1)
        cN = np.full(int(num_nsp), 0.1)
    else:
        raise ValueError("setup must be 'code_default_heterogeneous' or 'lightweight_demo'.")
    return MultiProviderProblem(aw=aw, theta=theta, local_cost=local_cost, F=F, B=B, cE=cE, cN=cN)


def trajectory_rows(result: Stage1Result) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for step in result.trajectory:
        row: Dict[str, object] = {
            "iteration": step.iteration,
            "updated_provider": step.updated_provider,
            "restricted_gap": step.restricted_gap,
            "social_cost": step.social_cost,
            "offloading_count": step.offloading_count,
            "stage2_calls": step.stage2_calls,
        }
        for idx, value in enumerate(step.pE, start=1):
            row["pE%d" % idx] = value
        for idx, value in enumerate(step.pN, start=1):
            row["pN%d" % idx] = value
        for idx, value in enumerate(step.esp_revenue, start=1):
            row["esp%d_revenue" % idx] = value
        for idx, value in enumerate(step.nsp_revenue, start=1):
            row["nsp%d_revenue" % idx] = value
        for idx, value in enumerate(step.provider_gains, start=1):
            row["provider%d_gain" % idx] = value
        rows.append(row)
    return rows


def assignment_rows(problem: MultiProviderProblem, result: Stage1Result) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    stage2 = result.stage2_result
    response = stage2.response
    for user_idx in range(response.assignment_e.size):
        offloading = bool(response.assignment_e[user_idx] >= 0)
        rows.append(
            {
                "user": int(user_idx),
                "offloading": int(offloading),
                "esp": "" if not offloading else int(response.assignment_e[user_idx] + 1),
                "nsp": "" if not offloading else int(response.assignment_n[user_idx] + 1),
                "f": float(response.f[user_idx]),
                "b": float(response.b[user_idx]),
                "local_cost": float(problem.local_cost[user_idx]),
                "offload_cost": "" if not offloading else float(response.offload_cost[user_idx]),
                "margin": "" if not offloading else float(response.margins[user_idx]),
            }
        )
    return rows


def provider_metric_rows(problem: MultiProviderProblem, result: Stage1Result) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    response = result.stage2_result.response
    for e_idx in range(problem.num_esp):
        served = response.assignment_e == e_idx
        rows.append(
            {
                "provider": "ESP%d" % (e_idx + 1),
                "kind": "ESP",
                "index": int(e_idx + 1),
                "price": float(result.pE[e_idx]),
                "cost": float(problem.cE[e_idx]),
                "capacity": float(problem.F[e_idx]),
                "demand": float(np.sum(response.f[served])),
                "utilization": float(np.sum(response.f[served]) / float(problem.F[e_idx])),
                "revenue": float(result.esp_revenue[e_idx]),
                "served_users": int(np.sum(served)),
            }
        )
    for n_idx in range(problem.num_nsp):
        served = response.assignment_n == n_idx
        rows.append(
            {
                "provider": "NSP%d" % (n_idx + 1),
                "kind": "NSP",
                "index": int(n_idx + 1),
                "price": float(result.pN[n_idx]),
                "cost": float(problem.cN[n_idx]),
                "capacity": float(problem.B[n_idx]),
                "demand": float(np.sum(response.b[served])),
                "utilization": float(np.sum(response.b[served]) / float(problem.B[n_idx])),
                "revenue": float(result.nsp_revenue[n_idx]),
                "served_users": int(np.sum(served)),
            }
        )
    return rows


def assignment_matrix(problem: MultiProviderProblem, result: Stage1Result) -> np.ndarray:
    matrix = np.zeros((problem.num_esp, problem.num_nsp), dtype=int)
    ae = result.stage2_result.assignment_e
    an = result.stage2_result.assignment_n
    for e_idx, n_idx in zip(ae, an):
        if e_idx >= 0 and n_idx >= 0:
            matrix[int(e_idx), int(n_idx)] += 1
    return matrix


def average_pair_offloading_cost(problem: MultiProviderProblem, result: Stage1Result) -> np.ndarray:
    totals = np.zeros((problem.num_esp, problem.num_nsp), dtype=float)
    counts = np.zeros((problem.num_esp, problem.num_nsp), dtype=int)
    response = result.stage2_result.response
    for user_idx, (e_idx, n_idx) in enumerate(zip(response.assignment_e, response.assignment_n)):
        if e_idx < 0 or n_idx < 0:
            continue
        totals[int(e_idx), int(n_idx)] += float(response.offload_cost[user_idx])
        counts[int(e_idx), int(n_idx)] += 1
    averages = np.full((problem.num_esp, problem.num_nsp), np.nan, dtype=float)
    np.divide(totals, counts, out=averages, where=counts > 0)
    return averages

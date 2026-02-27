from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable

import numpy as np

from .config import BaselineConfig, StackelbergConfig, SystemConfig
from .model import UserBatch, local_cost, theta
from .stackelberg import (
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    algorithm_4_optimal_rne_sampling,
    algorithm_5_stackelberg_guided_search,
)


@dataclass(frozen=True)
class BaselineOutcome:
    name: str
    price: tuple[float, float]
    offloading_set: tuple[int, ...]
    social_cost: float
    esp_revenue: float
    nsp_revenue: float
    epsilon_proxy: float
    meta: dict[str, float | int | str]


@dataclass(frozen=True)
class _Data:
    aw: np.ndarray
    th: np.ndarray
    cl: np.ndarray


def _build_data(users: UserBatch) -> _Data:
    return _Data(
        aw=users.alpha * users.w,
        th=theta(users),
        cl=local_cost(users),
    )


def _sorted_tuple(items: set[int] | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in set(items)))


def _set_allocations(
    data: _Data,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n = data.cl.size
    f = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)
    if not offloading_set:
        return f, b
    idx = np.asarray(offloading_set, dtype=int)
    sE = float(np.sum(np.sqrt(data.aw[idx])))
    sN = float(np.sum(np.sqrt(data.th[idx])))
    bar_pE = (sE / system.F) ** 2 if sE > 0 else 0.0
    bar_pN = (sN / system.B) ** 2 if sN > 0 else 0.0
    tE = max(pE, bar_pE)
    tN = max(pN, bar_pN)
    f[idx] = np.sqrt(data.aw[idx] / max(tE, 1e-12))
    b[idx] = np.sqrt(data.th[idx] / max(tN, 1e-12))
    return f, b


def _offload_costs(data: _Data, f: np.ndarray, b: np.ndarray, pE: float, pN: float) -> np.ndarray:
    ce = np.full_like(data.cl, np.inf)
    mask = (f > 0) & (b > 0)
    ce[mask] = data.aw[mask] / f[mask] + data.th[mask] / b[mask] + pE * f[mask] + pN * b[mask]
    return ce


def _evaluate(
    users: UserBatch,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    name: str,
    meta: dict[str, float | int | str] | None = None,
) -> BaselineOutcome:
    data = _build_data(users)
    f, b = _set_allocations(data, offloading_set, pE, pN, system)
    ce = _offload_costs(data, f, b, pE, pN)
    off = set(offloading_set)
    off_idx = np.asarray(sorted(off), dtype=int) if off else np.asarray([], dtype=int)
    loc_idx = np.asarray([i for i in range(users.n) if i not in off], dtype=int)
    social = float(np.sum(ce[off_idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)
    rev_e = float((pE - system.cE) * np.sum(f[off_idx])) if off_idx.size else 0.0
    rev_n = float((pN - system.cN) * np.sum(b[off_idx])) if off_idx.size else 0.0
    gE = algorithm_3_gain_approximation(users, offloading_set, pE, pN, "E", system).gain
    gN = algorithm_3_gain_approximation(users, offloading_set, pE, pN, "N", system).gain
    return BaselineOutcome(
        name=name,
        price=(float(pE), float(pN)),
        offloading_set=offloading_set,
        social_cost=social,
        esp_revenue=rev_e,
        nsp_revenue=rev_n,
        epsilon_proxy=float(max(gE, gN)),
        meta=meta or {},
    )


def baseline_stage2_centralized_solver(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    if users.n > base_cfg.exact_max_users:
        raise ValueError(
            f"Centralized Solver baseline only supports n_users <= {base_cfg.exact_max_users} "
            "to avoid exponential blow-up."
        )
    data = _build_data(users)
    best_set: tuple[int, ...] = tuple()
    best_social = float(np.sum(data.cl))

    for bits in itertools.product([0, 1], repeat=users.n):
        current = tuple(i for i, bit in enumerate(bits) if bit == 1)
        f, b = _set_allocations(data, current, pE, pN, system)
        ce = _offload_costs(data, f, b, pE, pN)
        if current:
            idx = np.asarray(current, dtype=int)
            if np.any(ce[idx] > data.cl[idx] + 1e-9):
                continue
        out = _evaluate(users, current, pE, pN, system, stack_cfg, "tmp")
        if out.social_cost < best_social:
            best_social = out.social_cost
            best_set = current

    return _evaluate(
        users,
        best_set,
        pE,
        pN,
        system,
        stack_cfg,
        "CS",
        meta={"searched_sets": int(2**users.n)},
    )


def baseline_stage2_ubrd(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    off = set(i for i in range(users.n) if 2.0 * math.sqrt(data.aw[i] * pE) + 2.0 * math.sqrt(data.th[i] * pN) < data.cl[i])
    changed = True
    rounds = 0
    while changed and rounds < base_cfg.ubrd_max_iters:
        changed = False
        rounds += 1
        for i in range(users.n):
            with_i = _sorted_tuple(list(off | {i}))
            w_f, w_b = _set_allocations(data, with_i, pE, pN, system)
            ce_with = _offload_costs(data, w_f, w_b, pE, pN)[i]
            should_offload = ce_with < data.cl[i]
            if should_offload and i not in off:
                off.add(i)
                changed = True
            if (not should_offload) and i in off:
                off.remove(i)
                changed = True
    return _evaluate(users, _sorted_tuple(list(off)), pE, pN, system, stack_cfg, "UBRD", meta={"rounds": rounds})


def baseline_stage2_ura(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    off = set(range(users.n))
    rounds = 0
    changed = True
    while changed and rounds < base_cfg.ura_max_iters:
        rounds += 1
        changed = False
        if not off:
            break
        f_u = system.F / len(off)
        b_u = system.B / len(off)
        leave = {i for i in off if (data.aw[i] / f_u + data.th[i] / b_u + pE * f_u + pN * b_u) >= data.cl[i]}
        if leave:
            off -= leave
            changed = True
        join = {
            i
            for i in range(users.n)
            if i not in off
            and (data.aw[i] / max(system.F / max(len(off) + 1, 1), 1e-12)
                 + data.th[i] / max(system.B / max(len(off) + 1, 1), 1e-12)
                 + pE * (system.F / max(len(off) + 1, 1))
                 + pN * (system.B / max(len(off) + 1, 1)))
            < data.cl[i]
        }
        if join:
            off |= join
            changed = True
    return _evaluate(users, _sorted_tuple(list(off)), pE, pN, system, stack_cfg, "URA", meta={"rounds": rounds})


def _stage2_solver(
    method: str,
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    if method == "CS":
        return baseline_stage2_centralized_solver(users, pE, pN, system, stack_cfg, base_cfg)
    if method == "UBRD":
        return baseline_stage2_ubrd(users, pE, pN, system, stack_cfg, base_cfg)
    if method == "URA":
        return baseline_stage2_ura(users, pE, pN, system, stack_cfg, base_cfg)
    if method == "DG":
        out = algorithm_2_heuristic_user_selection(users, pE, pN, system, stack_cfg)
        return _evaluate(
            users,
            out.offloading_set,
            pE,
            pN,
            system,
            stack_cfg,
            "DG",
            meta={"iterations": out.iterations},
        )
    raise ValueError(f"Unknown Stage-II method: {method}")


def baseline_stage1_grid_search_oracle(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    pE_grid = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.gso_grid_points)
    pN_grid = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.gso_grid_points)
    best: BaselineOutcome | None = None
    for pE in pE_grid:
        for pN in pN_grid:
            out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, float(pE), float(pN), system, stack_cfg, base_cfg)
            score = (-out.epsilon_proxy, out.esp_revenue + out.nsp_revenue, -out.social_cost)
            if best is None:
                best = out
                best_score = score
            elif score > best_score:
                best = out
                best_score = score
    assert best is not None
    return BaselineOutcome(
        name="GSO",
        price=best.price,
        offloading_set=best.offloading_set,
        social_cost=best.social_cost,
        esp_revenue=best.esp_revenue,
        nsp_revenue=best.nsp_revenue,
        epsilon_proxy=best.epsilon_proxy,
        meta={"grid_points": base_cfg.gso_grid_points},
    )


def _best_response_1d(
    provider: str,
    users: UserBatch,
    fixed_price: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> tuple[float, BaselineOutcome]:
    if provider == "E":
        grid = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.pbdr_grid_points)
        outs = [
            _stage2_solver(base_cfg.stage2_solver_for_pricing, users, float(pE), fixed_price, system, stack_cfg, base_cfg)
            for pE in grid
        ]
        best = max(outs, key=lambda o: o.esp_revenue)
        return best.price[0], best
    grid = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.pbdr_grid_points)
    outs = [
        _stage2_solver(base_cfg.stage2_solver_for_pricing, users, fixed_price, float(pN), system, stack_cfg, base_cfg)
        for pN in grid
    ]
    best = max(outs, key=lambda o: o.nsp_revenue)
    return best.price[1], best


def baseline_stage1_pbdr(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    last = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    for t in range(base_cfg.pbdr_max_iters):
        pE, _ = _best_response_1d("E", users, pN, system, stack_cfg, base_cfg)
        pN, last = _best_response_1d("N", users, pE, system, stack_cfg, base_cfg)
        if t > 0 and abs(pE - last.price[0]) + abs(pN - last.price[1]) <= base_cfg.pbdr_tol:
            break
    out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    return BaselineOutcome(
        name="PBRD",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=out.epsilon_proxy,
        meta={"iters": base_cfg.pbdr_max_iters},
    )


def _joint_objective(out: BaselineOutcome) -> float:
    return out.esp_revenue + out.nsp_revenue - 0.01 * out.social_cost


def baseline_stage1_bo(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    rng = np.random.default_rng(base_cfg.random_seed + 101)
    x: list[tuple[float, float]] = []
    y: list[float] = []
    best: BaselineOutcome | None = None

    def evaluate(pE: float, pN: float) -> BaselineOutcome:
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        return out

    for _ in range(base_cfg.bo_init_points):
        pE = float(rng.uniform(system.cE, base_cfg.max_price_E))
        pN = float(rng.uniform(system.cN, base_cfg.max_price_N))
        out = evaluate(pE, pN)
        val = _joint_objective(out)
        x.append((pE, pN))
        y.append(val)
        if best is None or val > _joint_objective(best):
            best = out

    for t in range(base_cfg.bo_iters):
        cand = np.column_stack(
            [
                rng.uniform(system.cE, base_cfg.max_price_E, size=base_cfg.bo_candidate_pool),
                rng.uniform(system.cN, base_cfg.max_price_N, size=base_cfg.bo_candidate_pool),
            ]
        )
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        scores = np.zeros(cand.shape[0], dtype=float)
        for i in range(cand.shape[0]):
            d2 = np.sum((x_arr - cand[i]) ** 2, axis=1)
            k = np.exp(-d2 / max(base_cfg.bo_kernel_bandwidth, 1e-12))
            wsum = float(np.sum(k))
            if wsum < 1e-12:
                mu = float(np.mean(y_arr))
                sigma = float(np.std(y_arr) + 1.0)
            else:
                w = k / wsum
                mu = float(np.sum(w * y_arr))
                sigma = float(np.sqrt(max(np.sum(w * (y_arr - mu) ** 2), 1e-12)))
            beta = base_cfg.bo_ucb_beta / math.sqrt(t + 1.0)
            scores[i] = mu + beta * sigma
        pick = int(np.argmax(scores))
        pE = float(cand[pick, 0])
        pN = float(cand[pick, 1])
        out = evaluate(pE, pN)
        val = _joint_objective(out)
        x.append((pE, pN))
        y.append(val)
        if best is None or val > _joint_objective(best):
            best = out

    assert best is not None
    return BaselineOutcome(
        name="BO",
        price=best.price,
        offloading_set=best.offloading_set,
        social_cost=best.social_cost,
        esp_revenue=best.esp_revenue,
        nsp_revenue=best.nsp_revenue,
        epsilon_proxy=best.epsilon_proxy,
        meta={"evals": len(x)},
    )


def baseline_stage1_drl(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    grid_e = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.drl_price_levels)
    grid_n = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.drl_price_levels)
    actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    q = np.zeros((grid_e.size, grid_n.size, len(actions)), dtype=float)
    rng = np.random.default_rng(base_cfg.random_seed + 303)

    def clamp(idx: int, n: int) -> int:
        return min(max(idx, 0), n - 1)

    for _ in range(base_cfg.drl_episodes):
        i = int(rng.integers(0, grid_e.size))
        j = int(rng.integers(0, grid_n.size))
        for _step in range(base_cfg.drl_steps_per_episode):
            if rng.uniform() < base_cfg.drl_epsilon:
                a_idx = int(rng.integers(0, len(actions)))
            else:
                a_idx = int(np.argmax(q[i, j]))
            di, dj = actions[a_idx]
            ni = clamp(i + di, grid_e.size)
            nj = clamp(j + dj, grid_n.size)
            out = _stage2_solver(
                base_cfg.stage2_solver_for_pricing,
                users,
                float(grid_e[ni]),
                float(grid_n[nj]),
                system,
                stack_cfg,
                base_cfg,
            )
            reward = _joint_objective(out)
            td_target = reward + base_cfg.drl_gamma * float(np.max(q[ni, nj]))
            q[i, j, a_idx] += base_cfg.drl_alpha * (td_target - q[i, j, a_idx])
            i, j = ni, nj

    best_idx = np.unravel_index(np.argmax(np.max(q, axis=2)), (grid_e.size, grid_n.size))
    pE = float(grid_e[int(best_idx[0])])
    pN = float(grid_n[int(best_idx[1])])
    out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    return BaselineOutcome(
        name="DRL",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=out.epsilon_proxy,
        meta={"episodes": base_cfg.drl_episodes},
    )


def baseline_market_equilibrium(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    data = _build_data(users)
    for _ in range(base_cfg.market_max_iters):
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        f, b = _set_allocations(data, out.offloading_set, pE, pN, system)
        dF = float(np.sum(f)) / system.F - 1.0
        dB = float(np.sum(b)) / system.B - 1.0
        pE = max(system.cE, pE * math.exp(base_cfg.market_step_size * dF))
        pN = max(system.cN, pN * math.exp(base_cfg.market_step_size * dB))
    out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    return BaselineOutcome(
        name="MarketEquilibrium",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=out.epsilon_proxy,
        meta={"iters": base_cfg.market_max_iters},
    )


def baseline_single_sp(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    off: set[int] = set()

    def utility(off_set: tuple[int, ...]) -> float:
        pE = system.cE
        pN = system.cN
        f, b = _set_allocations(data, off_set, pE, pN, system)
        ce = _offload_costs(data, f, b, pE, pN)
        idx = np.asarray(off_set, dtype=int) if off_set else np.asarray([], dtype=int)
        if idx.size and np.any(ce[idx] >= data.cl[idx]):
            return -1e18
        user_payment = float(np.sum(data.cl[idx] - ce[idx])) if idx.size else 0.0
        op_cost = float(system.cE * np.sum(f[idx]) + system.cN * np.sum(b[idx])) if idx.size else 0.0
        return user_payment - op_cost

    cur_u = utility(tuple())
    for _ in range(base_cfg.single_sp_max_iters):
        best_gain = 0.0
        best_user: int | None = None
        for j in range(users.n):
            if j in off:
                continue
            cand = _sorted_tuple(list(off | {j}))
            gain = utility(cand) - cur_u
            if gain > best_gain:
                best_gain = gain
                best_user = j
        if best_user is None:
            break
        off.add(best_user)
        cur_u = utility(_sorted_tuple(list(off)))

    pE = system.cE
    pN = system.cN
    out = _evaluate(
        users,
        _sorted_tuple(list(off)),
        pE,
        pN,
        system,
        stack_cfg,
        "SingleSP",
        meta={"utility": cur_u},
    )
    return out


def baseline_random_offloading(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    rng = np.random.default_rng(base_cfg.random_seed + 707)
    best: BaselineOutcome | None = None
    for _ in range(base_cfg.random_offloading_trials):
        mask = rng.uniform(size=users.n) < base_cfg.random_offloading_prob
        off = tuple(np.nonzero(mask)[0].tolist())
        rne = algorithm_4_optimal_rne_sampling(
            users,
            off,
            (max(system.cE, stack_cfg.initial_pE), max(system.cN, stack_cfg.initial_pN)),
            system,
            stack_cfg,
        )
        out = _evaluate(
            users,
            rne.offloading_set,
            rne.price[0],
            rne.price[1],
            system,
            stack_cfg,
            "RandomOffloading",
            meta={"source_set_size": len(off)},
        )
        if best is None or (out.esp_revenue + out.nsp_revenue) > (best.esp_revenue + best.nsp_revenue):
            best = out
    assert best is not None
    return best


def proposed_gsse(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
) -> BaselineOutcome:
    res = algorithm_5_stackelberg_guided_search(users, system, stack_cfg)
    return _evaluate(
        users,
        res.offloading_set,
        res.price[0],
        res.price[1],
        system,
        stack_cfg,
        "GSSE",
        meta={"search_steps": len(res.trajectory)},
    )


def run_all_baselines(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> list[BaselineOutcome]:
    outcomes: list[BaselineOutcome] = []
    init_pE = max(system.cE, stack_cfg.initial_pE)
    init_pN = max(system.cN, stack_cfg.initial_pN)

    outcomes.append(proposed_gsse(users, system, stack_cfg))
    outcomes.append(baseline_stage2_ubrd(users, init_pE, init_pN, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage2_ura(users, init_pE, init_pN, system, stack_cfg, base_cfg))
    if users.n <= base_cfg.exact_max_users:
        outcomes.append(baseline_stage2_centralized_solver(users, init_pE, init_pN, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_grid_search_oracle(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_pbdr(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_bo(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_drl(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_market_equilibrium(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_single_sp(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_random_offloading(users, system, stack_cfg, base_cfg))
    return outcomes


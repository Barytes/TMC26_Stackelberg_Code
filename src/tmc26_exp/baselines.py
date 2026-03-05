from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize, minimize_scalar

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


def _solve_inner_with_numerical_solver(
    data: _Data,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not offloading_set:
        n = data.cl.size
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    idx = np.asarray(offloading_set, dtype=int)
    m = idx.size
    eps = 1e-8

    # Feasible warm start: uniform split within capacity.
    x0 = np.concatenate(
        [
            np.full(m, max(system.F / max(m, 1), eps), dtype=float),
            np.full(m, max(system.B / max(m, 1), eps), dtype=float),
        ]
    )

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[:m], x[m:]

    def objective(x: np.ndarray) -> float:
        f_loc, b_loc = unpack(x)
        return float(
            np.sum(data.aw[idx] / f_loc + data.th[idx] / b_loc + pE * f_loc + pN * b_loc)
        )

    constraints = [
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
    # Final feasibility guard.
    if np.sum(f_sol) > system.F + 1e-6 or np.sum(b_sol) > system.B + 1e-6:
        return None
    ce = data.aw[idx] / f_sol + data.th[idx] / b_sol + pE * f_sol + pN * b_sol
    if np.any(ce > data.cl[idx] + 1e-6):
        return None

    f = np.zeros(data.cl.size, dtype=float)
    b = np.zeros(data.cl.size, dtype=float)
    f[idx] = f_sol
    b[idx] = b_sol
    return f, b


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


def _build_outcome_from_allocations(
    users: UserBatch,
    f: np.ndarray,
    b: np.ndarray,
    pE: float,
    pN: float,
    system: SystemConfig,
    name: str,
    meta: dict[str, float | int | str] | None = None,
) -> BaselineOutcome:
    data = _build_data(users)
    ce = _offload_costs(data, f, b, pE, pN)
    offloading_set = _sorted_tuple(tuple(int(i) for i in np.flatnonzero(np.isfinite(ce))))
    off_idx = np.asarray(offloading_set, dtype=int) if offloading_set else np.asarray([], dtype=int)
    loc_idx = np.asarray([i for i in range(users.n) if i not in set(offloading_set)], dtype=int)
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


def _repair_to_capacity(
    data: _Data,
    f: np.ndarray,
    b: np.ndarray,
    pE: float,
    pN: float,
    system: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    f_out = np.asarray(f, dtype=float).copy()
    b_out = np.asarray(b, dtype=float).copy()
    ce = _offload_costs(data, f_out, b_out, pE, pN)
    mask = np.isfinite(ce)
    if not np.any(mask):
        return f_out, b_out

    while np.sum(f_out[mask]) > system.F + 1e-9 or np.sum(b_out[mask]) > system.B + 1e-9:
        idx = np.flatnonzero(mask)
        gains = data.cl[idx] - ce[idx]
        drop = int(idx[np.argmin(gains)])
        f_out[drop] = 0.0
        b_out[drop] = 0.0
        ce[drop] = np.inf
        mask[drop] = False
        if not np.any(mask):
            break

    return f_out, b_out


def _vi_response_from_multipliers(
    data: _Data,
    lambda_F: float,
    lambda_B: float,
    pE: float,
    pN: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = data.cl.size
    f = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)
    tE = pE + max(lambda_F, 0.0)
    tN = pN + max(lambda_B, 0.0)
    augmented = 2.0 * np.sqrt(data.aw * tE) + 2.0 * np.sqrt(data.th * tN)
    off_idx = np.flatnonzero(augmented < data.cl)
    if off_idx.size:
        f[off_idx] = np.sqrt(data.aw[off_idx] / max(tE, 1e-12))
        b[off_idx] = np.sqrt(data.th[off_idx] / max(tN, 1e-12))
    return f, b


def _penalty_axis_best_response(
    a: float,
    price: float,
    residual: float,
    capacity: float,
    rho: float,
) -> float:
    eps = 1e-8
    upper = max(2.0 * capacity, 2.0 * math.sqrt(max(a, eps) / max(price, eps)), 1.0)

    def objective(x: float) -> float:
        overflow = max(0.0, residual + x - capacity)
        return a / x + price * x + 0.5 * rho * (overflow ** 2)

    res = minimize_scalar(objective, bounds=(eps, upper), method="bounded", options={"xatol": 1e-6})
    x_best = float(res.x if res.success else math.sqrt(max(a, eps) / max(price, eps)))
    return min(max(x_best, eps), upper)


def _check_bonmin_available() -> bool:
    """Check if BONMIN solver is available on the system."""
    try:
        import pyomo.environ as pyo
        solver = pyo.SolverFactory('bonmin')
        return solver.available()
    except ImportError:
        return False


def _build_minlp_model(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
) -> "pyomo.ConcreteModel":
    """
    Build Pyomo MINLP model for Social Cost Minimization.

    Variables:
        o[i] ∈ {0,1}: Binary offloading decision for user i
        f[i] ≥ 0: Edge computation resource allocated to user i (GHz)
        b[i] ≥ 0: Bandwidth allocated to user i (MHz)

    Objective: Minimize social cost
        Σ_i [o[i] * C^e_i(f[i], b[i]) + (1-o[i]) * C^l_i]

    Constraints:
        - Capacity: Σ_i f[i] ≤ F, Σ_i b[i] ≤ B
        - Variable linking (Big-M): f[i] ≤ M*o[i], b[i] ≤ M*o[i]
        - Individual rationality: C^e_i(f[i], b[i]) ≤ C^l_i + M*(1-o[i])
        - Lower bounds: f[i] ≥ ε*o[i], b[i] ≥ ε*o[i]
    """
    import pyomo.environ as pyo

    n = users.n
    data = _build_data(users)

    model = pyo.ConcreteModel(name="SCM_MINLP")

    # Index set
    model.I = pyo.Set(initialize=range(n))

    # Decision variables
    model.o = pyo.Var(model.I, domain=pyo.Binary)  # offloading decision
    model.f = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # edge CPU allocation
    model.b = pyo.Var(model.I, domain=pyo.NonNegativeReals)  # bandwidth allocation

    # Precomputed parameters
    model.aw = pyo.Param(model.I, initialize={i: float(data.aw[i]) for i in range(n)})
    model.th = pyo.Param(model.I, initialize={i: float(data.th[i]) for i in range(n)})
    model.cl = pyo.Param(model.I, initialize={i: float(data.cl[i]) for i in range(n)})

    # Big-M constant
    M = max(system.F, system.B) * 2.0

    # Small epsilon to avoid division by zero
    eps = 1e-8

    # Objective: minimize social cost
    def objective_rule(m):
        # Offloading cost for users who offload
        offload_cost = sum(
            m.o[i] * (m.aw[i] / (m.f[i] + eps) + m.th[i] / (m.b[i] + eps) + pE * m.f[i] + pN * m.b[i])
            for i in m.I
        )
        # Local cost for users who don't offload
        local_cost_sum = sum((1 - m.o[i]) * m.cl[i] for i in m.I)
        return offload_cost + local_cost_sum
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Capacity constraints
    model.cap_F = pyo.Constraint(expr=sum(model.f[i] for i in model.I) <= system.F)
    model.cap_B = pyo.Constraint(expr=sum(model.b[i] for i in model.I) <= system.B)

    # Big-M variable linking constraints: f[i] ≤ M*o[i], b[i] ≤ M*o[i]
    def link_f_rule(m, i):
        return m.f[i] <= M * m.o[i]
    model.link_f = pyo.Constraint(model.I, rule=link_f_rule)

    def link_b_rule(m, i):
        return m.b[i] <= M * m.o[i]
    model.link_b = pyo.Constraint(model.I, rule=link_b_rule)

    # Individual rationality constraints (with Big-M relaxation)
    # When o[i] = 1: C^e_i <= C^l_i
    # When o[i] = 0: constraint is relaxed (RHS = C^l_i + M)
    def ir_rule(m, i):
        return m.aw[i] / (m.f[i] + eps) + m.th[i] / (m.b[i] + eps) + pE * m.f[i] + pN * m.b[i] <= m.cl[i] + M * (1 - m.o[i])
    model.ir = pyo.Constraint(model.I, rule=ir_rule)

    # Lower bounds to ensure proper variable linking
    def f_lb_rule(m, i):
        return m.f[i] >= eps * m.o[i]
    model.f_lb = pyo.Constraint(model.I, rule=f_lb_rule)

    def b_lb_rule(m, i):
        return m.b[i] >= eps * m.o[i]
    model.b_lb = pyo.Constraint(model.I, rule=b_lb_rule)

    return model


def _solve_minlp_with_bonmin(
    model: "pyomo.ConcreteModel",
    base_cfg: BaselineConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], float, bool]:
    """
    Solve MINLP model using BONMIN solver.

    Returns:
        f: Array of edge CPU allocations
        b: Array of bandwidth allocations
        offloading_set: Tuple of user indices who offload
        social_cost: Optimal social cost
        success: Whether solver found optimal solution
    """
    import pyomo.environ as pyo

    solver = pyo.SolverFactory('bonmin')

    # Set solver options
    solver.options['bonmin.algorithm'] = base_cfg.bonmin_algorithm
    solver.options['bonmin.time_limit'] = base_cfg.bonmin_time_limit
    solver.options['bonmin.mip_allowable_gap'] = base_cfg.bonmin_mip_gap
    solver.options['print_level'] = 0

    results = solver.solve(model, tee=False)

    # Check solution status
    status = results.solver.status
    termination = results.solver.termination_condition

    success = (
        status == pyo.SolverStatus.ok and
        termination == pyo.TerminationCondition.optimal
    )

    n = len(model.I)
    f = np.zeros(n, dtype=float)
    b = np.zeros(n, dtype=float)

    for i in model.I:
        f[i] = float(pyo.value(model.f[i]))
        b[i] = float(pyo.value(model.b[i]))

    # Extract offloading set from binary variables
    offloading_set = tuple(
        int(i) for i in model.I
        if pyo.value(model.o[i]) > 0.5
    )

    social_cost = float(pyo.value(model.objective))

    return f, b, offloading_set, social_cost, success


def _solve_centralized_minlp(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """MINLP-based centralized solver using Pyomo + BONMIN."""
    import time

    start_time = time.perf_counter()

    # Build and solve model
    model = _build_minlp_model(users, pE, pN, system)
    f, b, offloading_set, social_cost, success = _solve_minlp_with_bonmin(model, base_cfg)

    runtime = time.perf_counter() - start_time

    # Build outcome using existing helper
    return _build_outcome_from_allocations(
        users,
        f,
        b,
        pE,
        pN,
        system,
        "CS_MINLP",
        meta={
            "solver": "bonmin",
            "algorithm": base_cfg.bonmin_algorithm,
            "social_cost": round(social_cost, 6),
            "offloading_size": len(offloading_set),
            "runtime_sec": round(runtime, 4),
            "success": success,
        },
    )


def _solve_centralized_enumeration(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Original enumeration-based solver."""
    if users.n > base_cfg.exact_max_users:
        raise ValueError(
            f"Enumeration-based centralized solver only supports n_users <= {base_cfg.exact_max_users}. "
            f"Use MINLP mode (cs_use_minlp=true) for larger problems."
        )

    data = _build_data(users)
    best_set: tuple[int, ...] = tuple()
    best_social = float(np.sum(data.cl))

    for bits in itertools.product([0, 1], repeat=users.n):
        current = tuple(i for i, bit in enumerate(bits) if bit == 1)
        alloc = _solve_inner_with_numerical_solver(data, current, pE, pN, system)
        if alloc is None:
            continue
        f, b = alloc
        ce = _offload_costs(data, f, b, pE, pN)
        idx = np.asarray(current, dtype=int) if current else np.asarray([], dtype=int)
        loc_idx = np.asarray([i for i in range(users.n) if i not in set(current)], dtype=int)
        social = float(np.sum(ce[idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)
        if social < best_social:
            best_social = social
            best_set = current

    return _evaluate(
        users,
        best_set,
        pE,
        pN,
        system,
        stack_cfg,
        "CS",
        meta={"searched_sets": int(2**users.n), "solver": "scipy_slsqp"},
    )


def baseline_stage2_centralized_solver(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """
    Solve Stage II using centralized optimization.

    By default, uses MINLP (Pyomo + BONMIN) for direct formulation.
    Falls back to enumeration for small n or if MINLP fails.
    """
    import warnings

    use_minlp = base_cfg.cs_use_minlp

    # Check if MINLP is viable
    if use_minlp:
        if not _check_bonmin_available():
            warnings.warn("BONMIN solver not found, falling back to enumeration.")
            use_minlp = False

    if use_minlp:
        try:
            outcome = _solve_centralized_minlp(users, pE, pN, system, stack_cfg, base_cfg)
            # Check if solution was successful
            if outcome.meta.get("success", True):
                return outcome
            # If MINLP failed and fallback is enabled
            if base_cfg.cs_fallback_to_enum and users.n <= base_cfg.exact_max_users:
                warnings.warn("MINLP did not find optimal solution, falling back to enumeration.")
                return _solve_centralized_enumeration(users, pE, pN, system, stack_cfg, base_cfg)
            return outcome
        except Exception as e:
            if base_cfg.cs_fallback_to_enum and users.n <= base_cfg.exact_max_users:
                warnings.warn(f"MINLP solver error: {e}. Falling back to enumeration.")
                return _solve_centralized_enumeration(users, pE, pN, system, stack_cfg, base_cfg)
            raise
    else:
        return _solve_centralized_enumeration(users, pE, pN, system, stack_cfg, base_cfg)


def baseline_stage2_ubrd(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    rng = np.random.default_rng(base_cfg.random_seed + users.n)
    off: set[int] = set()
    changed = True
    rounds = 0
    while changed and rounds < base_cfg.ubrd_max_iters:
        changed = False
        rounds += 1
        for i in rng.permutation(users.n):
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
    return _evaluate(
        users,
        _sorted_tuple(list(off)),
        pE,
        pN,
        system,
        stack_cfg,
        "UBRD",
        meta={"rounds": rounds, "init": "empty_set", "order": "random_per_round"},
    )


def baseline_stage2_vi(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    lambda_F = 0.0
    lambda_B = 0.0
    previous_set: tuple[int, ...] | None = None
    total_iters = base_cfg.vi_max_iters
    best_f = np.zeros(users.n, dtype=float)
    best_b = np.zeros(users.n, dtype=float)
    best_lambda_F = 0.0
    best_lambda_B = 0.0
    best_score = (float("inf"), float("inf"))

    for t in range(base_cfg.vi_max_iters):
        f, b = _vi_response_from_multipliers(data, lambda_F, lambda_B, pE, pN)
        current_set = _sorted_tuple(tuple(int(i) for i in np.flatnonzero((f > 0) & (b > 0))))
        excess_F = float(np.sum(f) - system.F)
        excess_B = float(np.sum(b) - system.B)
        ce = _offload_costs(data, f, b, pE, pN)
        off_idx = np.asarray(current_set, dtype=int) if current_set else np.asarray([], dtype=int)
        loc_idx = np.asarray([i for i in range(users.n) if i not in set(current_set)], dtype=int)
        actual_social = float(np.sum(ce[off_idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)
        score = (max(0.0, excess_F, excess_B), actual_social)
        if score < best_score:
            best_score = score
            best_f = f.copy()
            best_b = b.copy()
            best_lambda_F = lambda_F
            best_lambda_B = lambda_B
        step = base_cfg.vi_step_size / math.sqrt(t + 1.0)
        next_lambda_F = max(0.0, lambda_F + step * excess_F)
        next_lambda_B = max(0.0, lambda_B + step * excess_B)
        stable_set = previous_set == current_set
        if (
            stable_set
            and max(abs(next_lambda_F - lambda_F), abs(next_lambda_B - lambda_B), max(0.0, excess_F), max(0.0, excess_B))
            <= base_cfg.vi_tol
        ):
            lambda_F = next_lambda_F
            lambda_B = next_lambda_B
            total_iters = t + 1
            break
        lambda_F = next_lambda_F
        lambda_B = next_lambda_B
        previous_set = current_set

    f = best_f
    b = best_b
    before_repair = int(np.count_nonzero((f > 0) & (b > 0)))
    f, b = _repair_to_capacity(data, f, b, pE, pN, system)
    after_repair = int(np.count_nonzero((f > 0) & (b > 0)))
    return _build_outcome_from_allocations(
        users,
        f,
        b,
        pE,
        pN,
        system,
        "VI",
        meta={
            "iters": total_iters,
            "lambda_F": round(best_lambda_F, 6),
            "lambda_B": round(best_lambda_B, 6),
            "repaired_drop": before_repair - after_repair,
        },
    )


def baseline_stage2_penalty(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    data = _build_data(users)
    rng = np.random.default_rng(base_cfg.random_seed + 509 + users.n)
    f = np.zeros(users.n, dtype=float)
    b = np.zeros(users.n, dtype=float)
    rho = base_cfg.penalty_init_rho
    total_rounds = 0
    outer_used = 0

    for outer in range(base_cfg.penalty_outer_iters):
        outer_used = outer + 1
        for _inner in range(base_cfg.penalty_inner_iters):
            total_rounds += 1
            changed = False
            for i in rng.permutation(users.n):
                residual_F = float(np.sum(f) - f[i])
                residual_B = float(np.sum(b) - b[i])
                local_penalty = 0.5 * rho * (
                    max(0.0, residual_F - system.F) ** 2 + max(0.0, residual_B - system.B) ** 2
                )
                fi = _penalty_axis_best_response(float(data.aw[i]), pE, residual_F, system.F, rho)
                bi = _penalty_axis_best_response(float(data.th[i]), pN, residual_B, system.B, rho)
                off_penalty = 0.5 * rho * (
                    max(0.0, residual_F + fi - system.F) ** 2 + max(0.0, residual_B + bi - system.B) ** 2
                )
                penalized_offload = float(data.aw[i] / fi + data.th[i] / bi + pE * fi + pN * bi + off_penalty)
                penalized_local = float(data.cl[i] + local_penalty)
                if penalized_offload + 1e-9 < penalized_local:
                    if abs(f[i] - fi) > 1e-6 or abs(b[i] - bi) > 1e-6:
                        changed = True
                    f[i] = fi
                    b[i] = bi
                else:
                    if f[i] > 1e-9 or b[i] > 1e-9:
                        changed = True
                    f[i] = 0.0
                    b[i] = 0.0
            if not changed:
                break

        excess = max(0.0, float(np.sum(f) - system.F), float(np.sum(b) - system.B))
        if excess <= base_cfg.penalty_tol:
            break
        rho *= base_cfg.penalty_rho_scale

    before_repair = int(np.count_nonzero((f > 0) & (b > 0)))
    f, b = _repair_to_capacity(data, f, b, pE, pN, system)
    after_repair = int(np.count_nonzero((f > 0) & (b > 0)))
    return _build_outcome_from_allocations(
        users,
        f,
        b,
        pE,
        pN,
        system,
        "PEN",
        meta={
            "outer_iters": outer_used,
            "br_rounds": total_rounds,
            "final_rho": round(rho, 6),
            "repaired_drop": before_repair - after_repair,
        },
    )


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
    if method == "VI":
        return baseline_stage2_vi(users, pE, pN, system, stack_cfg, base_cfg)
    if method == "PEN":
        return baseline_stage2_penalty(users, pE, pN, system, stack_cfg, base_cfg)
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


def run_stage2_solver(
    method: str,
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Public API for running Stage-II solvers."""
    return _stage2_solver(method, users, pE, pN, system, stack_cfg, base_cfg)


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
    outcomes.append(baseline_stage2_vi(users, init_pE, init_pN, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage2_penalty(users, init_pE, init_pN, system, stack_cfg, base_cfg))
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

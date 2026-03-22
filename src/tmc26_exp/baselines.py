from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .config import BaselineConfig, StackelbergConfig, SystemConfig
from .model import UserBatch, local_cost, theta
from .stackelberg import (
    _refine_price_for_fixed_set,
    algorithm_2_heuristic_user_selection,
    algorithm_3_gain_approximation,
    run_stage1_solver,
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


@dataclass(frozen=True)
class Stage1GridEvaluation:
    pE_grid: np.ndarray
    pN_grid: np.ndarray
    outcomes: list[list[BaselineOutcome]]
    esp_rev: np.ndarray
    nsp_rev: np.ndarray
    eps: np.ndarray


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


def _build_outcome_from_solver_allocations(
    users: UserBatch,
    f: np.ndarray,
    b: np.ndarray,
    offloading_set: tuple[int, ...],
    pE: float,
    pN: float,
    system: SystemConfig,
    name: str,
    meta: dict[str, float | int | str] | None = None,
) -> BaselineOutcome:
    data = _build_data(users)
    offloading_set = _sorted_tuple(offloading_set)
    ce = _offload_costs(data, f, b, pE, pN)
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


def _check_gekko_available() -> bool:
    """Check if GEKKO solver is available."""
    try:
        from gekko import GEKKO
        return True
    except ImportError:
        return False


def _resolve_scip_executable(explicit_path: str | None = None) -> str | None:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
    env_path = os.environ.get("SCIP_EXECUTABLE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return str(p)
    default_path = Path.home() / "anaconda3" / "envs" / "scip_bin" / "Library" / "bin" / "scip.exe"
    if default_path.exists():
        return str(default_path)
    return "scip"


def _solve_with_pyomo_scip(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    scip_executable: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], float, bool]:
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition

    n = users.n
    data = _build_data(users)
    eps = 1e-8
    cap_tol = 1e-5
    ir_tol = 1e-5
    local_upper = float(np.sum(data.cl))

    warm_set: tuple[int, ...] = tuple()
    warm_f = np.zeros(n, dtype=float)
    warm_b = np.zeros(n, dtype=float)
    try:
        warm = algorithm_2_heuristic_user_selection(users, pE, pN, system, stack_cfg)
        warm_set = warm.offloading_set
        warm_f = np.asarray(warm.inner_result.f, dtype=float)
        warm_b = np.asarray(warm.inner_result.b, dtype=float)
    except Exception:
        pass
    warm_lookup = set(warm_set)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.o = pyo.Var(m.I, within=pyo.Binary)
    m.f = pyo.Var(m.I, within=pyo.NonNegativeReals, bounds=(0.0, float(system.F)))
    m.b = pyo.Var(m.I, within=pyo.NonNegativeReals, bounds=(0.0, float(system.B)))
    m.f_inv = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.b_inv = pyo.Var(m.I, within=pyo.NonNegativeReals)

    for i in range(n):
        m.o[i].set_value(1 if i in warm_lookup else 0)
        m.f[i].set_value(float(np.clip(warm_f[i], 0.0, system.F)) if i in warm_lookup else 0.0)
        m.b[i].set_value(float(np.clip(warm_b[i], 0.0, system.B)) if i in warm_lookup else 0.0)
        m.f_inv[i].set_value(1.0 / max(pyo.value(m.f[i]), eps) if i in warm_lookup else 0.0)
        m.b_inv[i].set_value(1.0 / max(pyo.value(m.b[i]), eps) if i in warm_lookup else 0.0)

    def _inv_f_rule(mm, i):
        return mm.f_inv[i] * (mm.f[i] + eps) >= mm.o[i]

    def _inv_b_rule(mm, i):
        return mm.b_inv[i] * (mm.b[i] + eps) >= mm.o[i]

    def _link_f_rule(mm, i):
        return mm.f[i] <= system.F * mm.o[i]

    def _link_b_rule(mm, i):
        return mm.b[i] <= system.B * mm.o[i]

    def _lb_f_rule(mm, i):
        return mm.f[i] >= eps * mm.o[i]

    def _lb_b_rule(mm, i):
        return mm.b[i] >= eps * mm.o[i]

    def _ir_rule(mm, i):
        m_cost_i = max(1.0, float(data.cl[i]))
        ce_i = data.aw[i] * mm.f_inv[i] + data.th[i] * mm.b_inv[i] + pE * mm.f[i] + pN * mm.b[i]
        return ce_i <= data.cl[i] + m_cost_i * (1 - mm.o[i])

    m.inv_f_con = pyo.Constraint(m.I, rule=_inv_f_rule)
    m.inv_b_con = pyo.Constraint(m.I, rule=_inv_b_rule)
    m.link_f_con = pyo.Constraint(m.I, rule=_link_f_rule)
    m.link_b_con = pyo.Constraint(m.I, rule=_link_b_rule)
    m.lb_f_con = pyo.Constraint(m.I, rule=_lb_f_rule)
    m.lb_b_con = pyo.Constraint(m.I, rule=_lb_b_rule)
    m.ir_con = pyo.Constraint(m.I, rule=_ir_rule)
    m.cap_f = pyo.Constraint(expr=sum(m.f[i] for i in m.I) <= system.F)
    m.cap_b = pyo.Constraint(expr=sum(m.b[i] for i in m.I) <= system.B)

    def _obj_rule(mm):
        offload = sum(
            mm.o[i] * (data.aw[i] * mm.f_inv[i] + data.th[i] * mm.b_inv[i] + pE * mm.f[i] + pN * mm.b[i])
            for i in mm.I
        )
        local = sum((1 - mm.o[i]) * data.cl[i] for i in mm.I)
        return offload + local

    m.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory("scip")
    scip_path = _resolve_scip_executable(scip_executable)
    if scip_path is None:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), tuple(), local_upper, False
    if scip_path != "scip":
        solver.set_executable(scip_path, validate=False)
    if not solver.available(False):
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), tuple(), local_upper, False
    solver.options["limits/time"] = float(base_cfg.gekko_time_limit)
    solver.options["limits/gap"] = float(base_cfg.gekko_mip_gap)
    solver.options["display/verblevel"] = 0

    try:
        results = solver.solve(m, tee=False)
        tc = results.solver.termination_condition
        solved = tc in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        }
    except Exception:
        solved = False

    if not solved:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), tuple(), local_upper, False

    f_vals = np.array([float(pyo.value(m.f[i])) for i in range(n)], dtype=float)
    b_vals = np.array([float(pyo.value(m.b[i])) for i in range(n)], dtype=float)
    o_raw = np.array([float(pyo.value(m.o[i])) for i in range(n)], dtype=float)
    offloading_set = tuple(int(i) for i in np.flatnonzero(o_raw > 0.5))
    ce = _offload_costs(data, f_vals, b_vals, pE, pN)
    off_idx = np.asarray(offloading_set, dtype=int) if offloading_set else np.asarray([], dtype=int)
    loc_idx = np.asarray([i for i in range(n) if i not in set(offloading_set)], dtype=int)
    social = float(np.sum(ce[off_idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)

    finite_ok = bool(np.all(np.isfinite(f_vals)) and np.all(np.isfinite(b_vals)) and np.isfinite(social))
    cap_ok = bool(float(np.sum(f_vals)) <= system.F + cap_tol and float(np.sum(b_vals)) <= system.B + cap_tol)
    if off_idx.size:
        ir_max = float(np.max(np.maximum(ce[off_idx] - data.cl[off_idx], 0.0)))
    else:
        ir_max = 0.0
    ir_ok = ir_max <= ir_tol
    objective_ok = social <= local_upper + 1e-4
    success = finite_ok and cap_ok and ir_ok and objective_ok
    if not success:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float), tuple(), local_upper, False
    return f_vals, b_vals, offloading_set, social, True


def _solve_centralized_pyomo_scip(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    scip_executable: str | None = None,
) -> BaselineOutcome:
    import time

    start_time = time.perf_counter()
    f, b, offloading_set, social_cost, success = _solve_with_pyomo_scip(
        users, pE, pN, system, stack_cfg, base_cfg, scip_executable
    )
    runtime = time.perf_counter() - start_time
    return _build_outcome_from_solver_allocations(
        users,
        f,
        b,
        offloading_set,
        pE,
        pN,
        system,
        "CS_PYOMO_SCIP",
        meta={
            "solver": "pyomo_scip",
            "social_cost": round(social_cost, 6),
            "offloading_size": len(offloading_set),
            "runtime_sec": round(runtime, 4),
            "success": success,
        },
    )


def _solve_with_gekko(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], float, bool]:
    """Solve MINLP with GEKKO+APOPT with warm-start, two-stage solve, and strict checks."""
    from gekko import GEKKO

    n = users.n
    data = _build_data(users)
    eps = 1e-8
    cap_tol = 1e-5
    ir_tol = 1e-5
    bin_tol = 1e-3
    local_upper = float(np.sum(data.cl))

    warm_set: tuple[int, ...] = tuple()
    warm_f = np.zeros(n, dtype=float)
    warm_b = np.zeros(n, dtype=float)
    try:
        warm = algorithm_2_heuristic_user_selection(users, pE, pN, system, stack_cfg)
        warm_set = warm.offloading_set
        warm_f = np.asarray(warm.inner_result.f, dtype=float)
        warm_b = np.asarray(warm.inner_result.b, dtype=float)
    except Exception:
        pass
    warm_lookup = set(warm_set)

    def _extract_value(var: object) -> float:
        try:
            value = getattr(var, "value", None)
            if value is None:
                return 0.0
            if isinstance(value, (list, np.ndarray)):
                return float(value[0]) if len(value) > 0 else 0.0
            return float(value)
        except Exception:
            return 0.0

    def _set_initial_values(mode: str, o: list, f: list, b: list, f_inv: list, b_inv: list) -> None:
        if mode == "warm":
            for i in range(n):
                is_off = 1.0 if i in warm_lookup else 0.0
                fi = float(np.clip(warm_f[i], 0.0, system.F)) if is_off > 0 else 0.0
                bi = float(np.clip(warm_b[i], 0.0, system.B)) if is_off > 0 else 0.0
                o[i].value = is_off
                f[i].value = fi
                b[i].value = bi
                f_inv[i].value = (1.0 / max(fi, eps)) if is_off > 0 else 0.0
                b_inv[i].value = (1.0 / max(bi, eps)) if is_off > 0 else 0.0
            return

        if mode == "heuristic":
            ce_star = 2.0 * np.sqrt(data.aw * max(pE, eps)) + 2.0 * np.sqrt(data.th * max(pN, eps))
            off = ce_star < data.cl
            f0 = np.zeros(n, dtype=float)
            b0 = np.zeros(n, dtype=float)
            idx = np.flatnonzero(off)
            if idx.size:
                f0[idx] = np.sqrt(data.aw[idx] / max(pE, eps))
                b0[idx] = np.sqrt(data.th[idx] / max(pN, eps))
                sf = float(np.sum(f0[idx]))
                sb = float(np.sum(b0[idx]))
                if sf > system.F and sf > 0.0:
                    f0[idx] *= system.F / sf
                if sb > system.B and sb > 0.0:
                    b0[idx] *= system.B / sb
            for i in range(n):
                is_off = 1.0 if bool(off[i]) else 0.0
                fi = float(f0[i]) if is_off > 0 else 0.0
                bi = float(b0[i]) if is_off > 0 else 0.0
                o[i].value = is_off
                f[i].value = fi
                b[i].value = bi
                f_inv[i].value = (1.0 / max(fi, eps)) if is_off > 0 else 0.0
                b_inv[i].value = (1.0 / max(bi, eps)) if is_off > 0 else 0.0
            return

        for i in range(n):
            o[i].value = 0.0
            f[i].value = 0.0
            b[i].value = 0.0
            f_inv[i].value = 0.0
            b_inv[i].value = 0.0

    def _run_attempt(init_mode: str) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], float, bool]:
        m = GEKKO(remote=False)
        m.options.SOLVER = 1
        m.options.IMODE = 3

        o = [m.Var(integer=True, lb=0, ub=1, name=f"o_{i}") for i in range(n)]
        f = [m.Var(lb=0, ub=system.F, name=f"f_{i}") for i in range(n)]
        b = [m.Var(lb=0, ub=system.B, name=f"b_{i}") for i in range(n)]
        f_inv = [m.Var(lb=0, name=f"f_inv_{i}") for i in range(n)]
        b_inv = [m.Var(lb=0, name=f"b_inv_{i}") for i in range(n)]

        for i in range(n):
            m.Equation(f_inv[i] * (f[i] + eps) >= o[i])
            m.Equation(b_inv[i] * (b[i] + eps) >= o[i])
            m.Equation(f[i] <= system.F * o[i])
            m.Equation(b[i] <= system.B * o[i])
            m.Equation(f[i] >= eps * o[i])
            m.Equation(b[i] >= eps * o[i])

            m_cost_i = max(1.0, float(data.cl[i]))
            ce_i = data.aw[i] * f_inv[i] + data.th[i] * b_inv[i] + pE * f[i] + pN * b[i]
            m.Equation(ce_i <= data.cl[i] + m_cost_i * (1 - o[i]))

        m.Equation(sum(f) <= system.F)
        m.Equation(sum(b) <= system.B)

        offload_cost = sum(
            o[i] * (data.aw[i] * f_inv[i] + data.th[i] * b_inv[i] + pE * f[i] + pN * b[i])
            for i in range(n)
        )
        local_cost_sum = sum((1 - o[i]) * data.cl[i] for i in range(n))
        m.Obj(offload_cost + local_cost_sum)

        _set_initial_values(init_mode, o, f, b, f_inv, b_inv)

        try:
            m.solver_options = [
                f"max_time {max(10, int(base_cfg.gekko_time_limit // 2))}",
                f"minlp_maximum_iterations {max(100, int(base_cfg.gekko_max_iter // 2))}",
                "minlp_as_nlp 1",
                f"minlp_gap_tol {base_cfg.gekko_mip_gap}",
            ]
            m.solve(disp=False, debug=0)
        except Exception:
            pass

        solved = False
        try:
            m.solver_options = [
                f"max_time {base_cfg.gekko_time_limit}",
                f"minlp_maximum_iterations {base_cfg.gekko_max_iter}",
                "minlp_as_nlp 0",
                "minlp_branch_method 1",
                f"minlp_gap_tol {base_cfg.gekko_mip_gap}",
            ]
            m.solve(disp=False, debug=0)
            solved = True
        except Exception:
            try:
                m.solver_options = [
                    f"max_time {base_cfg.gekko_time_limit}",
                    f"minlp_maximum_iterations {base_cfg.gekko_max_iter}",
                    "minlp_as_nlp 0",
                    "minlp_branch_method 3",
                    f"minlp_gap_tol {min(1e-2, max(base_cfg.gekko_mip_gap, 1e-3))}",
                ]
                m.solve(disp=False, debug=0)
                solved = True
            except Exception:
                solved = False

        f_vals = np.array([_extract_value(v) for v in f], dtype=float)
        b_vals = np.array([_extract_value(v) for v in b], dtype=float)
        o_raw = np.array([_extract_value(v) for v in o], dtype=float)
        offloading_set = tuple(int(i) for i in np.flatnonzero(o_raw > 0.5))

        ce = _offload_costs(data, f_vals, b_vals, pE, pN)
        off_idx = np.asarray(offloading_set, dtype=int) if offloading_set else np.asarray([], dtype=int)
        loc_idx = np.asarray([i for i in range(n) if i not in set(offloading_set)], dtype=int)
        social = float(np.sum(ce[off_idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)

        finite_ok = bool(np.all(np.isfinite(f_vals)) and np.all(np.isfinite(b_vals)) and np.isfinite(social))
        cap_ok = bool(float(np.sum(f_vals)) <= system.F + cap_tol and float(np.sum(b_vals)) <= system.B + cap_tol)
        integral_ok = bool(np.all((o_raw <= bin_tol) | (o_raw >= 1.0 - bin_tol)))
        if off_idx.size:
            ir_max = float(np.max(np.maximum(ce[off_idx] - data.cl[off_idx], 0.0)))
        else:
            ir_max = 0.0
        ir_ok = ir_max <= ir_tol
        objective_ok = social <= local_upper + 1e-4
        success = solved and finite_ok and cap_ok and integral_ok and ir_ok and objective_ok
        return f_vals, b_vals, offloading_set, social, success

    init_modes = ["warm", "heuristic", "empty"] if warm_set else ["heuristic", "empty"]
    best: tuple[np.ndarray, np.ndarray, tuple[int, ...], float, bool] | None = None
    for mode in init_modes:
        cand = _run_attempt(mode)
        if not cand[4]:
            continue
        if best is None or cand[3] < best[3]:
            best = cand

    if best is not None:
        return best
    return np.zeros(n, dtype=float), np.zeros(n, dtype=float), tuple(), local_upper, False


def _solve_centralized_minlp(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """MINLP-based centralized solver using GEKKO + APOPT."""
    import time

    start_time = time.perf_counter()

    # Solve with GEKKO
    f, b, offloading_set, social_cost, success = _solve_with_gekko(
        users, pE, pN, system, stack_cfg, base_cfg
    )

    runtime = time.perf_counter() - start_time

    # Build outcome using existing helper
    return _build_outcome_from_solver_allocations(
        users,
        f,
        b,
        offloading_set,
        pE,
        pN,
        system,
        "CS_MINLP",
        meta={
            "solver": "gekko_apopt",
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
    """Enumeration-based centralized solver over all user subsets."""

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
    Solve Stage II using exhaustive enumeration over all user sets.
    """
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


def evaluate_stage1_price_grid(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    pE_min: float,
    pE_max: float,
    pN_min: float,
    pN_max: float,
    pE_points: int,
    pN_points: int,
    stage2_method: str | None = None,
    progress_cb: Callable[[int, int, float, float], None] | None = None,
) -> Stage1GridEvaluation:
    if pE_points < 2 or pN_points < 2:
        raise ValueError("pE_points and pN_points must be at least 2.")
    if pE_max <= pE_min or pN_max <= pN_min:
        raise ValueError("Price upper bounds must be greater than lower bounds.")

    method = stage2_method or base_cfg.stage2_solver_for_pricing
    pE_grid = np.linspace(float(pE_min), float(pE_max), int(pE_points))
    pN_grid = np.linspace(float(pN_min), float(pN_max), int(pN_points))

    outcomes: list[list[BaselineOutcome]] = []
    esp_rev = np.zeros((pN_grid.size, pE_grid.size), dtype=float)
    nsp_rev = np.zeros((pN_grid.size, pE_grid.size), dtype=float)
    total_points = pN_grid.size * pE_grid.size
    done_points = 0

    for j, pN in enumerate(pN_grid):
        row: list[BaselineOutcome] = []
        for i, pE in enumerate(pE_grid):
            out = _stage2_solver(method, users, float(pE), float(pN), system, stack_cfg, base_cfg)
            row.append(out)
            esp_rev[j, i] = out.esp_revenue
            nsp_rev[j, i] = out.nsp_revenue
            done_points += 1
            if progress_cb is not None:
                progress_cb(done_points, total_points, float(pE), float(pN))
        outcomes.append(row)

    esp_max_per_pN = np.max(esp_rev, axis=1, keepdims=True)
    nsp_max_per_pE = np.max(nsp_rev, axis=0, keepdims=True)
    eps_E = esp_max_per_pN - esp_rev
    eps_N = nsp_max_per_pE - nsp_rev
    eps = np.maximum(eps_E, eps_N)

    return Stage1GridEvaluation(
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        outcomes=outcomes,
        esp_rev=esp_rev,
        nsp_rev=nsp_rev,
        eps=eps,
    )


def baseline_stage1_grid_search_oracle(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Grid-search SE proxy using *real-revenue deviation gap* definition.

    Definition aligned with plotting scripts:
      eps_E(i,j) = max_{i'} R_E(i',j) - R_E(i,j)
      eps_N(i,j) = max_{j'} R_N(i,j') - R_N(i,j)
      eps(i,j)   = max(eps_E(i,j), eps_N(i,j))

    Choose grid point with minimum eps(i,j) only.
    """
    grid = evaluate_stage1_price_grid(
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        base_cfg=base_cfg,
        pE_min=system.cE,
        pE_max=base_cfg.max_price_E,
        pN_min=system.cN,
        pN_max=base_cfg.max_price_N,
        pE_points=base_cfg.gso_grid_points,
        pN_points=base_cfg.gso_grid_points,
        stage2_method=base_cfg.stage2_solver_for_pricing,
    )
    bj, bi = np.unravel_index(int(np.argmin(grid.eps)), grid.eps.shape)
    best = grid.outcomes[bj][bi]
    return BaselineOutcome(
        name="GSO",
        price=best.price,
        offloading_set=best.offloading_set,
        social_cost=best.social_cost,
        esp_revenue=best.esp_revenue,
        nsp_revenue=best.nsp_revenue,
        epsilon_proxy=float(grid.eps[bj, bi]),
        meta={
            "grid_points": base_cfg.gso_grid_points,
            "oracle_eps_definition": "real_revenue_deviation_gap",
            "selection_rule": "min_eps_only",
        },
    )


def build_discrete_best_response_maps(
    esp_rev: np.ndarray,
    nsp_rev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if esp_rev.ndim != 2 or nsp_rev.ndim != 2 or esp_rev.shape != nsp_rev.shape:
        raise ValueError("esp_rev and nsp_rev must be 2D arrays with the same shape.")
    br_esp_idx_per_pN = np.argmax(esp_rev, axis=1).astype(int)  # row-wise best pE index
    br_nsp_idx_per_pE = np.argmax(nsp_rev, axis=0).astype(int)  # col-wise best pN index
    return br_esp_idx_per_pN, br_nsp_idx_per_pE


def run_discrete_br_dynamics(
    esp_rev: np.ndarray,
    nsp_rev: np.ndarray,
    start_idx: tuple[int, int],
    max_iters: int,
    mode: str = "alternating",
) -> list[tuple[int, int]]:
    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")
    if mode not in {"alternating", "greedy"}:
        raise ValueError("mode must be 'alternating' or 'greedy'.")

    br_esp_idx_per_pN, br_nsp_idx_per_pE = build_discrete_best_response_maps(esp_rev, nsp_rev)
    n_rows, n_cols = esp_rev.shape
    j, i = int(start_idx[0]), int(start_idx[1])
    if not (0 <= j < n_rows and 0 <= i < n_cols):
        raise ValueError("start_idx out of grid range.")

    trajectory: list[tuple[int, int]] = [(j, i)]
    for _ in range(max_iters):
        if mode == "alternating":
            changed = False
            i_next = int(br_esp_idx_per_pN[j])
            if i_next != i:
                i = i_next
                trajectory.append((j, i))
                changed = True
            j_next = int(br_nsp_idx_per_pE[i])
            if j_next != j:
                j = j_next
                trajectory.append((j, i))
                changed = True
            if not changed:
                break
            continue

        # greedy mode: choose the provider with larger one-step BR improvement.
        i_cand = int(br_esp_idx_per_pN[j])
        j_cand = int(br_nsp_idx_per_pE[i])
        delta_e = float(esp_rev[j, i_cand] - esp_rev[j, i])
        delta_n = float(nsp_rev[j_cand, i] - nsp_rev[j, i])
        if delta_e <= 1e-12 and delta_n <= 1e-12:
            break
        if delta_e >= delta_n and i_cand != i:
            i = i_cand
            trajectory.append((j, i))
        elif j_cand != j:
            j = j_cand
            trajectory.append((j, i))
        else:
            break

    return trajectory


def baseline_stage1_pbdr_discrete_br_map(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    mode: str = "alternating",
) -> BaselineOutcome:
    """
    Stage-I PBDR on a discrete BR map:
    1) Build one revenue heatmap grid.
    2) Compute discrete true BR maps for ESP/NSP from that grid.
    3) Run alternating/greedy BR dynamics on the BR map.
    """
    grid = evaluate_stage1_price_grid(
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        base_cfg=base_cfg,
        pE_min=system.cE,
        pE_max=base_cfg.max_price_E,
        pN_min=system.cN,
        pN_max=base_cfg.max_price_N,
        pE_points=base_cfg.pbdr_grid_points,
        pN_points=base_cfg.pbdr_grid_points,
        stage2_method=base_cfg.stage2_solver_for_pricing,
    )
    start_pE = max(system.cE, stack_cfg.initial_pE)
    start_pN = max(system.cN, stack_cfg.initial_pN)
    start_i = int(np.argmin(np.abs(grid.pE_grid - start_pE)))
    start_j = int(np.argmin(np.abs(grid.pN_grid - start_pN)))
    traj = run_discrete_br_dynamics(
        grid.esp_rev,
        grid.nsp_rev,
        start_idx=(start_j, start_i),
        max_iters=base_cfg.pbdr_max_iters,
        mode=mode,
    )
    end_j, end_i = traj[-1]
    out = grid.outcomes[end_j][end_i]
    return BaselineOutcome(
        name="PBRD_DISCRETE",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=float(grid.eps[end_j, end_i]),
        meta={
            "mode": mode,
            "trajectory_len": len(traj),
            "max_iters": base_cfg.pbdr_max_iters,
            "grid_points": base_cfg.pbdr_grid_points,
        },
    )


def _best_response_1d(
    provider: str,
    users: UserBatch,
    fixed_price: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    stage2_cache: dict[tuple[float, float], BaselineOutcome] | None = None,
) -> tuple[float, BaselineOutcome]:
    def cached_stage2(pE: float, pN: float) -> BaselineOutcome:
        key = _price_cache_key(pE, pN)
        if stage2_cache is not None and key in stage2_cache:
            return stage2_cache[key]
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        if stage2_cache is not None:
            stage2_cache[key] = out
        return out

    if provider == "E":
        grid = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.pbdr_grid_points)
        outs = [cached_stage2(float(pE), fixed_price) for pE in grid]
        best = max(outs, key=lambda o: o.esp_revenue)
        return best.price[0], best
    grid = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.pbdr_grid_points)
    outs = [cached_stage2(fixed_price, float(pN)) for pN in grid]
    best = max(outs, key=lambda o: o.nsp_revenue)
    return best.price[1], best


def _solve_leader_mpec_explicit_kkt(
    provider: str,
    users: UserBatch,
    own_price: float,
    fixed_price: float,
    offloading_set: tuple[int, ...],
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> tuple[float, BaselineOutcome]:
    """Solve a restricted leader MPEC with explicit follower KKT/complementarity conditions."""
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition

    if provider not in {"E", "N"}:
        raise ValueError("provider must be 'E' or 'N'.")

    data = _build_data(users)
    offloading_set = _sorted_tuple(offloading_set)
    if not offloading_set:
        if provider == "E":
            pE = min(max(float(own_price), float(system.cE)), float(base_cfg.max_price_E))
            pN = float(fixed_price)
        else:
            pE = float(fixed_price)
            pN = min(max(float(own_price), float(system.cN)), float(base_cfg.max_price_N))
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        return (float(out.price[0] if provider == "E" else out.price[1]), out)

    idx = np.asarray(offloading_set, dtype=int)
    aw = data.aw[idx]
    th = data.th[idx]
    cl = data.cl[idx]

    if provider == "E":
        price_lb = float(system.cE)
        price_ub = float(base_cfg.max_price_E)
        fixed_other = float(fixed_price)
        own_init = min(max(float(own_price), price_lb), price_ub)
    else:
        price_lb = float(system.cN)
        price_ub = float(base_cfg.max_price_N)
        fixed_other = float(fixed_price)
        own_init = min(max(float(own_price), price_lb), price_ub)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, idx.size - 1)
    m.f = pyo.Var(m.I, within=pyo.NonNegativeReals, bounds=(1e-8, float(system.F)))
    m.b = pyo.Var(m.I, within=pyo.NonNegativeReals, bounds=(1e-8, float(system.B)))
    m.mu = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.lambda_F = pyo.Var(within=pyo.NonNegativeReals)
    m.lambda_B = pyo.Var(within=pyo.NonNegativeReals)
    m.p = pyo.Var(within=pyo.NonNegativeReals, bounds=(price_lb, price_ub))
    m.p.set_value(own_init)
    pE0 = own_init if provider == "E" else fixed_other
    pN0 = fixed_other if provider == "E" else own_init
    sE = float(np.sum(np.sqrt(np.maximum(aw, 1e-12))))
    sN = float(np.sum(np.sqrt(np.maximum(th, 1e-12))))
    tE0 = max(pE0, (sE / max(system.F, 1e-9)) ** 2 if idx.size else pE0)
    tN0 = max(pN0, (sN / max(system.B, 1e-9)) ** 2 if idx.size else pN0)
    m.lambda_F.set_value(max(0.0, tE0 - pE0))
    m.lambda_B.set_value(max(0.0, tN0 - pN0))
    f0 = np.sqrt(np.maximum(aw, 1e-12) / max(tE0, 1e-9))
    b0 = np.sqrt(np.maximum(th, 1e-12) / max(tN0, 1e-9))
    for loc in range(idx.size):
        m.f[loc].set_value(float(np.clip(f0[loc], 1e-8, system.F)))
        m.b[loc].set_value(float(np.clip(b0[loc], 1e-8, system.B)))
        ce0 = aw[loc] / max(f0[loc], 1e-9) + th[loc] / max(b0[loc], 1e-9) + pE0 * f0[loc] + pN0 * b0[loc]
        m.mu[loc].set_value(1e-3 if abs(ce0 - cl[loc]) <= 1e-4 else 0.0)

    def _pE(mm):
        return mm.p if provider == "E" else fixed_other

    def _pN(mm):
        return fixed_other if provider == "E" else mm.p

    def _ce_expr(mm, i):
        return float(aw[i]) / mm.f[i] + float(th[i]) / mm.b[i] + _pE(mm) * mm.f[i] + _pN(mm) * mm.b[i]

    def _stationarity_f_rule(mm, i):
        return (1.0 + mm.mu[i]) * (_pE(mm) - float(aw[i]) / (mm.f[i] * mm.f[i])) + mm.lambda_F == 0.0

    def _stationarity_b_rule(mm, i):
        return (1.0 + mm.mu[i]) * (_pN(mm) - float(th[i]) / (mm.b[i] * mm.b[i])) + mm.lambda_B == 0.0

    def _ir_rule(mm, i):
        return _ce_expr(mm, i) <= float(cl[i])

    def _ir_comp_rule(mm, i):
        return mm.mu[i] * (float(cl[i]) - _ce_expr(mm, i)) == 0.0

    m.stationarity_f = pyo.Constraint(m.I, rule=_stationarity_f_rule)
    m.stationarity_b = pyo.Constraint(m.I, rule=_stationarity_b_rule)
    sum_f = sum(m.f[i] for i in m.I)
    sum_b = sum(m.b[i] for i in m.I)
    m.cap_f = pyo.Constraint(expr=sum_f <= float(system.F))
    m.cap_b = pyo.Constraint(expr=sum_b <= float(system.B))
    m.comp_f = pyo.Constraint(expr=m.lambda_F * (float(system.F) - sum_f) == 0.0)
    m.comp_b = pyo.Constraint(expr=m.lambda_B * (float(system.B) - sum_b) == 0.0)
    m.ir = pyo.Constraint(m.I, rule=_ir_rule)
    m.comp_ir = pyo.Constraint(m.I, rule=_ir_comp_rule)

    if provider == "E":
        m.obj = pyo.Objective(expr=(m.p - float(system.cE)) * sum_f, sense=pyo.maximize)
    else:
        m.obj = pyo.Objective(expr=(m.p - float(system.cN)) * sum_b, sense=pyo.maximize)

    solver = pyo.SolverFactory("ipopt")
    if not solver.available(False):
        raise RuntimeError("IPOPT is required for restricted follower-KKT MPEC solve but is not available.")
    solver.options["tol"] = max(float(base_cfg.pbdr_tol), 1e-6)
    solver.options["max_iter"] = max(300, int(base_cfg.gekko_max_iter))
    solver.options["print_level"] = 0
    results = solver.solve(m, tee=False)
    tc = results.solver.termination_condition
    solved = tc in {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    }
    if not solved:
        raise RuntimeError(
            f"Restricted explicit KKT MPEC solve failed for provider {provider} on set size {idx.size}: termination={tc}"
        )

    if provider == "E":
        pE = float(pyo.value(m.p))
        pN = float(fixed_other)
    else:
        pE = float(fixed_other)
        pN = float(pyo.value(m.p))

    out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    return (float(out.price[0] if provider == "E" else out.price[1]), out)


def _solve_bilevel_stage2_oracle(
    users: UserBatch,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    stage2_cache: dict[tuple[float, float], BaselineOutcome],
) -> BaselineOutcome:
    """Lower-level oracle for bilevel-MINLP leader search.

    Search-time oracle prefers the explicit centralized MINLP follower solve and
    only falls back to exact enumeration when the MINLP solve is unsuccessful
    and the instance is still small enough for the exact oracle.
    """

    key = _price_cache_key(pE, pN)
    if key in stage2_cache:
        return stage2_cache[key]

    out = _solve_centralized_minlp(users, pE, pN, system, stack_cfg, base_cfg)
    success = bool(out.meta.get("success"))
    if (not success) and bool(base_cfg.cs_fallback_to_enum) and users.n <= int(base_cfg.exact_max_users):
        out = baseline_stage2_centralized_solver(users, pE, pN, system, stack_cfg, base_cfg)
    stage2_cache[key] = out
    return out


def _solve_leader_bilevel_minlp(
    provider: str,
    users: UserBatch,
    own_price: float,
    fixed_price: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    stage2_cache: dict[tuple[float, float], BaselineOutcome],
) -> tuple[float, BaselineOutcome]:
    """Solve a leader subproblem as a bilevel-MINLP with a full follower MINLP oracle."""
    if provider not in {"E", "N"}:
        raise ValueError("provider must be 'E' or 'N'.")

    if provider == "E":
        lb = float(system.cE)
        ub = float(base_cfg.max_price_E)
        reward_getter = lambda out: float(out.esp_revenue)

        def eval_out(price: float) -> BaselineOutcome:
            pE = round(min(max(float(price), lb), ub), 6)
            return _solve_bilevel_stage2_oracle(users, pE, fixed_price, system, stack_cfg, base_cfg, stage2_cache)

    else:
        lb = float(system.cN)
        ub = float(base_cfg.max_price_N)
        reward_getter = lambda out: float(out.nsp_revenue)

        def eval_out(price: float) -> BaselineOutcome:
            pN = round(min(max(float(price), lb), ub), 6)
            return _solve_bilevel_stage2_oracle(users, fixed_price, pN, system, stack_cfg, base_cfg, stage2_cache)

    def objective(price: float) -> float:
        return -reward_getter(eval_out(float(price)))

    xatol = max(float(base_cfg.pbdr_tol), 1e-2)
    maxiter = max(12, min(24, int(base_cfg.pbdr_max_iters) * 4))
    res = minimize_scalar(
        objective,
        bounds=(lb, ub),
        method="bounded",
        options={"xatol": xatol, "maxiter": maxiter},
    )

    span = max(ub - lb, 1e-9)
    local_delta = max(span / 100.0, 2.0 * xatol)
    candidate_prices = [
        lb,
        ub,
        min(max(float(own_price), lb), ub),
        float(res.x) if getattr(res, "success", True) else min(max(float(own_price), lb), ub),
        0.5 * (lb + ub),
    ]
    center = min(max(float(candidate_prices[-1]), lb), ub)
    candidate_prices.extend(
        [
            min(max(center - local_delta, lb), ub),
            center,
            min(max(center + local_delta, lb), ub),
        ]
    )

    best_out: BaselineOutcome | None = None
    best_price = center
    best_reward = -float("inf")
    seen: set[float] = set()
    for price in candidate_prices:
        price = round(float(price), 12)
        if price in seen:
            continue
        seen.add(price)
        out = eval_out(price)
        reward = reward_getter(out)
        if reward > best_reward + 1e-12:
            best_reward = reward
            best_out = out
            best_price = float(out.price[0] if provider == "E" else out.price[1])

    assert best_out is not None
    return best_price, best_out


def run_stage1_epec_diagonalization(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    *,
    outcome_name: str = "EPEC-DIAG",
) -> tuple[BaselineOutcome, list[tuple[float, float]]]:
    """Alternating diagonalization for the two-leader EPEC via bilevel-MINLP subproblems."""
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    trajectory: list[tuple[float, float]] = [(float(pE), float(pN))]
    actual_iters = 0
    converged = False
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}

    for t in range(base_cfg.pbdr_max_iters):
        prev_pE, prev_pN = pE, pN
        pE_next, _ = _solve_leader_bilevel_minlp(
            "E",
            users,
            pE,
            pN,
            system,
            stack_cfg,
            base_cfg,
            stage2_cache,
        )
        pE = float(pE_next)
        if abs(pE - prev_pE) > 1e-12:
            trajectory.append((pE, float(pN)))

        pN_next, _ = _solve_leader_bilevel_minlp(
            "N",
            users,
            pN,
            pE,
            system,
            stack_cfg,
            base_cfg,
            stage2_cache,
        )
        pN = float(pN_next)
        if abs(pN - prev_pN) > 1e-12:
            trajectory.append((float(pE), pN))

        actual_iters = t + 1
        if abs(pE - prev_pE) + abs(pN - prev_pN) <= base_cfg.pbdr_tol:
            converged = True
            break

    final_key = _price_cache_key(pE, pN)
    if final_key in stage2_cache:
        out = stage2_cache[final_key]
    else:
        out = _solve_bilevel_stage2_oracle(users, pE, pN, system, stack_cfg, base_cfg, stage2_cache)
        stage2_cache[final_key] = out
    if users.n <= int(base_cfg.exact_max_users):
        out = baseline_stage2_centralized_solver(users, pE, pN, system, stack_cfg, base_cfg)
        stage2_cache[final_key] = out

    return (
        BaselineOutcome(
            name=outcome_name,
            price=out.price,
            offloading_set=out.offloading_set,
            social_cost=out.social_cost,
            esp_revenue=out.esp_revenue,
            nsp_revenue=out.nsp_revenue,
            epsilon_proxy=out.epsilon_proxy,
            meta={
                "iters": actual_iters,
                "converged": str(converged).lower(),
                "grid_points": base_cfg.pbdr_grid_points,
                "trajectory_len": len(trajectory),
                "leader_subproblem": "bilevel_minlp",
                "leader_subproblem_solver": "scipy_minimize_scalar_bounded",
                "leader_subproblem_formulation": "upper_continuous_price_plus_lower_level_minlp",
                "search_follower_oracle": "CS_MINLP_GEKKO_with_exact_enum_fallback",
                "final_validation": (
                    "CS_exact_enumeration"
                    if users.n <= int(base_cfg.exact_max_users)
                    else "CS_MINLP_GEKKO"
                ),
                "stage2_unique_prices": len(stage2_cache),
            },
        ),
        trajectory,
    )


def baseline_stage1_epec_diagonalization(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    outcome, _ = run_stage1_epec_diagonalization(users, system, stack_cfg, base_cfg, outcome_name="EPEC-DIAG")
    return outcome


def baseline_stage1_pbdr(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    actual_iters = 0
    for t in range(base_cfg.pbdr_max_iters):
        prev_pE, prev_pN = pE, pN
        pE, _ = _best_response_1d("E", users, pN, system, stack_cfg, base_cfg)
        pN, _ = _best_response_1d("N", users, pE, system, stack_cfg, base_cfg)
        actual_iters = t + 1
        if abs(pE - prev_pE) + abs(pN - prev_pN) <= base_cfg.pbdr_tol:
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
        meta={"iters": actual_iters, "grid_points": base_cfg.pbdr_grid_points},
    )


def _joint_action_q_greedy_action(
    q_esp: np.ndarray,
    q_nsp: np.ndarray,
    tol: float = 1e-12,
) -> tuple[int, int, bool]:
    """Select a greedy joint action from two joint-action Q tables.

    Preference order:
    1. Any pure-Nash joint action under the current Q stage game.
    2. Fallback to the joint action maximizing q_esp + q_nsp.
    """
    if q_esp.shape != q_nsp.shape:
        raise ValueError("ESP/NSP Q tables must share the same shape.")
    best_esp = np.max(q_esp, axis=0, keepdims=True)
    best_nsp = np.max(q_nsp, axis=1, keepdims=True)
    pure_nash_mask = (q_esp >= best_esp - tol) & (q_nsp >= best_nsp - tol)
    if np.any(pure_nash_mask):
        candidates = np.argwhere(pure_nash_mask)
        scores = (q_esp + q_nsp)[pure_nash_mask]
        pick = candidates[int(np.argmax(scores))]
        return int(pick[0]), int(pick[1]), True
    flat_idx = int(np.argmax(q_esp + q_nsp))
    a_esp, a_nsp = np.unravel_index(flat_idx, q_esp.shape)
    return int(a_esp), int(a_nsp), False


def _objective_score_and_candidate(
    out: BaselineOutcome,
    objective_name: str,
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    stage2_cache: dict[tuple[float, float], BaselineOutcome],
    pE_audit_grid: np.ndarray,
    pN_audit_grid: np.ndarray,
) -> tuple[float, BaselineOutcome]:
    if objective_name == "real_revenue_deviation_gap":
        score = _real_deviation_gap_audit(
            out,
            users,
            system,
            stack_cfg,
            base_cfg,
            stage2_cache,
            pE_audit_grid,
            pN_audit_grid,
        )
        return float(score), replace(out, epsilon_proxy=float(score))
    if objective_name == "epsilon_proxy":
        return float(out.epsilon_proxy), out
    if objective_name == "joint_revenue":
        return float(-(out.esp_revenue + out.nsp_revenue)), out
    if objective_name == "social_cost":
        return float(out.social_cost), out
    raise ValueError(f"Unknown objective: {objective_name}")


def _objective_is_better(
    candidate_score: float,
    candidate_out: BaselineOutcome,
    incumbent_score: float | None,
    incumbent_out: BaselineOutcome | None,
) -> bool:
    if incumbent_out is None or incumbent_score is None:
        return True
    if candidate_score < incumbent_score - 1e-12:
        return True
    if abs(candidate_score - incumbent_score) > 1e-12:
        return False
    if candidate_out.epsilon_proxy < incumbent_out.epsilon_proxy - 1e-12:
        return True
    if abs(candidate_out.epsilon_proxy - incumbent_out.epsilon_proxy) > 1e-12:
        return False

    cand_rev = candidate_out.esp_revenue + candidate_out.nsp_revenue
    inc_rev = incumbent_out.esp_revenue + incumbent_out.nsp_revenue
    if cand_rev > inc_rev + 1e-12:
        return True
    if abs(cand_rev - inc_rev) > 1e-12:
        return False

    return candidate_out.social_cost < incumbent_out.social_cost - 1e-12


def _price_cache_key(pE: float, pN: float) -> tuple[float, float]:
    return (round(float(pE), 12), round(float(pN), 12))


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    ls = max(float(length_scale), 1e-9)
    x1_scaled = np.asarray(x1, dtype=float) / ls
    x2_scaled = np.asarray(x2, dtype=float) / ls
    d2 = np.sum((x1_scaled[:, None, :] - x2_scaled[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * d2)


def _gp_predict(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    query_x: np.ndarray,
    length_scale: float,
    noise: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if observed_x.ndim != 2 or query_x.ndim != 2:
        raise ValueError("observed_x and query_x must be 2D arrays.")
    if observed_x.shape[0] != observed_y.shape[0]:
        raise ValueError("observed_x and observed_y must have the same number of samples.")
    if observed_x.shape[0] == 0:
        raise ValueError("At least one observed sample is required for GP prediction.")

    y = np.asarray(observed_y, dtype=float)
    y_mean = float(np.mean(y))
    y_scale = float(np.std(y))
    if y_scale < 1e-9:
        y_scale = 1.0
    y_norm = (y - y_mean) / y_scale

    k_xx = _rbf_kernel(observed_x, observed_x, length_scale)
    eye = np.eye(k_xx.shape[0], dtype=float)
    jitter = max(float(noise), 1e-9)
    chol: np.ndarray | None = None
    for _ in range(6):
        try:
            chol = np.linalg.cholesky(k_xx + jitter * eye)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0

    if chol is None:
        mu = np.full(query_x.shape[0], y_mean, dtype=float)
        sigma = np.full(query_x.shape[0], y_scale, dtype=float)
        return mu, sigma

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_norm))
    k_xs = _rbf_kernel(observed_x, query_x, length_scale)
    mu_norm = k_xs.T @ alpha
    v = np.linalg.solve(chol, k_xs)
    var_norm = np.maximum(1.0 - np.sum(v * v, axis=0), 1e-9)
    mu = y_mean + y_scale * mu_norm
    sigma = y_scale * np.sqrt(var_norm)
    return mu, sigma


def _real_deviation_gap_audit(
    current_out: BaselineOutcome,
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    stage2_cache: dict[tuple[float, float], BaselineOutcome],
    pE_audit_grid: np.ndarray,
    pN_audit_grid: np.ndarray,
) -> float:
    def cached_stage2(pE: float, pN: float) -> BaselineOutcome:
        key = _price_cache_key(pE, pN)
        if key not in stage2_cache:
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

    pE_cur, pN_cur = float(current_out.price[0]), float(current_out.price[1])
    best_esp_rev = float(current_out.esp_revenue)
    best_nsp_rev = float(current_out.nsp_revenue)

    for cand_pE in pE_audit_grid:
        cand_out = cached_stage2(float(cand_pE), pN_cur)
        if cand_out.esp_revenue > best_esp_rev:
            best_esp_rev = float(cand_out.esp_revenue)

    for cand_pN in pN_audit_grid:
        cand_out = cached_stage2(pE_cur, float(cand_pN))
        if cand_out.nsp_revenue > best_nsp_rev:
            best_nsp_rev = float(cand_out.nsp_revenue)

    eps_E = float(best_esp_rev - current_out.esp_revenue)
    eps_N = float(best_nsp_rev - current_out.nsp_revenue)
    return float(max(eps_E, eps_N))


def _run_stage1_gp_bo(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    *,
    outcome_name: str,
    objective_name: str,
    init_points: int,
    seed_offset: int,
) -> BaselineOutcome:
    rng = np.random.default_rng(base_cfg.random_seed + seed_offset)
    x: list[tuple[float, float]] = []
    y: list[float] = []
    best: BaselineOutcome | None = None
    best_score: float | None = None
    init_points = max(1, int(init_points))
    candidate_pool = max(1, int(base_cfg.bo_candidate_pool))
    n_iters = max(0, int(base_cfg.bo_iters))
    pE_min, pE_max = float(system.cE), float(base_cfg.max_price_E)
    pN_min, pN_max = float(system.cN), float(base_cfg.max_price_N)
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}
    pE_audit_grid = np.linspace(pE_min, pE_max, max(2, int(base_cfg.gso_grid_points)))
    pN_audit_grid = np.linspace(pN_min, pN_max, max(2, int(base_cfg.gso_grid_points)))

    def evaluate(pE: float, pN: float) -> None:
        nonlocal best, best_score
        key = _price_cache_key(pE, pN)
        if key not in stage2_cache:
            stage2_cache[key] = _stage2_solver(
                base_cfg.stage2_solver_for_pricing,
                users,
                pE,
                pN,
                system,
                stack_cfg,
                base_cfg,
            )
        out = stage2_cache[key]
        score, candidate = _objective_score_and_candidate(
            out,
            objective_name,
            users,
            system,
            stack_cfg,
            base_cfg,
            stage2_cache,
            pE_audit_grid,
            pN_audit_grid,
        )
        x.append((pE, pN))
        y.append(score)
        if _objective_is_better(score, candidate, best_score, best):
            best = candidate
            best_score = float(score)

    start_pE = min(max(float(stack_cfg.initial_pE), pE_min), pE_max)
    start_pN = min(max(float(stack_cfg.initial_pN), pN_min), pN_max)
    evaluate(start_pE, start_pN)

    for _ in range(init_points - 1):
        pE = float(rng.uniform(system.cE, base_cfg.max_price_E))
        pN = float(rng.uniform(system.cN, base_cfg.max_price_N))
        evaluate(pE, pN)

    for t in range(n_iters):
        cand = np.column_stack(
            [
                rng.uniform(pE_min, pE_max, size=candidate_pool),
                rng.uniform(pN_min, pN_max, size=candidate_pool),
            ]
        )
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mu, sigma = _gp_predict(
            x_arr,
            y_arr,
            cand,
            length_scale=base_cfg.bo_kernel_bandwidth,
        )
        beta = max(float(base_cfg.bo_ucb_beta), 0.0) * math.sqrt(2.0 * math.log(t + 2.0))
        acquisition = mu - beta * sigma
        order = np.argsort(acquisition)
        pick = int(order[0])
        for idx in order:
            d2 = np.sum((x_arr - cand[int(idx)]) ** 2, axis=1)
            if float(np.min(d2)) > 1e-12:
                pick = int(idx)
                break
        pE = float(cand[pick, 0])
        pN = float(cand[pick, 1])
        evaluate(pE, pN)

    assert best is not None
    return BaselineOutcome(
        name=outcome_name,
        price=best.price,
        offloading_set=best.offloading_set,
        social_cost=best.social_cost,
        esp_revenue=best.esp_revenue,
        nsp_revenue=best.nsp_revenue,
        epsilon_proxy=best.epsilon_proxy,
        meta={
            "evals": len(x),
            "objective": objective_name,
            "acquisition": "gp_lcb",
            "init_points": init_points,
            "stage2_unique_prices": len(stage2_cache),
            **(
                {"audit_grid_points": int(pE_audit_grid.size)}
                if objective_name == "real_revenue_deviation_gap"
                else {}
            ),
        },
    )


def baseline_stage1_bo(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """GP-based Bayesian optimization on epsilon proxy."""
    return _run_stage1_gp_bo(
        users,
        system,
        stack_cfg,
        base_cfg,
        outcome_name="BO",
        objective_name="epsilon_proxy",
        init_points=int(base_cfg.bo_init_points),
        seed_offset=101,
    )


def baseline_stage1_bo_online_real_gap(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Online GP-BO variant with |D0|=1 on real deviation gap."""
    return _run_stage1_gp_bo(
        users,
        system,
        stack_cfg,
        base_cfg,
        outcome_name="BO-online",
        objective_name="real_revenue_deviation_gap",
        init_points=1,
        seed_offset=151,
    )


def run_stage1_genetic_algorithm(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
    *,
    outcome_name: str = "GA",
    objective_name: str | None = None,
) -> tuple[BaselineOutcome, list[tuple[float, float]]]:
    objective_name = str(objective_name or base_cfg.ga_objective).strip().lower()
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
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}
    pE_audit_grid = np.linspace(pE_min, pE_max, max(2, int(base_cfg.gso_grid_points)))
    pN_audit_grid = np.linspace(pN_min, pN_max, max(2, int(base_cfg.gso_grid_points)))
    evals = 0
    best_out: BaselineOutcome | None = None
    best_score: float | None = None
    best_price: tuple[float, float] | None = None

    def _clip_price(pE: float, pN: float) -> tuple[float, float]:
        return (
            round(min(max(float(pE), pE_min), pE_max), 6),
            round(min(max(float(pN), pN_min), pN_max), 6),
        )

    def _evaluate_individual(pE: float, pN: float) -> tuple[tuple[float, float], float, BaselineOutcome]:
        nonlocal evals, best_out, best_score, best_price
        pE, pN = _clip_price(pE, pN)
        evals += 1
        key = _price_cache_key(pE, pN)
        if key not in stage2_cache:
            stage2_cache[key] = _stage2_solver(
                base_cfg.stage2_solver_for_pricing,
                users,
                pE,
                pN,
                system,
                stack_cfg,
                base_cfg,
            )
        out = stage2_cache[key]
        score, candidate = _objective_score_and_candidate(
            out,
            objective_name,
            users,
            system,
            stack_cfg,
            base_cfg,
            stage2_cache,
            pE_audit_grid,
            pN_audit_grid,
        )
        if _objective_is_better(score, candidate, best_score, best_out):
            best_out = candidate
            best_score = float(score)
            best_price = (pE, pN)
        return (pE, pN), float(score), candidate

    def _tournament_select(
        population: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
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

    init_price = _clip_price(float(stack_cfg.initial_pE), float(stack_cfg.initial_pN))
    trajectory: list[tuple[float, float]] = [init_price]

    def _evaluate_population(pop: np.ndarray) -> tuple[np.ndarray, list[BaselineOutcome], np.ndarray]:
        scored_prices: list[tuple[float, float]] = []
        scored_outcomes: list[BaselineOutcome] = []
        score_values: list[float] = []
        for pE, pN in pop:
            price, score, out = _evaluate_individual(float(pE), float(pN))
            scored_prices.append(price)
            scored_outcomes.append(out)
            score_values.append(score)
        return np.asarray(scored_prices, dtype=float), scored_outcomes, np.asarray(score_values, dtype=float)

    population, _, scores = _evaluate_population(population)
    if best_price is not None and (
        not trajectory
        or abs(best_price[0] - trajectory[-1][0]) + abs(best_price[1] - trajectory[-1][1]) > 1e-12
    ):
        trajectory.append(best_price)

    for _gen in range(n_gen):
        order = np.argsort(scores)
        next_population: list[np.ndarray] = [population[int(i)].copy() for i in order[:elite_size]]
        while len(next_population) < n_pop:
            parent_a = _tournament_select(population, scores)
            parent_b = _tournament_select(population, scores)
            child_a, child_b = _crossover(parent_a, parent_b)
            next_population.append(_mutate(child_a))
            if len(next_population) < n_pop:
                next_population.append(_mutate(child_b))
        population = np.asarray(next_population, dtype=float)
        population, _, scores = _evaluate_population(population)
        if best_price is not None and (
            not trajectory
            or abs(best_price[0] - trajectory[-1][0]) + abs(best_price[1] - trajectory[-1][1]) > 1e-12
        ):
            trajectory.append(best_price)

    assert best_out is not None and best_score is not None
    if not trajectory and best_price is not None:
        trajectory = [best_price]
    return (
        BaselineOutcome(
            name=outcome_name,
            price=best_out.price,
            offloading_set=best_out.offloading_set,
            social_cost=best_out.social_cost,
            esp_revenue=best_out.esp_revenue,
            nsp_revenue=best_out.nsp_revenue,
            epsilon_proxy=best_out.epsilon_proxy,
            meta={
                "objective": objective_name,
                "population_size": n_pop,
                "generations": n_gen,
                "elite_size": elite_size,
                "tournament_size": tournament_size,
                "crossover_rate": round(crossover_rate, 6),
                "mutation_rate": round(mutation_rate, 6),
                "mutation_std": round(mutation_std, 6),
                "trajectory_len": len(trajectory),
                "evals": evals,
                "stage2_unique_prices": len(stage2_cache),
                **(
                    {"audit_grid_points": int(pE_audit_grid.size)}
                    if objective_name == "real_revenue_deviation_gap"
                    else {}
                ),
            },
        ),
        trajectory,
    )


def baseline_stage1_ga(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    outcome, _ = run_stage1_genetic_algorithm(users, system, stack_cfg, base_cfg, outcome_name="GA")
    return outcome


def baseline_stage1_drl(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Legacy joint-action tabular RL proxy, not a full paper-facing MARL wrapper."""
    grid_e = np.linspace(system.cE, base_cfg.max_price_E, base_cfg.drl_price_levels)
    grid_n = np.linspace(system.cN, base_cfg.max_price_N, base_cfg.drl_price_levels)
    action_deltas = (-1, 0, 1)
    q_esp = np.zeros((grid_e.size, grid_n.size, len(action_deltas), len(action_deltas)), dtype=float)
    q_nsp = np.zeros((grid_e.size, grid_n.size, len(action_deltas), len(action_deltas)), dtype=float)
    rng = np.random.default_rng(base_cfg.random_seed + 303)
    stage2_cache: dict[tuple[int, int], BaselineOutcome] = {}

    def clamp(idx: int, n: int) -> int:
        return min(max(idx, 0), n - 1)

    def state_outcome(e_idx: int, n_idx: int) -> BaselineOutcome:
        key = (int(e_idx), int(n_idx))
        if key not in stage2_cache:
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

    for _ in range(base_cfg.drl_episodes):
        i = int(rng.integers(0, grid_e.size))
        j = int(rng.integers(0, grid_n.size))
        for _step in range(base_cfg.drl_steps_per_episode):
            if rng.uniform() < base_cfg.drl_epsilon:
                a_esp = int(rng.integers(0, len(action_deltas)))
                a_nsp = int(rng.integers(0, len(action_deltas)))
            else:
                a_esp, a_nsp, _ = _joint_action_q_greedy_action(q_esp[i, j], q_nsp[i, j])
            ni = clamp(i + action_deltas[a_esp], grid_e.size)
            nj = clamp(j + action_deltas[a_nsp], grid_n.size)
            out = state_outcome(ni, nj)
            reward_esp = float(out.esp_revenue)
            reward_nsp = float(out.nsp_revenue)
            next_a_esp, next_a_nsp, _ = _joint_action_q_greedy_action(q_esp[ni, nj], q_nsp[ni, nj])
            td_target_esp = reward_esp + base_cfg.drl_gamma * float(q_esp[ni, nj, next_a_esp, next_a_nsp])
            td_target_nsp = reward_nsp + base_cfg.drl_gamma * float(q_nsp[ni, nj, next_a_esp, next_a_nsp])
            q_esp[i, j, a_esp, a_nsp] += base_cfg.drl_alpha * (td_target_esp - q_esp[i, j, a_esp, a_nsp])
            q_nsp[i, j, a_esp, a_nsp] += base_cfg.drl_alpha * (td_target_nsp - q_nsp[i, j, a_esp, a_nsp])
            i, j = ni, nj

    start_i = int(np.argmin(np.abs(grid_e - max(system.cE, float(stack_cfg.initial_pE)))))
    start_j = int(np.argmin(np.abs(grid_n - max(system.cN, float(stack_cfg.initial_pN)))))
    i, j = start_i, start_j
    rollout_steps = 0
    pure_nash_steps = 0
    visited = {(i, j)}
    rollout_limit = max(1, int(base_cfg.drl_steps_per_episode))
    for _ in range(rollout_limit):
        a_esp, a_nsp, has_pure_nash = _joint_action_q_greedy_action(q_esp[i, j], q_nsp[i, j])
        pure_nash_steps += int(has_pure_nash)
        ni = clamp(i + action_deltas[a_esp], grid_e.size)
        nj = clamp(j + action_deltas[a_nsp], grid_n.size)
        if ni == i and nj == j:
            break
        i, j = ni, nj
        rollout_steps += 1
        if (i, j) in visited:
            break
        visited.add((i, j))

    out = state_outcome(i, j)
    return BaselineOutcome(
        name="DRL",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=out.epsilon_proxy,
        meta={
            "episodes": base_cfg.drl_episodes,
            "policy": "joint_action_q",
            "rewards": "esp_revenue,nsp_revenue",
            "rollout_steps": rollout_steps,
            "pure_nash_steps": pure_nash_steps,
            "stage2_unique_prices": len(stage2_cache),
        },
    )


def baseline_market_equilibrium(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Paper-facing ME baseline using tatonnement-style price adjustment."""
    pE = max(system.cE, stack_cfg.initial_pE)
    pN = max(system.cN, stack_cfg.initial_pN)
    data = _build_data(users)

    converged = False
    actual_iters = base_cfg.market_max_iters

    for t in range(base_cfg.market_max_iters):
        old_pE, old_pN = pE, pN
        out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
        f, b = _set_allocations(data, out.offloading_set, pE, pN, system)
        dF = float(np.sum(f)) / system.F - 1.0
        dB = float(np.sum(b)) / system.B - 1.0
        pE = max(system.cE, pE * math.exp(base_cfg.market_step_size * dF))
        pN = max(system.cN, pN * math.exp(base_cfg.market_step_size * dB))

        # Convergence check: price change below tolerance
        price_change = abs(pE - old_pE) + abs(pN - old_pN)
        if price_change < base_cfg.market_tol:
            actual_iters = t + 1
            converged = True
            break

    out = _stage2_solver(base_cfg.stage2_solver_for_pricing, users, pE, pN, system, stack_cfg, base_cfg)
    return BaselineOutcome(
        name="MarketEquilibrium",
        price=out.price,
        offloading_set=out.offloading_set,
        social_cost=out.social_cost,
        esp_revenue=out.esp_revenue,
        nsp_revenue=out.nsp_revenue,
        epsilon_proxy=out.epsilon_proxy,
        meta={"iters": actual_iters, "converged": converged},
    )


def baseline_single_sp(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """
    Paper-facing SingleSP baseline with personalized incentive-compatible pricing.

    Mechanism (from Tutuncuoglu et al. 2024):
    1. SP selects a subset of users to serve
    2. For each selected user i, sets price_i = cl_i - ce_i
    3. User i is indifferent: offload_cost + price = local_cost
    4. SP extracts all user surplus as utility

    SP Utility = 危_{i鈭圶} (cl_i - ce_i) - operational_cost
    """
    data = _build_data(users)
    off: set[int] = set()

    def sp_utility(off_set: tuple[int, ...]) -> float:
        """Calculate SP utility with personalized pricing."""
        if not off_set:
            return 0.0
        pE = system.cE
        pN = system.cN
        f, b = _set_allocations(data, off_set, pE, pN, system)
        # Check resource constraints
        if np.sum(f) > system.F + 1e-9 or np.sum(b) > system.B + 1e-9:
            return -1e18
        ce = _offload_costs(data, f, b, pE, pN)
        idx = np.asarray(off_set, dtype=int)
        # Check incentive constraint: ce[i] <= cl[i] for all i
        if np.any(ce[idx] >= data.cl[idx]):
            return -1e18
        # SP utility = sum of personalized prices - operational cost
        # price_i = cl_i - ce_i (extracts all user surplus)
        user_payments = float(np.sum(data.cl[idx] - ce[idx]))
        op_cost = float(system.cE * np.sum(f[idx]) + system.cN * np.sum(b[idx]))
        return user_payments - op_cost

    # Greedy user selection: add users one by one that increase SP utility
    current_utility = sp_utility(tuple())
    for _ in range(base_cfg.single_sp_max_iters):
        best_gain = 0.0
        best_user: int | None = None
        for j in range(users.n):
            if j in off:
                continue
            cand = _sorted_tuple(list(off | {j}))
            gain = sp_utility(cand) - current_utility
            if gain > best_gain:
                best_gain = gain
                best_user = j
        if best_user is None or best_gain <= 0:
            break
        off.add(best_user)
        current_utility = sp_utility(_sorted_tuple(list(off)))

    # Calculate final outcome with incentive-compatible pricing
    offloading_set = _sorted_tuple(list(off))
    f, b = _set_allocations(data, offloading_set, system.cE, system.cN, system)
    ce = _offload_costs(data, f, b, system.cE, system.cN)

    idx = np.asarray(offloading_set, dtype=int) if offloading_set else np.asarray([], dtype=int)
    loc_idx = np.asarray([i for i in range(users.n) if i not in off], dtype=int)

    # User payments with personalized pricing: price_i = cl_i - ce_i
    esp_revenue = float(np.sum(data.cl[idx] - ce[idx])) if idx.size else 0.0
    # Operational cost
    op_cost = float(system.cE * np.sum(f[idx]) + system.cN * np.sum(b[idx])) if idx.size else 0.0
    # SP utility = revenue - cost
    sp_utility_final = esp_revenue - op_cost

    # Social cost calculation
    social_cost = float(np.sum(ce[idx])) + (float(np.sum(data.cl[loc_idx])) if loc_idx.size else 0.0)

    return BaselineOutcome(
        name="SingleSP",
        price=(float(system.cE), float(system.cN)),  # Base prices for allocation
        offloading_set=offloading_set,
        social_cost=social_cost,
        esp_revenue=sp_utility_final,  # SP utility (non-zero!)
        nsp_revenue=0.0,  # No separate NSP in SingleSP model
        epsilon_proxy=0.0,  # Centralized, no deviation incentive
        meta={"sp_utility": sp_utility_final, "num_offloaders": len(offloading_set)},
    )


def baseline_random_offloading(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> BaselineOutcome:
    """Paper-facing Rand baseline based on randomized source offloading sets."""
    rng = np.random.default_rng(base_cfg.random_seed + 707)
    best: BaselineOutcome | None = None
    for _ in range(base_cfg.random_offloading_trials):
        mask = rng.uniform(size=users.n) < base_cfg.random_offloading_prob
        off = tuple(np.nonzero(mask)[0].tolist())
        price = _refine_price_for_fixed_set(
            users,
            off,
            (max(system.cE, stack_cfg.initial_pE), max(system.cN, stack_cfg.initial_pN)),
            system,
        )
        out = _evaluate(
            users,
            off,
            price[0],
            price[1],
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
    gsse_cfg = replace(stack_cfg, stage1_solver_variant="vbbr_brd")
    res = run_stage1_solver(users, system, gsse_cfg)
    return _evaluate(
        users,
        res.offloading_set,
        res.price[0],
        res.price[1],
        system,
        gsse_cfg,
        "GSSE",
        meta={"search_steps": len(res.trajectory), "solver_variant": "vbbr_brd"},
    )


def run_all_baselines(
    users: UserBatch,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    base_cfg: BaselineConfig,
) -> list[BaselineOutcome]:
    """Internal mixed launcher spanning proposal, formal baselines, and auxiliary legacy methods."""
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
    if users.n <= base_cfg.exact_max_users:
        outcomes.append(baseline_stage1_epec_diagonalization(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_bo(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_bo_online_real_gap(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_ga(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_stage1_drl(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_market_equilibrium(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_single_sp(users, system, stack_cfg, base_cfg))
    outcomes.append(baseline_random_offloading(users, system, stack_cfg, base_cfg))
    return outcomes

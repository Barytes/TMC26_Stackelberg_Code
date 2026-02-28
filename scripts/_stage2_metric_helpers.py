#!/usr/bin/env python3
"""Shared helpers for standalone Stage-II metric scripts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tmc26_exp.baselines import (
    baseline_stage2_centralized_solver,
    baseline_stage2_penalty,
    baseline_stage2_ubrd,
    baseline_stage2_vi,
)
from tmc26_exp.config import ExperimentConfig
from tmc26_exp.model import local_cost, theta
from tmc26_exp.stackelberg import GreedySelectionResult, algorithm_2_heuristic_user_selection


METHODS = ["Algorithm2", "UBRD", "VI", "PEN", "CS"]
STYLE = {
    "Algorithm2": {"color": "#1b5e20", "marker": "o", "label": "Algorithm 2 (DG)"},
    "UBRD": {"color": "#b71c1c", "marker": "s", "label": "UBRD"},
    "VI": {"color": "#0d47a1", "marker": "^", "label": "VI (shared multiplier)"},
    "PEN": {"color": "#e65100", "marker": "v", "label": "Penalty BRD"},
    "CS": {"color": "#4a148c", "marker": "D", "label": "CS (solver)"},
}


@dataclass(frozen=True)
class Stage2MethodRun:
    name: str
    social_cost: float
    offloading_size: int
    runtime_sec: float
    result: object


def parse_user_counts(raw: str) -> list[int]:
    counts = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not counts:
        raise ValueError("No user counts provided.")
    if any(x <= 0 for x in counts):
        raise ValueError("All user counts must be positive.")
    return counts


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def run_stage2_methods(users, pE: float, pN: float, cfg: ExperimentConfig) -> dict[str, Stage2MethodRun]:
    runs: dict[str, Stage2MethodRun] = {}

    t0 = time.perf_counter()
    alg2 = algorithm_2_heuristic_user_selection(users, pE, pN, cfg.system, cfg.stackelberg)
    t1 = time.perf_counter()
    runs["Algorithm2"] = Stage2MethodRun(
        name="Algorithm2",
        social_cost=alg2.social_cost,
        offloading_size=len(alg2.offloading_set),
        runtime_sec=t1 - t0,
        result=alg2,
    )

    t0 = time.perf_counter()
    ubrd = baseline_stage2_ubrd(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
    t1 = time.perf_counter()
    runs["UBRD"] = Stage2MethodRun(
        name="UBRD",
        social_cost=ubrd.social_cost,
        offloading_size=len(ubrd.offloading_set),
        runtime_sec=t1 - t0,
        result=ubrd,
    )

    t0 = time.perf_counter()
    vi = baseline_stage2_vi(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
    t1 = time.perf_counter()
    runs["VI"] = Stage2MethodRun(
        name="VI",
        social_cost=vi.social_cost,
        offloading_size=len(vi.offloading_set),
        runtime_sec=t1 - t0,
        result=vi,
    )

    t0 = time.perf_counter()
    pen = baseline_stage2_penalty(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
    t1 = time.perf_counter()
    runs["PEN"] = Stage2MethodRun(
        name="PEN",
        social_cost=pen.social_cost,
        offloading_size=len(pen.offloading_set),
        runtime_sec=t1 - t0,
        result=pen,
    )

    if users.n <= cfg.baselines.exact_max_users:
        t0 = time.perf_counter()
        cs = baseline_stage2_centralized_solver(users, pE, pN, cfg.system, cfg.stackelberg, cfg.baselines)
        t1 = time.perf_counter()
        runs["CS"] = Stage2MethodRun(
            name="CS",
            social_cost=cs.social_cost,
            offloading_size=len(cs.offloading_set),
            runtime_sec=t1 - t0,
            result=cs,
        )

    return runs


def _bounded_1d_min(a: float, price: float, upper: float) -> float:
    if upper <= 0:
        return float("inf")
    a_eff = max(a, 1e-12)
    p_eff = max(price, 1e-12)
    x_star = math.sqrt(a_eff / p_eff)
    if x_star <= upper:
        return 2.0 * math.sqrt(a_eff * p_eff)
    return a_eff / upper + p_eff * upper


def compute_algorithm2_exploitability(users, alg2: GreedySelectionResult, pE: float, pN: float, cfg: ExperimentConfig) -> tuple[float, float]:
    current_f = alg2.inner_result.f
    current_b = alg2.inner_result.b
    cl = local_cost(users)
    aw = users.alpha * users.w
    th = theta(users)

    current_cost = cl.copy()
    off_idx = np.asarray(alg2.offloading_set, dtype=int) if alg2.offloading_set else np.asarray([], dtype=int)
    if off_idx.size:
        current_cost[off_idx] = aw[off_idx] / np.maximum(current_f[off_idx], 1e-12) + th[off_idx] / np.maximum(
            current_b[off_idx], 1e-12
        ) + pE * current_f[off_idx] + pN * current_b[off_idx]

    gains = np.zeros(users.n, dtype=float)
    total_f = float(np.sum(current_f))
    total_b = float(np.sum(current_b))
    for i in range(users.n):
        residual_f = max(cfg.system.F - (total_f - float(current_f[i])), 0.0)
        residual_b = max(cfg.system.B - (total_b - float(current_b[i])), 0.0)
        best_offload = _bounded_1d_min(float(aw[i]), pE, residual_f) + _bounded_1d_min(float(th[i]), pN, residual_b)
        best_cost = min(float(cl[i]), best_offload)
        gains[i] = max(float(current_cost[i]) - best_cost, 0.0)

    return float(np.mean(gains)), float(np.max(gains))

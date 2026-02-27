from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .model import (
    UserBatch,
    local_cost,
    unconstrained_offload_cost,
    unconstrained_optimal_b,
    unconstrained_optimal_f,
)


MetricFn = Callable[[UserBatch, float, float], float]


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    description: str
    fn: MetricFn


METRICS: dict[str, Metric] = {}


def register_metric(metric: Metric) -> Metric:
    if metric.name in METRICS:
        raise ValueError(f"Duplicate metric name: {metric.name}")
    METRICS[metric.name] = metric
    return metric


def get_metric(name: str) -> Metric:
    if name not in METRICS:
        available = ", ".join(sorted(METRICS))
        raise KeyError(f"Unknown metric '{name}'. Available: {available}")
    return METRICS[name]


def _potential_offload_mask(users: UserBatch, pE: float, pN: float) -> np.ndarray:
    cl = local_cost(users)
    ce = unconstrained_offload_cost(users, pE, pN)
    return ce < cl


register_metric(
    Metric(
        name="potential_offload_ratio",
        label="Potential Offload Ratio",
        description="Fraction of users with unconstrained offloading cost < local cost.",
        fn=lambda users, pE, pN: float(np.mean(_potential_offload_mask(users, pE, pN))),
    )
)

register_metric(
    Metric(
        name="mean_cost_gap_local_minus_offload",
        label="Mean Cost Gap (Local - Offload)",
        description="Positive means offloading seems beneficial on average (proxy only).",
        fn=lambda users, pE, pN: float(
            np.mean(local_cost(users) - unconstrained_offload_cost(users, pE, pN))
        ),
    )
)

register_metric(
    Metric(
        name="esp_potential_revenue",
        label="ESP Potential Revenue Proxy",
        description="Proxy from users with potential offload under unconstrained resource demand.",
        fn=lambda users, pE, pN: _esp_potential_revenue(users, pE, pN),
    )
)

register_metric(
    Metric(
        name="nsp_potential_revenue",
        label="NSP Potential Revenue Proxy",
        description="Proxy from users with potential offload under unconstrained resource demand.",
        fn=lambda users, pE, pN: _nsp_potential_revenue(users, pE, pN),
    )
)

register_metric(
    Metric(
        name="mean_unconstrained_offload_cost",
        label="Mean Unconstrained Offload Cost",
        description="Average unconstrained offloading cost (no coupled capacity constraints).",
        fn=lambda users, pE, pN: float(np.mean(unconstrained_offload_cost(users, pE, pN))),
    )
)


def _esp_potential_revenue(users: UserBatch, pE: float, pN: float) -> float:
    mask = _potential_offload_mask(users, pE, pN)
    f = unconstrained_optimal_f(users, pE)
    return float(pE * np.sum(f[mask]))


def _nsp_potential_revenue(users: UserBatch, pE: float, pN: float) -> float:
    mask = _potential_offload_mask(users, pE, pN)
    b = unconstrained_optimal_b(users, pN)
    return float(pN * np.sum(b[mask]))

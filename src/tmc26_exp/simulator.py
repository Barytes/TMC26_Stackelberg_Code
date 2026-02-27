from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .distributions import sample_distribution
from .metrics import Metric
from .model import UserBatch


@dataclass(frozen=True)
class MetricSurface:
    metric_name: str
    metric_label: str
    pE_values: np.ndarray
    pN_values: np.ndarray
    mean_values: np.ndarray
    std_values: np.ndarray


def sample_users(config: ExperimentConfig, rng: np.random.Generator) -> UserBatch:
    n = config.n_users
    u = config.users
    return UserBatch(
        w=sample_distribution(u.w, n, rng),
        d=sample_distribution(u.d, n, rng),
        fl=sample_distribution(u.fl, n, rng),
        alpha=sample_distribution(u.alpha, n, rng),
        beta=sample_distribution(u.beta, n, rng),
        rho=sample_distribution(u.rho, n, rng),
        varpi=sample_distribution(u.varpi, n, rng),
        kappa=sample_distribution(u.kappa, n, rng),
        sigma=sample_distribution(u.sigma, n, rng),
    )


def evaluate_metric_surface(config: ExperimentConfig, metric: Metric) -> MetricSurface:
    rng = np.random.default_rng(config.seed)
    pE_values = np.linspace(config.pE.min, config.pE.max, config.pE.points)
    pN_values = np.linspace(config.pN.min, config.pN.max, config.pN.points)

    values_trials = np.zeros((config.n_trials, pN_values.size, pE_values.size), dtype=float)

    for t in range(config.n_trials):
        users = sample_users(config, rng)
        for n_idx, pN in enumerate(pN_values):
            for e_idx, pE in enumerate(pE_values):
                values_trials[t, n_idx, e_idx] = metric.fn(users, float(pE), float(pN))

    return MetricSurface(
        metric_name=metric.name,
        metric_label=metric.label,
        pE_values=pE_values,
        pN_values=pN_values,
        mean_values=np.mean(values_trials, axis=0),
        std_values=np.std(values_trials, axis=0),
    )

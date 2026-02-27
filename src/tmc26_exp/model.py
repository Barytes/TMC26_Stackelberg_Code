from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UserBatch:
    w: np.ndarray
    d: np.ndarray
    fl: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    rho: np.ndarray
    varpi: np.ndarray
    kappa: np.ndarray
    sigma: np.ndarray

    @property
    def n(self) -> int:
        return int(self.w.size)


def local_cost(users: UserBatch) -> np.ndarray:
    t_local = users.w / users.fl
    e_local = users.kappa * users.w * (users.fl ** 2)
    return users.alpha * t_local + users.beta * e_local


def theta(users: UserBatch) -> np.ndarray:
    return users.d * (users.alpha + users.beta * users.rho * users.varpi) / users.sigma


def unconstrained_optimal_f(users: UserBatch, pE: float) -> np.ndarray:
    return np.sqrt(users.alpha * users.w / pE)


def unconstrained_optimal_b(users: UserBatch, pN: float) -> np.ndarray:
    return np.sqrt(theta(users) / pN)


def unconstrained_offload_cost(users: UserBatch, pE: float, pN: float) -> np.ndarray:
    # Closed form from minimizing alpha*w/f + pE*f and theta/b + pN*b independently.
    return 2.0 * np.sqrt(users.alpha * users.w * pE) + 2.0 * np.sqrt(theta(users) * pN)

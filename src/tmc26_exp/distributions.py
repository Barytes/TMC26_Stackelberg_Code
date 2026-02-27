from __future__ import annotations

from typing import Any

import numpy as np

from .config import DistributionSpec


def sample_distribution(spec: DistributionSpec, size: int, rng: np.random.Generator) -> np.ndarray:
    kind = spec.kind
    p = spec.params

    if kind == "constant":
        value = float(p["value"])
        return np.full(size, value, dtype=float)

    if kind == "uniform":
        return rng.uniform(float(p["low"]), float(p["high"]), size=size)

    if kind == "normal":
        arr = rng.normal(float(p["mean"]), float(p["std"]), size=size)
        return _apply_clip(arr, p)

    if kind == "lognormal":
        arr = rng.lognormal(float(p["mean"]), float(p["sigma"]), size=size)
        return _apply_clip(arr, p)

    if kind == "choice":
        values = np.asarray(p["values"], dtype=float)
        probs = p.get("probs")
        prob_arr = None if probs is None else np.asarray(probs, dtype=float)
        return rng.choice(values, size=size, p=prob_arr).astype(float)

    if kind == "int_uniform":
        low = int(p["low"])
        high = int(p["high"])
        return rng.integers(low, high + 1, size=size).astype(float)

    raise ValueError(f"Unsupported distribution kind: {kind}")


def _apply_clip(arr: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    low = params.get("clip_min")
    high = params.get("clip_max")
    if low is not None:
        arr = np.maximum(arr, float(low))
    if high is not None:
        arr = np.minimum(arr, float(high))
    return arr

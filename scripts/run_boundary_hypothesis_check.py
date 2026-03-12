from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tmc26_exp.baselines import evaluate_stage1_price_grid
from tmc26_exp.config import StackelbergConfig, SystemConfig, load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import (
    _bar_prices,
    _boundary_price_for_provider,
    _build_data,
    _candidate_family,
    _heuristic_score_with_t,
    _provider_revenue_from_stage2_result,
    _solve_fixed_set_inner_exact,
    algorithm_1_distributed_primal_dual,
    algorithm_2_heuristic_user_selection,
)


Provider = str
_EPS = 1e-12


@dataclass(frozen=True)
class SliceEval:
    provider: Provider
    price: float
    fixed_price: float
    pE: float
    pN: float
    offloading_set: tuple[int, ...]
    esp_revenue: float
    nsp_revenue: float

    @property
    def provider_revenue(self) -> float:
        return self.esp_revenue if self.provider == "E" else self.nsp_revenue


@dataclass(frozen=True)
class BoundaryEvent:
    provider: Provider
    fixed_price: float
    boundary_price: float
    left_price: float
    right_price: float
    left_set: tuple[int, ...]
    right_set: tuple[int, ...]
    dropped: tuple[int, ...]
    added: tuple[int, ...]
    event_type: str


@dataclass(frozen=True)
class DGChainStep:
    order_idx: int
    prefix_before: tuple[int, ...]
    prefix_after: tuple[int, ...]
    user: int
    score_at_accept: float
    delta_true: float


@dataclass(frozen=True)
class PredictionEvent:
    provider: Provider
    current_price: float
    fixed_price: float
    predicted_price: float
    source_user: int
    order_idx: int
    prefix_before: tuple[int, ...]
    prefix_after: tuple[int, ...]
    delta_at_current: float
    delta_at_predicted: float
    current_set: tuple[int, ...]
    realized_next_set: tuple[int, ...]
    dropped: tuple[int, ...]
    added: tuple[int, ...]
    realized_event_type: str


@dataclass(frozen=True)
class ProviderStats:
    provider: Provider
    fixed_price: float
    exact_count: int
    old_raw_count: int
    old_unique_count: int
    hypothesis_count: int
    old_coverage_count: int
    old_coverage_ratio: float
    old_nearest_mean_abs_error: float
    old_nearest_max_abs_error: float
    ordered_match_count: int
    hypothesis_mean_abs_error: float
    hypothesis_max_abs_error: float


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0.")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0.")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be >= 1.")
    return value


def _min_two_int(raw: str) -> int:
    value = int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError("Value must be >= 2.")
    return value


def _serialize_indices(items: tuple[int, ...] | list[int] | set[int]) -> str:
    seq = sorted(int(x) for x in items)
    return ";".join(str(x) for x in seq)


def _event_type_from_sets(left_set: tuple[int, ...], right_set: tuple[int, ...]) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    dropped = tuple(sorted(set(left_set) - set(right_set)))
    added = tuple(sorted(set(right_set) - set(left_set)))
    if dropped and added:
        return "swap", dropped, added
    if dropped:
        return "drop", dropped, added
    if added:
        return "add", dropped, added
    return "none", dropped, added


def _price_pair(provider: Provider, axis_price: float, fixed_price: float) -> tuple[float, float]:
    if provider == "E":
        return float(axis_price), float(fixed_price)
    return float(fixed_price), float(axis_price)


def _axis_cost_floor(provider: Provider, system: SystemConfig) -> float:
    return float(system.cE if provider == "E" else system.cN)


def _load_summary_kv(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _load_grid_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    required = {"pE", "pN", "esp_revenue", "nsp_revenue"}
    if not required.issubset(rows[0].keys()):
        raise ValueError(f"CSV missing required columns: {sorted(required)}")

    pE_vals = sorted({float(r["pE"]) for r in rows})
    pN_vals = sorted({float(r["pN"]) for r in rows})
    pE_grid = np.asarray(pE_vals, dtype=float)
    pN_grid = np.asarray(pN_vals, dtype=float)
    esp = np.full((pN_grid.size, pE_grid.size), np.nan, dtype=float)
    nsp = np.full_like(esp, np.nan)
    e_map = {v: i for i, v in enumerate(pE_vals)}
    n_map = {v: j for j, v in enumerate(pN_vals)}
    for row in rows:
        i = e_map[float(row["pE"])]
        j = n_map[float(row["pN"])]
        esp[j, i] = float(row["esp_revenue"])
        nsp[j, i] = float(row["nsp_revenue"])
    if np.any(~np.isfinite(esp)) or np.any(~np.isfinite(nsp)):
        raise ValueError("Heatmap CSV grid is incomplete.")
    return pE_grid, pN_grid, esp, nsp


def _evaluate_slice_point(
    provider: Provider,
    axis_price: float,
    fixed_price: float,
    users,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
) -> SliceEval:
    pE, pN = _price_pair(provider, axis_price, fixed_price)
    s2 = algorithm_2_heuristic_user_selection(users, pE, pN, system, stack_cfg)
    esp_rev = float(_provider_revenue_from_stage2_result(s2, pE, pN, "E", system))
    nsp_rev = float(_provider_revenue_from_stage2_result(s2, pE, pN, "N", system))
    return SliceEval(
        provider=provider,
        price=float(axis_price),
        fixed_price=float(fixed_price),
        pE=float(pE),
        pN=float(pN),
        offloading_set=tuple(int(i) for i in s2.offloading_set),
        esp_revenue=esp_rev,
        nsp_revenue=nsp_rev,
    )


def _refine_boundary(
    provider: Provider,
    fixed_price: float,
    left_eval: SliceEval,
    right_eval: SliceEval,
    users,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    root_tol: float,
) -> BoundaryEvent:
    low = float(left_eval.price)
    high = float(right_eval.price)
    left_set = left_eval.offloading_set
    right_state = right_eval
    for _ in range(80):
        if high - low <= root_tol:
            break
        mid = 0.5 * (low + high)
        mid_eval = _evaluate_slice_point(provider, mid, fixed_price, users, system, stack_cfg)
        if mid_eval.offloading_set == left_set:
            low = mid
        else:
            high = mid
            right_state = mid_eval
    event_type, dropped, added = _event_type_from_sets(left_set, right_state.offloading_set)
    return BoundaryEvent(
        provider=provider,
        fixed_price=float(fixed_price),
        boundary_price=float(high),
        left_price=float(left_eval.price),
        right_price=float(right_eval.price),
        left_set=left_set,
        right_set=right_state.offloading_set,
        dropped=dropped,
        added=added,
        event_type=event_type,
    )


def _scan_exact_boundaries(
    provider: Provider,
    fixed_price: float,
    price_max: float,
    scan_points: int,
    users,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    root_tol: float,
) -> tuple[np.ndarray, list[SliceEval], list[BoundaryEvent]]:
    price_min = _axis_cost_floor(provider, system)
    grid = np.linspace(price_min, float(price_max), int(scan_points), dtype=float)
    evals = [
        _evaluate_slice_point(provider, float(p), fixed_price, users, system, stack_cfg)
        for p in grid
    ]
    events: list[BoundaryEvent] = []
    for left_eval, right_eval in zip(evals[:-1], evals[1:]):
        if left_eval.offloading_set == right_eval.offloading_set:
            continue
        events.append(
            _refine_boundary(
                provider=provider,
                fixed_price=fixed_price,
                left_eval=left_eval,
                right_eval=right_eval,
                users=users,
                system=system,
                stack_cfg=stack_cfg,
                root_tol=root_tol,
            )
        )
    return grid, evals, events


def _compute_old_boundary_points(
    data,
    current_set: tuple[int, ...],
    start_pE: float,
    start_pN: float,
    system: SystemConfig,
) -> tuple[list[dict[str, object]], dict[Provider, list[float]], dict[Provider, int]]:
    family = _candidate_family(data, current_set, start_pE, start_pN, system)
    rows: list[dict[str, object]] = []
    unique: dict[Provider, list[float]] = {"E": [], "N": []}
    raw_counts = {"E": 0, "N": 0}
    seen = {"E": set(), "N": set()}

    for provider in ("E", "N"):
        opponent_price = start_pN if provider == "E" else start_pE
        for candidate_set in family:
            if not candidate_set:
                continue
            boundary_price = _boundary_price_for_provider(data, candidate_set, opponent_price, provider, system)
            if boundary_price is None:
                continue
            raw_counts[provider] += 1
            pE, pN = _price_pair(provider, boundary_price, opponent_price)
            price_key = float(boundary_price)
            point_group = "unique"
            if price_key in seen[provider]:
                point_group = "raw_duplicate"
            else:
                seen[provider].add(price_key)
                unique[provider].append(price_key)
            rows.append(
                {
                    "provider": provider,
                    "point_group": point_group,
                    "boundary_price": float(boundary_price),
                    "pE": float(pE),
                    "pN": float(pN),
                    "candidate_set_size": len(candidate_set),
                    "candidate_set": _serialize_indices(candidate_set),
                }
            )
    unique["E"].sort()
    unique["N"].sort()
    return rows, unique, raw_counts


def _solve_inner_for_trace(users, offloading_set: set[int], pE: float, pN: float, system: SystemConfig, stack_cfg: StackelbergConfig):
    inner = _solve_fixed_set_inner_exact(users, offloading_set, pE, pN, system)
    if inner is None:
        inner = algorithm_1_distributed_primal_dual(users, offloading_set, pE, pN, system, stack_cfg)
    return inner


def _trace_dg_chain(
    users,
    data,
    pE: float,
    pN: float,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
) -> tuple[list[DGChainStep], tuple[int, ...]]:
    active_users: set[int] = set(range(users.n))
    offloading_set: set[int] = set()
    previous_ve = 0.0
    last_added: int | None = None
    steps: list[DGChainStep] = []

    for _ in range(stack_cfg.greedy_max_iters):
        inner = _solve_inner_for_trace(users, offloading_set, pE, pN, system, stack_cfg)
        ve = inner.offloading_objective
        if last_added is not None and steps:
            delta_true = float(ve - previous_ve - data.cl[last_added])
            if delta_true >= 0.0:
                offloading_set.discard(last_added)
                active_users.discard(last_added)
                steps.pop()
                inner = _solve_inner_for_trace(users, offloading_set, pE, pN, system, stack_cfg)
                ve = inner.offloading_objective
            else:
                last_step = steps[-1]
                steps[-1] = DGChainStep(
                    order_idx=last_step.order_idx,
                    prefix_before=last_step.prefix_before,
                    prefix_after=last_step.prefix_after,
                    user=last_step.user,
                    score_at_accept=last_step.score_at_accept,
                    delta_true=delta_true,
                )

        candidates = sorted(active_users - offloading_set)
        if not candidates:
            break

        lambda_F, lambda_B = (inner.lambda_F, inner.lambda_B) if offloading_set else (0.0, 0.0)
        best_user = min(
            candidates,
            key=lambda j: _heuristic_score_with_t(data, j, pE + lambda_F, pN + lambda_B, system),
        )
        best_score = float(_heuristic_score_with_t(data, best_user, pE + lambda_F, pN + lambda_B, system))
        if best_score >= 0.0:
            break

        prefix_before = tuple(sorted(offloading_set))
        offloading_set.add(best_user)
        prefix_after = tuple(sorted(offloading_set))
        previous_ve = ve
        last_added = best_user
        steps.append(
            DGChainStep(
                order_idx=len(steps),
                prefix_before=prefix_before,
                prefix_after=prefix_after,
                user=best_user,
                score_at_accept=best_score,
                delta_true=float("nan"),
            )
        )

    return steps, tuple(sorted(offloading_set))


def _phi_value(data, offloading_set: tuple[int, ...], pE: float, pN: float, system: SystemConfig) -> float:
    if not offloading_set:
        return 0.0
    idx = np.asarray(offloading_set, dtype=int)
    sE = float(np.sum(np.sqrt(data.aw[idx])))
    sN = float(np.sum(np.sqrt(data.th[idx])))
    bar_pE, bar_pN = _bar_prices(data, offloading_set, system)
    tE = max(float(pE), float(bar_pE))
    tN = max(float(pN), float(bar_pN))
    return float(2.0 * sE * math.sqrt(max(tE, _EPS)) + 2.0 * sN * math.sqrt(max(tN, _EPS)))


def _delta_v_for_step(
    provider: Provider,
    axis_price: float,
    fixed_price: float,
    step: DGChainStep,
    data,
    system: SystemConfig,
) -> float:
    pE, pN = _price_pair(provider, axis_price, fixed_price)
    return float(
        _phi_value(data, step.prefix_after, pE, pN, system)
        - _phi_value(data, step.prefix_before, pE, pN, system)
        - float(data.cl[step.user])
    )


def _slack_regime_root_for_step(
    provider: Provider,
    fixed_price: float,
    current_price: float,
    step: DGChainStep,
    data,
    system: SystemConfig,
) -> float | None:
    user = int(step.user)
    if provider == "E":
        coeff = math.sqrt(float(data.aw[user]))
        if coeff <= 0.0:
            return None
        idx_before = np.asarray(step.prefix_before, dtype=int)
        idx_after = np.asarray(step.prefix_after, dtype=int)
        s_other_before = float(np.sum(np.sqrt(data.th[idx_before]))) if idx_before.size else 0.0
        s_other_after = float(np.sum(np.sqrt(data.th[idx_after]))) if idx_after.size else 0.0
        _, bar_n_before = _bar_prices(data, step.prefix_before, system)
        _, bar_n_after = _bar_prices(data, step.prefix_after, system)
        t_other_before = max(float(fixed_price), float(bar_n_before))
        t_other_after = max(float(fixed_price), float(bar_n_after))
        numerator = float(data.cl[user]) - 2.0 * (
            s_other_after * math.sqrt(max(t_other_after, _EPS))
            - s_other_before * math.sqrt(max(t_other_before, _EPS))
        )
        bar_axis_before, _ = _bar_prices(data, step.prefix_before, system)
        bar_axis_after, _ = _bar_prices(data, step.prefix_after, system)
        axis_floor = max(float(current_price), float(system.cE), float(bar_axis_before), float(bar_axis_after))
    else:
        coeff = math.sqrt(float(data.th[user]))
        if coeff <= 0.0:
            return None
        idx_before = np.asarray(step.prefix_before, dtype=int)
        idx_after = np.asarray(step.prefix_after, dtype=int)
        s_other_before = float(np.sum(np.sqrt(data.aw[idx_before]))) if idx_before.size else 0.0
        s_other_after = float(np.sum(np.sqrt(data.aw[idx_after]))) if idx_after.size else 0.0
        bar_e_before, _ = _bar_prices(data, step.prefix_before, system)
        bar_e_after, _ = _bar_prices(data, step.prefix_after, system)
        t_other_before = max(float(fixed_price), float(bar_e_before))
        t_other_after = max(float(fixed_price), float(bar_e_after))
        numerator = float(data.cl[user]) - 2.0 * (
            s_other_after * math.sqrt(max(t_other_after, _EPS))
            - s_other_before * math.sqrt(max(t_other_before, _EPS))
        )
        _, bar_axis_before = _bar_prices(data, step.prefix_before, system)
        _, bar_axis_after = _bar_prices(data, step.prefix_after, system)
        axis_floor = max(float(current_price), float(system.cN), float(bar_axis_before), float(bar_axis_after))

    if numerator <= 0.0:
        return None
    root = float((numerator / (2.0 * coeff)) ** 2)
    if not math.isfinite(root):
        return None
    if root <= axis_floor + 1e-9:
        return None
    return root


def _build_hypothesis_chain(
    provider: Provider,
    start_pE: float,
    start_pN: float,
    price_max: float,
    users,
    data,
    system: SystemConfig,
    stack_cfg: StackelbergConfig,
    root_tol: float,
) -> list[PredictionEvent]:
    current_eval = _evaluate_slice_point(
        provider,
        start_pE if provider == "E" else start_pN,
        start_pN if provider == "E" else start_pE,
        users,
        system,
        stack_cfg,
    )
    current_price = float(current_eval.price)
    fixed_price = float(current_eval.fixed_price)
    probe = max(1e-6, 8.0 * root_tol)
    predictions: list[PredictionEvent] = []

    max_steps = 4 * users.n + 8
    for _ in range(max_steps):
        chain, chain_final = _trace_dg_chain(users, data, current_eval.pE, current_eval.pN, system, stack_cfg)
        if not chain:
            break

        candidates: list[tuple[float, DGChainStep, float, float]] = []
        for step in chain:
            candidate_price = _slack_regime_root_for_step(provider, fixed_price, current_price, step, data, system)
            if candidate_price is None or candidate_price > price_max:
                continue
            delta_current = _delta_v_for_step(provider, current_price, fixed_price, step, data, system)
            delta_pred = _delta_v_for_step(provider, candidate_price, fixed_price, step, data, system)
            candidates.append((candidate_price, step, delta_current, delta_pred))

        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        predicted_price, step, delta_current, delta_pred = candidates[0]
        probe_price = min(float(price_max), float(predicted_price + probe))
        if probe_price <= current_price + 0.5 * probe:
            break
        next_eval = _evaluate_slice_point(provider, probe_price, fixed_price, users, system, stack_cfg)
        realized_event_type, dropped, added = _event_type_from_sets(chain_final, next_eval.offloading_set)
        predictions.append(
            PredictionEvent(
                provider=provider,
                current_price=float(current_price),
                fixed_price=float(fixed_price),
                predicted_price=float(predicted_price),
                source_user=int(step.user),
                order_idx=int(step.order_idx),
                prefix_before=step.prefix_before,
                prefix_after=step.prefix_after,
                delta_at_current=float(delta_current),
                delta_at_predicted=float(delta_pred),
                current_set=chain_final,
                realized_next_set=next_eval.offloading_set,
                dropped=dropped,
                added=added,
                realized_event_type=realized_event_type,
            )
        )
        current_eval = next_eval
        current_price = float(current_eval.price)
        if current_price >= price_max - probe:
            break

    return predictions


def _ordered_matching_errors(exact_prices: list[float], predicted_prices: list[float]) -> list[float]:
    match_count = min(len(exact_prices), len(predicted_prices))
    return [abs(float(exact_prices[i]) - float(predicted_prices[i])) for i in range(match_count)]


def _cloud_coverage_errors(exact_prices: list[float], cloud_prices: list[float]) -> list[float]:
    if not exact_prices or not cloud_prices:
        return []
    cloud = np.asarray(cloud_prices, dtype=float)
    return [float(np.min(np.abs(cloud - exact_price))) for exact_price in exact_prices]


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_max(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=float)))


def _build_provider_stats(
    provider: Provider,
    fixed_price: float,
    exact_events: list[BoundaryEvent],
    old_raw_count: int,
    old_unique_prices: list[float],
    hypothesis_events: list[PredictionEvent],
    match_tol: float,
) -> ProviderStats:
    exact_prices = [float(event.boundary_price) for event in exact_events]
    hypothesis_prices = [float(event.predicted_price) for event in hypothesis_events]
    old_nearest = _cloud_coverage_errors(exact_prices, old_unique_prices)
    ordered_errors = _ordered_matching_errors(exact_prices, hypothesis_prices)
    old_coverage_count = int(sum(err <= match_tol for err in old_nearest))
    old_coverage_ratio = float(old_coverage_count / len(exact_prices)) if exact_prices else float("nan")
    return ProviderStats(
        provider=provider,
        fixed_price=float(fixed_price),
        exact_count=len(exact_prices),
        old_raw_count=int(old_raw_count),
        old_unique_count=len(old_unique_prices),
        hypothesis_count=len(hypothesis_prices),
        old_coverage_count=old_coverage_count,
        old_coverage_ratio=old_coverage_ratio,
        old_nearest_mean_abs_error=_safe_mean(old_nearest),
        old_nearest_max_abs_error=_safe_max(old_nearest),
        ordered_match_count=min(len(exact_prices), len(hypothesis_prices)),
        hypothesis_mean_abs_error=_safe_mean(ordered_errors),
        hypothesis_max_abs_error=_safe_max(ordered_errors),
    )


def _unique_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8, frameon=True)


def _plot_slice_comparison(
    provider: Provider,
    evals: list[SliceEval],
    exact_events: list[BoundaryEvent],
    old_unique_prices: list[float],
    hypothesis_events: list[PredictionEvent],
    start_price: float,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prices = np.asarray([float(item.price) for item in evals], dtype=float)
    revenues = np.asarray([float(item.provider_revenue) for item in evals], dtype=float)
    if prices.size == 0:
        raise ValueError("Slice evaluation is empty.")

    y_min = float(np.min(revenues))
    y_max = float(np.max(revenues))
    y_span = max(y_max - y_min, 1.0)
    rug_y_old = y_min + 0.04 * y_span
    rug_y_hyp = y_min + 0.10 * y_span
    rug_y_exact = y_min + 0.16 * y_span

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.plot(prices, revenues, color="black", linewidth=1.4, label=f"{provider} revenue")

    for idx, event in enumerate(exact_events):
        ax.axvline(
            event.boundary_price,
            color="red",
            linewidth=0.9,
            alpha=0.65,
            linestyle="-",
            label="exact DG boundary" if idx == 0 else None,
        )

    old_arr = np.asarray(old_unique_prices, dtype=float)
    if old_arr.size > 0:
        ax.scatter(
            old_arr,
            np.full(old_arr.shape, rug_y_old, dtype=float),
            s=18,
            marker="o",
            facecolors="none",
            edgecolors="deepskyblue",
            linewidths=0.8,
            alpha=0.85,
            label="old boundary points",
        )

    hyp_arr = np.asarray([float(event.predicted_price) for event in hypothesis_events], dtype=float)
    if hyp_arr.size > 0:
        ax.scatter(
            hyp_arr,
            np.full(hyp_arr.shape, rug_y_hyp, dtype=float),
            s=28,
            marker="D",
            c="limegreen",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
            label="hypothesis points",
        )

    exact_arr = np.asarray([float(event.boundary_price) for event in exact_events], dtype=float)
    if exact_arr.size > 0:
        ax.scatter(
            exact_arr,
            np.full(exact_arr.shape, rug_y_exact, dtype=float),
            s=24,
            marker="x",
            c="red",
            alpha=0.9,
            label="exact DG markers",
        )

    ax.scatter(
        [float(start_price)],
        [float(np.interp(start_price, prices, revenues))],
        s=90,
        marker="*",
        c="gold",
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
        label="start",
    )

    ax.set_xlabel("pE" if provider == "E" else "pN")
    ax.set_ylabel(f"{provider} revenue")
    axis_fixed = "pN" if provider == "E" else "pE"
    fixed_val = evals[0].fixed_price
    ax.set_title(f"{provider} revenue slice with boundary comparisons ({axis_fixed}={fixed_val:.4g})")
    _unique_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _scatter_points_in_extent(
    ax,
    x: list[float],
    y: list[float],
    extent: tuple[float, float, float, float],
    *,
    label: str,
    marker: str,
    edgecolors: str,
    facecolors: str | None = None,
    color: str | None = None,
    size: float = 28.0,
    alpha: float = 0.9,
) -> int:
    x0, x1, y0, y1 = extent
    kept = [
        (float(px), float(py))
        for px, py in zip(x, y)
        if x0 - 1e-9 <= float(px) <= x1 + 1e-9 and y0 - 1e-9 <= float(py) <= y1 + 1e-9
    ]
    if not kept:
        return 0
    xs = [px for px, _ in kept]
    ys = [py for _, py in kept]
    scatter_kwargs = {"s": size, "marker": marker, "alpha": alpha, "label": label}
    unfilled_markers = {"x", "+", "1", "2", "3", "4", "|", "_"}
    if marker in unfilled_markers:
        scatter_kwargs["c"] = color if color is not None else edgecolors
    else:
        if color is not None:
            scatter_kwargs["c"] = color
        if facecolors is not None:
            scatter_kwargs["facecolors"] = facecolors
        scatter_kwargs["edgecolors"] = edgecolors
    ax.scatter(xs, ys, **scatter_kwargs)
    return len(kept)


def _plot_revenue_overlay(
    values: np.ndarray,
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    start_pE: float,
    start_pN: float,
    old_esp: list[float],
    exact_esp: list[float],
    hyp_esp: list[float],
    old_nsp: list[float],
    exact_nsp: list[float],
    hyp_nsp: list[float],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    extent = (float(pE_grid[0]), float(pE_grid[-1]), float(pN_grid[0]), float(pN_grid[-1]))
    im = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.axhline(start_pN, color="white", linewidth=0.8, linestyle="--", alpha=0.75, label="ESP slice")
    ax.axvline(start_pE, color="black", linewidth=0.8, linestyle="--", alpha=0.75, label="NSP slice")

    _scatter_points_in_extent(
        ax,
        old_esp,
        [start_pN] * len(old_esp),
        extent,
        label="old ESP boundary points",
        marker="o",
        edgecolors="deepskyblue",
        facecolors="none",
        size=22.0,
    )
    _scatter_points_in_extent(
        ax,
        exact_esp,
        [start_pN] * len(exact_esp),
        extent,
        label="exact ESP boundaries",
        marker="x",
        edgecolors="red",
        color="red",
        size=28.0,
    )
    _scatter_points_in_extent(
        ax,
        hyp_esp,
        [start_pN] * len(hyp_esp),
        extent,
        label="hypothesis ESP points",
        marker="D",
        edgecolors="black",
        color="limegreen",
        size=28.0,
    )

    _scatter_points_in_extent(
        ax,
        [start_pE] * len(old_nsp),
        old_nsp,
        extent,
        label="old NSP boundary points",
        marker="s",
        edgecolors="orange",
        facecolors="none",
        size=20.0,
    )
    _scatter_points_in_extent(
        ax,
        [start_pE] * len(exact_nsp),
        exact_nsp,
        extent,
        label="exact NSP boundaries",
        marker="+",
        edgecolors="magenta",
        color="magenta",
        size=34.0,
    )
    _scatter_points_in_extent(
        ax,
        [start_pE] * len(hyp_nsp),
        hyp_nsp,
        extent,
        label="hypothesis NSP points",
        marker="^",
        edgecolors="black",
        color="yellow",
        size=30.0,
    )

    ax.scatter(
        [start_pE],
        [start_pN],
        s=120,
        marker="*",
        c="gold",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
        label="start",
    )
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    ax.set_title(title)
    _unique_legend(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_csv_rows(out_path: Path, header: list[str], rows: list[list[object]]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_exact_csv(out_path: Path, events: list[BoundaryEvent]) -> None:
    rows: list[list[object]] = []
    for event in events:
        pE, pN = _price_pair(event.provider, event.boundary_price, event.fixed_price)
        rows.append(
            [
                event.provider,
                f"{event.fixed_price:.12g}",
                f"{event.boundary_price:.12g}",
                f"{pE:.12g}",
                f"{pN:.12g}",
                event.event_type,
                len(event.left_set),
                len(event.right_set),
                _serialize_indices(event.left_set),
                _serialize_indices(event.right_set),
                _serialize_indices(event.dropped),
                _serialize_indices(event.added),
            ]
        )
    _write_csv_rows(
        out_path,
        [
            "provider",
            "fixed_price",
            "boundary_price",
            "pE",
            "pN",
            "event_type",
            "left_size",
            "right_size",
            "left_set",
            "right_set",
            "dropped",
            "added",
        ],
        rows,
    )


def _write_old_csv(out_path: Path, rows_in: list[dict[str, object]]) -> None:
    rows: list[list[object]] = []
    for row in rows_in:
        rows.append(
            [
                row["provider"],
                row["point_group"],
                f"{float(row['boundary_price']):.12g}",
                f"{float(row['pE']):.12g}",
                f"{float(row['pN']):.12g}",
                row["candidate_set_size"],
                row["candidate_set"],
            ]
        )
    _write_csv_rows(
        out_path,
        ["provider", "point_group", "boundary_price", "pE", "pN", "candidate_set_size", "candidate_set"],
        rows,
    )


def _write_hypothesis_csv(out_path: Path, events: list[PredictionEvent]) -> None:
    rows: list[list[object]] = []
    for event in events:
        pE, pN = _price_pair(event.provider, event.predicted_price, event.fixed_price)
        rows.append(
            [
                event.provider,
                f"{event.current_price:.12g}",
                f"{event.fixed_price:.12g}",
                f"{event.predicted_price:.12g}",
                f"{pE:.12g}",
                f"{pN:.12g}",
                event.source_user,
                event.order_idx,
                f"{event.delta_at_current:.12g}",
                f"{event.delta_at_predicted:.12g}",
                event.realized_event_type,
                len(event.current_set),
                len(event.realized_next_set),
                _serialize_indices(event.prefix_before),
                _serialize_indices(event.prefix_after),
                _serialize_indices(event.current_set),
                _serialize_indices(event.realized_next_set),
                _serialize_indices(event.dropped),
                _serialize_indices(event.added),
            ]
        )
    _write_csv_rows(
        out_path,
        [
            "provider",
            "current_price",
            "fixed_price",
            "predicted_price",
            "pE",
            "pN",
            "source_user",
            "order_idx",
            "delta_at_current",
            "delta_at_predicted",
            "realized_event_type",
            "current_size",
            "realized_next_size",
            "prefix_before",
            "prefix_after",
            "current_set",
            "realized_next_set",
            "dropped",
            "added",
        ],
        rows,
    )


def _write_stats_csv(out_path: Path, stats: list[ProviderStats], match_tol: float) -> None:
    rows = [
        [
            stat.provider,
            f"{stat.fixed_price:.12g}",
            stat.exact_count,
            stat.old_raw_count,
            stat.old_unique_count,
            stat.hypothesis_count,
            f"{match_tol:.12g}",
            stat.old_coverage_count,
            f"{stat.old_coverage_ratio:.12g}",
            f"{stat.old_nearest_mean_abs_error:.12g}",
            f"{stat.old_nearest_max_abs_error:.12g}",
            stat.ordered_match_count,
            f"{stat.hypothesis_mean_abs_error:.12g}",
            f"{stat.hypothesis_max_abs_error:.12g}",
        ]
        for stat in stats
    ]
    _write_csv_rows(
        out_path,
        [
            "provider",
            "fixed_price",
            "exact_count",
            "old_raw_count",
            "old_unique_count",
            "hypothesis_count",
            "match_tol",
            "old_coverage_count",
            "old_coverage_ratio",
            "old_nearest_mean_abs_error",
            "old_nearest_max_abs_error",
            "ordered_match_count",
            "hypothesis_mean_abs_error",
            "hypothesis_max_abs_error",
        ],
        rows,
    )


def _match_summary_lines(
    provider: Provider,
    exact_events: list[BoundaryEvent],
    old_unique_prices: list[float],
    hypothesis_events: list[PredictionEvent],
) -> list[str]:
    exact_prices = [float(event.boundary_price) for event in exact_events]
    hypothesis_prices = [float(event.predicted_price) for event in hypothesis_events]
    old_nearest = _cloud_coverage_errors(exact_prices, old_unique_prices)
    ordered_errors = _ordered_matching_errors(exact_prices, hypothesis_prices)
    lines = [f"[{provider}]"]
    preview = min(5, min(len(exact_prices), len(hypothesis_prices)))
    for idx in range(preview):
        lines.append(
            f"matched_{idx} = exact:{exact_prices[idx]:.8f} hypothesis:{hypothesis_prices[idx]:.8f} abs_err:{ordered_errors[idx]:.8f}"
        )
    preview_old = min(5, len(exact_prices))
    for idx in range(preview_old):
        nearest = old_nearest[idx] if idx < len(old_nearest) else float("nan")
        lines.append(
            f"old_nearest_{idx} = exact:{exact_prices[idx]:.8f} nearest_old_abs_err:{nearest:.8f}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone diagnostic script for testing DG slice boundaries against old boundary points and a closed-form hypothesis."
    )
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--start-pE", type=float, default=0.5, help="Start pE for the two slices.")
    parser.add_argument("--start-pN", type=float, default=0.5, help="Start pN for the two slices.")
    parser.add_argument("--csv", type=str, default=None, help="Optional existing price_grid_metrics.csv.")
    parser.add_argument("--summary", type=str, default=None, help="Optional summary.txt for CSV reuse.")
    parser.add_argument("--pEmax", type=_positive_float, default=None, help="Optional pE upper bound.")
    parser.add_argument("--pNmax", type=_positive_float, default=None, help="Optional pN upper bound.")
    parser.add_argument("--pE-points", type=_min_two_int, default=81, help="Heatmap pE points when recomputing.")
    parser.add_argument("--pN-points", type=_min_two_int, default=81, help="Heatmap pN points when recomputing.")
    parser.add_argument("--scan-pE-points", type=_min_two_int, default=801, help="Slice scan points for the ESP axis.")
    parser.add_argument("--scan-pN-points", type=_min_two_int, default=1001, help="Slice scan points for the NSP axis.")
    parser.add_argument("--root-tol", type=_nonnegative_float, default=1e-5, help="Bisection tolerance for exact slice boundaries.")
    parser.add_argument("--match-tol", type=_nonnegative_float, default=None, help="Optional tolerance for cloud coverage metrics.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    rng = np.random.default_rng(seed)
    users = sample_users(cfg, rng)
    data = _build_data(users)
    system = cfg.system
    stack_cfg = cfg.stackelberg
    start_pE = max(float(args.start_pE), float(system.cE))
    start_pN = max(float(args.start_pN), float(system.cN))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = Path("outputs") / f"boundary_hypothesis_check_{timestamp}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    start_eval = _evaluate_slice_point("E", start_pE, start_pN, users, system, stack_cfg)
    current_set = start_eval.offloading_set
    old_rows, old_unique, old_raw_counts = _compute_old_boundary_points(data, current_set, start_pE, start_pN, system)

    summary_path = Path(args.summary) if args.summary else None
    if summary_path is None and args.csv is not None:
        candidate = Path(args.csv).parent / "summary.txt"
        if candidate.exists():
            summary_path = candidate
    summary_kv = _load_summary_kv(summary_path)

    csv_pE_max = None
    csv_pN_max = None
    heatmap_source = "recomputed"
    if args.csv is not None:
        pE_grid, pN_grid, esp_grid, nsp_grid = _load_grid_csv(Path(args.csv))
        csv_pE_max = float(pE_grid[-1])
        csv_pN_max = float(pN_grid[-1])
        heatmap_source = f"csv:{Path(args.csv)}"
    else:
        pE_grid = pN_grid = esp_grid = nsp_grid = None  # type: ignore[assignment]

    inferred_pE_max = max([start_pE, *(old_unique["E"] or [start_pE])])
    inferred_pN_max = max([start_pN, *(old_unique["N"] or [start_pN])])
    scan_pE_max = float(args.pEmax) if args.pEmax is not None else max(v for v in [inferred_pE_max, csv_pE_max or 0.0, cfg.pE.max])
    scan_pN_max = float(args.pNmax) if args.pNmax is not None else max(v for v in [inferred_pN_max, csv_pN_max or 0.0, cfg.pN.max])

    t0 = time.perf_counter()
    _, esp_slice, exact_esp = _scan_exact_boundaries(
        provider="E",
        fixed_price=start_pN,
        price_max=scan_pE_max,
        scan_points=args.scan_pE_points,
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    _, nsp_slice, exact_nsp = _scan_exact_boundaries(
        provider="N",
        fixed_price=start_pE,
        price_max=scan_pN_max,
        scan_points=args.scan_pN_points,
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    hypothesis_esp = _build_hypothesis_chain(
        provider="E",
        start_pE=start_pE,
        start_pN=start_pN,
        price_max=scan_pE_max,
        users=users,
        data=data,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    hypothesis_nsp = _build_hypothesis_chain(
        provider="N",
        start_pE=start_pE,
        start_pN=start_pN,
        price_max=scan_pN_max,
        users=users,
        data=data,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(args.root_tol),
    )
    elapsed = time.perf_counter() - t0

    if args.csv is None:
        grid = evaluate_stage1_price_grid(
            users=users,
            system=system,
            stack_cfg=stack_cfg,
            base_cfg=cfg.baselines,
            pE_min=0.0,
            pE_max=float(scan_pE_max),
            pN_min=0.0,
            pN_max=float(scan_pN_max),
            pE_points=int(args.pE_points),
            pN_points=int(args.pN_points),
            stage2_method="DG",
        )
        pE_grid = grid.pE_grid
        pN_grid = grid.pN_grid
        esp_grid = grid.esp_rev
        nsp_grid = grid.nsp_rev

    assert pE_grid is not None and pN_grid is not None and esp_grid is not None and nsp_grid is not None
    joint_grid = esp_grid + nsp_grid

    step_e = float(scan_pE_max - _axis_cost_floor("E", system)) / max(int(args.scan_pE_points) - 1, 1)
    step_n = float(scan_pN_max - _axis_cost_floor("N", system)) / max(int(args.scan_pN_points) - 1, 1)
    match_tol = float(args.match_tol) if args.match_tol is not None else max(float(args.root_tol) * 8.0, 1.25 * max(step_e, step_n))

    stats = [
        _build_provider_stats("E", start_pN, exact_esp, old_raw_counts["E"], old_unique["E"], hypothesis_esp, match_tol),
        _build_provider_stats("N", start_pE, exact_nsp, old_raw_counts["N"], old_unique["N"], hypothesis_nsp, match_tol),
    ]

    _plot_slice_comparison("E", esp_slice, exact_esp, old_unique["E"], hypothesis_esp, start_pE, out_dir / "esp_slice_boundary_comparison.png")
    _plot_slice_comparison("N", nsp_slice, exact_nsp, old_unique["N"], hypothesis_nsp, start_pN, out_dir / "nsp_slice_boundary_comparison.png")
    _plot_revenue_overlay(
        esp_grid,
        pE_grid,
        pN_grid,
        title="ESP revenue with old/exact/hypothesis boundaries",
        cbar_label="esp_revenue",
        out_path=out_dir / "esp_revenue_boundary_overlay.png",
        start_pE=start_pE,
        start_pN=start_pN,
        old_esp=old_unique["E"],
        exact_esp=[event.boundary_price for event in exact_esp],
        hyp_esp=[event.predicted_price for event in hypothesis_esp],
        old_nsp=[],
        exact_nsp=[],
        hyp_nsp=[],
    )
    _plot_revenue_overlay(
        nsp_grid,
        pE_grid,
        pN_grid,
        title="NSP revenue with old/exact/hypothesis boundaries",
        cbar_label="nsp_revenue",
        out_path=out_dir / "nsp_revenue_boundary_overlay.png",
        start_pE=start_pE,
        start_pN=start_pN,
        old_esp=[],
        exact_esp=[],
        hyp_esp=[],
        old_nsp=old_unique["N"],
        exact_nsp=[event.boundary_price for event in exact_nsp],
        hyp_nsp=[event.predicted_price for event in hypothesis_nsp],
    )
    _plot_revenue_overlay(
        joint_grid,
        pE_grid,
        pN_grid,
        title="Joint revenue with old/exact/hypothesis boundaries",
        cbar_label="joint_revenue",
        out_path=out_dir / "joint_revenue_boundary_overlay.png",
        start_pE=start_pE,
        start_pN=start_pN,
        old_esp=old_unique["E"],
        exact_esp=[event.boundary_price for event in exact_esp],
        hyp_esp=[event.predicted_price for event in hypothesis_esp],
        old_nsp=old_unique["N"],
        exact_nsp=[event.boundary_price for event in exact_nsp],
        hyp_nsp=[event.predicted_price for event in hypothesis_nsp],
    )

    _write_exact_csv(out_dir / "exact_dg_boundaries.csv", [*exact_esp, *exact_nsp])
    _write_old_csv(out_dir / "old_boundary_points.csv", old_rows)
    _write_hypothesis_csv(out_dir / "hypothesis_boundary_points.csv", [*hypothesis_esp, *hypothesis_nsp])
    _write_stats_csv(out_dir / "boundary_match_summary.csv", stats, match_tol)

    summary_lines = [
        f"config = {args.config}",
        f"seed = {seed}",
        f"start_pE = {start_pE}",
        f"start_pN = {start_pN}",
        f"current_set_size = {len(current_set)}",
        f"current_set = {_serialize_indices(current_set)}",
        f"heatmap_source = {heatmap_source}",
        f"scan_pE_max = {scan_pE_max}",
        f"scan_pN_max = {scan_pN_max}",
        f"scan_pE_points = {args.scan_pE_points}",
        f"scan_pN_points = {args.scan_pN_points}",
        f"root_tol = {args.root_tol}",
        f"match_tol = {match_tol}",
        f"elapsed_seconds = {elapsed:.3f}",
        f"summary_hint_config = {summary_kv.get('config', '')}",
        "",
    ]
    for stat in stats:
        summary_lines.extend(
            [
                f"[{stat.provider}]",
                f"fixed_price = {stat.fixed_price}",
                f"exact_count = {stat.exact_count}",
                f"old_raw_count = {stat.old_raw_count}",
                f"old_unique_count = {stat.old_unique_count}",
                f"hypothesis_count = {stat.hypothesis_count}",
                f"old_coverage_count = {stat.old_coverage_count}",
                f"old_coverage_ratio = {stat.old_coverage_ratio:.8f}",
                f"old_nearest_mean_abs_error = {stat.old_nearest_mean_abs_error:.8f}",
                f"old_nearest_max_abs_error = {stat.old_nearest_max_abs_error:.8f}",
                f"ordered_match_count = {stat.ordered_match_count}",
                f"hypothesis_mean_abs_error = {stat.hypothesis_mean_abs_error:.8f}",
                f"hypothesis_max_abs_error = {stat.hypothesis_max_abs_error:.8f}",
                "",
            ]
        )
        exact_provider = exact_esp if stat.provider == "E" else exact_nsp
        old_provider = old_unique["E"] if stat.provider == "E" else old_unique["N"]
        hyp_provider = hypothesis_esp if stat.provider == "E" else hypothesis_nsp
        summary_lines.extend(_match_summary_lines(stat.provider, exact_provider, old_provider, hyp_provider))
        summary_lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main()

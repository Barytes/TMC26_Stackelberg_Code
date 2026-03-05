#!/usr/bin/env python3
"""
Simple experiment suite runner for Stage-II experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import itertools
from pathlib import Path
import time
from typing import Any

import numpy as np

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from tmc26_exp.baselines import BaselineOutcome, run_stage2_solver
from tmc26_exp.config import ExperimentConfig, SystemConfig, load_config
from tmc26_exp.model import UserBatch
from tmc26_exp.simulator import sample_users


ALLOWED_METHODS = {"DG", "CS", "UBRD", "VI", "PEN"}
ALLOWED_SWEEP_KEYS = ("B", "F", "n_users", "pE", "pN")
SUPPORTED_METRICS = (
    "social_cost",
    "offloading_size",
    "epsilon_proxy",
    "runtime_sec",
    "esp_revenue",
    "nsp_revenue",
)
RESULT_DIMENSIONS = ("pE", "pN", "n_users", "F", "B")


def load_suite(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def get_sweep_params(sweep_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not sweep_cfg:
        return [{}]

    keys = sorted(sweep_cfg.keys())
    values_list: list[list[Any]] = []
    for key in keys:
        raw_value = sweep_cfg[key]
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        if not values:
            raise ValueError(f"Sweep key '{key}' must not be an empty list.")
        values_list.append(values)

    combos = itertools.product(*values_list)
    return [dict(zip(keys, combo)) for combo in combos]


def sample_users_for_trial(cfg: ExperimentConfig, n_users: int, seed: int) -> UserBatch:
    if n_users <= 0:
        raise ValueError("n_users must be positive.")
    user_cfg = replace(cfg, n_users=int(n_users), seed=int(seed))
    rng = np.random.default_rng(int(seed))
    return sample_users(user_cfg, rng)


def build_system_for_run(cfg: ExperimentConfig, F: float | None = None, B: float | None = None) -> SystemConfig:
    resolved_F = cfg.system.F if F is None else float(F)
    resolved_B = cfg.system.B if B is None else float(B)
    if resolved_F <= 0 or resolved_B <= 0:
        raise ValueError("F and B must be positive.")
    return replace(cfg.system, F=resolved_F, B=resolved_B)


def resolve_run_params(cfg: ExperimentConfig, params: dict[str, Any]) -> dict[str, float | int]:
    resolved = {
        "pE": float(params.get("pE", cfg.stackelberg.initial_pE)),
        "pN": float(params.get("pN", cfg.stackelberg.initial_pN)),
        "n_users": int(params.get("n_users", cfg.n_users)),
        "F": float(params.get("F", cfg.system.F)),
        "B": float(params.get("B", cfg.system.B)),
    }
    if resolved["n_users"] <= 0:
        raise ValueError("n_users must be positive.")
    if resolved["F"] <= 0 or resolved["B"] <= 0:
        raise ValueError("F and B must be positive.")
    return resolved


def validate_suite(
    suite: dict[str, Any],
    cfg: ExperimentConfig,
    param_combinations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    experiment = suite.get("experiment")
    if not isinstance(experiment, dict) or not str(experiment.get("name", "")).strip():
        raise ValueError("Suite must define [experiment].name.")

    methods_cfg = suite.get("methods", {})
    if not isinstance(methods_cfg, dict):
        raise ValueError("[methods] must be a table.")
    methods = methods_cfg.get("target", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("Suite must define a non-empty [methods].target list.")
    invalid_methods = [str(method) for method in methods if str(method) not in ALLOWED_METHODS]
    if invalid_methods:
        raise ValueError(
            f"Unsupported methods: {', '.join(invalid_methods)}. "
            f"Allowed methods are: {', '.join(sorted(ALLOWED_METHODS))}."
        )

    sweep_cfg = suite.get("sweep", {})
    if not isinstance(sweep_cfg, dict):
        raise ValueError("[sweep] must be a table.")
    invalid_sweep_keys = [key for key in sweep_cfg if key not in ALLOWED_SWEEP_KEYS]
    if invalid_sweep_keys:
        raise ValueError(
            f"Unsupported sweep keys: {', '.join(sorted(invalid_sweep_keys))}. "
            f"Allowed keys are: {', '.join(ALLOWED_SWEEP_KEYS)}."
        )

    data_cfg = suite.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("[data] must be a table.")
    trials = int(data_cfg.get("trials", 1))
    if trials <= 0:
        raise ValueError("[data].trials must be positive.")
    for key in ("reuse_users_across_methods", "reuse_users_across_sweep"):
        raw_value = data_cfg.get(key, True if key == "reuse_users_across_methods" else False)
        if not isinstance(raw_value, bool):
            raise ValueError(f"[data].{key} must be a boolean.")

    metrics_cfg = suite.get("metrics", {})
    if not isinstance(metrics_cfg, dict):
        raise ValueError("[metrics] must be a table.")
    metrics = metrics_cfg.get("include", ["social_cost"])
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Suite must define a non-empty [metrics].include list.")
    invalid_metrics = [str(metric) for metric in metrics if str(metric) not in SUPPORTED_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metrics: {', '.join(invalid_metrics)}. "
            f"Supported metrics are: {', '.join(SUPPORTED_METRICS)}."
        )

    if "CS" in methods:
        for params in param_combinations:
            resolved = resolve_run_params(cfg, params)
            if int(resolved["n_users"]) > cfg.baselines.exact_max_users:
                raise ValueError(
                    "Method 'CS' requires n_users <= "
                    f"{cfg.baselines.exact_max_users}, got {resolved['n_users']}."
                )

    normalized_methods = list(dict.fromkeys(str(method) for method in methods))
    normalized_metrics = list(dict.fromkeys(str(metric) for metric in metrics))
    return normalized_methods, normalized_metrics


def extract_metric(outcome: BaselineOutcome, metric: str, runtime_sec: float) -> float:
    if metric == "runtime_sec":
        return float(runtime_sec)
    if metric == "offloading_size":
        return float(len(outcome.offloading_set))
    value = getattr(outcome, metric)
    return float(value)


def write_csv(rows: list[dict[str, Any]], columns: list[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    group_columns = ["method", *RESULT_DIMENSIONS]
    metric_columns = [metric for metric in SUPPORTED_METRICS]

    for row in rows:
        key = tuple(row[column] for column in group_columns)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary_row = {column: value for column, value in zip(group_columns, key)}
        summary_row["count"] = len(group)
        for metric in metric_columns:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            summary_row[f"{metric}_mean"] = float(np.mean(values))
            summary_row[f"{metric}_std"] = float(np.std(values))
        summary_rows.append(summary_row)

    return summary_rows


def run_suite(config_path: str, suite_path: str) -> None:
    cfg = load_config(config_path)
    suite_file = Path(suite_path)
    suite = load_suite(suite_file)
    param_combinations = get_sweep_params(suite.get("sweep", {}))
    methods, metrics = validate_suite(suite, cfg, param_combinations)

    experiment = suite["experiment"]
    exp_name = str(experiment["name"]).strip()
    out_dir = Path("outputs/suites") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "suite_snapshot.toml").write_text(suite_file.read_text(encoding="utf-8"), encoding="utf-8")

    data_cfg = suite.get("data", {})
    base_seed = int(data_cfg.get("seed", cfg.seed))
    trials = int(data_cfg.get("trials", 1))
    reuse_users_across_methods = bool(data_cfg.get("reuse_users_across_methods", True))
    reuse_users_across_sweep = bool(data_cfg.get("reuse_users_across_sweep", False))

    results: list[dict[str, Any]] = []
    sweep_user_cache: dict[tuple[int, int], UserBatch] = {}
    total_runs = trials * len(param_combinations) * len(methods)
    print(f"Running suite '{exp_name}' with {total_runs} runs...")

    for trial_idx in range(trials):
        trial_seed = base_seed + trial_idx

        for combo_idx, params in enumerate(param_combinations):
            resolved = resolve_run_params(cfg, params)
            system = build_system_for_run(cfg, F=float(resolved["F"]), B=float(resolved["B"]))

            shared_users: UserBatch | None = None
            if reuse_users_across_methods:
                if reuse_users_across_sweep:
                    cache_key = (trial_seed, int(resolved["n_users"]))
                    shared_users = sweep_user_cache.get(cache_key)
                    if shared_users is None:
                        shared_users = sample_users_for_trial(cfg, int(resolved["n_users"]), trial_seed)
                        sweep_user_cache[cache_key] = shared_users
                else:
                    sample_seed = trial_seed * 1000 + combo_idx
                    shared_users = sample_users_for_trial(cfg, int(resolved["n_users"]), sample_seed)

            for method_idx, method in enumerate(methods):
                users = shared_users
                if users is None:
                    if reuse_users_across_sweep:
                        sample_seed = trial_seed * 1000 + method_idx
                    else:
                        sample_seed = trial_seed * 100000 + combo_idx * 100 + method_idx
                    users = sample_users_for_trial(cfg, int(resolved["n_users"]), sample_seed)

                started_at = time.perf_counter()
                outcome = run_stage2_solver(
                    method=method,
                    users=users,
                    pE=float(resolved["pE"]),
                    pN=float(resolved["pN"]),
                    system=system,
                    stack_cfg=cfg.stackelberg,
                    base_cfg=cfg.baselines,
                )
                runtime_sec = time.perf_counter() - started_at

                row: dict[str, Any] = {
                    "trial": trial_idx,
                    "seed": trial_seed,
                    "method": method,
                    "pE": float(resolved["pE"]),
                    "pN": float(resolved["pN"]),
                    "n_users": int(resolved["n_users"]),
                    "F": float(resolved["F"]),
                    "B": float(resolved["B"]),
                }
                for metric in SUPPORTED_METRICS:
                    row[metric] = extract_metric(outcome, metric, runtime_sec)
                results.append(row)

    result_columns = ["trial", "seed", "method", *RESULT_DIMENSIONS, *metrics]
    write_csv(results, result_columns, out_dir / "results.csv")

    summary_rows = build_summary_rows(results)
    summary_columns = [
        "method",
        *RESULT_DIMENSIONS,
        "count",
        *(f"{metric}_mean" for metric in SUPPORTED_METRICS),
        *(f"{metric}_std" for metric in SUPPORTED_METRICS),
    ]
    write_csv(summary_rows, summary_columns, out_dir / "summary.csv")

    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to base config.toml")
    parser.add_argument("--suite", required=True, help="Path to suite.toml")
    args = parser.parse_args()
    run_suite(args.config, args.suite)

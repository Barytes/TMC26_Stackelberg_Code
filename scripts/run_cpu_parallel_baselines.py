#!/usr/bin/env python3
"""Parallel experiment runner for multi-core CPU servers.

This script is intentionally standalone and minimally invasive:
- reuses existing package APIs (`load_config`, `sample_users`, `run_all_baselines`, `proposed_gsse`)
- adds process-level parallelism across independent trials/seeds
- writes both per-trial raw results and method-level summary tables

Example:
  uv run python scripts/run_cpu_parallel_baselines.py \
    --config configs/default.toml \
    --trials 100 \
    --workers 16 \
    --methods GSSE,GSO,PBRD,BO,DRL,MarketEquilibrium,SingleSP,RandomOffloading
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tmc26_exp.baselines import BaselineOutcome, proposed_gsse, run_all_baselines
from tmc26_exp.config import ExperimentConfig, load_config
from tmc26_exp.simulator import sample_users


DEFAULT_METHODS = [
    "GSSE",
    "CS",
    "UBRD",
    "URA",
    "GSO",
    "PBRD",
    "BO",
    "DRL",
    "MarketEquilibrium",
    "SingleSP",
    "RandomOffloading",
]


def _parse_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods provided.")
    unknown = [m for m in methods if m not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {DEFAULT_METHODS}")
    return methods


def _pick_methods(all_outcomes: Iterable[BaselineOutcome], selected: set[str]) -> list[BaselineOutcome]:
    return [o for o in all_outcomes if o.name in selected]


def _run_one_trial(config_path: str, seed: int, methods: list[str]) -> list[dict[str, object]]:
    cfg = load_config(config_path)
    rng = np.random.default_rng(seed)
    users = sample_users(cfg, rng)

    started = time.perf_counter()
    if methods == ["GSSE"]:
        outcomes = [proposed_gsse(users, cfg.system, cfg.stackelberg)]
    else:
        outcomes = run_all_baselines(users, cfg.system, cfg.stackelberg, cfg.baselines)
        outcomes = _pick_methods(outcomes, set(methods))
    elapsed = time.perf_counter() - started

    rows: list[dict[str, object]] = []
    denom = max(len(outcomes), 1)
    est_method_time = elapsed / denom
    for out in outcomes:
        rows.append(
            {
                "seed": seed,
                "method": out.name,
                "pE": out.price[0],
                "pN": out.price[1],
                "offloading_size": len(out.offloading_set),
                "social_cost": out.social_cost,
                "esp_revenue": out.esp_revenue,
                "nsp_revenue": out.nsp_revenue,
                "epsilon_proxy": out.epsilon_proxy,
                "runtime_sec_est": est_method_time,
                "meta": str(out.meta),
            }
        )
    return rows


def _write_raw_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "seed",
        "method",
        "pE",
        "pN",
        "offloading_size",
        "social_cost",
        "esp_revenue",
        "nsp_revenue",
        "epsilon_proxy",
        "runtime_sec_est",
        "meta",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _write_summary_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)

    fields = [
        "method",
        "count",
        "social_cost_mean",
        "social_cost_std",
        "esp_revenue_mean",
        "esp_revenue_std",
        "nsp_revenue_mean",
        "nsp_revenue_std",
        "epsilon_proxy_mean",
        "epsilon_proxy_std",
        "offloading_size_mean",
        "offloading_size_std",
        "runtime_sec_est_mean",
        "runtime_sec_est_std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method in sorted(grouped):
            g = grouped[method]
            social = [float(x["social_cost"]) for x in g]
            esp = [float(x["esp_revenue"]) for x in g]
            nsp = [float(x["nsp_revenue"]) for x in g]
            eps = [float(x["epsilon_proxy"]) for x in g]
            off = [float(x["offloading_size"]) for x in g]
            rt = [float(x["runtime_sec_est"]) for x in g]
            social_m, social_s = _mean_std(social)
            esp_m, esp_s = _mean_std(esp)
            nsp_m, nsp_s = _mean_std(nsp)
            eps_m, eps_s = _mean_std(eps)
            off_m, off_s = _mean_std(off)
            rt_m, rt_s = _mean_std(rt)
            w.writerow(
                {
                    "method": method,
                    "count": len(g),
                    "social_cost_mean": social_m,
                    "social_cost_std": social_s,
                    "esp_revenue_mean": esp_m,
                    "esp_revenue_std": esp_s,
                    "nsp_revenue_mean": nsp_m,
                    "nsp_revenue_std": nsp_s,
                    "epsilon_proxy_mean": eps_m,
                    "epsilon_proxy_std": eps_s,
                    "offloading_size_mean": off_m,
                    "offloading_size_std": off_s,
                    "runtime_sec_est_mean": rt_m,
                    "runtime_sec_est_std": rt_s,
                }
            )


def _write_run_meta(
    out_dir: Path,
    cfg: ExperimentConfig,
    config_path: str,
    methods: list[str],
    workers: int,
    trials: int,
    start_seed: int,
    wall_sec: float,
) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "run_name": cfg.run_name,
        "n_users": cfg.n_users,
        "methods": methods,
        "workers": workers,
        "trials": trials,
        "start_seed": start_seed,
        "seed_range": [start_seed, start_seed + trials - 1],
        "wall_time_sec": wall_sec,
    }
    out = out_dir / "run_meta.txt"
    lines = [f"{k} = {v}" for k, v in meta.items()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel CPU baseline experiment runner")
    parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    parser.add_argument("--trials", type=int, default=20, help="Number of independent trials")
    parser.add_argument("--workers", type=int, default=8, help="Number of process workers")
    parser.add_argument("--start-seed", type=int, default=20260000, help="Seed for first trial")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated subset of methods. Allowed: {DEFAULT_METHODS}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Optional output dir. Default: outputs/<run_name>/cpu_parallel_<timestamp>",
    )
    parser.add_argument(
        "--max-pending-factor",
        type=int,
        default=4,
        help="Queue size factor to keep memory usage controlled (pending <= workers*factor).",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="auto",
        choices=["auto", "process", "thread"],
        help="Parallel backend. 'auto' tries process pool first then falls back to thread pool.",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.max_pending_factor <= 0:
        raise ValueError("--max-pending-factor must be positive.")

    cfg = load_config(args.config)
    methods = _parse_methods(args.methods)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.outdir:
        out_dir = Path(args.outdir)
    else:
        out_dir = Path(cfg.output_dir) / cfg.run_name / f"cpu_parallel_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[info] start parallel run: trials={args.trials}, workers={args.workers}, "
        f"methods={methods}, out={out_dir}, executor={args.executor}"
    )

    start = time.perf_counter()
    all_rows: list[dict[str, object]] = []
    pending = {}
    trial_ids = list(range(args.trials))
    next_idx = 0
    max_pending = args.workers * args.max_pending_factor

    def run_with_executor(executor_cls) -> None:
        nonlocal next_idx
        with executor_cls(max_workers=args.workers) as ex:
            while next_idx < len(trial_ids) or pending:
                while next_idx < len(trial_ids) and len(pending) < max_pending:
                    t = trial_ids[next_idx]
                    seed = args.start_seed + t
                    fut = ex.submit(_run_one_trial, args.config, seed, methods)
                    pending[fut] = t
                    next_idx += 1

                done_batch = []
                for fut in as_completed(list(pending.keys()), timeout=None):
                    done_batch.append(fut)
                    # process one-by-one to keep progress smooth
                    break
                for fut in done_batch:
                    t = pending.pop(fut)
                    rows = fut.result()
                    all_rows.extend(rows)
                    print(f"[progress] trial {t + 1}/{args.trials} done, rows={len(rows)}")

    if args.executor == "process":
        run_with_executor(ProcessPoolExecutor)
    elif args.executor == "thread":
        run_with_executor(ThreadPoolExecutor)
    else:
        try:
            run_with_executor(ProcessPoolExecutor)
        except (PermissionError, OSError) as exc:
            print(f"[warn] process pool unavailable ({exc}); falling back to thread pool.")
            pending.clear()
            next_idx = 0
            run_with_executor(ThreadPoolExecutor)

    wall_sec = time.perf_counter() - start
    raw_csv = out_dir / "raw_results.csv"
    summary_csv = out_dir / "summary_by_method.csv"
    _write_raw_csv(all_rows, raw_csv)
    _write_summary_csv(all_rows, summary_csv)
    _write_run_meta(out_dir, cfg, args.config, methods, args.workers, args.trials, args.start_seed, wall_sec)

    print(f"[done] wall_time={wall_sec:.2f}s")
    print(f"[done] raw={raw_csv}")
    print(f"[done] summary={summary_csv}")


if __name__ == "__main__":
    main()

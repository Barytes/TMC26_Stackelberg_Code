from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import TextIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import run_stage1_budget_matched_quality_vs_users as budget_matched
import run_stage1_final_grid_ne_gap_vs_users as quality
from _figure_missing_impl import _load_cfg, _sample_users, _write_summary
from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
from tmc26_exp.baselines import BaselineOutcome, _price_cache_key, _stage2_solver


TRIAL_FIELDS = [
    "method",
    "n_users",
    "trial",
    "source",
    "success",
    "objective_eval_budget",
    "objective_evals_completed",
    "reference_stage2_calls",
    "search_budget_exhausted",
    "budget_stop_mode",
    "final_pE",
    "final_pN",
    "offloading_size",
    "final_grid_ne_gap",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "runtime_sec",
    "stage2_solver_calls",
    "audit_stage2_solver_calls",
    "total_stage2_solver_calls",
    "error",
]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _parse_n_users_list(raw: str) -> list[int]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("n-users-list cannot be empty.")
    values = [int(x) for x in items]
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Each n in n-users-list must be > 0.")
    return values


def _parse_methods(raw: str) -> list[str]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("methods cannot be empty.")
    allowed = {
        "bo": "BO-online",
        "bo-online": "BO-online",
        "bo_online": "BO-online",
        "ga": "GA",
    }
    methods: list[str] = []
    for item in items:
        key = item.lower()
        if key not in allowed:
            raise argparse.ArgumentTypeError("Allowed methods: BO-online, GA.")
        methods.append(allowed[key])
    return methods


def _trial_row_key(row: dict[str, object]) -> tuple[str, int, int]:
    return (str(row["method"]), int(float(row["n_users"])), int(float(row["trial"])))


def _load_checkpoint_rows(csv_path: Path | None) -> tuple[list[dict[str, object]], set[tuple[str, int, int]]]:
    if csv_path is None or not csv_path.exists():
        return [], set()
    rows = list(load_csv_rows(csv_path))
    return rows, {_trial_row_key(row) for row in rows}


def _clone_same_first_price_peer(
    rows: list[dict[str, object]],
    *,
    method: str,
    n_users: int,
    trial: int,
) -> dict[str, object] | None:
    if method not in {"BO-online", "GA"}:
        return None
    peer_method = "GA" if method == "BO-online" else "BO-online"
    for row in reversed(rows):
        if str(row.get("method")) != peer_method:
            continue
        if int(float(row.get("n_users", -1))) != int(n_users) or int(float(row.get("trial", -1))) != int(trial):
            continue
        if int(float(row.get("success", 0))) != 1:
            continue
        if str(row.get("budget_stop_mode")) != "after_one_objective_eval":
            continue
        cloned = dict(row)
        cloned["method"] = method
        cloned["source"] = "one_objective_eval_reused_same_first_price"
        cloned["error"] = ""
        return cloned
    return None


def _format_progress_message(
    *,
    completed: int,
    total: int,
    n_users: int,
    trial: int,
    trials: int,
    method: str,
    phase: str,
) -> str:
    current = min(int(completed) + 1, int(total)) if int(total) > 0 else 0
    return (
        f"[{current}/{int(total)}] "
        f"phase={phase} "
        f"n_users={int(n_users)} "
        f"trial={int(trial)}/{int(trials)} "
        f"method={method} "
        "objective_eval_budget=1"
    )


def _print_progress(
    *,
    completed: int,
    total: int,
    n_users: int,
    trial: int,
    trials: int,
    method: str,
    phase: str,
    stream: TextIO = sys.stdout,
) -> None:
    print(
        _format_progress_message(
            completed=completed,
            total=total,
            n_users=n_users,
            trial=trial,
            trials=trials,
            method=method,
            phase=phase,
        ),
        file=stream,
        flush=True,
    )


def _first_price_for_method(method: str, system, stack_cfg, base_cfg) -> tuple[float, float]:
    pE_min, pE_max = float(system.cE), float(base_cfg.max_price_E)
    pN_min, pN_max = float(system.cN), float(base_cfg.max_price_N)
    if method == "BO-online":
        pE = min(max(float(stack_cfg.initial_pE), pE_min), pE_max)
        pN = min(max(float(stack_cfg.initial_pN), pN_min), pN_max)
        return round(float(pE), 6), round(float(pN), 6)
    if method == "GA":
        # Match the first individual evaluated by the current GA runner.
        pE = min(max(float(stack_cfg.initial_pE), pE_min), pE_max)
        pN = min(max(float(stack_cfg.initial_pN), pN_min), pN_max)
        return round(float(pE), 6), round(float(pN), 6)
    raise ValueError(f"Unsupported method={method}")


def _stage2_price_task(payload):
    users, system, stack_cfg, base_cfg, pE, pN = payload
    key = _price_cache_key(float(pE), float(pN))
    out = _stage2_solver(
        base_cfg.stage2_solver_for_pricing,
        users,
        float(pE),
        float(pN),
        system,
        stack_cfg,
        base_cfg,
    )
    return key, out


def _run_one_objective_eval_parallel(
    *,
    users,
    system,
    stack_cfg,
    base_cfg,
    pE: float,
    pN: float,
    audit_points: int,
    workers: int,
) -> tuple[BaselineOutcome, dict[str, object]]:
    t0 = time.perf_counter()
    pE_grid = np.linspace(float(system.cE), float(base_cfg.max_price_E), max(2, int(audit_points)))
    pN_grid = np.linspace(float(system.cN), float(base_cfg.max_price_N), max(2, int(audit_points)))
    price_by_key: dict[tuple[float, float], tuple[float, float]] = {}
    for price_E, price_N in [(pE, pN), *[(float(x), pN) for x in pE_grid], *[(pE, float(x)) for x in pN_grid]]:
        price_by_key[_price_cache_key(float(price_E), float(price_N))] = (float(price_E), float(price_N))

    payloads = [(users, system, stack_cfg, base_cfg, price_E, price_N) for price_E, price_N in price_by_key.values()]
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for key, out in pool.map(_stage2_price_task, payloads):
            stage2_cache[key] = out

    direct_out = stage2_cache[_price_cache_key(float(pE), float(pN))]
    best_esp_rev = float(direct_out.esp_revenue)
    best_nsp_rev = float(direct_out.nsp_revenue)
    for cand_pE in pE_grid:
        cand_out = stage2_cache[_price_cache_key(float(cand_pE), float(pN))]
        best_esp_rev = max(best_esp_rev, float(cand_out.esp_revenue))
    for cand_pN in pN_grid:
        cand_out = stage2_cache[_price_cache_key(float(pE), float(cand_pN))]
        best_nsp_rev = max(best_nsp_rev, float(cand_out.nsp_revenue))
    final_gap = float(max(best_esp_rev - float(direct_out.esp_revenue), best_nsp_rev - float(direct_out.nsp_revenue)))
    candidate = replace(direct_out, grid_ne_gap=final_gap)
    meta = {
        "runtime_sec": float(time.perf_counter() - t0),
        "stage2_calls": int(len(stage2_cache)),
        "objective_evals_completed": 1,
        "final_grid_ne_gap": float(final_gap),
    }
    return candidate, meta


def _run_one_objective_eval(
    method: str,
    *,
    users,
    system,
    stack_cfg,
    base_cfg,
    audit_points: int,
    workers: int,
) -> tuple[BaselineOutcome, dict[str, object]]:
    t0 = time.perf_counter()
    pE, pN = _first_price_for_method(method, system, stack_cfg, base_cfg)
    if int(workers) > 1:
        return _run_one_objective_eval_parallel(
            users=users,
            system=system,
            stack_cfg=stack_cfg,
            base_cfg=base_cfg,
            pE=pE,
            pN=pN,
            audit_points=audit_points,
            workers=workers,
        )
    stage2_cache: dict[tuple[float, float], BaselineOutcome] = {}

    def cached_stage2(price_E: float, price_N: float) -> BaselineOutcome:
        key = _price_cache_key(float(price_E), float(price_N))
        if key not in stage2_cache:
            stage2_cache[key] = _stage2_solver(
                base_cfg.stage2_solver_for_pricing,
                users,
                float(price_E),
                float(price_N),
                system,
                stack_cfg,
                base_cfg,
            )
        return stage2_cache[key]

    direct_out = cached_stage2(pE, pN)
    pE_grid = np.linspace(float(system.cE), float(base_cfg.max_price_E), max(2, int(audit_points)))
    pN_grid = np.linspace(float(system.cN), float(base_cfg.max_price_N), max(2, int(audit_points)))
    final_gap, candidate = budget_matched._budgeted_grid_gap_candidate(
        direct_out,
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        base_cfg=base_cfg,
        cached_stage2=cached_stage2,
        pE_audit_grid=pE_grid,
        pN_audit_grid=pN_grid,
    )
    meta = {
        "runtime_sec": float(time.perf_counter() - t0),
        "stage2_calls": int(len(stage2_cache)),
        "objective_evals_completed": 1,
        "final_grid_ne_gap": float(final_gap),
    }
    return candidate, meta


def _reference_rows_and_stage2_calls(
    reference_csv: Path,
    n_list: list[int],
    trials: int,
) -> tuple[list[dict[str, str]], dict[tuple[int, int], int]]:
    rows = list(load_csv_rows(reference_csv))
    proposed_rows = [row for row in rows if str(row["method"]) == "Proposed"]
    stage2_map: dict[tuple[int, int], int] = {}
    for row in proposed_rows:
        stage2_map[(int(row["n_users"]), int(row["trial"]))] = int(float(row["stage2_solver_calls"]))
    missing = [
        (n, trial)
        for n in n_list
        for trial in range(1, trials + 1)
        if (int(n), int(trial)) not in stage2_map
    ]
    if missing:
        raise ValueError(f"Reference proposed CSV missing rows for: {missing[:5]}")
    return proposed_rows, stage2_map


def _load_proposed_audit_overrides(audit_csv: Path | None) -> dict[tuple[int, int], dict[str, float]]:
    if audit_csv is None:
        return {}
    if not audit_csv.exists():
        raise FileNotFoundError(f"Proposed audit CSV not found: {audit_csv}")
    overrides: dict[tuple[int, int], dict[str, float]] = {}
    for row in load_csv_rows(audit_csv):
        if str(row.get("error", "")).strip():
            continue
        key = (int(float(row["n_users"])), int(float(row["trial"])))
        overrides[key] = {
            "final_grid_ne_gap": float(row["true_grid_ne_gap"]),
            "audit_stage2_solver_calls": float(row.get("audit_stage2_solver_calls", 0) or 0),
        }
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary Stage-I baseline experiment: reuse Proposed rows from a reference run, "
            "but stop GA/BO-online after one objective evaluation and record NE gap and joint revenue."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=_parse_n_users_list, default="10,15,20,25,30")
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument("--methods", type=_parse_methods, default="BO-online,GA")
    parser.add_argument("--reference-run-dir", type=str, required=True)
    parser.add_argument("--proposed-audit-csv", type=Path, default=None)
    parser.add_argument("--final-audit-grid-points", type=_positive_int, default=120)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    n_list = list(args.n_users_list) if not isinstance(args.n_users_list, str) else _parse_n_users_list(args.n_users_list)
    methods = list(args.methods) if not isinstance(args.methods, str) else _parse_methods(args.methods)
    reference_dir = Path(args.reference_run_dir)
    reference_csv = reference_dir / "stage1_final_grid_ne_gap_vs_users.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    proposed_rows, reference_stage2_map = _reference_rows_and_stage2_calls(reference_csv, n_list, int(args.trials))
    proposed_audit_overrides = _load_proposed_audit_overrides(args.proposed_audit_csv)
    cfg = _load_cfg(str(args.config))
    out_dir = resolve_out_dir("run_stage1_one_eval_baseline_quality_vs_users", args.out_dir)
    trials_csv = out_dir / "stage1_one_eval_baseline_quality_vs_users.csv"

    rows, completed_keys = _load_checkpoint_rows(trials_csv)
    proposed_subset = [
        row
        for row in proposed_rows
        if int(row["n_users"]) in n_list and 1 <= int(row["trial"]) <= int(args.trials)
    ]
    for row in proposed_subset:
        key = ("Proposed", int(row["n_users"]), int(row["trial"]))
        if key in completed_keys:
            continue
        audit_override = proposed_audit_overrides.get((int(row["n_users"]), int(row["trial"])), {})
        audit_calls = int(audit_override.get("audit_stage2_solver_calls", float(row["audit_stage2_solver_calls"])))
        stage2_calls = int(float(row["stage2_solver_calls"]))
        rows.append(
            {
                "method": "Proposed",
                "n_users": int(row["n_users"]),
                "trial": int(row["trial"]),
                "source": "reused_reference",
                "success": int(float(row["success"])),
                "objective_eval_budget": "",
                "objective_evals_completed": "",
                "reference_stage2_calls": int(float(row["stage2_solver_calls"])),
                "search_budget_exhausted": 0,
                "budget_stop_mode": "reference_reuse",
                "final_pE": float(row["final_pE"]),
                "final_pN": float(row["final_pN"]),
                "offloading_size": int(float(row["offloading_size"])),
                "final_grid_ne_gap": float(audit_override.get("final_grid_ne_gap", float(row["final_grid_ne_gap"]))),
                "esp_revenue": float(row["esp_revenue"]),
                "nsp_revenue": float(row["nsp_revenue"]),
                "joint_revenue": float(row["joint_revenue"]),
                "runtime_sec": float(row["runtime_sec"]),
                "stage2_solver_calls": stage2_calls,
                "audit_stage2_solver_calls": audit_calls,
                "total_stage2_solver_calls": stage2_calls + audit_calls,
                "error": str(row.get("error", "")),
            }
        )
        completed_keys.add(key)
    write_csv_rows(trials_csv, TRIAL_FIELDS, rows)

    baseline_keys = {
        (method, int(n), int(trial))
        for method in methods
        for n in n_list
        for trial in range(1, int(args.trials) + 1)
    }
    completed_baseline_runs = len(completed_keys & baseline_keys)
    total_baseline_runs = len(n_list) * int(args.trials) * len(methods)
    failures = sum(1 for row in rows if int(float(row.get("success", 0))) != 1)

    for n in n_list:
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users(cfg, int(n), int(args.seed), int(trial))
            base_cfg = quality._apply_baseline_overrides(
                cfg.baselines,
                bo_candidate_pool=None,
                bo_iters=None,
                ga_population_size=None,
                ga_generations=None,
                marl_price_levels=None,
                marl_episodes=None,
                marl_steps_per_episode=None,
            )
            for method in methods:
                key = (str(method), int(n), int(trial))
                if key in completed_keys:
                    continue
                cloned = _clone_same_first_price_peer(rows, method=method, n_users=int(n), trial=int(trial))
                if cloned is not None:
                    rows.append(cloned)
                    completed_keys.add(key)
                    write_csv_rows(trials_csv, TRIAL_FIELDS, rows)
                    completed_baseline_runs += 1
                    continue
                _print_progress(
                    completed=completed_baseline_runs,
                    total=total_baseline_runs,
                    n_users=int(n),
                    trial=int(trial),
                    trials=int(args.trials),
                    method=method,
                    phase="start",
                )
                try:
                    out, meta = _run_one_objective_eval(
                        method,
                        users=users,
                        system=cfg.system,
                        stack_cfg=cfg.stackelberg,
                        base_cfg=base_cfg,
                        audit_points=int(args.final_audit_grid_points),
                        workers=int(args.workers),
                    )
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "source": "one_objective_eval_rerun",
                            "success": 1,
                            "objective_eval_budget": 1,
                            "objective_evals_completed": int(meta["objective_evals_completed"]),
                            "reference_stage2_calls": int(reference_stage2_map[(int(n), int(trial))]),
                            "search_budget_exhausted": 1,
                            "budget_stop_mode": "after_one_objective_eval",
                            "final_pE": float(out.price[0]),
                            "final_pN": float(out.price[1]),
                            "offloading_size": int(len(out.offloading_set)),
                            "final_grid_ne_gap": float(meta["final_grid_ne_gap"]),
                            "esp_revenue": float(out.esp_revenue),
                            "nsp_revenue": float(out.nsp_revenue),
                            "joint_revenue": float(out.esp_revenue + out.nsp_revenue),
                            "runtime_sec": float(meta["runtime_sec"]),
                            "stage2_solver_calls": int(meta["stage2_calls"]),
                            "audit_stage2_solver_calls": 0,
                            "total_stage2_solver_calls": int(meta["stage2_calls"]),
                            "error": "",
                        }
                    )
                    completed_keys.add(key)
                    write_csv_rows(trials_csv, TRIAL_FIELDS, rows)
                    _print_progress(
                        completed=completed_baseline_runs,
                        total=total_baseline_runs,
                        n_users=int(n),
                        trial=int(trial),
                        trials=int(args.trials),
                        method=method,
                        phase="done",
                    )
                except Exception as exc:
                    failures += 1
                    rows.append(
                        {
                            "method": method,
                            "n_users": int(n),
                            "trial": int(trial),
                            "source": "one_objective_eval_rerun",
                            "success": 0,
                            "objective_eval_budget": 1,
                            "objective_evals_completed": 0,
                            "reference_stage2_calls": int(reference_stage2_map[(int(n), int(trial))]),
                            "search_budget_exhausted": 0,
                            "budget_stop_mode": "error",
                            "final_pE": float("nan"),
                            "final_pN": float("nan"),
                            "offloading_size": -1,
                            "final_grid_ne_gap": float("nan"),
                            "esp_revenue": float("nan"),
                            "nsp_revenue": float("nan"),
                            "joint_revenue": float("nan"),
                            "runtime_sec": float("nan"),
                            "stage2_solver_calls": float("nan"),
                            "audit_stage2_solver_calls": float("nan"),
                            "total_stage2_solver_calls": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    completed_keys.add(key)
                    write_csv_rows(trials_csv, TRIAL_FIELDS, rows)
                    _print_progress(
                        completed=completed_baseline_runs,
                        total=total_baseline_runs,
                        n_users=int(n),
                        trial=int(trial),
                        trials=int(args.trials),
                        method=method,
                        phase="error",
                    )
                completed_baseline_runs += 1

    method_order = ["Proposed", *methods]
    summary_rows = quality.summarize_trials(rows, method_order, n_list)
    write_csv_rows(trials_csv, TRIAL_FIELDS, rows)
    write_csv_rows(out_dir / "stage1_one_eval_baseline_quality_vs_users_stats.csv", quality._summary_fieldnames(), summary_rows)
    quality.plot_metric_summary(
        summary_rows,
        out_dir / "stage1_one_eval_gap_vs_users.png",
        methods=method_order,
        metric="final_grid_ne_gap",
        statistic="median_iqr",
        ylabel="NE gap",
        title="One-evaluation baseline NE gap vs. number of users",
    )
    quality.plot_metric_summary(
        summary_rows,
        out_dir / "stage1_one_eval_joint_revenue_vs_users.png",
        methods=method_order,
        metric="joint_revenue",
        statistic="median_iqr",
        ylabel="Joint revenue",
        title="One-evaluation baseline joint revenue vs. number of users",
    )
    _write_summary(
        out_dir / "stage1_one_eval_baseline_quality_vs_users_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"trials = {args.trials}",
            f"n_users_list = {','.join(str(x) for x in n_list)}",
            f"methods = {','.join(method_order)}",
            f"reference_run_dir = {reference_dir}",
            f"proposed_audit_csv = {'' if args.proposed_audit_csv is None else args.proposed_audit_csv}",
            "objective_eval_budget = 1",
            "stop_rule = GA and BO-online stop immediately after one objective evaluation",
            "first_price_rule = GA and BO-online use the first price evaluated by the current baseline runners; both start from the clipped Stage-I initial price",
            f"final_audit_grid_points = {int(args.final_audit_grid_points)}",
            f"workers = {int(args.workers)}",
            "final_grid_ne_gap_definition = max unilateral provider revenue improvement on the audit price grid with the other provider price fixed",
            "stage2_solver_calls_definition = all Stage-II solves used inside the single objective evaluation, including unilateral grid evaluations for NE-gap scoring",
            "proposed_rows = reused unchanged from reference_run_dir",
            f"failed_runs = {int(failures)}",
        ],
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import os
import sys

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", str(ROOT / "outputs" / "_tmp_cache")))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

matplotlib.use("Agg")
import numpy as np

import run_stage1_final_grid_ne_gap_vs_users as quality
from _figure_missing_impl import (
    _load_cfg,
    _load_grid_ne_gap_surface,
    _nearest_grid_ne_gap,
    _run_stage1_method,
    _sample_users,
    _write_summary,
)
from _figure_wrapper_utils import load_csv_rows, resolve_out_dir, write_csv_rows
from tmc26_exp.baselines import BaselineOutcome, _grid_ne_gap_audit, _price_cache_key

TRIAL_FIELDS = [
    "method",
    "n_users",
    "trial",
    "success",
    "final_pE",
    "final_pN",
    "offloading_size",
    "restricted_gap",
    "final_grid_ne_gap",
    "final_grid_ne_gap_source",
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


def _resolve_gap_heatmap_csv_path(template: str | None, n_users: int) -> Path | None:
    if template is None:
        return None
    candidate = Path(str(template).format(n=int(n_users)))
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    return candidate if candidate.exists() else None


def _build_current_outcome(
    price: tuple[float, float],
    offloading_set: tuple[int, ...],
    social_cost: float,
    esp_revenue: float,
    nsp_revenue: float,
) -> BaselineOutcome:
    return BaselineOutcome(
        name="Proposed",
        price=(float(price[0]), float(price[1])),
        offloading_set=tuple(int(x) for x in offloading_set),
        social_cost=float(social_cost),
        esp_revenue=float(esp_revenue),
        nsp_revenue=float(nsp_revenue),
        grid_ne_gap=float("nan"),
        legacy_gain_proxy=float("nan"),
        meta={},
    )


def _run_proposed_point(
    config_path: str,
    seed: int,
    n_users: int,
    trial: int,
    gap_heatmap_csv_template: str | None,
) -> dict[str, object]:
    cfg = _load_cfg(config_path)
    users = _sample_users(cfg, int(n_users), int(seed), int(trial))
    try:
        price, offloading_set, restricted_gap, esp_revenue, nsp_revenue, meta = _run_stage1_method(
            users,
            cfg.system,
            cfg.stackelberg,
            cfg.baselines,
            "Proposed",
        )
        audit_stage2_calls = 0
        if abs(float(restricted_gap)) <= 1e-15:
            final_grid_gap = 0.0
            gap_source = "restricted_gap_zero_certified"
        else:
            heatmap_path = _resolve_gap_heatmap_csv_path(gap_heatmap_csv_template, int(n_users))
            if heatmap_path is not None:
                heatmap_pE_grid, heatmap_pN_grid, heatmap_grid_ne_gap = _load_grid_ne_gap_surface(heatmap_path)
                final_grid_gap = _nearest_grid_ne_gap(
                    float(price[0]),
                    float(price[1]),
                    pE_grid=heatmap_pE_grid,
                    pN_grid=heatmap_pN_grid,
                    grid_ne_gap=heatmap_grid_ne_gap,
                )
                gap_source = "heatmap_csv_nearest"
            else:
                current_out = _build_current_outcome(
                    price=price,
                    offloading_set=offloading_set,
                    social_cost=float(meta.get("social_cost", float("nan"))),
                    esp_revenue=float(esp_revenue),
                    nsp_revenue=float(nsp_revenue),
                )
                stage2_cache = {_price_cache_key(float(price[0]), float(price[1])): current_out}
                pE_audit_grid = np.linspace(float(cfg.system.cE), float(cfg.baselines.max_price_E), max(2, int(cfg.baselines.gso_grid_points)))
                pN_audit_grid = np.linspace(float(cfg.system.cN), float(cfg.baselines.max_price_N), max(2, int(cfg.baselines.gso_grid_points)))
                final_grid_gap = _grid_ne_gap_audit(
                    current_out,
                    users,
                    cfg.system,
                    cfg.stackelberg,
                    cfg.baselines,
                    stage2_cache,
                    pE_audit_grid,
                    pN_audit_grid,
                )
                audit_stage2_calls = max(0, int(len(stage2_cache) - 1))
                gap_source = "audit_grid"
        stage2_calls = int(meta.get("stage2_calls", 0))
        return {
            "method": "Proposed",
            "n_users": int(n_users),
            "trial": int(trial),
            "success": 1,
            "final_pE": float(price[0]),
            "final_pN": float(price[1]),
            "offloading_size": int(len(offloading_set)),
            "restricted_gap": float(restricted_gap),
            "final_grid_ne_gap": float(final_grid_gap),
            "final_grid_ne_gap_source": str(gap_source),
            "esp_revenue": float(esp_revenue),
            "nsp_revenue": float(nsp_revenue),
            "joint_revenue": float(esp_revenue + nsp_revenue),
            "runtime_sec": float(meta.get("runtime_sec", float("nan"))),
            "stage2_solver_calls": int(stage2_calls),
            "audit_stage2_solver_calls": int(audit_stage2_calls),
            "total_stage2_solver_calls": int(stage2_calls + audit_stage2_calls),
            "error": "",
        }
    except Exception as exc:
        return {
            "method": "Proposed",
            "n_users": int(n_users),
            "trial": int(trial),
            "success": 0,
            "final_pE": float("nan"),
            "final_pN": float("nan"),
            "offloading_size": -1,
            "restricted_gap": float("nan"),
            "final_grid_ne_gap": float("nan"),
            "final_grid_ne_gap_source": "",
            "esp_revenue": float("nan"),
            "nsp_revenue": float("nan"),
            "joint_revenue": float("nan"),
            "runtime_sec": float("nan"),
            "stage2_solver_calls": float("nan"),
            "audit_stage2_solver_calls": float("nan"),
            "total_stage2_solver_calls": float("nan"),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _ordered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: (int(row["n_users"]), int(row["trial"])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable Proposed/VBBR n-sweep, writing runtime, Stage-II calls, and final grid-gap summaries "
            "with incremental CSV flushing."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=_parse_n_users_list, default="200,400,600,800,1000,1200,1400,1600,1800,2000")
    parser.add_argument("--trials", type=_positive_int, default=10)
    parser.add_argument("--jobs", type=_positive_int, default=8)
    parser.add_argument("--gap-heatmap-csv-template", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    n_list = list(args.n_users_list) if not isinstance(args.n_users_list, str) else _parse_n_users_list(args.n_users_list)
    out_dir = resolve_out_dir("run_stage1_proposed_n_sweep", args.out_dir)
    trials_csv = out_dir / "stage1_proposed_n_sweep.csv"
    stats_csv = out_dir / "stage1_proposed_n_sweep_stats.csv"
    summary_path = out_dir / "stage1_proposed_n_sweep_summary.txt"

    rows: list[dict[str, object]] = list(load_csv_rows(trials_csv)) if trials_csv.exists() else []
    completed_points = {
        (int(row["n_users"]), int(row["trial"]))
        for row in rows
        if str(row.get("method", "")) == "Proposed"
    }
    total_points = len(n_list) * int(args.trials)

    def _flush_progress(*, completed: bool) -> None:
        ordered_rows = _ordered_rows(rows)
        write_csv_rows(trials_csv, TRIAL_FIELDS, ordered_rows)
        if ordered_rows:
            summary_rows = quality.summarize_trials(ordered_rows, ["Proposed"], n_list)
            write_csv_rows(stats_csv, quality._summary_fieldnames(), summary_rows)
            quality.plot_gap_summary(
                summary_rows,
                out_dir / "stage1_final_grid_ne_gap_vs_users.png",
                methods=["Proposed"],
                statistic="median_iqr",
            )
            quality.plot_metric_summary(
                summary_rows,
                out_dir / "stage1_runtime_vs_users.png",
                methods=["Proposed"],
                metric="runtime_sec",
                statistic="median_iqr",
                ylabel="Stage-I runtime (s)",
                title="Stage-I runtime vs. number of users",
            )
            quality.plot_metric_summary(
                summary_rows,
                out_dir / "stage1_stage2_calls_vs_users.png",
                methods=["Proposed"],
                metric="stage2_solver_calls",
                statistic="median_iqr",
                ylabel="Stage-II solver calls",
                title="Stage-II solver calls vs. number of users",
            )
        _write_summary(
            summary_path,
            [
                f"config = {args.config}",
                f"seed = {args.seed}",
                f"trials = {args.trials}",
                f"n_users_list = {','.join(str(x) for x in n_list)}",
                "methods = Proposed",
                f"jobs = {int(args.jobs)}",
                f"gap_heatmap_csv_template = {'' if args.gap_heatmap_csv_template is None else str(args.gap_heatmap_csv_template)}",
                "final_grid_ne_gap_definition = max unilateral provider revenue improvement on the audit price grid with the other provider price fixed",
                "grid_ne_gap_evaluation_mode = Proposed runs with restricted_gap==0 are certified to have final_grid_ne_gap=0 because the audit grid is a subset of unilateral deviations; otherwise matching gap-heatmap CSV reuse is tried first and direct audit is the fallback",
                f"completed_points = {len(completed_points)}",
                f"total_points = {total_points}",
                f"progress_complete = {'true' if completed else 'false'}",
            ],
        )

    _flush_progress(completed=(len(completed_points) >= total_points and total_points > 0))

    pending_points = [
        (int(n_users), int(trial))
        for n_users in n_list
        for trial in range(1, int(args.trials) + 1)
        if (int(n_users), int(trial)) not in completed_points
    ]

    if int(args.jobs) <= 1:
        for n_users, trial in pending_points:
            rows.append(
                _run_proposed_point(
                    str(args.config),
                    int(args.seed),
                    int(n_users),
                    int(trial),
                    args.gap_heatmap_csv_template,
                )
            )
            completed_points.add((int(n_users), int(trial)))
            _flush_progress(completed=False)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
            future_to_point = {
                executor.submit(
                    _run_proposed_point,
                    str(args.config),
                    int(args.seed),
                    int(n_users),
                    int(trial),
                    args.gap_heatmap_csv_template,
                ): (int(n_users), int(trial))
                for n_users, trial in pending_points
            }
            for future in concurrent.futures.as_completed(future_to_point):
                n_users, trial = future_to_point[future]
                rows.append(future.result())
                completed_points.add((int(n_users), int(trial)))
                _flush_progress(completed=False)

    _flush_progress(completed=True)


if __name__ == "__main__":
    main()

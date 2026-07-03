from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, Path(__file__).resolve().parent):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", "/tmp/tmc26_cache"))
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

from _figure_missing_impl import _load_cfg, _sample_users
from _figure_wrapper_utils import write_csv_rows
from run_stage1_final_grid_ne_gap_vs_users import _build_current_outcome
from tmc26_exp.baselines import _grid_ne_gap_audit, _price_cache_key


FIELDS = [
    "n_users",
    "trial",
    "final_pE",
    "final_pN",
    "restricted_gap",
    "recorded_final_grid_ne_gap",
    "true_grid_ne_gap",
    "audit_grid_points",
    "audit_stage2_solver_calls",
    "runtime_sec",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "source_row_final_grid_ne_gap_source",
    "error",
]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_completed(path: Path) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    if not path.exists():
        return [], set()
    rows = _read_rows(path)
    completed: set[tuple[int, int]] = set()
    for row in rows:
        if str(row.get("error", "")).strip():
            continue
        completed.add((int(row["n_users"]), int(row["trial"])))
    return list(rows), completed


def _source_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["n_users"]), int(row["trial"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a direct posthoc audit-grid NE-gap evaluation for Proposed rows, "
            "without the restricted-gap-zero certification shortcut."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    parser.add_argument("--audit-grid-points", type=_positive_int, default=120)
    args = parser.parse_args()

    cfg = _load_cfg(str(args.config))
    source_csv = Path(args.source_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    source_rows = [
        row
        for row in _read_rows(source_csv)
        if row.get("method") == "Proposed" and int(float(row.get("success", 0))) == 1
    ]
    source_rows.sort(key=lambda row: (int(row["n_users"]), int(row["trial"])))

    out_rows, completed = _load_completed(out_csv)
    pE_grid = np.linspace(float(cfg.system.cE), float(cfg.baselines.max_price_E), int(args.audit_grid_points))
    pN_grid = np.linspace(float(cfg.system.cN), float(cfg.baselines.max_price_N), int(args.audit_grid_points))

    total = len(source_rows)
    for row in source_rows:
        key = _source_key(row)
        if key in completed:
            continue

        n_users, trial = key
        print(
            f"[{len(completed) + 1}/{total}] phase=start n_users={n_users} trial={trial} "
            f"audit_grid_points={int(args.audit_grid_points)}",
            flush=True,
        )
        start = time.perf_counter()
        try:
            users = _sample_users(cfg, n_users, int(args.seed), trial)
            price = (float(row["final_pE"]), float(row["final_pN"]))
            current_out = _build_current_outcome(
                method="Proposed",
                price=price,
                offloading_set=(),
                social_cost=float("nan"),
                esp_revenue=float(row["esp_revenue"]),
                nsp_revenue=float(row["nsp_revenue"]),
            )
            stage2_cache = {_price_cache_key(float(price[0]), float(price[1])): current_out}
            true_gap = _grid_ne_gap_audit(
                current_out,
                users,
                cfg.system,
                cfg.stackelberg,
                cfg.baselines,
                stage2_cache,
                pE_grid,
                pN_grid,
            )
            audit_calls = max(0, len(stage2_cache) - 1)
            out_rows.append(
                {
                    "n_users": int(n_users),
                    "trial": int(trial),
                    "final_pE": float(price[0]),
                    "final_pN": float(price[1]),
                    "restricted_gap": float(row["restricted_gap"]),
                    "recorded_final_grid_ne_gap": float(row["final_grid_ne_gap"]),
                    "true_grid_ne_gap": float(true_gap),
                    "audit_grid_points": int(args.audit_grid_points),
                    "audit_stage2_solver_calls": int(audit_calls),
                    "runtime_sec": float(time.perf_counter() - start),
                    "esp_revenue": float(row["esp_revenue"]),
                    "nsp_revenue": float(row["nsp_revenue"]),
                    "joint_revenue": float(row["joint_revenue"]),
                    "source_row_final_grid_ne_gap_source": str(row["final_grid_ne_gap_source"]),
                    "error": "",
                }
            )
            completed.add(key)
            print(
                f"[{len(completed)}/{total}] phase=done n_users={n_users} trial={trial} "
                f"true_grid_ne_gap={float(true_gap):.12g} audit_calls={audit_calls}",
                flush=True,
            )
        except Exception as exc:
            out_rows.append(
                {
                    "n_users": int(n_users),
                    "trial": int(trial),
                    "final_pE": row.get("final_pE", ""),
                    "final_pN": row.get("final_pN", ""),
                    "restricted_gap": row.get("restricted_gap", ""),
                    "recorded_final_grid_ne_gap": row.get("final_grid_ne_gap", ""),
                    "true_grid_ne_gap": "",
                    "audit_grid_points": int(args.audit_grid_points),
                    "audit_stage2_solver_calls": "",
                    "runtime_sec": float(time.perf_counter() - start),
                    "esp_revenue": row.get("esp_revenue", ""),
                    "nsp_revenue": row.get("nsp_revenue", ""),
                    "joint_revenue": row.get("joint_revenue", ""),
                    "source_row_final_grid_ne_gap_source": str(row.get("final_grid_ne_gap_source", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{len(completed) + 1}/{total}] phase=error n_users={n_users} trial={trial}: {exc}", flush=True)
        write_csv_rows(out_csv, FIELDS, out_rows)

    print(f"completed={len(completed)}/{total} out_csv={out_csv}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics

from _figure_wrapper_utils import write_csv_rows


FIELDS = [
    "n_users",
    "matched_trials",
    "old_mean_true_grid_ne_gap",
    "new_mean_true_grid_ne_gap",
    "delta_mean_new_minus_old",
    "old_median_true_grid_ne_gap",
    "new_median_true_grid_ne_gap",
    "delta_median_new_minus_old",
    "old_max_true_grid_ne_gap",
    "new_max_true_grid_ne_gap",
    "delta_max_new_minus_old",
]


def _read_audit(path: Path) -> dict[tuple[int, int], float]:
    rows: dict[tuple[int, int], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("error", "")).strip():
                continue
            gap_text = str(row.get("true_grid_ne_gap", "")).strip()
            if not gap_text:
                continue
            key = (int(row["n_users"]), int(row["trial"]))
            rows[key] = float(gap_text)
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _max(values: list[float]) -> float:
    return float(max(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old and new Proposed true grid NE-gap audits.")
    parser.add_argument("--old-audit-csv", required=True)
    parser.add_argument("--new-audit-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    old_rows = _read_audit(Path(args.old_audit_csv).resolve())
    new_rows = _read_audit(Path(args.new_audit_csv).resolve())
    matched_keys = sorted(set(old_rows) & set(new_rows))
    by_n: dict[int, list[tuple[float, float]]] = {}
    for key in matched_keys:
        n_users, _trial = key
        by_n.setdefault(int(n_users), []).append((float(old_rows[key]), float(new_rows[key])))

    out_rows: list[dict[str, object]] = []
    for n_users in sorted(by_n):
        pairs = by_n[n_users]
        old_values = [old for old, _new in pairs]
        new_values = [new for _old, new in pairs]
        old_mean = _mean(old_values)
        new_mean = _mean(new_values)
        old_median = _median(old_values)
        new_median = _median(new_values)
        old_max = _max(old_values)
        new_max = _max(new_values)
        out_rows.append(
            {
                "n_users": int(n_users),
                "matched_trials": int(len(pairs)),
                "old_mean_true_grid_ne_gap": old_mean,
                "new_mean_true_grid_ne_gap": new_mean,
                "delta_mean_new_minus_old": float(new_mean - old_mean),
                "old_median_true_grid_ne_gap": old_median,
                "new_median_true_grid_ne_gap": new_median,
                "delta_median_new_minus_old": float(new_median - old_median),
                "old_max_true_grid_ne_gap": old_max,
                "new_max_true_grid_ne_gap": new_max,
                "delta_max_new_minus_old": float(new_max - old_max),
            }
        )

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_path, FIELDS, out_rows)
    print(f"matched_pairs={len(matched_keys)} groups={len(out_rows)} out_csv={out_path}")


if __name__ == "__main__":
    main()

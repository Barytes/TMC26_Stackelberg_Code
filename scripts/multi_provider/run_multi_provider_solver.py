from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multi_provider.multi_provider_core import (
    assignment_matrix,
    assignment_rows,
    generate_random_problem,
    provider_metric_rows,
    solve_multi_provider_stage1,
    trajectory_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3-ESP 3-NSP multi-provider Stackelberg extension.")
    parser.add_argument("--n-users", type=int, default=60)
    parser.add_argument("--num-esp", type=int, default=3)
    parser.add_argument("--num-nsp", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--setup",
        choices=["code_default_heterogeneous", "lightweight_demo"],
        default="code_default_heterogeneous",
    )
    parser.add_argument("--capacity-mode", choices=["total_equal", "per_provider_paper"], default="total_equal")
    parser.add_argument("--nsp-total-bandwidth", type=float, default=None)
    parser.add_argument("--initial-pE", type=str, default="")
    parser.add_argument("--initial-pN", type=str, default="")
    parser.add_argument("--stage1-iters", type=int, default=8)
    parser.add_argument("--stage2-iters", type=int, default=64)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-price-E", type=float, default=4.0)
    parser.add_argument("--max-price-N", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _parse_vector(raw: str, expected: int, default: list[float]) -> list[float]:
    if not raw.strip():
        return list(default)
    values = [float(part) for part in raw.split(",") if part.strip()]
    if len(values) != expected:
        raise ValueError("Expected %d comma-separated values, got %d." % (expected, len(values)))
    return values


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (ROOT / "outputs" / "multi_provider" / ("run_multi_provider_%s" % stamp))
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    problem = generate_random_problem(
        n_users=args.n_users,
        num_esp=args.num_esp,
        num_nsp=args.num_nsp,
        seed=args.seed,
        capacity_mode=args.capacity_mode,
        setup=args.setup,
        nsp_total_bandwidth=args.nsp_total_bandwidth,
    )
    default_pE = [0.42, 0.50, 0.58] if args.num_esp == 3 and args.setup == "code_default_heterogeneous" else [0.35] * args.num_esp
    default_pN = [0.40, 0.50, 0.62] if args.num_nsp == 3 and args.setup == "code_default_heterogeneous" else [0.35] * args.num_nsp
    initial_pE = _parse_vector(args.initial_pE, args.num_esp, default_pE)
    initial_pN = _parse_vector(args.initial_pN, args.num_nsp, default_pN)
    result = solve_multi_provider_stage1(
        problem,
        initial_pE=initial_pE,
        initial_pN=initial_pN,
        max_iters=args.stage1_iters,
        q=args.q,
        tol=args.tol,
        max_price_E=args.max_price_E,
        max_price_N=args.max_price_N,
        stage2_max_iters=args.stage2_iters,
    )

    _write_csv(out_dir / "multi_provider_trajectory.csv", trajectory_rows(result))
    _write_csv(out_dir / "multi_provider_assignment.csv", assignment_rows(problem, result))
    _write_csv(out_dir / "multi_provider_provider_metrics.csv", provider_metric_rows(problem, result))
    matrix = assignment_matrix(problem, result)

    summary = {
        "n_users": int(args.n_users),
        "num_esp": int(args.num_esp),
        "num_nsp": int(args.num_nsp),
        "seed": int(args.seed),
        "setup": args.setup,
        "capacity_mode": args.capacity_mode,
        "nsp_total_bandwidth": "" if args.nsp_total_bandwidth is None else float(args.nsp_total_bandwidth),
        "initial_pE": [float(x) for x in initial_pE],
        "initial_pN": [float(x) for x in initial_pN],
        "stage1_iters_requested": int(args.stage1_iters),
        "stage1_iters_completed": len(result.trajectory),
        "stage2_calls": int(result.stage2_calls),
        "runtime_sec": float(result.runtime_sec),
        "restricted_gap": float(result.restricted_gap),
        "social_cost": float(result.stage2_result.social_cost),
        "offloading_count": int(result.stage2_result.offloading_count),
        "pE": [float(x) for x in result.pE],
        "pN": [float(x) for x in result.pN],
        "esp_revenue": [float(x) for x in result.esp_revenue],
        "nsp_revenue": [float(x) for x in result.nsp_revenue],
        "assignment_matrix": matrix.tolist(),
        "artifacts": [
            "multi_provider_trajectory.csv",
            "multi_provider_assignment.csv",
            "multi_provider_provider_metrics.csv",
            "multi_provider_summary.json",
            "summary.txt",
            "multi_provider_average_offloading_cost_heatmap.png",
            "multi_provider_provider_revenue.png",
        ],
    }
    (out_dir / "multi_provider_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        "multi_provider_experiment = 3ESP_3NSP_extension",
        "n_users = %d" % args.n_users,
        "num_esp = %d" % args.num_esp,
        "num_nsp = %d" % args.num_nsp,
        "seed = %d" % args.seed,
        "setup = %s" % args.setup,
        "capacity_mode = %s" % args.capacity_mode,
        "nsp_total_bandwidth = %s" % ("" if args.nsp_total_bandwidth is None else ("%.10g" % args.nsp_total_bandwidth)),
        "initial_pE = %s" % ",".join("%.6g" % x for x in initial_pE),
        "initial_pN = %s" % ",".join("%.6g" % x for x in initial_pN),
        "stage1_iters_completed = %d" % len(result.trajectory),
        "stage2_calls = %d" % result.stage2_calls,
        "runtime_sec = %.10g" % result.runtime_sec,
        "restricted_gap = %.10g" % result.restricted_gap,
        "social_cost = %.10g" % result.stage2_result.social_cost,
        "offloading_count = %d" % result.stage2_result.offloading_count,
        "pE = %s" % ",".join("%.6g" % x for x in result.pE),
        "pN = %s" % ",".join("%.6g" % x for x in result.pN),
        "esp_revenue = %s" % ",".join("%.6g" % x for x in result.esp_revenue),
        "nsp_revenue = %s" % ",".join("%.6g" % x for x in result.nsp_revenue),
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()

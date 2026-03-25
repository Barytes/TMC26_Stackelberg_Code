from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from _figure_missing_impl import _load_cfg, _sample_users, _run_stage1_method
from _figure_wrapper_utils import load_csv_rows, write_csv_rows
from tmc26_exp.stackelberg import solve_stage2_scm


FIELDNAMES = [
    "method",
    "n_users",
    "trial",
    "social_cost",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "comp_utilization",
    "band_utilization",
    "final_pE",
    "final_pN",
    "offloading_size",
    "restricted_gap",
]

METHOD_ORDER = ["Full model", "ME", "SingleSP", "Coop", "Rand"]
METHOD_RANK = {name: idx for idx, name in enumerate(METHOD_ORDER)}


def _row_sort_key(row: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(row["n_users"]),
        int(row["trial"]),
        METHOD_RANK.get(str(row["method"]), 999),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute only the Rand rows in an existing E2 CSV.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    backup_path = csv_path.with_name(csv_path.stem + "_before_rand_refresh.csv")
    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)

    cfg = _load_cfg(args.config)
    n_list = [int(x) for x in args.n_users_list.split(",") if x.strip()]

    existing_rows = load_csv_rows(csv_path)
    retained_rows: list[dict[str, object]] = []
    target_keys = {("Rand", int(n), int(trial)) for n in n_list for trial in range(1, int(args.trials) + 1)}

    for row in existing_rows:
        key = (str(row["method"]), int(row["n_users"]), int(row["trial"]))
        if key in target_keys:
            continue
        retained_rows.append(
            {
                "method": str(row["method"]),
                "n_users": int(row["n_users"]),
                "trial": int(row["trial"]),
                "social_cost": float(row["social_cost"]),
                "esp_revenue": float(row["esp_revenue"]),
                "nsp_revenue": float(row["nsp_revenue"]),
                "joint_revenue": float(row["joint_revenue"]),
                "comp_utilization": float(row["comp_utilization"]),
                "band_utilization": float(row["band_utilization"]),
                "final_pE": float(row["final_pE"]),
                "final_pN": float(row["final_pN"]),
                "offloading_size": float(row["offloading_size"]),
                "restricted_gap": float(row["restricted_gap"]),
            }
        )

    refreshed_rows: list[dict[str, object]] = []
    for n in n_list:
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users(cfg, n, int(args.seed), int(trial))
            price, offloading_set, gap, esp_rev, nsp_rev, meta = _run_stage1_method(
                users,
                cfg.system,
                cfg.stackelberg,
                cfg.baselines,
                "Rand",
            )
            stage2 = solve_stage2_scm(
                users,
                float(price[0]),
                float(price[1]),
                cfg.system,
                cfg.stackelberg,
                inner_solver_mode="primal_dual",
            )
            refreshed_rows.append(
                {
                    "method": "Rand",
                    "n_users": int(n),
                    "trial": int(trial),
                    "social_cost": float(meta["social_cost"]),
                    "esp_revenue": float(esp_rev),
                    "nsp_revenue": float(nsp_rev),
                    "joint_revenue": float(esp_rev + nsp_rev),
                    "comp_utilization": float(stage2.inner_result.f.sum() / cfg.system.F),
                    "band_utilization": float(stage2.inner_result.b.sum() / cfg.system.B),
                    "final_pE": float(price[0]),
                    "final_pN": float(price[1]),
                    "offloading_size": int(len(offloading_set)),
                    "restricted_gap": float(gap),
                }
            )

    all_rows = retained_rows + refreshed_rows
    all_rows.sort(key=_row_sort_key)
    write_csv_rows(csv_path, list(FIELDNAMES), all_rows)

    summary_path = csv_path.with_name("E2_rand_recompute_summary.txt")
    summary_path.write_text(
        "\n".join(
            [
                f"config = {args.config}",
                f"csv = {csv_path}",
                f"backup_csv = {backup_path}",
                f"seed = {args.seed}",
                f"n_users_list = {','.join(str(x) for x in n_list)}",
                f"trials = {args.trials}",
                "method = Rand",
                "refresh_mode = recompute_rand_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

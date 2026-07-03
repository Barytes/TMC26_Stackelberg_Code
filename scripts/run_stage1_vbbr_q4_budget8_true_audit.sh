#!/usr/bin/env zsh
set -euo pipefail

export MPLCONFIGDIR=/private/tmp/tmc26_mpl_cache
export TMC26_CACHE_DIR=/private/tmp/tmc26_cache
export UV_CACHE_DIR=/private/tmp/tmc26_uv_cache

OUT_DIR="outputs/2. comp stackelberg baseline/run_stage1_vbbr_q4_budget8_true_grid_ne_gap_vs_users_20260618_n50_250_t10"
OLD_AUDIT="outputs/2. comp stackelberg baseline/run_stage1_final_grid_ne_gap_vs_users_20260608_n50_250_t10/proposed_true_grid_ne_gap_audit_120.csv"
RUN_CSV="${OUT_DIR}/stage1_final_grid_ne_gap_vs_users.csv"
NEW_AUDIT="${OUT_DIR}/proposed_true_grid_ne_gap_audit_120.csv"
COMPARE_CSV="${OUT_DIR}/proposed_true_grid_ne_gap_compare_old_vs_q4_budget8.csv"

uv run python scripts/run_stage1_final_grid_ne_gap_vs_users.py \
  --config configs/stage1_vbbr_q4_budget8.toml \
  --seed 2026 \
  --n-users-list 50,100,150,200,250 \
  --trials 10 \
  --methods Proposed \
  --statistic median_iqr \
  --final-audit-grid-points 120 \
  --out-dir "${OUT_DIR}"

uv run python scripts/audit_proposed_true_grid_ne_gap.py \
  --config configs/stage1_vbbr_q4_budget8.toml \
  --seed 2026 \
  --source-csv "${RUN_CSV}" \
  --out-csv "${NEW_AUDIT}" \
  --audit-grid-points 120

uv run python scripts/compare_proposed_true_grid_ne_gap.py \
  --old-audit-csv "${OLD_AUDIT}" \
  --new-audit-csv "${NEW_AUDIT}" \
  --out-csv "${COMPARE_CSV}"

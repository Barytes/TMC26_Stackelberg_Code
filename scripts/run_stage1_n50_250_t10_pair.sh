#!/usr/bin/env zsh
set -euo pipefail

export MPLCONFIGDIR=/private/tmp/tmc26_mpl_cache
export TMC26_CACHE_DIR=/private/tmp/tmc26_cache
export UV_CACHE_DIR=/private/tmp/tmc26_uv_cache
export UV_PROJECT_ENVIRONMENT=/private/tmp/tmc26_uv_env

uv run python scripts/run_stage1_final_grid_ne_gap_vs_users.py \
  --config configs/default.toml \
  --seed 2026 \
  --n-users-list 50,100,150,200,250 \
  --trials 10 \
  --methods Proposed,GA,BO-online,MARL \
  --statistic median_iqr \
  --final-audit-grid-points 120 \
  --bo-candidate-pool 48 \
  --bo-iters 20 \
  --ga-population-size 12 \
  --ga-generations 8 \
  --marl-price-levels 11 \
  --marl-episodes 60 \
  --marl-steps-per-episode 20 \
  --out-dir "outputs/2. comp stackelberg baseline/run_stage1_final_grid_ne_gap_vs_users_20260608_n50_250_t10"

uv run python scripts/run_stage1_budget_matched_quality_vs_users.py \
  --config configs/default.toml \
  --seed 2026 \
  --n-users-list 50,100,150,200,250 \
  --trials 10 \
  --methods BO-online,GA,MARL \
  --reference-run-dir "outputs/2. comp stackelberg baseline/run_stage1_final_grid_ne_gap_vs_users_20260608_n50_250_t10" \
  --final-audit-grid-points 120 \
  --bo-candidate-pool 48 \
  --bo-iters 20 \
  --ga-population-size 12 \
  --ga-generations 8 \
  --marl-price-levels 11 \
  --marl-episodes 60 \
  --marl-steps-per-episode 20 \
  --out-dir "outputs/2. comp stackelberg baseline/run_stage1_budget_matched_quality_vs_users_20260608_n50_250_t10"

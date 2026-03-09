# Stage I Gap-Alignment Report (hat-G ε vs real-revenue ε)

Date: 2026-03-09 08:42 (Asia/Shanghai)

## Objective
Quantify and visualize the mismatch between:
- `epsilon_hat(pE,pN) = max{hat G_E, hat G_N}` from Algorithm 3 estimator
- `epsilon_real(pE,pN)` from real Stage-II revenue surfaces (best unilateral real-revenue improvement)

on the **same user instance + same (pE, pN) grid**.

## Run Setup
- Script: `scripts/run_stage1_gap_alignment.py`
- Output folder: `outputs/stage1_gap_alignment_20260309_084228`
- Config base: `configs/default.toml`
- Stage-II solver: `DG`
- Users: `n=40` (single sampled instance), seed `20260309`
- Grid: `21 x 21`
- Domain: `pE in [0.01, 6.0]`, `pN in [0.01, 6.0]`

## Produced Artifacts
- Surfaces (CSV):
  - `epsilon_real.csv`
  - `epsilon_hat_boundary.csv`
  - `epsilon_hat_refined_price.csv`
  - `delta_boundary_minus_real.csv`
  - `delta_refined_minus_real.csv`
- Figures:
  - `landscape_alignment_boundary.png`
  - `landscape_alignment_refined_price.png`
- Metrics:
  - `summary_metrics.csv`
  - `metadata.json`

Each map marks:
- **red star**: unified SE proxy point (argmin of `epsilon_real`)
- **black x**: argmin of `epsilon_hat` for that estimator variant

## Quantitative Alignment Summary
Unified SE proxy (from real ε landscape):
- `(pE, pN) = (1.208, 1.5075)`
- `epsilon_real = 0.0`

### boundary variant
- MAE: `133.2543`
- RMSE: `157.6427`
- Max |diff|: `305.1474`
- argmin(`epsilon_hat_boundary`): `(6.0, 0.609)`
- Distance to real argmin:
  - grid-index distance: `16.2788`
  - price-space distance: `4.8755`

### refined_price variant
- MAE: `285.1082`
- RMSE: `289.2278`
- Max |diff|: `393.7816`
- argmin(`epsilon_hat_refined_price`): `(1.5075, 3.6040)`
- Distance to real argmin:
  - grid-index distance: `7.0711`
  - price-space distance: `2.1178`

## Main Findings
1. **Strong scale mismatch**: both `epsilon_hat` variants are much larger than `epsilon_real` over most of the grid.
2. **Landscape mismatch exists in both variants**:
   - `boundary` has lower global error (MAE/RMSE) but argmin is far from real proxy.
   - `refined_price` has closer argmin location than boundary, but much worse global error.
3. **Estimator variant tradeoff**: lower average calibration error does not guarantee better minimizer alignment, and vice versa.

## Implications for Algorithm 5 Convergence
- Algorithm 5 uses `hat-G` as the search signal. If `epsilon_hat` landscape is misaligned from real exploitability (`epsilon_real`), then:
  1. search can converge to a point that is locally good under surrogate but suboptimal under real revenue gap;
  2. stopping criteria based only on surrogate ε can be over-conservative (if overestimated) or directionally biased;
  3. local refinement / occasional real-gap validation checkpoints are likely needed to reduce surrogate-induced drift.

Suggested practical follow-up:
- Introduce sparse real-gap anchor evaluations during Stage-I iterations (e.g., every K iterations or near candidate minima), and re-rank final candidates by real ε.

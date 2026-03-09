# Stage-I `topk_brd` Variant (ESP/NSP Alternating Best-Response)

## What was added

A new Stage-I solver variant `topk_brd` is added in `src/tmc26_exp/stackelberg.py`:

- Function: `algorithm_topk_brd_stage1(...)`
- Entry selector: `run_stage1_solver(...)`
- Config switch: `stackelberg.stage1_solver_variant = "topk_brd"`

It performs per-iteration updates:

1. Given current price `p=(pE,pN)`, compute ESP best response using Algorithm-3 with `estimator_variant="topk_real_reval"`.
2. Update `pE` via ESP boundary best-response price on selected candidate set.
3. Compute NSP best response similarly and update `pN`.
4. Re-evaluate epsilon at updated `(pE,pN)` and iterate.

## Top-k estimator reuse

`algorithm_3_gain_approximation(...)` now accepts `top_k` (default `4`), and `topk_brd` uses:

- `stackelberg.gain_topk_k` (default: `4`)

This keeps existing estimator behavior intact while allowing controlled top-k re-evaluation.

## Diagnostics

Per iteration trajectory includes:

- `pE`, `pN`, `epsilon`
- selected BR candidate info proxy:
  - `esp_best_set_size`, `nsp_best_set_size`
  - `esp_gain`, `nsp_gain`
- final stop reason in result summary (`stopping_reason`)

`stackelberg_trajectory.csv` columns were extended accordingly.

## New config flags (variant-focused)

Under `[stackelberg]`:

- `stage1_solver_variant = "algorithm5" | "topk_brd"` (default: `algorithm5`)
- `gain_topk_k = 4`
- `topk_brd_price_tol = 1e-6`
- `topk_brd_epsilon_tol = 1e-7`
- `topk_brd_cycle_window = 6`

## Comparison script

Added:

- `scripts/run_stage1_topk_brd_comparison.py`

Run:

```bash
python scripts/run_stage1_topk_brd_comparison.py --config configs/stage1_fast_diag.toml --seed 20260309
```

Outputs in `outputs/stage1_topk_brd_comparison/`:

- `trajectory_overlay.png`
- `epsilon_vs_iteration.png`
- `summary.csv` (steps, final epsilon, dist-to-SE proxy, runtime, stop reason)

## Interpretation notes

- Lower `epsilon` indicates better approximate stability.
- Smaller distance to SE proxy means trajectory endpoint is closer to dense-grid oracle reference.
- `topk_brd` is expected to be lighter/greedy in price BR updates; Algorithm 5 may still be stronger on some instances due to broader neighborhood checks.

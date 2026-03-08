# Stage-I convergence fix validation (2026-03-09)

## Scope
Complete the remaining Stage-I convergence milestone from SPEC:
- patch Algorithm 5 to avoid premature 1-2 step termination when set-switch improvement is unavailable;
- run fresh convergence experiments;
- produce commit-ready outputs.

## Code patch
File:
- `src/tmc26_exp/stackelberg.py`

Changes:
1. Added `_best_local_price_on_fixed_set(...)` to run boundary-guided local price updates on the current offloading set.
2. In `algorithm_5_stackelberg_guided_search(...)`, when no improving set candidate exists:
   - try local fixed-set price improvement first;
   - continue search if epsilon improves;
   - use a small plateau tolerance window (`no_improve_rounds >= 3`) before stopping.
3. Limited true-SE proxy computation to small instances (`n <= 16`) to prevent unnecessary heavy oracle calls in medium/large-scale Stage-I runs.

## Fresh validation run
Command:

```bash
uv run python scripts/run_stage1_deviation_gap_convergence.py \
  --config configs/stage1_fast_diag.toml \
  --n-users 20,30 \
  --trials 2 \
  --seed 20260309 \
  --run-name fig5_stage1_convergence_fixcheck_v3
```

Output directory:
- `outputs/fig5_stage1_convergence_fixcheck_v3/`

Generated artifacts:
- `raw_stage1_convergence.csv`
- `summary_stage1_convergence.csv`
- `stage1_deviation_gap_convergence.png`
- `run_meta.txt`

## Key results (from fresh run)
- n=20, trial0: 9 iterations, epsilon `318.3568 -> 6.07e-10`
- n=20, trial1: 9 iterations, epsilon `585.1669 -> 3.57e-10`
- n=30, trial0: 9 iterations, epsilon `823.3632 -> 1.17e-09`
- n=30, trial1: 5 iterations, epsilon `371.4360 -> 0.0`

Conclusion: Stage-I convergence trajectory is now sustained beyond the previous short-stop behavior and reaches near-zero final deviation gap on fresh seeds.

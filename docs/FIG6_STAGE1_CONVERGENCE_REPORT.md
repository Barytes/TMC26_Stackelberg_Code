# FIG6 & Stage-I Convergence Report

## Scope
This report covers:
1. Fig.6 enhancement (true-SE marker + baseline trajectories + CSV exports)
2. Root-cause diagnosis of short Algorithm-5 runs
3. Stage-I algorithm changes and A/B convergence diagnostics

---

## (a) Why only ~2 iterations happened before

### Root cause
In the original Algorithm 5 loop, termination is triggered as soon as **no candidate offloading set gives immediate epsilon improvement** under strict criterion:

- improve only if `cand_eps + tol < best_eps`
- otherwise stop with `no_improving_candidate`

This means the loop is effectively a **set-switch local search**, not a sustained price-trajectory process. If set changes stop helping, the algorithm exits quickly even when price is still far from the true Stackelberg point.

### Evidence
Previous Fig6 trajectory exports (already in repo) show very short trajectories and stop behavior:
- `outputs/fig6_boundary_final/trajectory_trial0.csv`

New diagnostics (minimal-patch A) still show short stopping on this seed:
- `outputs/stage1_convergence_diag_ab/variant_a_minimal.csv` (2 steps)
- `outputs/stage1_convergence_diag_ab/ab_summary.txt` (`variant_a_stopping_reason = no_improving_candidate`)

---

## (b) What changed

## 1) Figure 6 enhancements implemented
File changed:
- `scripts/run_stage1_boundary_visualization.py`

Enhancements:
- Explicit **true Stackelberg equilibrium marker** using grid-search oracle baseline (`X` marker)
- Overlay **multiple Stage-I baseline trajectories** (not only endpoints):
  - PBRD trajectory
  - BO trajectory (best-so-far path)
  - DRL trajectory (state walk path)
- Export per-method trajectory CSVs:
  - `baseline_trajectory_pbdr_trial*.csv`
  - `baseline_trajectory_bo_trial*.csv`
  - `baseline_trajectory_drl_trial*.csv`
- Extended Algorithm-5 trajectory CSV fields:
  - `iteration,pE,pN,epsilon,dist_to_se,epsilon_delta`
- Added diagnostics text export with:
  - iteration count
  - stopping reason
  - epsilon progression
  - distance-to-SE progression

## 2) Stage-I algorithm patch (A: theory-preserving minimal patch)
File changed:
- `src/tmc26_exp/stackelberg.py`

Changes:
- Added fixed-set damped BR price refinement helper:
  - `_refine_price_for_fixed_set(...)`
- Integrated refinement into outer loop before gain evaluation
- Added richer trajectory/log fields:
  - `SearchStep.dist_to_se`, `SearchStep.epsilon_delta`
  - `StackelbergResult.stopping_reason`

### Theory consistency risk note
- This keeps the original Algorithm-5 structure (deviation-target first, neighborhood fallback, improvement-based stopping), but adds a local price-refinement operator.
- This is **not** a formal proof extension in current codebase; refinement is heuristic and may alter proof assumptions tied to exact original update sequence.

## 3) Stage-I variant (B: stronger but less theory-aligned)
File added:
- `scripts/run_stage1_convergence_diagnostics.py`

Variant B implemented in diagnostics script:
- Synchronize with Stage-II each iteration
- Continue for multiple iterations with plateau stopping

This variant improves trajectory length but can be unstable in epsilon objective.

---

## (c) Evidence of improved convergence / diagnostics

## Enhanced Fig6 outputs
Run (fast diagnostic config):
- `configs/stage1_fast_diag.toml`
- output dir: `outputs/fig6_stage1_enhanced_convergence/`

Key artifacts:
- Figure: `outputs/fig6_stage1_enhanced_convergence/boundary_visualization_trial0.png`
- Algo5 trajectory: `outputs/fig6_stage1_enhanced_convergence/trajectory_trial0.csv`
- Baseline trajectories:
  - `.../baseline_trajectory_pbdr_trial0.csv`
  - `.../baseline_trajectory_bo_trial0.csv`
  - `.../baseline_trajectory_drl_trial0.csv`
- Diagnostics: `outputs/fig6_stage1_enhanced_convergence/algorithm5_diagnostics_trial0.txt`

## A/B convergence comparison
Artifacts:
- `outputs/stage1_convergence_diag_ab/ab_summary.txt`
- `outputs/stage1_convergence_diag_ab/variant_a_minimal.csv`
- `outputs/stage1_convergence_diag_ab/variant_b_sync.csv`

Summary table (seed=20260002, fast diag config):

| Variant | Steps | Final epsilon | Dist to true SE (grid oracle) | Stop reason |
|---|---:|---:|---:|---|
| A (minimal patch) | 2 | 0.0 | 10.7523 | no_improving_candidate |
| B (sync variant) | 12 | 581.993 | 5.4313 | max_iters |

Interpretation:
- A achieves very low epsilon quickly but can be far from true-SE proxy.
- B explores longer and gets closer in price-space distance, but epsilon can worsen dramatically.

This confirms an objective mismatch / non-monotonicity issue under current formulation.

---

## (d) Remaining limitations

1. **Objective mismatch**: epsilon proxy (max gain wrt candidate family/boundary) is not guaranteed to align with global true-SE price from grid oracle.
2. **No monotonic global Lyapunov** in current implementation; variants can trade off epsilon vs price distance.
3. **True-SE computation is expensive** (grid-search oracle), so sustained large-scale verification is costly.
4. **Formal theorem compatibility** of added price-refinement and sync-style updates is not established in this codebase.

---

## Recommendation
- Keep Variant A as default (minimal perturbation, stable runtime, low epsilon behavior).
- Use Variant B only for exploratory analysis, not as default algorithm.
- For publication-grade “converges to true Stackelberg equilibrium” claim, current objective/algorithm needs a stronger unified potential or bilevel-consistent stopping condition.

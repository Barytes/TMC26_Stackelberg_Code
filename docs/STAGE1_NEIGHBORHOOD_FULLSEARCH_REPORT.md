# Stage I Neighborhood Full-Search Exploration Report

Date: 2026-03-09

## Scope

This report covers requested Stage I explorations:

1. Audit Algorithm 3 deviation-gap approximation errors.
2. Implement Algorithm 5 exploratory variant that searches full neighborhood `N(p)` (with safe compute cap).
3. Add at least one improved Algorithm 3 estimator variant.
4. Compare baseline vs new variants on iterations / final epsilon / distance-to-true-SE proxy / runtime.
5. Investigate and fix Figure 6 heatmap-domain / overlay mismatch issue.

---

## What was wrong

## 1) Algorithm 5 candidate selection logic

Previous behavior in `algorithm_5_stackelberg_guided_search`:
- It first checked two candidates from Algorithm 3 (ESP/NSP best-set estimates).
- It only scanned the rest of `N(p)` if those two failed to improve.

Risk:
- If one of those two improves *somewhat*, search could stop early without finding the true minimum-gap set in `N(p)`.

## 2) Algorithm 3 gain approximation error source

Algorithm 3 used boundary revenue (`_boundary_revenue_for_provider`) as proxy gain.

Audit indicates large absolute errors in some states, especially for NSP-side gains in larger instances. Main source is **revenue model mismatch**, not only family-miss:
- `N(p)` hit rate was moderate (~0.625 in this run),
- but biggest errors occurred even when exact best set was inside family.

This means candidate-family coverage alone is not enough; gain evaluator itself needs calibration.

## 3) Figure 6 visualization mismatch

Observed issue root cause:
- heatmap domain from contour CSV was finite (example run: `pE,pN in [0.05, 5.0]`),
- overlay trajectories/markers could include points outside domain (example: DRL endpoint had `pE=0.01`),
- matplotlib autoscaling expanded axes to include out-of-domain markers, visually compressing heatmap into a corner and making equilibrium marker appear detached.

---

## What changed

## A) Algorithm 5: full-neighborhood mode + safe cap

Updated `StackelbergConfig` with:
- `stage1_neighborhood_mode`: `two_stage` (default) or `full_search`
- `stage1_neighborhood_max_candidates`: positive integer cap (default 256)

Updated search loop:
- In `full_search`, evaluate all sets in `N(p)` (excluding current set), then pick minimum-epsilon candidate.
- In `two_stage`, keep legacy order preference but still under unified candidate-evaluation path.
- All candidate evaluations respect `stage1_neighborhood_max_candidates`.

## B) Algorithm 3: improved estimator variant

Added estimator selector to Algorithm 3:
- `boundary` (legacy default)
- `refined_price` (new calibration variant)

`refined_price` method:
- For each candidate set `Y`, refine prices on `Y` via `_refine_price_for_fixed_set`,
- Evaluate provider revenue at refined candidate prices,
- Use that as candidate revenue estimate.

This reduces approximation error relative to legacy boundary-only proxy in the audit run.

Config field added:
- `gain_estimator_variant`: `boundary` (default) or `refined_price`

## C) Figure 6 domain/scale fix

In `scripts/run_stage1_boundary_visualization.py`:
- Added explicit axis locking to heatmap domain:
  - `ax.set_xlim(min(pE_grid), max(pE_grid))`
  - `ax.set_ylim(min(pN_grid), max(pN_grid))`
- Added domain check helper `_inside_domain(...)`.
- Out-of-domain baseline markers are **not plotted** on heatmap axes; they are recorded and annotated.
- Added domain annotation text to figure footer.
- Returned and persisted heatmap-domain diagnostics in JSON summary.

Result: heatmap and overlays are now aligned/readable; no corner-compression artifact.

---

## Before/After metrics (exploration run)

Run artifact:
- `outputs/stage1_fullsearch_explore_fast_20260309/`
- Config: `configs/stage1_fast_diag.toml`
- Users: `n={8,12}`
- Trials: `2` each variant

## Algorithm 3 audit summary

From `audit_summary.csv`:

- boundary:
  - mean abs error: **94.87**
  - p90 abs error: **199.42**
  - family hit rate: **0.625**

- refined_price:
  - mean abs error: **79.67**
  - p90 abs error: **169.59**
  - family hit rate: **0.625**

Interpretation:
- `refined_price` reduced mean absolute error by ~**16.0%** vs boundary proxy.

## Algorithm 5 variant comparison

From `comparison_summary.csv`:

- baseline_two_stage_boundary
  - iterations: 8.5
  - final epsilon: 9.19e-10
  - dist to true-SE proxy: 3.694
  - runtime: 28.54s

- full_search_boundary
  - iterations: 8.5
  - final epsilon: 9.19e-10
  - dist to true-SE proxy: 3.694
  - runtime: 27.54s

- full_search_refined
  - iterations: 8.5
  - final epsilon: 6.43e-10
  - dist to true-SE proxy: 3.694
  - runtime: 29.02s

Interpretation:
- On this small diagnostic batch, full-search did **not** materially change iterations or SE-distance.
- `refined_price` gave a slightly lower final epsilon (same order of magnitude), with small runtime overhead.

---

## Figure 6 fix output

Corrected visualization run:
- `outputs/stage1_boundary_fix_20260309/boundary_visualization_trial0.png`
- Diagnostics:
  - `outputs/stage1_boundary_fix_20260309/boundary_summary_trial0.json`

Recorded domain diagnostics (example):
- heatmap domain: `pE=[0.05,5.0], pN=[0.05,5.0]`
- out-of-domain overlay detected/hidden: `DRL=(0.01,0.609)`

---

## Recommended default

For default reproducible pipeline:
- Keep `stage1_neighborhood_mode = "two_stage"` (current behavior is stable and not worse in this batch).
- Prefer `gain_estimator_variant = "refined_price"` when compute budget allows, because it improves Algorithm 3 gain-estimation fidelity.

If doing targeted robustness sweeps:
- Use `stage1_neighborhood_mode = "full_search"` with bounded `stage1_neighborhood_max_candidates`.

---

## Risks / theory consistency notes

1. `refined_price` estimator changes Algorithm 3 from pure boundary approximation to calibrated heuristic.
   - Better empirical fidelity, but strict theorem statements tied to boundary-only approximation should be worded carefully.

2. Full-neighborhood search increases per-iteration cost with neighborhood size.
   - Safe cap (`stage1_neighborhood_max_candidates`) is important for scalability.

3. Figure 6 now clips/hides out-of-domain markers by design.
   - This preserves heatmap readability; out-of-domain points are still disclosed in annotations/JSON diagnostics.

---

## Changed files

- `src/tmc26_exp/config.py`
- `src/tmc26_exp/stackelberg.py`
- `scripts/run_stage1_neighborhood_fullsearch_exploration.py` (new)
- `scripts/run_stage1_boundary_visualization.py`
- `docs/STAGE1_NEIGHBORHOOD_FULLSEARCH_REPORT.md` (this report)

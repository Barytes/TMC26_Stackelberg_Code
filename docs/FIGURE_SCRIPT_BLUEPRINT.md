# Figure Script Blueprint

## 1. Purpose

This document is the pre-workflow-5 figure blueprint.

It freezes three things:

- the canonical figure list for Block A-F;
- the one-figure-one-script rule;
- the figure-level experiment specification for each planned script.

The goal is to make workflow 5 implementation mechanical: each figure below should map to one canonical script, one primary output image, one CSV schema, and one summary file.

Current implementation snapshot:

- all blueprint figure IDs A1-F4 now exist as direct, runnable `scripts/run_figure_*.py` entries;
- legacy multi-output scripts remain as reusable implementation references, not public wrapper layers.

---

## 2. Figure Rules

### 2.1 One Figure, One Script

From this document onward, the canonical rule is:

- each main figure corresponds to exactly one canonical script;
- if a current script already produces the right figure, reuse its plot form;
- if a current script previously produced multiple figures, workflow 5 has already split it into figure-specific direct runners rather than keeping one multi-purpose public entry.

Canonical naming convention:

- `scripts/run_figure_<FigureID>_<short_name>.py`

Examples:

- `scripts/run_figure_A1_stage2_social_cost_trace.py`
- `scripts/run_figure_B4_nsp_slice_boundary_comparison.py`

### 2.2 Reuse Policy

If an existing script already has a suitable figure form:

- keep the figure form;
- keep the current plotting helper logic where possible;
- move or wrap it behind a figure-specific script if the current script emits multiple figures.

### 2.3 Base Scenario

Unless a figure explicitly overrides a parameter, use a paper-aligned base scenario rather than blindly inheriting `configs/default.toml`.

Paper-aligned base scenario:

- user count:
  - representative geometry / convergence figure: `n = 12`
  - small exact regime: `n in {8, 12, 16}`
  - medium / large statistical regime: `n in {20, 40, 60, 80, 100}`
- default user model:
  - `rho_i ~ U[0.1, 2]`
  - `d_i ~ U[1, 5] Mbits`
  - `w_i ~ U[0.1, 1.0] Gcycles`
  - `f_i^l in {0.5, 0.8, 1.0, 1.2} GHz`
  - `alpha_i ~ U[1, 2]`
  - `beta_i ~ U[0.1, 0.5]`
  - `kappa_i = 1e-27`
  - `varpi_i = 1 / 0.35`
- system parameters:
  - `F = 20 GHz`
  - `B = 50 MHz`
  - `c_E = c_N = 0.1`
- representative seed: `2026`
- statistical seed list:
  - `2026, 2027, ..., 2045`

Implementation note:

- when current code defaults differed from the paper defaults, workflow 5 created figure-specific config files under `configs/figures/`, and the canonical scripts now use those configs.

### 2.4 Repeat Policy

Use these default repeat policies unless a figure overrides them:

- representative geometry figures:
  - `1` instance, seed `2026`
- small exact statistical figures:
  - `50` trials per point
- medium / large statistical figures:
  - `20` trials per point
- heavy runtime figures:
  - `10` trials per point

### 2.5 Output Convention

Each canonical figure script should write:

- one primary image:
  - `<figure_id>_<short_name>.png`
- one primary CSV:
  - `<figure_id>_<short_name>.csv`
- one summary text file:
  - `<figure_id>_<short_name>_summary.txt`
- all outputs, including temporary smoke-test outputs, must be written under the repository `outputs/` directory
- if `--out-dir` is provided, it must still resolve inside `outputs/`

---

## Paper Subsection Layout

### Convergence of the Proposed Algorithms

- `A1`
  `Convergence Of Algorithm 2 Across User Scales`

### Comparison with Baselines for Stackelberg Equilibrium

### Impact of Alternative Strategic Settings

### Sensitivity and Scalability Analysis

---

## 3. Figure Index

| ID | Block | Canonical script | Current reusable source | Status |
| --- | --- | --- | --- | --- |
| A1 | A | `scripts/run_figure_A1_stage2_social_cost_trace.py` | `scripts/run_stage2_social_cost_compare.py` | direct runner |
| A3 | A | `scripts/run_figure_A3_stage2_approx_ratio_bound.py` | `scripts/run_stage2_approximation_ratio.py` | direct runner |
| A4 | A | `scripts/run_figure_A4_stage2_runtime_vs_users.py` | none | direct runner |
| A5 | A | `scripts/run_figure_A5_stage2_rollback_diagnostics.py` | none | direct runner |
| A6 | A | `scripts/run_figure_A6_stage2_exploitability_supp.py` | `scripts/run_algorithm2_exploitability_vs_users.py` | direct runner |
| B1 | B | `scripts/run_figure_B1_stage1_joint_revenue_heatmap.py` | `scripts/run_stage1_price_heatmaps.py` | direct runner |
| B2 | B | `scripts/run_figure_B2_stage1_restricted_gap_heatmap.py` | `scripts/run_stage1_price_heatmaps.py` | direct runner |
| B3 | B | `scripts/run_figure_B3_esp_slice_boundary_comparison.py` | `scripts/run_boundary_hypothesis_check.py` | direct runner |
| B4 | B | `scripts/run_figure_B4_nsp_slice_boundary_comparison.py` | `scripts/run_boundary_hypothesis_check.py` | direct runner |
| B5 | B | `scripts/run_figure_B5_joint_revenue_boundary_overlay.py` | `scripts/run_boundary_hypothesis_check.py` | direct runner |
| B6 | B | `scripts/run_figure_B6_candidate_family_diagnostics.py` | none | direct runner |
| C1 | C | `scripts/run_figure_C1_restricted_gap_trajectory.py` | none | direct runner |
| C2 | C | `scripts/run_figure_C2_best_response_gain_trajectory.py` | none | direct runner |
| C3 | C | `scripts/run_figure_C3_price_trajectory_on_gap_heatmap.py` | `scripts/run_stage1_vbbr_trajectory_on_heatmap.py` | direct runner |
| C4 | C | `scripts/run_figure_C4_final_gap_vs_budget.py` | none | direct runner |
| C5 | C | `scripts/run_figure_C5_trajectory_compare_supp.py` | `scripts/plot_stage1_trajectories_on_heatmap.py` | direct runner |
| D1 | D | `scripts/run_figure_D1_stage2_runtime_vs_users.py` | none | direct runner |
| D2 | D | `scripts/run_figure_D2_stage1_runtime_vs_users.py` | none | direct runner |
| D3 | D | `scripts/run_figure_D3_stage2_calls_inside_stage1.py` | none | direct runner |
| D4 | D | `scripts/run_figure_D4_exact_runtime_feasibility.py` | none | direct runner |
| E1 | E | `scripts/run_figure_E1_user_social_cost_compare.py` | none | direct runner |
| E2 | E | `scripts/run_figure_E2_provider_revenue_compare.py` | none | direct runner |
| E3 | E | `scripts/run_figure_E3_resource_utilization_compare.py` | none | direct runner |
| E4 | E | `scripts/run_figure_E4_price_and_offloading_compare.py` | none | direct runner |
| F1 | F | `scripts/run_figure_F1_q_sensitivity.py` | none | direct runner |
| F2 | F | `scripts/run_figure_F2_resource_asymmetry_sensitivity.py` | none | direct runner |
| F3 | F | `scripts/run_figure_F3_provider_cost_sensitivity.py` | none | direct runner |
| F4 | F | `scripts/run_figure_F4_user_distribution_sensitivity.py` | none | direct runner |

---

## 4. Block A: Stage II SCM Quality

### A1. Convergence Of Algorithm 2 Across User Scales

- Canonical script:
  `scripts/run_figure_A1_stage2_social_cost_trace.py`
- Reuse source:
  `scripts/run_stage2_social_cost_compare.py`
- Plot type:
  single-panel multi-line plot with one Algorithm-2 trace per user scale and one horizontal centralized-reference line per user scale
- X axis:
  `Iterations`
- Y axis:
  `Social Cost`
- Methods:
  proposed Stage II SCM solver (`DG` / Algorithm 2)
- Baseline / reference:
  centralized Stage II reference solved by `pyomo_scip`, drawn as dashed horizontal lines
- Repeats:
  `1` representative instance per `n`
- Base parameters:
  use `configs/default.toml`, seed `2026`, fixed prices `p_E = 0.5`, `p_N = 0.5`, `n in {50, 100, 150, 200}`
- Key overrides:
  `inner_solver_mode = primal_dual`, `centralized_solver = pyomo_scip`, `max_iter_plot = 12`
- Output images:
  primary English image `A1_stage2_social_cost_trace.png`
  secondary Chinese image `A1_stage2_social_cost_trace_zh.png`
- Primary CSV:
  `A1_stage2_social_cost_trace.csv`
- Summary file:
  `A1_stage2_social_cost_trace_summary.txt`
- Language rule:
  the script always emits one English figure and one Chinese figure from the same experiment results; the primary image language is controlled by `--lang`, and the other language is emitted as the secondary image
- Expected takeaway:
  across `n = 50, 100, 150, 200`, Algorithm 2 social cost should decrease over iterations and approach the corresponding centralized reference level.

### A3. Empirical Approximation Ratio Versus Theorem Bound

- Canonical script:
  `scripts/run_figure_A3_stage2_approx_ratio_bound.py`
- Reuse source:
  `scripts/run_stage2_approximation_ratio.py`
- Plot type:
  scatter plot
- X axis:
  theorem upper bound
- Y axis:
  empirical ratio `V(X_DG) / V(X*)`
- Methods:
  proposed Stage II SCM solver
- Baseline / reference:
  centralized exact Stage II reference provides `V(X*)`
- Repeats:
  `25` trials per `n`
- Base parameters:
  use `configs/default.toml`, `n in {50, 100, 150, 200}`, seeds `2026..2050`, fixed prices `p_E = 0.5`, `p_N = 0.5`
- Key overrides:
  use `linear` scatter as the canonical main-text figure; `centralized_solver = pyomo_scip`; log transforms are supplementary only
- Primary output:
  `A3_stage2_approx_ratio_bound.png`
- Expected takeaway:
  most points lie below the `y = x` line and the theorem bound is empirically conservative.

### A4. Stage II Runtime Versus Number Of Users

- Canonical script:
  `scripts/run_figure_A4_stage2_runtime_vs_users.py`
- Reuse source:
  none
- Plot type:
  line plot with mean ± std error bars
- X axis:
  number of users
- Y axis:
  runtime in seconds
- Methods:
  proposed Stage II SCM solver
- Baseline / reference:
  centralized exact Stage II reference where feasible
- Repeats:
  `10` trials per `n`
- Base parameters:
  `n in {8, 12, 16, 20, 40, 60, 80, 100}`, seeds `2026..2035`, fixed prices `p_E = 0.5`, `p_N = 0.5`
- Key overrides:
  exact runtime only plotted for `n <= 16`
- Primary output:
  `A4_stage2_runtime_vs_users.png`
- Expected takeaway:
  the proposed Stage II runtime scales much better than exact reference solving.

### A5. Stage II Rollback Diagnostics

- Canonical script:
  `scripts/run_figure_A5_stage2_rollback_diagnostics.py`
- Reuse source:
  none
- Plot type:
  three-panel line plot
- X axis:
  number of users
- Y axes:
  panel 1: average rollback count
  panel 2: average accepted admissions
  panel 3: average final offloading-set size
- Methods:
  proposed Stage II SCM solver
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`, fixed prices `p_E = 0.5`, `p_N = 0.5`
- Primary output:
  `A5_stage2_rollback_diagnostics.png`
- Expected takeaway:
  rollback is active but controlled, and accepted admissions remain interpretable as scale grows.

### A6. Supplementary Exploitability Diagnostic

- Canonical script:
  `scripts/run_figure_A6_stage2_exploitability_supp.py`
- Reuse source:
  `scripts/run_algorithm2_exploitability_vs_users.py`
- Plot type:
  mean ± std error-bar plot
- X axis:
  number of users
- Y axis:
  average exploitability
- Methods:
  proposed Stage II SCM solver
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`, fixed prices `p_E = 0.5`, `p_N = 0.5`
- Status:
  supplementary only
- Primary output:
  `A6_stage2_exploitability_supp.png`

---

## 5. Block B: Stage I Geometry And Boundary Diagnostics

### B1. Joint Revenue Heatmap

- Canonical script:
  `scripts/run_figure_B1_stage1_joint_revenue_heatmap.py`
- Reuse source:
  `scripts/run_stage1_price_heatmaps.py`
- Plot type:
  2D heatmap
- X axis:
  `p_E`
- Y axis:
  `p_N`
- Color:
  joint provider revenue `R_E + R_N`
- Methods:
  proposed full model
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, price grid `81 x 81`, price range `[c_E, 6.0] x [c_N, 6.0]`
- Overlay:
  low-gap set marker and representative point
- Primary output:
  `B1_stage1_joint_revenue_heatmap.png`
- Expected takeaway:
  profitable pricing regions are structured and not diffuse over the whole plane.

### B2. Restricted-Gap Heatmap

- Canonical script:
  `scripts/run_figure_B2_stage1_restricted_gap_heatmap.py`
- Reuse source:
  `scripts/run_stage1_price_heatmaps.py`
- Plot type:
  2D heatmap
- X axis:
  `p_E`
- Y axis:
  `p_N`
- Color:
  boundary-price-restricted NE gap
- Methods:
  proposed full model
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, price grid `81 x 81`, price range `[c_E, 6.0] x [c_N, 6.0]`
- Overlay:
  low-gap set marker
- Primary output:
  `B2_stage1_restricted_gap_heatmap.png`
- Expected takeaway:
  small-gap regions are localized, which motivates structured search instead of dense brute force.

### B3. ESP Slice Boundary Comparison

- Canonical script:
  `scripts/run_figure_B3_esp_slice_boundary_comparison.py`
- Reuse source:
  `scripts/run_boundary_hypothesis_check.py`
- Plot type:
  1D line plot with vertical boundary markers
- X axis:
  `p_E`
- Y axis:
  ESP revenue under fixed `p_N`
- Methods:
  proposed full model
- Boundary markers:
  old candidate boundaries, exact switching prices, hypothesis boundaries
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, fixed `p_N = p_N^{start}` from the representative Stage I run
- Primary output:
  `B3_esp_slice_boundary_comparison.png`
- Expected takeaway:
  relevant price changes cluster around switching or boundary prices rather than arbitrary interior points.

### B4. NSP Slice Boundary Comparison

- Canonical script:
  `scripts/run_figure_B4_nsp_slice_boundary_comparison.py`
- Reuse source:
  `scripts/run_boundary_hypothesis_check.py`
- Plot type:
  1D line plot with vertical boundary markers
- X axis:
  `p_N`
- Y axis:
  NSP revenue under fixed `p_E`
- Methods:
  proposed full model
- Boundary markers:
  old candidate boundaries, exact switching prices, hypothesis boundaries
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, fixed `p_E = p_E^{start}` from the representative Stage I run
- Primary output:
  `B4_nsp_slice_boundary_comparison.png`
- Expected takeaway:
  NSP-side geometry exhibits the same switching-set concentration predicted by the theory.

### B5. Joint-Revenue Boundary Overlay

- Canonical script:
  `scripts/run_figure_B5_joint_revenue_boundary_overlay.py`
- Reuse source:
  `scripts/run_boundary_hypothesis_check.py`
- Plot type:
  2D heatmap with overlaid curves / points
- X axis:
  `p_E`
- Y axis:
  `p_N`
- Color:
  joint provider revenue
- Overlay:
  old boundaries, exact switching prices, hypothesis boundaries, representative start point
- Methods:
  proposed full model
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, price grid `81 x 81`, price range `[c_E, 6.0] x [c_N, 6.0]`
- Primary output:
  `B5_joint_revenue_boundary_overlay.png`
- Expected takeaway:
  the boundary-price construction tracks the regime-change geometry well enough to guide search.

### B6. Local Candidate-Family Diagnostics

- Canonical script:
  `scripts/run_figure_B6_candidate_family_diagnostics.py`
- Reuse source:
  none
- Plot type:
  two-panel figure
- X axes:
  panel 1: candidate rank sorted by surrogate score
  panel 2: candidate operation type (`drop`, `add`, `swap`, `mixed`)
- Y axes:
  panel 1: surrogate provider revenue or estimated gain
  panel 2: candidate count
- Methods:
  proposed full model
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`, anchor price = current Stage I iterate, `Q` as in the paper-aligned config
- Primary output:
  `B6_candidate_family_diagnostics.png`
- Expected takeaway:
  the local family is structured, small enough to search, and concentrated on interpretable add/drop changes.

---

## 6. Block C: Stage I Iterative-Pricing Convergence

### C1. Restricted-Gap Trajectory

- Canonical script:
  `scripts/run_figure_C1_restricted_gap_trajectory.py`
- Reuse source:
  none
- Plot type:
  line plot
- X axis:
  Stage I outer iteration
- Y axis:
  boundary-price-restricted NE gap
- Methods:
  proposed Stage I solver
- Repeats:
  `20` trials
- Base parameters:
  `n = 20`, seeds `2026..2045`, `stage1_solver_variant = vbbr_brd`
- Summary statistic:
  mean ± std across trials
- Primary output:
  `C1_restricted_gap_trajectory.png`
- Expected takeaway:
  restricted gap decreases quickly and stabilizes near zero or a small tolerance.

### C2. Best-Response Gain Trajectories

- Canonical script:
  `scripts/run_figure_C2_best_response_gain_trajectory.py`
- Reuse source:
  none
- Plot type:
  two-line plot
- X axis:
  Stage I outer iteration
- Y axis:
  unilateral gain
- Methods:
  proposed Stage I solver
- Lines:
  ESP gain, NSP gain
- Repeats:
  `20` trials
- Base parameters:
  `n = 20`, seeds `2026..2045`, `stage1_solver_variant = vbbr_brd`
- Summary statistic:
  mean ± std across trials
- Primary output:
  `C2_best_response_gain_trajectory.png`
- Expected takeaway:
  both provider-side gains shrink, confirming convergence of the iterative pricing process.

### C5. Supplementary Trajectory Comparison On Gap Heatmap

- Canonical script:
  `scripts/run_figure_C5_trajectory_compare_supp.py`
- Reuse source:
  `scripts/plot_stage1_trajectories_on_heatmap.py`
- Plot type:
  2D heatmap with multiple trajectory overlays
- X axis:
  `p_E`
- Y axis:
  `p_N`
- Color:
  restricted gap
- Methods:
  proposed Stage I solver, `GA`, `BO`, `MARL`
- Repeats:
  `1` representative instance
- Base parameters:
  `n = 12`, seed `2026`
- Status:
  supplementary only
- Primary output:
  `C5_trajectory_compare_supp.png`

---

## 7. Block D: Runtime And Scalability

### D2. Stage I Runtime Versus Number Of Users

- Canonical script:
  `scripts/run_figure_D2_stage1_runtime_vs_users.py`
- Reuse source:
  none
- Plot type:
  line plot with mean ± std error bars
- X axis:
  number of users
- Y axis:
  runtime in seconds
- Methods:
  proposed Stage I solver, `GA`, `BO`, `MARL`
- Baseline / reference:
  `GSO` only where feasible
- Repeats:
  `10` trials per `n`
- Base parameters:
  `n in {8, 12, 16, 20, 40, 60, 80, 100}`, seeds `2026..2035`
- Budget rule:
  auxiliary baselines use a matched evaluation budget
- Primary output:
  `D2_stage1_runtime_vs_users.png`

### D3. Number Of Stage II Solves Inside Stage I

- Canonical script:
  `scripts/run_figure_D3_stage2_calls_inside_stage1.py`
- Reuse source:
  none
- Plot type:
  line plot with mean ± std error bars
- X axis:
  number of users
- Y axis:
  number of Stage II calls invoked by Stage I
- Methods:
  proposed Stage I solver
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`
- Primary output:
  `D3_stage2_calls_inside_stage1.png`
- Expected takeaway:
  the pricing solver keeps the number of follower-side solves controlled as the system scales.

### D4. Exact Reference Runtime And Feasibility

- Canonical script:
  `scripts/run_figure_D4_exact_runtime_feasibility.py`
- Reuse source:
  none
- Plot type:
  two-panel figure
- X axis:
  number of users
- Y axes:
  panel 1: exact runtime in seconds
  panel 2: exact-solver success rate
- Methods:
  centralized exact Stage II solver
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {8, 12, 16, 20, 24}`, seeds `2026..2045`, fixed prices `p_E = 0.5`, `p_N = 0.5`
- Primary output:
  `D4_exact_runtime_feasibility.png`
- Expected takeaway:
  exact reference solving rapidly becomes expensive or unreliable outside the small-instance regime.

---

## 8. Block E: Strategic-Setting Comparisons

### E1. User Social Cost Comparison

- Canonical script:
  `scripts/run_figure_E1_user_social_cost_compare.py`
- Reuse source:
  none
- Plot type:
  line plot with mean ± std error bars
- X axis:
  number of users
- Y axis:
  total user social cost
- Methods:
  full model, `ME`, `SingleSP`, `Coop`, `Rand`
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`
- Primary output:
  `E1_user_social_cost_compare.png`

### E2. Provider Revenue Comparison

- Canonical script:
  `scripts/run_figure_E2_provider_revenue_compare.py`
- Reuse source:
  none
- Plot type:
  three-panel line plot
- X axis:
  number of users
- Y axes:
  panel 1: ESP revenue
  panel 2: NSP revenue
  panel 3: joint provider revenue
- Methods:
  full model, `ME`, `SingleSP`, `Coop`, `Rand`
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`
- Primary output:
  `E2_provider_revenue_compare.png`

### E3. Resource Utilization Comparison

- Canonical script:
  `scripts/run_figure_E3_resource_utilization_compare.py`
- Reuse source:
  none
- Plot type:
  two-panel line plot
- X axis:
  number of users
- Y axes:
  panel 1: computation utilization `sum f_i / F`
  panel 2: bandwidth utilization `sum b_i / B`
- Methods:
  full model, `ME`, `SingleSP`, `Coop`, `Rand`
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`
- Primary output:
  `E3_resource_utilization_compare.png`

### E4. Final Price And Offloading-Size Comparison

- Canonical script:
  `scripts/run_figure_E4_price_and_offloading_compare.py`
- Reuse source:
  none
- Plot type:
  three-panel line plot
- X axis:
  number of users
- Y axes:
  panel 1: final `p_E`
  panel 2: final `p_N`
  panel 3: final offloading-set size
- Methods:
  full model, `ME`, `SingleSP`, `Coop`, `Rand`
- Repeats:
  `20` trials per `n`
- Base parameters:
  `n in {20, 40, 60, 80, 100}`, seeds `2026..2045`
- Primary output:
  `E4_price_and_offloading_compare.png`

---

## 9. Block F: Robustness And Sensitivity

### F1. Q Sensitivity

- Canonical script:
  `scripts/run_figure_F1_q_sensitivity.py`
- Reuse source:
  none
- Plot type:
  three-panel line plot
- X axis:
  local candidate-family parameter `Q`
- Y axes:
  panel 1: average candidate-family size
  panel 2: final restricted gap
  panel 3: Stage I runtime
- Methods:
  proposed Stage I solver
- Repeats:
  `20` trials per `Q`
- Base parameters:
  `Q in {1, 2, 3, 4, 5, 6}`, `n = 40`, seeds `2026..2045`
- Primary output:
  `F1_q_sensitivity.png`
- Expected takeaway:
  increasing `Q` trades runtime for better local search quality without a sharp instability cliff.

### F2. Resource-Asymmetry Sensitivity

- Canonical script:
  `scripts/run_figure_F2_resource_asymmetry_sensitivity.py`
- Reuse source:
  none
- Plot type:
  three-panel line plot
- X axis:
  resource asymmetry `F / B`
- Y axes:
  panel 1: final Stage II social cost
  panel 2: final restricted gap
  panel 3: final offloading-set size
- Methods:
  proposed full model
- Repeats:
  `20` trials per point
- Base parameters:
  keep `F * B` scale fixed while sweeping `F / B in {0.5, 0.75, 1.0, 1.5, 2.0}`, `n = 40`, seeds `2026..2045`
- Primary output:
  `F2_resource_asymmetry_sensitivity.png`

### F3. Provider-Cost Sensitivity

- Canonical script:
  `scripts/run_figure_F3_provider_cost_sensitivity.py`
- Reuse source:
  none
- Plot type:
  two-panel heatmap
- X axis:
  `c_E`
- Y axis:
  `c_N`
- Colors:
  panel 1: final Stage II social cost
  panel 2: final restricted gap
- Methods:
  proposed full model
- Repeats:
  `20` trials per grid point
- Base parameters:
  `c_E, c_N in {0.05, 0.1, 0.15, 0.2, 0.25}`, `n = 40`, seeds `2026..2045`
- Primary output:
  `F3_provider_cost_sensitivity.png`

### F4. User-Distribution Sensitivity

- Canonical script:
  `scripts/run_figure_F4_user_distribution_sensitivity.py`
- Reuse source:
  none
- Plot type:
  three-panel grouped boxplot or point-range plot
- X axis:
  distribution scenario
- Y axes:
  panel 1: final Stage II social cost
  panel 2: final restricted gap
  panel 3: runtime
- Methods:
  proposed full model
- Repeats:
  `20` trials per scenario
- Base parameters:
  `n = 40`, seeds `2026..2045`
- Scenarios:
  paper default
  heavy-workload
  bandwidth-poor
  high-delay-sensitivity
  high-energy-sensitivity
- Primary output:
  `F4_user_distribution_sensitivity.png`

---

## 10. Workflow-5 Implementation Notes

Workflow 5 was implemented according to the following order:

1. direct figure runners for all A-F figures;
2. shared helper logic reused from legacy multi-output scripts where appropriate;
3. figure-specific configs under `configs/figures/`;
4. figure-level CSV schemas and summary templates.

Current status after workflow 5 completion:

- all figure IDs already have public `run_figure_*.py` entries;
- Block E figures can use the public baseline set `ME / SingleSP / Coop / Rand`;
- Block C/D comparison figures can use `MARL` directly as a paper-facing baseline label.

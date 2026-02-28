# Detailed Experiment Plan

Detailed plan derived from review.md: fill missing results, quantify approximation error, run ablations on L/candidate-family, and report scalability plus robustness.

## Cases
### A1 - Core Performance Comparison
- Motivation: Fill missing quantitative results and validate superiority claims.
- Baselines: GSSE, CS, UBRD, URA, GSO, PBRD, BO, DRL, MarketEquilibrium, SingleSP, RandomOffloading
- Metrics: social_cost, esp_revenue, nsp_revenue, epsilon_proxy, runtime_sec
- Trials: 100
- Settings:
  - n_users: 20, 40, 80, 120
  - capacities: default F/B
  - pricing_bounds: [0.01, 6.0] x [0.01, 6.0]
- Expected outputs: table_core.csv, fig_core_tradeoff.png

### A2 - Gain Approximation Fidelity
- Motivation: Address review concern on epsilon certificate accuracy.
- Baselines: Algorithm3Approx, ExactBestResponseSmallN
- Metrics: gain_abs_error, gain_rel_error, epsilon_gap
- Trials: 100
- Settings:
  - n_users: <= 16
  - candidate_family: default and expanded
  - prices: randomly sampled feasible points
- Expected outputs: table_gain_error.csv, fig_gain_error_boxplot.png

### A3 - Boundary Sampling Sensitivity
- Motivation: Quantify quality/runtime trade-off for Algorithm 4 direction count L.
- Baselines: GSSE
- Metrics: epsilon_proxy, provider_revenue_sum, runtime_sec
- Trials: 100
- Settings:
  - L_values: 4, 8, 12, 20, 32, 48
- Expected outputs: table_L_sensitivity.csv, fig_L_vs_quality_runtime.png

### A4 - Candidate Family Ablation
- Motivation: Validate impact of sensitive-user candidate family size in Algorithm 3.
- Baselines: GSSE, GSSE-ReducedFamily, GSSE-ExpandedFamily
- Metrics: epsilon_proxy, runtime_sec, candidate_count
- Trials: 100
- Settings:
  - family_rules: boundary-only / medium / exhaustive-smallN
- Expected outputs: table_family_ablation.csv, fig_family_ablation.png

### A5 - Scalability and Communication
- Motivation: Address complexity/scalability concerns from review.
- Baselines: GSSE, PBRD, BO, DRL
- Metrics: runtime_sec, inner_iterations, search_steps, message_count_proxy
- Trials: 100
- Settings:
  - n_users: 20 to 500
  - log_scale: true
- Expected outputs: table_scalability.csv, fig_scalability_curves.png

### A6 - Robustness to Modeling Assumptions
- Motivation: Assess sensitivity to heterogeneity and capacity settings.
- Baselines: GSSE, MarketEquilibrium, SingleSP
- Metrics: social_cost, provider_revenue_sum, offloading_ratio
- Trials: 200
- Settings:
  - task_mix: light/medium/heavy
  - channel_quality: low/medium/high sigma
  - capacity: F and B sweep
- Expected outputs: table_robustness.csv, fig_robustness_heatmaps.png

### A7 - Convergence Profiles
- Motivation: Provide per-iteration convergence evidence for Algorithms 1/2/5.
- Baselines: Algorithm1, Algorithm2, GSSE
- Metrics: duality_gap_proxy, constraint_violation, epsilon_proxy
- Trials: 100
- Settings:
  - episodes: multiple seeds
  - trace_logging: enabled
- Expected outputs: trace_inner.csv, trace_search.csv, fig_convergence.png

## Notes
- All summary statistics should include mean, std, and 95% CI across seeds.
- Keep a small-scale sanity subset for reproducibility and exact-oracle comparisons.
- Do not execute this plan on constrained devices; use dedicated compute nodes.

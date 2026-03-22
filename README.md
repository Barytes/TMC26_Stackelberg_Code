# TMC26 Stackelberg Experiment Code

This repository contains experiment infrastructure for the paper *Strategic User Offloading and Service Provider Pricing in Mobile Edge Computing*.

The latest source of truth is `TMC26_Stackelberg.tex`. Some scripts and configuration keys still carry legacy names from earlier implementation stages, but the documentation in this repository follows the current paper terminology.

## Paper Structure

The paper studies a two-stage multi-leader multi-follower Stackelberg game.

- Stage I: the ESP and NSP strategically price computation and bandwidth resources.
- Stage II: users jointly decide whether to offload and how much computation and bandwidth to purchase under coupled capacity constraints.

The current paper's algorithmic structure is:

- Stage II SCM solver formed by:
  - Algorithm 2 for offloading-user selection with rollback
  - Algorithm 1 for the inner resource-allocation subproblem
- Stage I restricted pricing-space analysis with fixed-set closed forms
- boundary-price-based best-response estimation
- iterative pricing for an epsilon-approximate Nash equilibrium of the Stage I pricing game

In code, Stage I now has two interchangeable pipelines behind the same public entry `solve_stage1_pricing(...)`:

- `paper_iterative_pricing`:
  the paper-facing pipeline, implemented as `Algorithm 3` boundary-price best-response estimation plus the paper's iterative pricing loop.
- `vbbr_brd`:
  a retained backup pipeline based on the earlier VBBR oracle and dynamics code.
  It is kept because its empirical behavior has been useful in prior experiments, but it should be described as an alternative implementation path rather than the paper's canonical algorithm.

## Environment

```bash
uv sync
```

## Main Entry Points

```bash
uv run python scripts/run_figure_A1_stage2_social_cost_trace.py
```

Figure-level scripts under `scripts/run_figure_*.py` are the public experiment entry points. Each script writes exactly one main figure, one CSV, and one summary file.

Output rule:

- all generated artifacts, including temporary test runs, must stay under the project `outputs/` directory;
- if `--out-dir` is given as a relative path, it is interpreted under `outputs/`;
- absolute `--out-dir` values outside the project `outputs/` directory are rejected.

Example:

```bash
uv run python scripts/run_figure_B1_stage1_joint_revenue_heatmap.py
```

The legacy package CLI still exists, but it should be treated as an internal umbrella runner rather than the paper-facing interface:

```bash
uv run python -m tmc26_exp --config configs/default.toml
```

That internal CLI can still:

- compute configured simulation-based metric surfaces;
- run one configured Stage I solve on a sampled user batch;
- optionally emit a non-executed detailed experiment plan.

## Repository Layout

```text
.
├── configs/
│   ├── default.toml
│   ├── figures/
│   │   └── paper_base.toml
│   └── *.toml
├── docs/
│   ├── BASELINE_SCRIPT_MAPPING.md
│   ├── CHANGE_PLAN.md
│   ├── DEV.md
│   ├── FINAL_CONSISTENCY_AUDIT.md
│   ├── FIGURE_SCRIPT_BLUEPRINT.md
│   ├── OUTPUT_SCHEMA.md
│   ├── SPEC.md
│   ├── STAGE1_PRICING_AUDIT.md
│   ├── STAGE2_SCM_AUDIT.md
│   ├── TERM_INTERFACE_FREEZE.md
│   └── md_alignment_audit.md
├── scripts/
│   ├── run_figure_*.py
│   ├── run_stage2_social_cost_compare.py
│   ├── run_stage2_approximation_ratio.py
│   ├── run_boundary_hypothesis_check.py
│   ├── run_stage1_price_heatmaps.py
│   ├── run_stage1_price_heatmaps_cs_gekko.py
│   ├── run_stage1_vbbr_trajectory_on_heatmap.py
│   ├── run_algorithm2_exploitability_vs_users.py
│   ├── plot_stage1_trajectories_on_heatmap.py
│   ├── plot_pbdr_trajectory_from_heatmap_csv.py
│   ├── collect_figure_manifests.py
│   └── reprint_stage1_selected_figures.py
├── src/
│   └── tmc26_exp/
│       ├── baselines.py
│       ├── cli.py
│       ├── config.py
│       ├── distributions.py
│       ├── metrics.py
│       ├── model.py
│       ├── plotting.py
│       ├── simulator.py
│       └── stackelberg.py
└── TMC26_Stackelberg.tex
```

## Current Script Inventory

The repository now exposes a figure-level public runner matrix under `scripts/run_figure_*.py`.

All canonical figure scripts are now direct, runnable public entries:

- `scripts/run_figure_A1_stage2_social_cost_trace.py`
- `scripts/run_figure_A2_stage2_social_cost_multiscale.py`
- `scripts/run_figure_A3_stage2_approx_ratio_bound.py`
- `scripts/run_figure_A4_stage2_runtime_vs_users.py`
- `scripts/run_figure_A5_stage2_rollback_diagnostics.py`
- `scripts/run_figure_A6_stage2_exploitability_supp.py`
- `scripts/run_figure_B1_stage1_joint_revenue_heatmap.py`
- `scripts/run_figure_B2_stage1_restricted_gap_heatmap.py`
- `scripts/run_figure_B3_esp_slice_boundary_comparison.py`
- `scripts/run_figure_B4_nsp_slice_boundary_comparison.py`
- `scripts/run_figure_B5_joint_revenue_boundary_overlay.py`
- `scripts/run_figure_B6_candidate_family_diagnostics.py`
- `scripts/run_figure_C1_restricted_gap_trajectory.py`
- `scripts/run_figure_C2_best_response_gain_trajectory.py`
- `scripts/run_figure_C3_price_trajectory_on_gap_heatmap.py`
- `scripts/run_figure_C4_final_gap_vs_budget.py`
- `scripts/run_figure_C5_trajectory_compare_supp.py`
- `scripts/run_figure_D1_stage2_runtime_vs_users.py`
- `scripts/run_figure_D2_stage1_runtime_vs_users.py`
- `scripts/run_figure_D3_stage2_calls_inside_stage1.py`
- `scripts/run_figure_D4_exact_runtime_feasibility.py`
- `scripts/run_figure_E1_user_social_cost_compare.py`
- `scripts/run_figure_E2_provider_revenue_compare.py`
- `scripts/run_figure_E3_resource_utilization_compare.py`
- `scripts/run_figure_E4_price_and_offloading_compare.py`
- `scripts/run_figure_F1_q_sensitivity.py`
- `scripts/run_figure_F2_resource_asymmetry_sensitivity.py`
- `scripts/run_figure_F3_provider_cost_sensitivity.py`
- `scripts/run_figure_F4_user_distribution_sensitivity.py`

Legacy multi-output scripts are still kept as shared helper modules and diagnostic references:

- `scripts/run_stage2_social_cost_compare.py`
- `scripts/run_stage2_approximation_ratio.py`
- `scripts/run_algorithm2_exploitability_vs_users.py`
- `scripts/run_boundary_hypothesis_check.py`
- `scripts/run_stage1_price_heatmaps.py`
- `scripts/run_stage1_price_heatmaps_cs_gekko.py`
- `scripts/run_stage1_vbbr_trajectory_on_heatmap.py`
- `scripts/plot_stage1_trajectories_on_heatmap.py`
- `scripts/plot_pbdr_trajectory_from_heatmap_csv.py`
- `scripts/reprint_stage1_selected_figures.py`

Some filenames still use legacy names such as `vbbr` or `pbdr`. Read them according to the current paper's Stage I boundary-price and trajectory diagnostics, not as authoritative paper terminology.

For Stage II, read the scripts as diagnostics for the integrated SCM solution route rather than as separate paper tracks for Algorithm 1 and Algorithm 2.

The figure-level canonical plan is frozen in `docs/FIGURE_SCRIPT_BLUEPRINT.md`. Workflow 5 has established one figure per script as the public interface; current multi-output scripts remain only as reusable implementation references.

Workflow 6 output schema is frozen in `docs/OUTPUT_SCHEMA.md`. Each figure run now also writes a standard `figure_manifest.json`, and cross-run aggregation is available through `scripts/collect_figure_manifests.py`.

## Configuration

The legacy project-wide configuration file is `configs/default.toml`. The figure-level paper-aligned default is `configs/figures/paper_base.toml`.

Important sections:

- top-level run controls: `run_name`, `output_dir`, `n_users`, `n_trials`, `seed`
- `[system]`: system capacities and provider costs
- `[stackelberg]`: Stage I and Stage II solver hyper-parameters
- `[baselines]`: baseline method settings
- `[detailed_experiment]`: plan emission settings
- `[price_grid]`: diagnostic surface ranges
- `[user_distributions]`: random-instance generation

When interpreting configuration names, prefer the paper's current concepts over legacy key names. For example, the paper's formal Stage I metrics are the NE gap and the boundary-price-restricted NE gap, even if some implementation knobs still use older naming conventions.

Important Stage I config keys:

- `stage1_solver_variant = "paper_iterative_pricing"`:
  use the paper-facing Stage I pipeline.
- `stage1_solver_variant = "vbbr_brd"`:
  use the retained backup Stage I pipeline.
- `paper_local_Q`:
  the `Q` in the paper's local candidate family `N_Q(p)`.
- `paper_restricted_gap_tol`:
  stopping tolerance for the paper-facing restricted-gap iterative pricing loop.

Current config files may still default to `vbbr_brd` for continuity with existing experiments.
For paper-aligned Stage I runs, set `stage1_solver_variant = "paper_iterative_pricing"` explicitly.

## Baselines

The current paper groups baselines into two categories.

Stackelberg-equilibrium baselines:

- GSO
- GA
- BO
- MARL

Strategic-setting baselines:

- ME
- SingleSP
- Coop
- Rand

Current code status is narrower than the paper taxonomy:

- `GSO`, `GA`, `BO`, `ME`, `SingleSP`, and `Rand` have direct code paths.
- `MARL` is not yet a dedicated public baseline; the current code only contains a legacy `DRL` proxy.
- `Coop` does not yet have a dedicated public wrapper.

The repository also contains auxiliary or legacy diagnostics beyond this list. Those should not be treated as the paper's primary baseline taxonomy unless the paper source is updated accordingly. See `docs/BASELINE_SCRIPT_MAPPING.md` for the frozen mapping.

## Documentation

Use the following order when reading the repository:

1. `TMC26_Stackelberg.tex`
2. `docs/SPEC.md`
3. `docs/TERM_INTERFACE_FREEZE.md`
4. `docs/DEV.md`
5. `docs/CHANGE_PLAN.md`
6. `docs/FIGURE_SCRIPT_BLUEPRINT.md`
7. `docs/OUTPUT_SCHEMA.md`
8. `docs/FINAL_CONSISTENCY_AUDIT.md`

If a script name, config key, or old note conflicts with the paper, follow the paper.

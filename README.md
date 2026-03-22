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

## Environment

```bash
uv sync
```

## Main Run

```bash
uv run tmc26-exp --config configs/default.toml
```

If the entry point is unavailable in your environment, use:

```bash
uv run python -m tmc26_exp --config configs/default.toml
```

This command currently does three things:

- computes configured simulation-based metric surfaces;
- runs the configured Stage I solver on one sampled user batch if `[stackelberg].enabled = true`;
- optionally emits a non-executed detailed experiment plan.

Typical outputs in `outputs/<run_name>/`:

- `metrics_summary.txt`
- `*.csv`, `*_heatmap.png`, `*_contour.png` for metric surfaces
- `stackelberg_summary.txt`
- `stackelberg_trajectory.csv`
- `stackelberg_allocation.csv`
- `baselines_summary.csv` when baselines are enabled

## Repository Layout

```text
.
├── configs/
│   ├── default.toml
│   └── *.toml
├── docs/
│   ├── DEV.md
│   ├── SPEC.md
│   └── md_alignment_audit.md
├── scripts/
│   ├── run_stage2_social_cost_compare.py
│   ├── run_stage2_approximation_ratio.py
│   ├── run_boundary_hypothesis_check.py
│   ├── run_stage1_price_heatmaps.py
│   ├── run_stage1_price_heatmaps_cs_gekko.py
│   ├── run_stage1_vbbr_trajectory_on_heatmap.py
│   ├── run_algorithm2_exploitability_vs_users.py
│   ├── plot_stage1_trajectories_on_heatmap.py
│   ├── plot_pbdr_trajectory_from_heatmap_csv.py
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

The repository does not currently expose a one-to-one "all paper figures" runner matrix. Instead, it contains targeted scripts for diagnostics and figure construction.

Stage II focused scripts:

- `scripts/run_stage2_social_cost_compare.py`
- `scripts/run_stage2_approximation_ratio.py`
- `scripts/run_algorithm2_exploitability_vs_users.py`

Stage I focused scripts:

- `scripts/run_boundary_hypothesis_check.py`
- `scripts/run_stage1_price_heatmaps.py`
- `scripts/run_stage1_price_heatmaps_cs_gekko.py`
- `scripts/run_stage1_vbbr_trajectory_on_heatmap.py`

Plotting and reprint helpers:

- `scripts/plot_stage1_trajectories_on_heatmap.py`
- `scripts/plot_pbdr_trajectory_from_heatmap_csv.py`
- `scripts/reprint_stage1_selected_figures.py`

Some filenames still use legacy names such as `vbbr` or `pbdr`. Read them according to the current paper's Stage I boundary-price and trajectory diagnostics, not as authoritative paper terminology.

For Stage II, read the scripts as diagnostics for the integrated SCM solution route rather than as separate paper tracks for Algorithm 1 and Algorithm 2.

## Configuration

The main configuration file is `configs/default.toml`.

Important sections:

- top-level run controls: `run_name`, `output_dir`, `n_users`, `n_trials`, `seed`
- `[system]`: system capacities and provider costs
- `[stackelberg]`: Stage I and Stage II solver hyper-parameters
- `[baselines]`: baseline method settings
- `[detailed_experiment]`: plan emission settings
- `[price_grid]`: diagnostic surface ranges
- `[user_distributions]`: random-instance generation

When interpreting configuration names, prefer the paper's current concepts over legacy key names. For example, the paper's formal Stage I metrics are the NE gap and the boundary-price-restricted NE gap, even if some implementation knobs still use older naming conventions.

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

The repository may also contain auxiliary or legacy diagnostics beyond this list. Those should not be treated as the paper's primary baseline taxonomy unless the paper source is updated accordingly.

## Documentation

Use the following order when reading the repository:

1. `TMC26_Stackelberg.tex`
2. `docs/SPEC.md`
3. `docs/DEV.md`
4. `docs/md_alignment_audit.md`

If a script name, config key, or old note conflicts with the paper, follow the paper.

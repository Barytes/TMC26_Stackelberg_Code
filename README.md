# TMC26 Stackelberg Experiment Code

This repository provides a compact Python implementation for the paper *Strategic User Offloading and Service Provider Pricing in Mobile Edge Computing*.

---

## Experiment Scripts (17 Figures)

The repository includes comprehensive experiment scripts covering all paper figures. See `docs/experiment_figures.md` for full details.

### Phase 1: Stage I Core Experiments (Figures 5-9)

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 5 | `run_stage1_deviation_gap_convergence.py` | Algorithm 5 epsilon convergence |
| Figure 6 | `run_stage1_boundary_visualization.py` | Price space boundary + trajectory |
| Figure 7 | `run_stage1_gain_approximation_accuracy.py` | Algorithm 3 gain approximation |
| Figure 8 | `run_stage1_candidate_family_hit_rate.py` | Candidate family N(p) hit rate |
| Figure 9 | `run_stage1_scalability.py` | Stage I scalability |

### Phase 2: Stage II Completion (Figures 1-4)

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 1 | `run_stage2_convergence_plot.py` | Iteration convergence |
| Figure 2 | `run_stage2_approximation_ratio.py` | Theorem 2 validation |
| Figure 3 | `run_stage2_communication_rounds.py` | Communication rounds |
| Figure 4 | `run_stage2_exploitability_comparison.py` | Exploitability comparison |

### Phase 3: Strategic Settings (Figures 10-13)

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 10 | `run_strategic_social_cost.py` | User social cost |
| Figure 11 | `run_strategic_joint_revenue.py` | Provider revenue |
| Figure 12 | `run_strategic_pareto_tradeoff.py` | Pareto tradeoff |
| Figure 13 | `run_strategic_fb_sensitivity.py` | F/B sensitivity |

### Phase 4: Ablation Studies (Figures 14-15)

| Figure | Script | Description |
|--------|--------|-------------|
| Figure 14 | `run_ablation_L_sensitivity.py` | Sampling density L |
| Figure 15 | `run_ablation_guided_search.py` | Guided search ablation |

### Phase 5: Appendix (A1-A2)

| Figure | Script | Description |
|--------|--------|-------------|
| Appendix A1 | `run_appendix_final_epsilon_vs_users.py` | Final epsilon scaling |
| Appendix A2 | `run_appendix_exploitability_vs_users.py` | Full exploitability |

### Quick Test

```bash
# Stage I
uv run python scripts/run_stage1_deviation_gap_convergence.py --trials 3
uv run python scripts/run_stage1_scalability.py --trials 3

# Stage II
uv run python scripts/run_stage2_convergence_plot.py --trials 3

# Strategic
uv run python scripts/run_strategic_social_cost.py --trials 5

# Ablation
uv run python scripts/run_ablation_L_sensitivity.py --trials 3
```

---

## 1) Environment

```bash
uv sync
```

## 2) Run

```bash
uv run tmc26-exp --config configs/default.toml
```

By default this command:
- computes all configured metric surfaces
- runs the Stackelberg guided-search pipeline once on one sampled user batch
- emits detailed experiment plan files only (does not run detailed heavy experiments)

Outputs:
- `raw_results.csv` (per-trial/per-method records)
- `summary_by_method.csv` (mean/std summary)
- `run_meta.txt`

## 4) Project structure

```text
.
├── configs/
│   └── default.toml
├── src/
│   └── tmc26_exp/
│       ├── cli.py
│       ├── config.py
│       ├── distributions.py
│       ├── metrics.py
│       ├── model.py
│       ├── plotting.py
│       ├── simulator.py
│       └── stackelberg.py
├── pyproject.toml
└── TMC26_Stackelberg.tex
```

## 5) Outputs

All outputs are written to `outputs/<run_name>/`.

Metric outputs:
- `metrics_summary.txt`
- `*.csv` metric tables (`pE,pN,value_mean,value_std`)
- `*_heatmap.png`, `*_contour.png`

Stackelberg outputs (when `[stackelberg].enabled = true`):
- `stackelberg_summary.txt`
- `stackelberg_trajectory.csv`
- `stackelberg_allocation.csv`

Baseline outputs (when `[baselines].enabled = true` or `--run-baselines`):
- `baselines_summary.csv`



## 7) Configuration

Main sections in `configs/default.toml`:
- `[price_grid]`: metric visualization range
- `[user_distributions]`: user parameter sampling
- `[system]`: `F, B, cE, cN`
- `[stackelberg]`: Algorithm 1~5 hyper-parameters
- `[baselines]`: all baseline methods hyper-parameters
- `[detailed_experiment]`: detailed plan generation controls

Set `[stackelberg].enabled = false` if you only want metric surfaces.
Set `[baselines].enabled = true` to run baseline methods on the sampled batch.
Detailed experiment code is provided but must be executed manually on high-compute hardware.

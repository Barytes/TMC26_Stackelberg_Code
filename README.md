# TMC26 Stackelberg Experiment Code

This repository provides a compact Python implementation for:
- metric surface simulation on the `(p_E, p_N)` plane
- the five algorithms in `TMC26_Stackelberg.tex` (Algorithm 1~5)
- all baseline methods listed in the updated Experimental Setup section
- a detailed experiment plan generator based on `review.md`

Implemented algorithms:
1. Distributed primal-dual solver for the Stage-II inner problem
2. Heuristic distributed offloading user selection (outer SCM)
3. Approximated best-response gain computation
4. Optimal RNE computation via multi-direction boundary sampling
5. Guided search for Stackelberg equilibrium over offloading user sets

Implemented baselines:
1. Stage-II baselines: `CS`, `UBRD`, `VI`, `PEN`
2. Stage-I baselines: `GSO`, `PBRD`, `BO`, `DRL`
3. Strategic-setting baselines: `MarketEquilibrium`, `SingleSP`, `RandomOffloading`

Stage-II derivations for the new GNEP baselines are documented in `docs/stage2_gnep_baselines.md`.

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

For multi-core CPU cloud servers (parallel trials), use:

```bash
uv run python scripts/run_cpu_parallel_baselines.py \
  --config configs/default.toml \
  --trials 100 \
  --workers 16 \
  --methods GSSE,GSO,PBRD,BO,DRL,MarketEquilibrium,SingleSP,RandomOffloading
```

Outputs:
- `raw_results.csv` (per-trial/per-method records)
- `summary_by_method.csv` (mean/std summary)
- `run_meta.txt`

## 3) Project structure

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

## 4) Outputs

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

Detailed plan outputs (when `[detailed_experiment].emit_plan = true` or `--emit-detailed-plan`):
- `detailed_plan/detailed_experiment_plan.md`
- `detailed_plan/detailed_experiment_plan.json`
- `detailed_plan/run_detailed_experiments.py` (runner stub, not auto-executed)

## 6) Run The Detailed Experiment Plan

Use high-compute CPU servers for this section.

### Step 1: Generate plan files only

```bash
uv run tmc26-exp --config configs/default.toml --emit-detailed-plan
```

This writes plan files under:
`outputs/<run_name>/detailed_plan/`

### Step 2: Execute the detailed experiments (manual)

After Step 1, run the generated stub on the server:

```bash
uv run python outputs/default_run/detailed_plan/run_detailed_experiments.py \
  --config configs/default.toml \
  --outdir outputs/default_run/detailed_exec
```

Or call the module directly:

```bash
uv run python - <<'PY'
from pathlib import Path
from tmc26_exp.detailed_experiments import run_detailed_experiments
run_detailed_experiments("configs/default.toml", Path("outputs/default_run/detailed_exec"))
PY
```

Main outputs include:
- `A1_core_comparison.csv`
- `A2_gain_fidelity.csv`
- `A3_L_sensitivity.csv`
- `A4_candidate_family_ablation.csv`
- `A5_scalability.csv`
- `A6_robustness.csv`
- `A7_convergence_trace.csv`

## 5) Configuration

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

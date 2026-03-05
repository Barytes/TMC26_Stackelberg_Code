# TMC26 Stackelberg Experiment Code

This repository provides a compact Python implementation for the paper *Strategic User Offloading and Service Provider Pricing in Mobile Edge Computing*.

---

## Paper Summary

### System Model

The paper considers a **Mobile Edge Computing (MEC)** environment with:
- **One ESP (Edge Service Provider)**: Manages computation resources at an edge server with capacity `F` (GHz)
- **One NSP (Network Service Provider)**: Manages wireless bandwidth at a base station with capacity `B` (MHz)
- **Set of users I = {1, ..., I}**: Each user has an indivisible task characterized by:
  - Workload `w_i` (in CPU cycles)
  - Input data size `d_i` (in bits)

Each user makes a **binary offloading decision**:
- `o_i = 0`: Local computation
- `o_i = 1`: Offload to edge server via base station

If offloading, user `i` also decides computation resource `f_i` (from ESP) and bandwidth `b_i` (from NSP).

### User Costs

- **Local computation cost**: `C_i^l = α_i * T_i^l + β_i * E_i^l`
  - `T_i^l = w_i / f_i^l` (delay)
  - `E_i^l = κ_i * w_i * (f_i^l)²` (energy)
  - `α_i`, `β_i`: User's marginal monetary costs for delay and energy

- **Offloading cost**: `C_i^e(f_i, b_i) = α_i * T_i^e + β_i * E_i^u + p_E * f_i + p_N * b_i`
  - `T_i^e = T_i^u + T_i^c` (transmission + computation delay)
  - `E_i^u`: Transmission energy
  - `p_E`, `p_N`: Unit prices set by ESP and NSP

### Service Provider Revenue

- **ESP Revenue**: `U_E = (p_E - c_E) * Σ f_i` (where `c_E` is unit cost)
- **NSP Revenue**: `U_N = (p_N - c_N) * Σ b_i` (where `c_N` is unit cost)

### Problem Formulation

The interactions form a **two-stage Stackelberg game**:

**Stage I (Leaders)**: ESP and NSP set prices `p_E` and `p_N` to maximize revenue.

**Stage II (Followers)**: Given prices, users decide offloading decisions and resource demands to minimize their costs, subject to capacity constraints.

**Goal**: Find a **Stackelberg Equilibrium** where:
- Users' strategies form a **Generalized Nash Equilibrium (GNE)** of the Stage II game
- Prices form a **Nash Equilibrium (NE)** of the Stage I pricing game

### Methodology

#### Stage II: User Offloading Game

1. **Social Cost Minimization (SCM) Problem**: Minimize total cost over all users by selecting the optimal offloading set and allocating resources.

2. **Inner Problem**: Given a fixed offloading set X, allocate resources to minimize total offloading cost (strictly convex → unique solution).

3. **Outer Problem**: Select the optimal offloading user set X (NP-hard).

4. **Algorithms**:
   - **Algorithm 1**: Distributed primal-dual algorithm for inner problem
   - **Algorithm 2**: Heuristic distributed greedy algorithm for outer problem (using shadow prices)

#### Stage I: Service Provider Pricing Game

1. **Key Insight**: Any Stackelberg equilibrium must be a **Restricted Nash Equilibrium (RNE)** on the **boundary** of the restricted pricing space P_X.

2. **Best Response Gain Minimization (BRGM)**: Reformulate the NE problem as searching over offloading user sets.

3. **Algorithms**:
   - **Algorithm 3**: Approximated best response gain computation (quadratic complexity)
   - **Algorithm 4**: Optimal RNE via multi-direction boundary sampling
   - **Algorithm 5**: Guided search for Stackelberg equilibrium

---

Implemented algorithms:
1. Distributed primal-dual solver for the Stage-II inner problem
2. Heuristic distributed offloading user selection (outer SCM)
3. Approximated best-response gain computation
4. Optimal RNE computation via multi-direction boundary sampling
5. Guided search for Stackelberg equilibrium over offloading user sets

Implemented baselines:

**Stage-II baselines (User Offloading Game solvers):**
- `CS` (Centralized Solver): Direct MINLP formulation (Pyomo + BONMIN). Solves SCM problem exactly without enumeration. Requires BONMIN solver installation.
- `UBRD` (User Best Response Dynamics): Iterative greedy where users update offloading decisions. Converges to GNE.
- `VI` (Variational Inequality): Primal-dual subgradient method for GNEP. Updates shadow prices until convergence.
- `PEN` (Penalty Method): Augmented Lagrangian approach for GNEP. Alternates between best-response and penalty updates.

**Stage-I baselines (Pricing Game solvers):**
- `GSO` (Grid Search Oracle): Brute-force discretizes (pE, pN) space. Serves as upper-bound oracle.
- `PBRD` (Provider Best Response Dynamics): Alternating optimization between ESP and NSP prices.
- `BO` (Bayesian Optimization): GP-based black-box optimization with UCB acquisition.
- `DRL` (Deep Reinforcement Learning): Q-learning with epsilon-greedy over discretized price grid.

**Strategic-setting baselines (Different model assumptions):**
- `MarketEquilibrium`: Non-strategic multi-provider, competitive users. Prices clear market (demand = supply).
- `SingleSP`: Single provider managing both resources. Greedy offloader selection, non-competitive users.
- `RandomOffloading`: Strategic multi-provider, non-strategic users. Random offloading set + pricing equilibrium.

**Proposed method:**
- `GSSE` (Guided Stackelberg Equilibrium): Algorithm 5 - guided search over offloading user sets to find ε-NE.

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

## 3) Run A Simple Suite Experiment

For simple agent-driven Stage-II experiments (for example: fixed `n_users`, sweep `pE`/`pN`, compare `DG` vs `VI` vs other Stage-II methods), use the suite runner:

```bash
uv run python scripts/run_suite.py \
  --config configs/default.toml \
  --suite suites/example_stage2_price_sweep.toml
```

This path is intentionally narrow and simple:
- it is for fixed-price or price-sweep experiments
- it currently supports Stage-II methods only: `DG`, `CS`, `UBRD`, `VI`, `PEN`
- it is the recommended interface when an AI agent generates experiment configs from natural-language requests

### Suite file format

Suite files live under `suites/*.toml`.

Minimal example:

```toml
[experiment]
name = "compare_social_cost_pE"
description = "Impact of pE on social cost for fixed n_users"

[data]
n_users = 100
seed = 2026
trials = 20
reuse_users_across_methods = true
reuse_users_across_sweep = false

[sweep]
pE = [0.1, 0.5, 1.0, 1.5, 2.0]
pN = [0.5]

[methods]
target = ["DG", "UBRD", "VI", "PEN"]

[metrics]
include = [
  "social_cost",
  "offloading_size",
  "epsilon_proxy",
  "runtime_sec",
  "esp_revenue",
  "nsp_revenue"
]
```

Supported sweep keys:
- `pE`
- `pN`
- `n_users`
- `F`
- `B`

Supported metric keys:
- `social_cost`
- `offloading_size`
- `epsilon_proxy`
- `runtime_sec`
- `esp_revenue`
- `nsp_revenue`

Execution semantics:
- all sweep keys are expanded via Cartesian product
- scalar sweep values are treated as length-1 lists
- trial `t` uses seed `seed + t`
- when `reuse_users_across_methods = true`, all methods at the same trial and sweep point use the same sampled user batch
- when `reuse_users_across_sweep = false`, different sweep points resample users by default

### Suite outputs

Suite outputs are written to:
`outputs/suites/<experiment.name>/`

The runner writes:
- `suite_snapshot.toml`
- `results.csv`
- `summary.csv`

`results.csv` contains one row per actual run.
`summary.csv` aggregates by `method`, `pE`, `pN`, `n_users`, `F`, and `B`, and reports count plus mean/std metrics.

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

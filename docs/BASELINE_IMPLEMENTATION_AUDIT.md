# Baseline Implementation Audit

This document audits the implementation status of the eight baselines listed in the current paper version
([`TMC26_Stackelberg.tex`](../TMC26_Stackelberg.tex)).

Scope note:

- This audit intentionally excludes the known engineering deviation that `GSO / GA / BO / MARL`
  do not use centralized MINLP as the default follower solver in everyday runs.
- That deviation is currently treated as an accepted performance concession rather than a blocking
  implementation error.
- The findings below therefore focus only on the remaining implementation gaps.

## Paper Baselines

The paper lists the following eight baselines:

1. `GSO`
2. `GA`
3. `BO`
4. `MARL`
5. `ME`
6. `SingleSP`
7. `Coop`
8. `Rand`

Reference:

- [`TMC26_Stackelberg.tex`](../TMC26_Stackelberg.tex): lines around the baseline section
- Code implementation file: [`src/tmc26_exp/baselines.py`](../src/tmc26_exp/baselines.py)

## Executive Summary

Current implementation status, excluding the accepted `DG`-default deviation:

- `Implemented and structurally usable, but not fully paper-faithful`: `GSO`, `GA`, `BO`, `MARL`, `ME`, `Coop`
- `Implemented and not flagged by this audit`: `SingleSP`, `Rand`

The most important remaining issues are:

1. `run_all_baselines()` is an internal mixed launcher, not a clean launcher for the paper's eight baselines.
2. `GA / BO` remain configurable frameworks rather than narrow paper-only runners.

## Findings

### 1. `Coop` baseline now has a direct implementation

Paper definition:

- The ESP and NSP cooperate and jointly choose `(p_E, p_N)` to maximize total revenue while users remain competitive in Stage II.

Code status:

- Implemented by
  [`baseline_coop()`](../src/tmc26_exp/baselines.py).
- The current implementation reuses the Stage-I price grid evaluation and selects the grid point with maximum joint provider revenue `U_E + U_N`.

Assessment:

- The baseline now exists as a dedicated public path.
- Its main limitation is grid resolution, not absence of implementation.

Priority:

- Low

### 2. `MARL` baseline now has a direct implementation

Paper definition:

- A multi-agent RL baseline in which ESP and NSP act as separate agents and jointly learn pricing through joint-action Q-learning.

Code status:

- The paper-facing implementation is now
  [`baseline_stage1_marl()`](../src/tmc26_exp/baselines.py).
- It uses discrete price levels as the action space and joint-action Q-learning over price pairs.
- Each provider receives its own revenue as reward, and the final price pair is selected directly from the learned joint Q tables.

Assessment:

- This baseline now exists as a dedicated public path.
- Remaining non-blocking differences are implementation details, not a missing-baseline issue.

Priority:

- Low

### 3. `GSO` baseline has a public implementation

Paper definition:

- `GSO` performs grid search over `(p_E, p_N)` and locates the Stage-I equilibrium by computing the paper NE gap.

Code status:

- Implemented by
  [`baseline_stage1_grid_search_oracle()`](../src/tmc26_exp/baselines.py).
- It exposes a dedicated public baseline path rather than being folded into a mixed helper only.

Assessment:

- From an implementation-audit perspective, the main point is that the baseline exists and is independently callable.
- This document does not use `GSO` to define repository-wide naming or objective-policy rules.

Priority:

- Low

### 4. `GA` baseline has a public implementation

Paper definition:

- `GA` searches the 2D pricing space and evaluates candidates using the paper's approximate equilibrium-gap criterion.

Code status:

- Implemented by
  [`run_stage1_genetic_algorithm()`](../src/tmc26_exp/baselines.py) and
  [`baseline_stage1_ga()`](../src/tmc26_exp/baselines.py).
- The implementation remains a configurable GA framework rather than a single hard-coded experimental runner.

Assessment:

- The baseline exists as a dedicated public implementation.
- The main residual issue is that the optimization target remains configurable, so callers can move it away from the paper-facing setup.

Priority:

- Low

### 5. `BO` baseline has a public implementation

Paper definition:

- `BO` should minimize the same approximate equilibrium-gap objective used by `GA`.

Code status:

- Implemented by
  [`baseline_stage1_bo()`](../src/tmc26_exp/baselines.py).
- The internal helper
  [`_objective_score_and_candidate()`](../src/tmc26_exp/baselines.py)
  is now routed through a dedicated public `BO` baseline path.

Assessment:

- The BO machinery exists and is usable.
- The main residual issue is the same as for `GA`: the helper layer remains more general than the narrow paper-facing experiment definition.

Priority:

- Low

### 6. `ME` is structurally close, and now reports through the unified baseline schema

Paper definition:

- `ME` should search for market-clearing prices in the non-strategic multi-provider setting.

Code status:

- Implemented by
  [`baseline_market_equilibrium()`](../src/tmc26_exp/baselines.py).
- The update rule is tatonnement-style and is structurally close to the paper description.
- The returned outcome now uses the shared `BaselineOutcome` schema with `grid_ne_gap` and `legacy_gain_proxy`.

Assessment:

- Among the eight baselines, `ME` is one of the closer ones.
- The residual issue is methodological rather than naming-related.

Priority:

- Low

### 7. `run_all_baselines()` is not a paper-8 baseline launcher

Code status:

- [`run_all_baselines()`](../src/tmc26_exp/baselines.py) mixes:
  - proposal methods,
  - Stage-II auxiliary baselines,
  - legacy Stage-I methods,
  - formal paper baselines.

Assessment:

- This function is suitable as an internal compatibility launcher.
- It should not be treated as the canonical launcher for the paper's eight baselines.

Priority:

- Low

## Baseline-by-Baseline Status Table

| Baseline | Paper baseline exists in code | Paper-faithful status | Main issue |
| --- | --- | --- | --- |
| `GSO` | Yes | Partial | Public implementation exists; paper-faithfulness still depends on experiment configuration |
| `GA` | Yes | Partial | Public implementation exists; optimization target remains configurable |
| `BO` | Yes | Partial | Public implementation exists; helper layer remains broader than the paper-facing run |
| `MARL` | Yes | Partial | Direct implementation exists; residual details are in policy-extraction and discretization choices |
| `ME` | Yes | Mostly structural | Structurally close; remaining issues are methodological |
| `SingleSP` | Yes | Accepted as-is in this audit | No blocking issue recorded here |
| `Coop` | Yes | Partial | Current implementation is grid-based joint-revenue maximization |
| `Rand` | Yes | Accepted as-is in this audit | No blocking issue recorded here |

## Recommended Fix Order

If these baseline issues are to be addressed, the recommended order is:

1. Decide whether `GA / BO` should remain configurable frameworks or be narrowed to stricter paper-facing runners.
2. Decide whether `Coop` should remain a grid-search baseline or gain local refinement around the best joint-revenue point.
3. Add a dedicated paper-baseline launcher instead of relying on `run_all_baselines()`.

## Audit Scope Boundary

This document does not treat the following item as a blocking mismatch:

- `GSO / GA / BO / MARL` using a fast follower solver by default instead of centralized MINLP in routine runs.

That choice is currently accepted as an engineering tradeoff due to the prohibitive runtime of the centralized MINLP solver.

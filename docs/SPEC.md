# SPEC.md

## Objective

Design the experiment section so that it directly validates the current paper's actual theoretical and algorithmic contributions in `TMC26_Stackelberg.tex`.

This document is not a generic benchmark wishlist. It should stay aligned with the current paper's formal structure:

- Stage II:
  - GNE existence and multiple equilibria
  - SOE selection through SCM
  - Algorithm 2 + Algorithm 1 as the integrated SCM solver
- Stage I:
  - restricted pricing spaces
  - switching-set structure
  - candidate boundary prices
  - best-response estimation via boundary prices
  - iterative pricing for an epsilon-approximate Nash equilibrium

The main question is:

> Does the implemented experiment suite validate the mechanisms that the current paper actually proves and uses?

---

## Source of Truth

When there is any conflict, the priority order is:

1. `TMC26_Stackelberg.tex`
2. this file
3. `docs/DEV.md`
4. script names and config-key names

Legacy implementation names such as `vbbr`, `pbdr`, or old "verified-gap" labels must not redefine the paper narrative.

Implementation note:

- the repository now keeps two interchangeable Stage I pipelines behind `solve_stage1_pricing(...)`;
- `paper_iterative_pricing` is the paper-facing pipeline and should be the default choice for paper-aligned experiments;
- `vbbr_brd` remains available as a backup implementation for robustness checks, ablations, or fallback runs, but it should not replace the paper's main Stage I narrative in the main text.

---

## Core Claims to Validate

### Stage II

The experiments should validate the following claims.

1. The Stage II user game admits a meaningful SOE target because the game is an exact generalized potential game and multiple GNE may arise.
2. Algorithm 2 together with Algorithm 1 is a practical approximation method for SCM:
   - rollback prevents harmful user admissions;
   - the achieved social cost is competitive with centralized exact solving on small instances;
   - the empirical approximation ratio is consistent with the theorem-based upper bound.
3. Stage II should be treated as one integrated follower-side SCM solver in experiments, rather than split into a separate Algorithm 1 experiment track and a separate Algorithm 2 experiment track.

### Stage I

The experiments should validate the following claims.

1. Restricted pricing spaces and switching boundaries are structurally useful:
   good unilateral price moves concentrate near switching prices rather than arbitrary points in the full 2D plane.
2. The fixed-set closed-form analysis is operationally useful:
   it provides the margins, candidate families, and boundary prices needed to estimate best responses.
3. The local candidate family `N_Q(p)` is a tractable approximation of the locally relevant regime changes.
4. Boundary-price-based best-response estimation is effective:
   evaluating candidate boundary prices can recover strong unilateral price improvements without exhaustive search over all offloading sets.
5. The iterative pricing algorithm reduces the boundary-price-restricted NE gap and produces a reasonable epsilon-approximate equilibrium of the Stage I pricing game.
6. The full Stackelberg framework produces economically meaningful differences in prices, revenues, user social cost, and resource utilization relative to baseline strategic settings.
7. The main conclusions should remain reasonably stable under key sensitivity knobs such as:
   - the local candidate-family size `Q`;
   - resource capacities `F` and `B`;
   - provider costs `c_E` and `c_N`;
   - user-population distributions and heterogeneity levels.

Stage I implementation policy for experiments:

- main paper-facing Stage I figures should use `stage1_solver_variant = "paper_iterative_pricing"`;
- if `vbbr_brd` is reported, it should be labeled as an alternative Stage I implementation path, not as the canonical paper algorithm;
- direct comparisons between `paper_iterative_pricing` and `vbbr_brd` belong in supplementary diagnostics, robustness checks, or implementation notes rather than the main theorem-validation storyline.

---

## Baseline Taxonomy

Use the current paper's baseline grouping.

### Baselines for Stackelberg Equilibrium

- GSO
- GA
- BO
- MARL

### Baselines for Strategic Settings

- ME
- SingleSP
- Coop
- Rand

Do not use old baseline groupings such as `PBRD` or `DRL` in the main paper-aligned narrative unless the paper source is explicitly revised.

The figure-level canonical plan is frozen in `docs/FIGURE_SCRIPT_BLUEPRINT.md`. The public experiment entry points have been implemented according to that one-figure-one-script mapping.

---

## Recommended Experiment Blocks

The experiments should be organized by paper contribution, not by legacy implementation stage names.

### Block A: Stage II SCM quality

Purpose:

- validate the integrated Stage II SCM route formed by Algorithm 2 and Algorithm 1.

Recommended outputs:

- final social cost compared with centralized exact solving on small instances;
- empirical approximation ratio
  `V(X_DG) / V(X*)`;
- theorem upper bound from the paper;
- overall Stage II runtime;
- rollback diagnostics:
  - number of rollback events,
  - accepted admissions,
  - final offloading-set size.

Baselines:

- centralized exact solver on small instances.

### Block B: Stage I geometry and boundary-price diagnostics

Purpose:

- validate the structural analysis used by the Stage I solver.

Recommended outputs:

- small-instance price heatmaps or contours for social cost / revenue / NE-gap-style diagnostics;
- fixed-opponent revenue slices;
- overlays of switching-price or boundary-price candidates;
- local candidate-family diagnostics around the current SOE.

Expected takeaway:

- useful price moves cluster near switching boundaries;
- the boundary-price construction is a practical approximation to locally relevant regime changes.

### Block C: Stage I iterative-pricing convergence

Purpose:

- validate the iterative pricing algorithm itself.

Recommended outputs:

- current and final boundary-price-restricted NE gap over iterations;
- ESP / NSP best-response gains over iterations;
- price trajectory in the joint pricing space;
- final price pair and Stage II SOE outcome.

Baselines:

- GSO on small instances;
- GA, BO, MARL under comparable evaluation budgets where appropriate.

Implementation rule:

- Block C should treat `paper_iterative_pricing` as the primary proposed method;
- `vbbr_brd` may be added as a supplementary implementation comparison, but should not displace the paper-facing trajectory as the main curve.

### Block D: Runtime and scalability

Purpose:

- quantify practical cost of the proposed framework.

Recommended outputs:

- Stage II SCM runtime vs. number of users;
- Stage I runtime vs. number of users;
- centralized exact solving time where feasible;
- number of Stage II solves invoked by Stage I;

### Block E: Strategic-setting comparisons

Purpose:

- validate the paper's economic conclusions.

Recommended outputs:

- total user social cost;
- ESP revenue;
- NSP revenue;
- joint provider revenue;
- computation and bandwidth utilization.

Compared methods:

- full model;
- ME;
- SingleSP;
- Coop;
- Rand.

Suggested sweep variables:

- number of users;
- resource asymmetry such as `F/B`;
- provider-cost settings where relevant.

### Block F: Robustness and sensitivity

Purpose:

- show that the main conclusions are not artifacts of one narrow parameter setting.

Recommended placement:

- supplementary section or appendix if the main text is crowded.

Recommended sweeps:

- `Q`:
  sensitivity of the Stage I local candidate family size
- `F`, `B`, or `F/B`:
  resource-capacity and asymmetry sensitivity
- `c_E`, `c_N`:
  provider-cost sensitivity
- user distributions:
  task size, workload, channel quality, or preference heterogeneity

Recommended outputs:

- final Stage II social cost;
- approximation ratio on small instances where exact comparison is feasible;
- final Stage I boundary-price-restricted NE gap;
- final price pair;
- ESP / NSP revenue;
- offloading-set size;
- runtime.

Expected takeaway:

- the qualitative conclusions should remain stable across moderate parameter changes;
- `Q` should exhibit a cost-quality tradeoff rather than an unstable behavior cliff;
- changes in `F/B`, `c_E/c_N`, and user heterogeneity should shift equilibrium outcomes in interpretable ways rather than invalidate the method.

---

## Metrics to Report

Primary paper-aligned metrics:

- Stage II social cost
- approximation ratio
- rollback count
- runtime
- offloading-set size
- ESP revenue
- NSP revenue
- joint provider revenue
- resource utilization
- NE gap `delta(p)` where available
- boundary-price-restricted NE gap

Secondary diagnostics that may still be useful but should not replace the formal paper metrics:

- exploitability-style checks
- dense-grid heatmaps
- auxiliary proxy surfaces

If secondary diagnostics are included, they should be labeled as supplementary.

---

## Experimental Design Rules

1. Small-scale instances are for exact comparison:
   use them when centralized exact solving is needed for ground truth.
2. Medium- and large-scale instances are for scalability:
   use them to evaluate Stage II SCM runtime and the iterative Stage I pipeline.
3. Any figure that uses legacy script names must still be described using current paper terminology.
4. Do not elevate implementation-only heuristics to paper contributions unless they appear in the paper source.
5. Keep Stage II and Stage I logically separated in reporting:
   Stage II validates the follower-side solution mechanism;
   Stage I validates the pricing-game structure and equilibrium computation.
6. Every main figure in Block A-F should map to exactly one canonical script;
   current multi-output scripts are reusable sources, not the final public figure interface.

---

## Implementation Notes

- Preserve the distinction between:
  - the exact paper concepts;
  - simulation diagnostics;
  - legacy implementation names.
- Fixed-set closed forms are important for Stage I structural analysis, but they do not replace the paper's formal role of Algorithm 1 and Algorithm 2 in Stage II.
- The current paper's Stage I search should be described in terms of boundary prices, candidate families, and best-response gains, not in terms of old "verified-gap" language.
- When scripts expose old names, prefer adding a documentation mapping rather than rewriting the paper around those names.

---

## Recommended Main-Text Storyline

1. Introduce the simulation settings and the two baseline groups.
2. Validate the integrated Stage II SCM solver on social cost, approximation ratio, and rollback behavior.
3. Show the Stage I pricing-space structure through fixed-opponent slices, heatmaps, and boundary-price diagnostics.
4. Show convergence of the iterative pricing algorithm and compare against Stackelberg-equilibrium baselines.
5. Report strategic-setting comparisons to demonstrate the economic consequences of modeling both selfish providers and competitive users.

This storyline matches the current paper's mathematical structure and avoids importing claims from older experiment plans.

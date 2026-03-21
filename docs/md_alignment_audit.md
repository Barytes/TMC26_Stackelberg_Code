# Markdown Documentation Alignment Audit

## Purpose

This document summarizes how the current Markdown documents in the repository differ from the latest paper source in `TMC26_Stackelberg.tex`.

The paper source is treated as the ground truth. In particular, the current paper defines:

- a two-stage MLMF Stackelberg game;
- Stage II as a user offloading game with GNE existence, SOE selection, SCM formulation, Algorithm 1 (distributed primal-dual inner solver), and Algorithm 2 (heuristic distributed offloading-user selection with rollback and an approximation bound);
- Stage I as a pricing game analyzed through restricted pricing spaces, switching sets, local candidate family construction, candidate boundary prices, best-response estimation via boundary prices, and an iterative pricing algorithm for an epsilon-approximate Nash equilibrium;
- experiments centered on simulation settings and two baseline groups:
  - Stackelberg-equilibrium baselines: GSO, GA, BO, MARL;
  - strategic-setting baselines: ME, SingleSP, Coop, Rand.

Key paper anchors:

- `TMC26_Stackelberg.tex`: abstract and contributions
- `TMC26_Stackelberg.tex`: Stage II analysis and Algorithms 1-2
- `TMC26_Stackelberg.tex`: Stage I analysis and Algorithms 3-5
- `TMC26_Stackelberg.tex`: experimental setup and baseline methods

---

## Overall Assessment

The current Markdown documents fall into two groups.

Documents with major divergence from the latest paper:

- `README.md`
- `docs/SPEC.md`
- `docs/stage2_vi_penalty_implementation_guide.md`
- `CLAUDE.md`

Documents that are partly aligned but still contain stale terminology or old experimental framing:

- `docs/DEV.md`
- `review.md`

The main source of divergence is that several Markdown documents still follow an older VBBR / verified-gap / heavy-ablation narrative, while the current paper is written around:

- SOE / SCM in Stage II;
- restricted pricing space and boundary-price best-response estimation in Stage I;
- NE gap / restricted NE gap;
- baseline sets `{GSO, GA, BO, MARL}` and `{ME, SingleSP, Coop, Rand}`.

---

## File-by-File Audit

### `README.md`

Severity: high

Issues:

- The `17 Figures` and `Phase 1-5` structure is not the current paper structure.
- The README uses old language such as `guided-search pipeline`, `Algorithm 5 epsilon convergence`, and figure-grouping around old experiment planning.
- It references `docs/experiment_figures.md`, which does not exist.
- It lists many scripts that do not exist in `scripts/`.
- The quick-test section uses commands for non-existent scripts.
- The run/output/configuration explanation is still centered on old metric-surface and planning language, not on the current paper's Stage II SOE computation and Stage I iterative pricing framework.

Required correction:

- Rewrite the README around the current paper contributions and the actual scripts that exist.
- Remove the old 17-figure matrix unless a new figure matrix is rebuilt from the latest paper and the actual repository state.
- Replace old terms with paper-consistent terminology:
  - `SOE`, `SCM`, `Algorithm 1`, `Algorithm 2`
  - `restricted pricing space`
  - `candidate boundary price`
  - `best-response estimation`
  - `iterative pricing`
  - `NE gap` / `restricted NE gap`

---

### `docs/SPEC.md`

Severity: high

Issues:

- The entire top-level objective is built around `VBBR-BRD`, `verified unilateral gain`, and `verified epsilon`.
- The main claims section assumes old mechanisms such as:
  - best-so-far tracking
  - monotone acceptance
  - safe cyclic policy
  - lazy verification
  - singleton-first candidate generation
  - periodic full-epsilon audit
  - coarse-to-fine refinement
- The figure plan is written as a large VBBR-centered experiment program, not as the current paper's algorithm-and-baseline structure.
- The document uses outdated baseline sets such as `PBRD` and `DRL`, while the current paper uses `GA` and `MARL`, and explicitly includes `Coop` in strategic-setting baselines.
- It elevates `exploitability`, `verified epsilon`, and Stage-II call breakdown to primary metrics, but these are not the paper's central formal metrics.
- The implementation notes and storyline continue to treat Stage I as a verified-gap search process rather than the current boundary-price-restricted best-response dynamics.

Required correction:

- Rewrite the document from scratch.
- Redefine the experiment objective around the current paper's real claims:
  - Stage II:
    - GNE existence and multiple equilibria context
    - SOE via SCM
    - Algorithm 1 convergence / runtime / communication
    - Algorithm 2 approximation quality and rollback behavior
  - Stage I:
    - restricted pricing spaces
    - switching set structure
    - local candidate family `N_Q(p)`
    - candidate boundary prices
    - best-response estimation via boundary prices
    - iterative pricing under alternating updates
    - `delta(p)` and `hat-delta(p)` rather than `verified epsilon`
- Replace stale baselines with the current paper baselines:
  - GSO, GA, BO, MARL
  - ME, SingleSP, Coop, Rand
- If a figure plan is retained, rebuild it from the current paper rather than patching the old plan.

---

### `docs/DEV.md`

Severity: medium

Issues:

- The document is broadly aligned with the latest paper, but it still contains some wording that can mislead implementation toward the old narrative.
- It suggests that the fixed-set closed-form inner solution is mainly an appendix-side implementation trick; in the current paper, the closed-form fixed-set expressions are already part of the formal Stage I restricted-space analysis.
- It still uses old naming such as `vbbr trajectory`.
- It should make clearer that the formal Stage I metric in the current paper is the restricted NE gap, not a verified-gap metric.

Required correction:

- Keep the document, but edit terminology and factual phrasing.
- Make the following consistent with the paper:
  - the role of Algorithm 1 vs. fixed-set closed forms;
  - the Stage I metric names;
  - trajectory naming around boundary-price / iterative-pricing search;
  - baseline names and experiment naming.

---

### `docs/stage2_vi_penalty_implementation_guide.md`

Severity: high

Issues:

- It references `bare_jrnl_new_sample4.tex`, which is not the current paper source.
- It presents VI and PEN as if they are central Stage II implementation guidance, but they are not part of the current paper's formal algorithm set or formal baseline list.
- It risks confusing auxiliary exploratory baselines with the paper's official Stage II solution route.

Required correction:

- Either:
  - move this file into an archive / optional-baselines area, or
  - rename and rewrite it as an optional non-paper baseline note.
- Explicitly state that the paper's official Stage II route is:
  - Algorithm 1: distributed primal-dual inner solver
  - Algorithm 2: heuristic distributed offloading-user selection
- Remove the obsolete paper reference and re-anchor the document to `TMC26_Stackelberg.tex`, if the file is retained.

---

### `review.md`

Severity: medium

Issues:

- This is not a main implementation guide, but it still contains old terminology from an earlier draft line of thinking.
- It uses terms such as `RNE`, `BRGM`, `sampling density L`, `PBRD`, and `DRL`, which are not the naming used in the current paper.
- Some concerns remain useful, but the terminology should be updated so the memo is interpretable against the current paper.

Required correction:

- Either:
  - label it clearly as a historical review memo, or
  - revise it to the current terminology.
- If revised, align the document to:
  - boundary-price-based best-response estimation
  - iterative pricing
  - candidate family size parameter `Q`
  - baseline sets using `GA` and `MARL`

---

### `CLAUDE.md`

Severity: high

Issues:

- It instructs agents to follow `docs/SPEC.md`.
- Since `docs/SPEC.md` is currently the most outdated planning document, this creates a persistent source-of-truth error for any future agent work.

Required correction:

- Change the priority order to:
  1. `TMC26_Stackelberg.tex` is the primary source of truth.
  2. `docs/DEV.md` is an implementation aid.
  3. `docs/SPEC.md` should only be followed after it has been rewritten to match the latest paper.

---

## Terminology Replacement Guide

The following replacements should be applied consistently when revising the docs.

- `VBBR-BRD` -> `boundary-price-based best-response estimation + iterative pricing algorithm`
- `verified epsilon` -> `NE gap` or `restricted NE gap`, depending on context
- `verified unilateral gain` -> `best-response gain` or `boundary-price-restricted best-response gain`
- `outer policy ablation` -> remove unless redefined under the current alternating-update algorithm
- `PBRD` / `DRL` -> remove from the main paper narrative
- `DRL` -> `MARL` where the paper refers to the learning baseline
- old stage-I call-reduction mechanisms -> remove from the paper-aligned main narrative unless explicitly marked as implementation extensions

---

## Repair Plan

### Phase 1: Establish the source of truth

- Create a concise terminology and algorithm map from `TMC26_Stackelberg.tex`.
- Fix the source-of-truth order in `CLAUDE.md`.

### Phase 2: Rewrite the public-facing docs

- Rewrite `README.md`.
- Rewrite `docs/SPEC.md`.

### Phase 3: Clean the implementation-support docs

- Edit `docs/DEV.md` to remove stale VBBR-era wording.
- Reclassify or archive `docs/stage2_vi_penalty_implementation_guide.md`.

### Phase 4: Clean residual notes

- Update or relabel `review.md`.
- Remove broken links, missing-file references, and dead script commands.

### Phase 5: Consistency verification

- Re-scan every Markdown file against:
  - the latest paper terminology;
  - actual files in `scripts/`;
  - actual files in `docs/`;
  - baseline names and algorithm names used in the paper.

---

## Recommended Next Action

The next practical step is:

1. rewrite `README.md`;
2. rewrite `docs/SPEC.md`;
3. then update `CLAUDE.md` and `docs/DEV.md`.

This order removes the highest-risk source-of-truth conflicts first.

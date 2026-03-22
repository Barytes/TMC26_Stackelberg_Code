# Final Consistency Audit

## Purpose

This document is the workflow-7 final consistency audit for the current repository state.

It checks four layers together:

- paper-facing documentation;
- public figure-script entry points;
- figure output schema and aggregation;
- remaining internal legacy naming that is still tolerated.

## Audit Scope

Audited sources:

- `README.md`
- `docs/SPEC.md`
- `docs/DEV.md`
- `docs/CHANGE_PLAN.md`
- `docs/TERM_INTERFACE_FREEZE.md`
- `docs/FIGURE_SCRIPT_BLUEPRINT.md`
- `docs/BASELINE_SCRIPT_MAPPING.md`
- `docs/OUTPUT_SCHEMA.md`
- `scripts/run_figure_*.py`
- `scripts/collect_figure_manifests.py`
- `src/tmc26_exp/stackelberg.py`
- `src/tmc26_exp/baselines.py`
- `src/tmc26_exp/config.py`

## Final Status

Overall result:

- paper-facing structure is now consistent enough to treat the repository as aligned with the current `TMC26_Stackelberg.tex` narrative;
- the public experiment interface is consistently `one figure -> one script`;
- outputs are consistently constrained under the repository `outputs/` directory;
- workflow 1-7 deliverables now exist on disk.

## Passed Checks

### 1. Public Experiment Interface

- all 29 planned figure IDs now exist as `scripts/run_figure_*.py` entries;
- public figure scripts no longer depend on placeholder behavior;
- public figure scripts no longer rely on subprocess wrapper chaining.

### 2. Stage II Narrative

- Stage II is consistently documented as the integrated follower-side SCM solver;
- paper-facing documentation no longer frames Algorithm 1 as a separate experimental track;
- Block A scripts and docs remain aligned with the `Algorithm 2 + Algorithm 1` integrated route.

### 3. Stage I Narrative

- paper-facing docs consistently describe Stage I as the restricted-pricing / boundary-price / iterative-pricing route;
- public figure script names use canonical figure IDs rather than legacy `vbbr` naming;
- old gap language is no longer used as the paper-facing main terminology.

### 4. Output Contract

- figure outputs are constrained under `outputs/`;
- relative `--out-dir` values resolve under `outputs/`;
- absolute `--out-dir` values outside `outputs/` are rejected;
- each figure output now contains:
  - primary `.png`
  - primary `.csv`
  - primary `_summary.txt`
  - `figure_manifest.json`
- figure CSV files now carry standard columns:
  - `figure_id`
  - `block`
- cross-run aggregation is available through `scripts/collect_figure_manifests.py`.

### 5. Documentation State

- `README.md` now matches the current figure-script-first usage model;
- `docs/FIGURE_SCRIPT_BLUEPRINT.md` now reflects that all figure scripts are direct runners;
- `docs/TERM_INTERFACE_FREEZE.md` now references the actual existing canonical figure scripts;
- `docs/OUTPUT_SCHEMA.md` now freezes the workflow-6 output contract.

## Fixes Applied During Workflow 7

- corrected stale canonical script names in `docs/TERM_INTERFACE_FREEZE.md`;
- updated figure-status rows in `docs/FIGURE_SCRIPT_BLUEPRINT.md` from old `new` / wrapper-era wording to `direct runner`;
- added this final audit document;
- updated workflow-6/7 documentation state in `docs/CHANGE_PLAN.md`.

## Accepted Residual Legacy

These items still exist, but they are now explicitly treated as internal or compatibility-only:

- code/config names such as `vbbr_*`, `pbdr_*`, `stage1_solver_variant = vbbr_brd`;
- internal result objects such as `VBBROracleResult`;
- legacy helper scripts such as `run_stage1_vbbr_trajectory_on_heatmap.py` and `plot_pbdr_trajectory_from_heatmap_csv.py`;
- `DRL` as the current code proxy for paper-facing `MARL`;
- missing dedicated public `Coop` baseline implementation.

These residuals are acceptable for the current state because they no longer define the public interface or the paper-facing documentation.

## Remaining Non-Blocking Gaps

- some figure summaries still leave `runtime_sec` blank because per-figure runtime instrumentation is not yet wired through every script;
- several internal helper scripts still expose legacy help text or legacy option names;
- the baseline layer still contains auxiliary and legacy methods that should not be confused with the paper-facing baseline taxonomy.

These are cleanup items, not blockers for the current aligned public interface.

## Completion Decision

Workflow 7 is considered complete when judged against the current plan:

- documentation, public scripts, and output schema are aligned;
- no public figure entry remains placeholder-based;
- no public figure entry writes outside the project `outputs/` tree;
- the repository has a final machine-readable aggregation path for figure outputs.

# Output Schema

## Purpose

This document is the workflow-6 output contract for all `scripts/run_figure_*.py` entries.

It freezes three things:

- the required artifact set in each figure output directory;
- the standard columns automatically added to figure CSV files;
- the standard summary and manifest structure used for downstream aggregation.

## Output Directory Rule

- all outputs must stay under the repository `outputs/` directory;
- relative `--out-dir` values are interpreted under `outputs/`;
- absolute `--out-dir` values outside `outputs/` are rejected.

## Required Artifacts

Each figure run directory must contain:

- one primary image: `<figure_id>_<short_name>.png`
- one primary CSV: `<figure_id>_<short_name>.csv`
- one primary summary: `<figure_id>_<short_name>_summary.txt`
- one standard manifest: `figure_manifest.json`

## Standard CSV Columns

Figure CSV files keep their figure-specific columns, but workflow 6 adds these standard leading columns automatically:

- `figure_id`
- `block`

This means different blocks can still preserve their native data schema while remaining uniformly indexable.

## Standard Summary Template

Every figure summary now uses the same section order:

- `[run]`
- `[artifacts]`
- `[inputs]`
- `[primary_metrics]`
- `[runtime]`
- `[details]` if extra lines are present

Minimum normalized fields:

- `figure_id`
- `block`
- `script`
- `output_dir`
- `primary_image`
- `primary_csv`
- `summary_file`
- `config`
- `seed`
- `n_users`
- `n_trials`
- `runtime_sec`

## Standard Manifest

`figure_manifest.json` is the machine-readable companion to the summary file.

Current normalized fields:

- `schema_version`
- `generated_at`
- `figure_id`
- `block`
- `base_stem`
- `script`
- `output_dir`
- `summary_file`
- `primary_image`
- `primary_csv`
- `config`
- `seed`
- `n_users`
- `n_trials`
- `runtime_sec`
- `primary_metrics`
- `detail_lines`

## Aggregation Entry

Workflow 6 adds:

- `scripts/collect_figure_manifests.py`

This script scans `outputs/` for `figure_manifest.json` files and produces:

- `figure_run_index.csv`
- `figure_run_index_summary.txt`

Both outputs are also written under `outputs/`.

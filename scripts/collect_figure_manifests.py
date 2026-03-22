from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _figure_output_schema import load_figure_manifest
from _figure_wrapper_utils import OUTPUTS_ROOT, resolve_out_dir, write_csv_rows


def _find_manifests(outputs_root: Path) -> list[Path]:
    return sorted(outputs_root.rglob("figure_manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect standardized figure manifests from outputs/ and build a unified run index.")
    parser.add_argument("--outputs-root", type=str, default=str(OUTPUTS_ROOT))
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    manifests = _find_manifests(outputs_root)
    out_dir = resolve_out_dir("collect_figure_manifests", args.out_dir)

    rows: list[dict[str, object]] = []
    block_counter: Counter[str] = Counter()
    figure_counter: Counter[str] = Counter()

    for manifest_path in manifests:
        manifest = load_figure_manifest(manifest_path)
        figure_id = str(manifest.get("figure_id", ""))
        block = str(manifest.get("block", ""))
        block_counter[block] += 1
        figure_counter[figure_id] += 1
        primary_metrics = manifest.get("primary_metrics", {})
        metric_keys = ""
        if isinstance(primary_metrics, dict):
            metric_keys = ";".join(sorted(str(key) for key in primary_metrics.keys()))
        rows.append(
            {
                "figure_id": figure_id,
                "block": block,
                "base_stem": manifest.get("base_stem", ""),
                "script": manifest.get("script", ""),
                "config": manifest.get("config", ""),
                "seed": manifest.get("seed", ""),
                "n_users": manifest.get("n_users", ""),
                "n_trials": manifest.get("n_trials", ""),
                "runtime_sec": manifest.get("runtime_sec", ""),
                "output_dir": manifest.get("output_dir", ""),
                "primary_image": manifest.get("primary_image", ""),
                "primary_csv": manifest.get("primary_csv", ""),
                "summary_file": manifest.get("summary_file", ""),
                "generated_at": manifest.get("generated_at", ""),
                "primary_metric_keys": metric_keys,
                "manifest_path": str(manifest_path),
            }
        )

    write_csv_rows(
        out_dir / "figure_run_index.csv",
        [
            "figure_id",
            "block",
            "base_stem",
            "script",
            "config",
            "seed",
            "n_users",
            "n_trials",
            "runtime_sec",
            "output_dir",
            "primary_image",
            "primary_csv",
            "summary_file",
            "generated_at",
            "primary_metric_keys",
            "manifest_path",
        ],
        rows,
    )

    summary_lines = [
        f"outputs_root = {outputs_root}",
        f"manifest_count = {len(manifests)}",
        "",
        "[blocks]",
    ]
    for block in sorted(block_counter):
        summary_lines.append(f"{block} = {block_counter[block]}")
    summary_lines.extend(["", "[figures]"])
    for figure_id in sorted(figure_counter):
        summary_lines.append(f"{figure_id} = {figure_counter[figure_id]}")
    (out_dir / "figure_run_index_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

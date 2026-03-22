from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re


_SUMMARY_SUFFIX = "_summary"
_FIGURE_RE = re.compile(r"^(?P<figure_id>[A-F]\d+)_(?P<short_name>.+)$")
_RESERVED_KEYS = {
    "config",
    "seed",
    "n_users",
    "n_users_list",
    "trials",
    "trials_per_n",
    "runtime_sec",
    "output_dir",
    "primary_image",
    "primary_csv",
    "summary_file",
    "figure_id",
    "block",
    "script",
}


def infer_figure_output_info(path: Path) -> dict[str, str] | None:
    stem = path.stem
    base_stem = stem[: -len(_SUMMARY_SUFFIX)] if stem.endswith(_SUMMARY_SUFFIX) else stem
    match = _FIGURE_RE.match(base_stem)
    if match is None:
        return None
    figure_id = match.group("figure_id")
    return {
        "figure_id": figure_id,
        "block": figure_id[0],
        "base_stem": base_stem,
        "script": f"run_figure_{base_stem}.py",
        "primary_image": f"{base_stem}.png",
        "primary_csv": f"{base_stem}.csv",
        "summary_file": f"{base_stem}_summary.txt",
    }


def augment_figure_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    info = infer_figure_output_info(path)
    if info is None or path.suffix.lower() != ".csv":
        return fieldnames, rows

    prefixed = list(fieldnames)
    prefix_cols: list[str] = []
    for col in ("figure_id", "block"):
        if col not in prefixed:
            prefix_cols.append(col)
    if prefix_cols:
        prefixed = prefix_cols + prefixed

    out_rows: list[dict[str, object]] = []
    for row in rows:
        merged = dict(row)
        merged.setdefault("figure_id", info["figure_id"])
        merged.setdefault("block", info["block"])
        out_rows.append(merged)
    return prefixed, out_rows


def _parse_summary_lines(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    kv: dict[str, str] = {}
    extra: list[str] = []
    for raw in lines:
        line = str(raw).rstrip()
        if not line:
            extra.append("")
            continue
        if "=" not in line or line.startswith("[") or line.startswith("---"):
            extra.append(line)
            continue
        key, value = line.split("=", 1)
        kv[key.strip()] = value.strip()
    return kv, extra


def _pick_runtime(kv: dict[str, str]) -> str:
    for key in (
        "runtime_sec",
        "algorithm2_runtime_sec",
        "mean_runtime_sec",
        "ga_runtime_sec",
        "bo_runtime_sec",
    ):
        value = kv.get(key, "")
        if value:
            return value
    return ""


def write_standard_figure_summary(path: Path, lines: list[str]) -> None:
    info = infer_figure_output_info(path)
    if info is None:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    kv, extra_lines = _parse_summary_lines(lines)
    output_dir = path.parent
    runtime_sec = _pick_runtime(kv)
    n_trials = kv.get("trials", kv.get("trials_per_n", ""))
    n_users_value = kv.get("n_users", kv.get("n_users_list", ""))

    primary_metrics = {
        key: value
        for key, value in kv.items()
        if key not in _RESERVED_KEYS
    }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "figure_id": info["figure_id"],
        "block": info["block"],
        "base_stem": info["base_stem"],
        "script": info["script"],
        "output_dir": str(output_dir),
        "summary_file": info["summary_file"],
        "primary_image": info["primary_image"],
        "primary_csv": info["primary_csv"],
        "config": kv.get("config", ""),
        "seed": kv.get("seed", ""),
        "n_users": n_users_value,
        "n_trials": n_trials,
        "runtime_sec": runtime_sec,
        "primary_metrics": primary_metrics,
        "detail_lines": extra_lines,
    }

    section_lines = [
        "[run]",
        f"figure_id = {info['figure_id']}",
        f"block = {info['block']}",
        f"script = {info['script']}",
        f"output_dir = {output_dir}",
        "",
        "[artifacts]",
        f"primary_image = {info['primary_image']}",
        f"primary_csv = {info['primary_csv']}",
        f"summary_file = {info['summary_file']}",
        "",
        "[inputs]",
        f"config = {kv.get('config', '')}",
        f"seed = {kv.get('seed', '')}",
        f"n_users = {n_users_value}",
        f"n_trials = {n_trials}",
        "",
        "[primary_metrics]",
    ]
    if primary_metrics:
        section_lines.extend(f"{key} = {value}" for key, value in primary_metrics.items())
    else:
        section_lines.append("none =")
    section_lines.extend(["", "[runtime]", f"runtime_sec = {runtime_sec}"])

    if extra_lines:
        section_lines.extend(["", "[details]", *extra_lines])

    path.write_text("\n".join(section_lines) + "\n", encoding="utf-8")
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_figure_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

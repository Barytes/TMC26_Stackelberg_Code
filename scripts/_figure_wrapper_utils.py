from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from _figure_output_schema import augment_figure_csv_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = ROOT / "outputs"


def default_out_dir(script_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUTS_ROOT / f"{script_stem}_{timestamp}"


def resolve_out_dir(script_stem: str, raw_out_dir: str | None) -> Path:
    if raw_out_dir is None:
        out_dir = default_out_dir(script_stem)
    else:
        requested = Path(raw_out_dir)
        if requested.is_absolute():
            try:
                requested.relative_to(OUTPUTS_ROOT)
            except ValueError as exc:
                raise ValueError(f"--out-dir must stay under {OUTPUTS_ROOT}") from exc
            out_dir = requested
        elif requested.parts and requested.parts[0] == OUTPUTS_ROOT.name:
            out_dir = ROOT / requested
        else:
            out_dir = OUTPUTS_ROOT / requested
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    fieldnames, rows = augment_figure_csv_rows(path, fieldnames, rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

#!/usr/bin/env python3
"""Runner stub for detailed experiments.

Intentionally not executed by default. Run manually on high-compute machines only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tmc26_exp.detailed_experiments import run_detailed_experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_detailed_experiments(args.config, Path(args.outdir))


if __name__ == "__main__":
    main()

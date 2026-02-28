from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .config import DetailedExperimentConfig, ExperimentConfig


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    title: str
    motivation: str
    baselines: list[str]
    metrics: list[str]
    settings: dict[str, str]
    trials: int
    expected_outputs: list[str]


@dataclass(frozen=True)
class ExperimentPlan:
    overview: str
    cases: list[ExperimentCase]
    notes: list[str]


def build_detailed_plan(cfg: ExperimentConfig) -> ExperimentPlan:
    cases = [
        ExperimentCase(
            case_id="A1",
            title="Core Performance Comparison",
            motivation="Fill missing quantitative results and validate superiority claims.",
            baselines=[
                "GSSE",
                "CS",
                "UBRD",
                "VI",
                "PEN",
                "GSO",
                "PBRD",
                "BO",
                "DRL",
                "MarketEquilibrium",
                "SingleSP",
                "RandomOffloading",
            ],
            metrics=["social_cost", "esp_revenue", "nsp_revenue", "epsilon_proxy", "runtime_sec"],
            settings={
                "n_users": "20, 40, 80, 120",
                "capacities": "default F/B",
                "pricing_bounds": f"[{cfg.system.cE}, {cfg.baselines.max_price_E}] x [{cfg.system.cN}, {cfg.baselines.max_price_N}]",
            },
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["table_core.csv", "fig_core_tradeoff.png"],
        ),
        ExperimentCase(
            case_id="A2",
            title="Gain Approximation Fidelity",
            motivation="Address review concern on epsilon certificate accuracy.",
            baselines=["Algorithm3Approx", "ExactBestResponseSmallN"],
            metrics=["gain_abs_error", "gain_rel_error", "epsilon_gap"],
            settings={
                "n_users": f"<= {cfg.baselines.exact_max_users}",
                "candidate_family": "default and expanded",
                "prices": "randomly sampled feasible points",
            },
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["table_gain_error.csv", "fig_gain_error_boxplot.png"],
        ),
        ExperimentCase(
            case_id="A3",
            title="Boundary Sampling Sensitivity",
            motivation="Quantify quality/runtime trade-off for Algorithm 4 direction count L.",
            baselines=["GSSE"],
            metrics=["epsilon_proxy", "provider_revenue_sum", "runtime_sec"],
            settings={"L_values": "4, 8, 12, 20, 32, 48"},
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["table_L_sensitivity.csv", "fig_L_vs_quality_runtime.png"],
        ),
        ExperimentCase(
            case_id="A4",
            title="Candidate Family Ablation",
            motivation="Validate impact of sensitive-user candidate family size in Algorithm 3.",
            baselines=["GSSE", "GSSE-ReducedFamily", "GSSE-ExpandedFamily"],
            metrics=["epsilon_proxy", "runtime_sec", "candidate_count"],
            settings={"family_rules": "boundary-only / medium / exhaustive-smallN"},
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["table_family_ablation.csv", "fig_family_ablation.png"],
        ),
        ExperimentCase(
            case_id="A5",
            title="Scalability and Communication",
            motivation="Address complexity/scalability concerns from review.",
            baselines=["GSSE", "PBRD", "BO", "DRL"],
            metrics=["runtime_sec", "inner_iterations", "search_steps", "message_count_proxy"],
            settings={"n_users": "20 to 500", "log_scale": "true"},
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["table_scalability.csv", "fig_scalability_curves.png"],
        ),
        ExperimentCase(
            case_id="A6",
            title="Robustness to Modeling Assumptions",
            motivation="Assess sensitivity to heterogeneity and capacity settings.",
            baselines=["GSSE", "MarketEquilibrium", "SingleSP"],
            metrics=["social_cost", "provider_revenue_sum", "offloading_ratio"],
            settings={
                "task_mix": "light/medium/heavy",
                "channel_quality": "low/medium/high sigma",
                "capacity": "F and B sweep",
            },
            trials=cfg.detailed_experiment.heavy_trials,
            expected_outputs=["table_robustness.csv", "fig_robustness_heatmaps.png"],
        ),
        ExperimentCase(
            case_id="A7",
            title="Convergence Profiles",
            motivation="Provide per-iteration convergence evidence for Algorithms 1/2/5.",
            baselines=["Algorithm1", "Algorithm2", "GSSE"],
            metrics=["duality_gap_proxy", "constraint_violation", "epsilon_proxy"],
            settings={"episodes": "multiple seeds", "trace_logging": "enabled"},
            trials=cfg.detailed_experiment.suggested_trials,
            expected_outputs=["trace_inner.csv", "trace_search.csv", "fig_convergence.png"],
        ),
    ]
    notes = [
        "All summary statistics should include mean, std, and 95% CI across seeds.",
        "Keep a small-scale sanity subset for reproducibility and exact-oracle comparisons.",
        "Do not execute this plan on constrained devices; use dedicated compute nodes.",
    ]
    return ExperimentPlan(
        overview=(
            "Detailed plan derived from review.md: fill missing results, quantify approximation "
            "error, run ablations on L/candidate-family, and report scalability plus robustness."
        ),
        cases=cases,
        notes=notes,
    )


def write_plan_markdown(plan: ExperimentPlan, out_path: Path) -> None:
    lines: list[str] = [
        "# Detailed Experiment Plan",
        "",
        plan.overview,
        "",
        "## Cases",
    ]
    for c in plan.cases:
        lines.append(f"### {c.case_id} - {c.title}")
        lines.append(f"- Motivation: {c.motivation}")
        lines.append(f"- Baselines: {', '.join(c.baselines)}")
        lines.append(f"- Metrics: {', '.join(c.metrics)}")
        lines.append(f"- Trials: {c.trials}")
        lines.append("- Settings:")
        for k, v in c.settings.items():
            lines.append(f"  - {k}: {v}")
        lines.append(f"- Expected outputs: {', '.join(c.expected_outputs)}")
        lines.append("")
    lines.append("## Notes")
    for note in plan.notes:
        lines.append(f"- {note}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plan_manifest_json(plan: ExperimentPlan, out_path: Path) -> None:
    data = {
        "overview": plan.overview,
        "cases": [asdict(c) for c in plan.cases],
        "notes": plan.notes,
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_nonexecuted_runner_stub(out_path: Path) -> None:
    script = """#!/usr/bin/env python3
\"\"\"Runner stub for detailed experiments.

Intentionally not executed by default. Run manually on high-compute machines only.
\"\"\"

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
"""
    out_path.write_text(script, encoding="utf-8")

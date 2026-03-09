from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .baselines import run_all_baselines
from .config import load_config
from .experiment_plan import (
    build_detailed_plan,
    write_nonexecuted_runner_stub,
    write_plan_manifest_json,
    write_plan_markdown,
)
from .metrics import METRICS, get_metric
from .plotting import plot_surface_contour, plot_surface_heatmap
from .simulator import evaluate_metric_surface, sample_users
from .stackelberg import run_stage1_solver, summarize_stackelberg_result


def main() -> None:
    parser = argparse.ArgumentParser(description="TMC26 experiment infrastructure")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML config")
    parser.add_argument("--run-baselines", action="store_true", help="Run baseline methods on one sampled user batch.")
    parser.add_argument(
        "--emit-detailed-plan",
        action="store_true",
        help="Generate detailed experiment plan files only (does not run heavy experiments).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_names = cfg.metrics if cfg.metrics else sorted(METRICS)

    for name in metric_names:
        metric = get_metric(name)
        surface = evaluate_metric_surface(cfg, metric)

        save_surface_csv(surface, run_dir / f"{metric.name}.csv")
        plot_surface_heatmap(surface, run_dir / f"{metric.name}_heatmap.png")
        plot_surface_contour(surface, run_dir / f"{metric.name}_contour.png")

    if cfg.stackelberg.enabled:
        users = sample_users(cfg, np.random.default_rng(cfg.seed))
        result = run_stage1_solver(users, cfg.system, cfg.stackelberg)
        (run_dir / "stackelberg_summary.txt").write_text(
            summarize_stackelberg_result(users, result, cfg.system),
            encoding="utf-8",
        )
        save_stackelberg_trajectory_csv(result, run_dir / "stackelberg_trajectory.csv")
        save_stackelberg_allocation_csv(result, run_dir / "stackelberg_allocation.csv")

        if cfg.baselines.enabled or args.run_baselines:
            baseline_rows = run_all_baselines(users, cfg.system, cfg.stackelberg, cfg.baselines)
            save_baselines_csv(baseline_rows, run_dir / "baselines_summary.csv")

    if cfg.detailed_experiment.emit_plan or args.emit_detailed_plan:
        plan = build_detailed_plan(cfg)
        plan_dir = run_dir / cfg.detailed_experiment.output_subdir
        plan_dir.mkdir(parents=True, exist_ok=True)
        write_plan_markdown(plan, plan_dir / "detailed_experiment_plan.md")
        write_plan_manifest_json(plan, plan_dir / "detailed_experiment_plan.json")
        write_nonexecuted_runner_stub(plan_dir / "run_detailed_experiments.py")

    write_summary(cfg, metric_names, run_dir / "metrics_summary.txt")
    print(f"Done. Results written to: {run_dir}")


def save_surface_csv(surface, out_path: Path) -> None:
    rows: list[str] = ["pE,pN,value_mean,value_std"]
    for n_idx, pN in enumerate(surface.pN_values):
        for e_idx, pE in enumerate(surface.pE_values):
            rows.append(
                f"{pE:.10g},{pN:.10g},{surface.mean_values[n_idx, e_idx]:.10g},{surface.std_values[n_idx, e_idx]:.10g}"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_summary(cfg, metric_names: list[str], out_path: Path) -> None:
    text = "\n".join(
        [
            f"run_name = {cfg.run_name}",
            f"n_users = {cfg.n_users}",
            f"n_trials = {cfg.n_trials}",
            f"seed = {cfg.seed}",
            f"pE_range = [{cfg.pE.min}, {cfg.pE.max}], points={cfg.pE.points}",
            f"pN_range = [{cfg.pN.min}, {cfg.pN.max}], points={cfg.pN.points}",
            f"metrics = {', '.join(metric_names)}",
            f"stackelberg_enabled = {cfg.stackelberg.enabled}",
            f"baselines_enabled = {cfg.baselines.enabled}",
            f"detailed_plan_emit = {cfg.detailed_experiment.emit_plan}",
            "",
            "Metric surfaces are simulation-based diagnostics.",
            "Stackelberg algorithm outputs (if enabled) are saved as stackelberg_*.{txt,csv}.",
            "Detailed experiment plan files only describe/prepare heavy runs and are not executed automatically.",
        ]
    )
    out_path.write_text(text + "\n", encoding="utf-8")


def save_stackelberg_trajectory_csv(result, out_path: Path) -> None:
    rows = [
        "iteration,pE,pN,epsilon,offloading_size,epsilon_delta,esp_best_set_size,nsp_best_set_size,esp_gain,nsp_gain"
    ]
    for step in result.trajectory:
        rows.append(
            f"{step.iteration},{step.pE:.10g},{step.pN:.10g},{step.epsilon:.10g},{len(step.offloading_set)},"
            f"{step.epsilon_delta:.10g},{step.esp_best_set_size},{step.nsp_best_set_size},{step.esp_gain:.10g},{step.nsp_gain:.10g}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_stackelberg_allocation_csv(result, out_path: Path) -> None:
    off = set(result.offloading_set)
    inner = result.inner_result
    rows = ["user_idx,offload,f,b,mu"]
    for i in range(inner.f.size):
        rows.append(
            f"{i},{1 if i in off else 0},{inner.f[i]:.10g},{inner.b[i]:.10g},{inner.mu[i]:.10g}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_baselines_csv(results, out_path: Path) -> None:
    rows = ["method,pE,pN,offloading_size,social_cost,esp_revenue,nsp_revenue,epsilon_proxy,meta_json"]
    for r in results:
        meta_json = str(r.meta).replace(",", ";")
        rows.append(
            f"{r.name},{r.price[0]:.10g},{r.price[1]:.10g},{len(r.offloading_set)},"
            f"{r.social_cost:.10g},{r.esp_revenue:.10g},{r.nsp_revenue:.10g},{r.epsilon_proxy:.10g},{meta_json}"
        )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

cache_root = Path(os.environ.get("TMC26_CACHE_DIR", str(ROOT / "outputs" / "_tmp_cache" / "tmc26_cache")))
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from _figure_wrapper_utils import resolve_out_dir, write_csv_rows
from _figure_missing_impl import _load_cfg, _sample_users, _write_summary
import plot_stage1_trajectories_on_heatmap as s1_compare
import run_algorithm2_exploitability_vs_users as s2_exploit
import run_boundary_hypothesis_check as boundary_diag
import run_stage1_price_heatmaps as s1_heatmaps
import run_stage1_vbbr_trajectory_on_heatmap as s1_traj
import run_stage2_approximation_ratio as s2_ratio
import run_stage2_social_cost_compare as s2_compare
from tmc26_exp.baselines import evaluate_stage1_price_grid
from tmc26_exp.config import load_config
from tmc26_exp.simulator import sample_users
from tmc26_exp.stackelberg import _build_data, solve_stage1_pricing, solve_stage2_scm


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _sample_once(cfg, n_users: int, seed: int):
    cfg_n = replace(cfg, n_users=int(n_users))
    return sample_users(cfg_n, np.random.default_rng(int(seed)))


def _run_stage2_trace_bundle(
    *,
    config_path: str,
    cfg,
    seed: int,
    n_list: list[int],
    pE: float,
    pN: float,
    inner_solver: str,
    centralized_solver: str,
    centralized_max_n: int | None,
) -> tuple[dict[int, list[float]], dict[int, float | None], list[dict[str, object]], list[str]]:
    traces_by_n: dict[int, list[float]] = {}
    centralized_by_n: dict[int, float | None] = {}
    rows: list[dict[str, object]] = []
    summary_lines = [
        f"config = {config_path}",
        f"seed = {seed}",
        f"pE = {float(pE):.10g}",
        f"pN = {float(pN):.10g}",
        f"algorithm2_inner_solver = {inner_solver}",
        f"centralized_mode = {centralized_solver}",
        f"n_users_list = {','.join(str(n) for n in n_list)}",
    ]
    for n in n_list:
        users = s2_compare._sample_users_for_n(cfg, n_users=n, seed=int(seed) + int(n))
        greedy_result = solve_stage2_scm(
            users=users,
            pE=float(pE),
            pN=float(pN),
            system=cfg.system,
            cfg=cfg.stackelberg,
            inner_solver_mode=inner_solver,
        )
        trace = list(greedy_result.social_cost_trace)
        centralized_social: float | None = None
        centralized_name = "skipped"
        if centralized_solver != "skip" and (centralized_max_n is None or int(n) <= int(centralized_max_n)):
            cs_out, success = s2_ratio._run_centralized(users, float(pE), float(pN), cfg.system, cfg, centralized_solver)
            centralized_name = str(cs_out.meta.get("solver", centralized_solver))
            if success:
                centralized_social = float(cs_out.social_cost)
            else:
                centralized_name = f"{centralized_name}_failed"
        traces_by_n[int(n)] = trace
        centralized_by_n[int(n)] = centralized_social
        summary_lines.extend(
            [
                f"--- n={n} ---",
                f"algorithm2_iterations = {greedy_result.iterations}",
                f"algorithm2_final_social_cost = {float(greedy_result.social_cost):.10g}",
                f"algorithm2_runtime_sec = {float(greedy_result.runtime_sec):.10g}",
                f"algorithm2_rollbacks = {greedy_result.rollback_count}",
                f"algorithm2_accepted_admissions = {greedy_result.accepted_admissions}",
                f"algorithm2_inner_calls = {greedy_result.inner_call_count}",
                f"algorithm2_final_offloading_size = {len(greedy_result.offloading_set)}",
                f"algorithm2_used_exact_inner = {int(greedy_result.used_exact_inner)}",
                f"centralized_social_cost = {'' if centralized_social is None else f'{centralized_social:.10g}'}",
                f"centralized_solver = {centralized_name}",
            ]
        )
        for i, value in enumerate(trace, start=1):
            rows.append(
                {
                    "n_users": int(n),
                    "iteration": int(i),
                    "social_cost": float(value),
                    "centralized_social_cost": "" if centralized_social is None else float(centralized_social),
                }
            )
    return traces_by_n, centralized_by_n, rows, summary_lines


def _grid_eval_bundle(
    *,
    cfg,
    n_users: int,
    seed: int,
    pEmax: float,
    pNmax: float,
    pE_points: int,
    pN_points: int,
):
    cfg_n = replace(cfg, n_users=int(n_users))
    users = _sample_once(cfg_n, n_users, seed)
    grid = evaluate_stage1_price_grid(
        users=users,
        system=cfg_n.system,
        stack_cfg=cfg_n.stackelberg,
        base_cfg=cfg_n.baselines,
        pE_min=0.0,
        pE_max=float(pEmax),
        pN_min=0.0,
        pN_max=float(pNmax),
        pE_points=int(pE_points),
        pN_points=int(pN_points),
        stage2_method=None,
    )
    joint_rev = grid.esp_rev + grid.nsp_rev
    legacy_gain_proxy = np.asarray([[float(out.legacy_gain_proxy) for out in row] for row in grid.outcomes], dtype=float)
    eq_mask = grid.grid_ne_gap <= 1e-12
    representative = s1_heatmaps._select_equilibrium_representative(eq_mask, grid.grid_ne_gap, joint_rev, joint_rev)
    return cfg_n, users, grid, joint_rev, legacy_gain_proxy, eq_mask, representative


def _grid_alias_rows(grid, legacy_gain_proxy: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for j, pN in enumerate(grid.pN_grid):
        for i, pE in enumerate(grid.pE_grid):
            rows.append(
                {
                    "pE": float(pE),
                    "pN": float(pN),
                    "esp_revenue": float(grid.esp_rev[j, i]),
                    "nsp_revenue": float(grid.nsp_rev[j, i]),
                    "joint_revenue": float(grid.esp_rev[j, i] + grid.nsp_rev[j, i]),
                    "grid_ne_gap": float(grid.grid_ne_gap[j, i]),
                    "legacy_gain_proxy": float(legacy_gain_proxy[j, i]),
                }
            )
    return rows


def _boundary_bundle(
    *,
    cfg,
    n_users: int,
    seed: int,
    start_pE: float,
    start_pN: float,
    pEmax: float,
    pNmax: float,
    pE_points: int = 81,
    pN_points: int = 81,
    scan_pE_points: int = 801,
    scan_pN_points: int = 1001,
    root_tol: float = 1e-5,
):
    cfg_n = replace(cfg, n_users=int(n_users))
    users = _sample_once(cfg_n, n_users, seed)
    data = _build_data(users)
    system = cfg_n.system
    stack_cfg = cfg_n.stackelberg
    start_pE = max(float(start_pE), float(system.cE))
    start_pN = max(float(start_pN), float(system.cN))
    start_eval = boundary_diag._evaluate_slice_point("E", start_pE, start_pN, users, system, stack_cfg)
    current_set = start_eval.offloading_set
    old_rows, old_unique, old_raw_counts = boundary_diag._compute_old_boundary_points(data, current_set, start_pE, start_pN, system)
    _, esp_slice, exact_esp = boundary_diag._scan_exact_boundaries(
        provider="E",
        fixed_price=start_pN,
        price_max=float(pEmax),
        scan_points=int(scan_pE_points),
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(root_tol),
    )
    _, nsp_slice, exact_nsp = boundary_diag._scan_exact_boundaries(
        provider="N",
        fixed_price=start_pE,
        price_max=float(pNmax),
        scan_points=int(scan_pN_points),
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(root_tol),
    )
    hypothesis_esp = boundary_diag._build_hypothesis_chain(
        provider="E",
        start_pE=start_pE,
        start_pN=start_pN,
        price_max=float(pEmax),
        users=users,
        data=data,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(root_tol),
    )
    hypothesis_nsp = boundary_diag._build_hypothesis_chain(
        provider="N",
        start_pE=start_pE,
        start_pN=start_pN,
        price_max=float(pNmax),
        users=users,
        data=data,
        system=system,
        stack_cfg=stack_cfg,
        root_tol=float(root_tol),
    )
    grid = evaluate_stage1_price_grid(
        users=users,
        system=system,
        stack_cfg=stack_cfg,
        base_cfg=cfg_n.baselines,
        pE_min=0.0,
        pE_max=float(pEmax),
        pN_min=0.0,
        pN_max=float(pNmax),
        pE_points=int(pE_points),
        pN_points=int(pN_points),
        stage2_method="DG",
    )
    joint_grid = grid.esp_rev + grid.nsp_rev
    step_e = float(pEmax - boundary_diag._axis_cost_floor("E", system)) / max(int(scan_pE_points) - 1, 1)
    step_n = float(pNmax - boundary_diag._axis_cost_floor("N", system)) / max(int(scan_pN_points) - 1, 1)
    match_tol = max(float(root_tol) * 8.0, 1.25 * max(step_e, step_n))
    stats = [
        boundary_diag._build_provider_stats("E", start_pN, exact_esp, old_raw_counts["E"], old_unique["E"], hypothesis_esp, match_tol),
        boundary_diag._build_provider_stats("N", start_pE, exact_nsp, old_raw_counts["N"], old_unique["N"], hypothesis_nsp, match_tol),
    ]
    return {
        "cfg": cfg_n,
        "users": users,
        "system": system,
        "stack_cfg": stack_cfg,
        "start_pE": start_pE,
        "start_pN": start_pN,
        "current_set": current_set,
        "old_rows": old_rows,
        "old_unique": old_unique,
        "exact_esp": exact_esp,
        "exact_nsp": exact_nsp,
        "hypothesis_esp": hypothesis_esp,
        "hypothesis_nsp": hypothesis_nsp,
        "esp_slice": esp_slice,
        "nsp_slice": nsp_slice,
        "grid": grid,
        "joint_grid": joint_grid,
        "match_tol": match_tol,
        "stats": stats,
    }


def _boundary_alias_rows(bundle: dict[str, object], provider: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in bundle["old_rows"]:
        if provider is not None and row["provider"] != provider:
            continue
        rows.append(
            {
                "provider": row["provider"],
                "source_type": "old",
                "price_value": float(row["boundary_price"]),
                "pE": float(row["pE"]),
                "pN": float(row["pN"]),
                "fixed_price": "",
                "event_type": "",
                "point_group": row["point_group"],
            }
        )
    event_groups = [
        ("exact", bundle["exact_esp"]),
        ("exact", bundle["exact_nsp"]),
        ("hypothesis", bundle["hypothesis_esp"]),
        ("hypothesis", bundle["hypothesis_nsp"]),
    ]
    for source_type, events in event_groups:
        for event in events:
            if provider is not None and event.provider != provider:
                continue
            price_value = event.boundary_price if source_type == "exact" else event.predicted_price
            pE, pN = boundary_diag._price_pair(event.provider, price_value, event.fixed_price)
            event_type = event.event_type if source_type == "exact" else event.realized_event_type
            rows.append(
                {
                    "provider": event.provider,
                    "source_type": source_type,
                    "price_value": float(price_value),
                    "pE": float(pE),
                    "pN": float(pN),
                    "fixed_price": float(event.fixed_price),
                    "event_type": event_type,
                    "point_group": "",
                }
            )
    return rows


def _trajectory_alias_rows(
    traj: list[tuple[float, float]],
    pE_grid: np.ndarray,
    pN_grid: np.ndarray,
    grid_ne_gap: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step, (pE, pN) in enumerate(traj):
        i = s1_traj._nearest_idx(pE_grid, float(pE))
        j = s1_traj._nearest_idx(pN_grid, float(pN))
        rows.append(
            {
                "step": int(step),
                "pE": float(pE),
                "pN": float(pN),
                "nearest_grid_pE": float(pE_grid[i]),
                "nearest_grid_pN": float(pN_grid[j]),
                "nearest_grid_grid_ne_gap": float(grid_ne_gap[j, i]),
            }
        )
    return rows


def _configure_multilang_font_support() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in available]
    if chosen:
        existing = list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["font.sans-serif"] = chosen + [name for name in existing if name not in chosen]
    plt.rcParams["axes.unicode_minus"] = False


def _stage2_trace_labels(lang: str) -> dict[str, str]:
    if lang == "zh":
        return {
            "title": "\u7b97\u6cd52\u7684\u6536\u655b\u6027",
            "xlabel": "\u8fed\u4ee3\u6b21\u6570",
            "ylabel": "\u793e\u4f1a\u6210\u672c",
            "alg2_prefix": "\u7b97\u6cd52",
            "cs_prefix": "\u96c6\u4e2d\u5f0f\u6700\u4f18",
        }
    if lang == "bilingual":
        return {
            "title": "\u7b2c\u4e8c\u9636\u6bb5\u793e\u4f1a\u6210\u672c\uff1a\u7b97\u6cd52\u4e0e\u96c6\u4e2d\u5f0f\u6700\u4f18\u5728\u4e0d\u540c\u7528\u6237\u89c4\u6a21\u4e0b\u7684\u5bf9\u6bd4 / Stage-II Social Cost: Algorithm 2 vs CS Across User Scales",
            "xlabel": "\u8fed\u4ee3\u6b21\u6570\uff08\u7b97\u6cd52\uff09 / Iteration (Algorithm 2)",
            "ylabel": "\u793e\u4f1a\u6210\u672c / Social Cost",
            "alg2_prefix": "\u7b97\u6cd52/Alg2",
            "cs_prefix": "\u96c6\u4e2d\u5f0f\u6700\u4f18/CS",
        }
    return {
        "title": "Convergence of Algorithm 2",
        "xlabel": "Iterations",
        "ylabel": "Social Cost",
        "alg2_prefix": "Alg2",
        "cs_prefix": "CS",
    }


def _plot_A1_multiscale_trace(
    *,
    traces_by_n: dict[int, list[float]],
    centralized_by_n: dict[int, float | None],
    out_path: Path,
    max_iter_plot: int | None,
    lang: str,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> None:
    _configure_multilang_font_support()
    labels = _stage2_trace_labels(lang)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    cmap = plt.get_cmap("tab10")
    ns = sorted(traces_by_n.keys())
    for idx, n in enumerate(ns):
        color = cmap(idx % 10)
        trace = traces_by_n[n]
        if max_iter_plot is not None:
            trace = trace[:max_iter_plot]
        x = np.arange(1, len(trace) + 1, dtype=int)
        ax.plot(
            x,
            np.asarray(trace, dtype=float),
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{labels['alg2_prefix']} n={n}",
        )
        cs_val = centralized_by_n.get(n)
        if cs_val is not None:
            ax.axhline(
                y=float(cs_val),
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                label=f"{labels['cs_prefix']} n={n}",
            )
    ax.set_xlabel(xlabel or labels["xlabel"])
    ax.set_ylabel(ylabel or labels["ylabel"])
    ax.set_title(title or labels["title"])
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _flush_A3_progress(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    args,
    seed: int,
    completed_trials: int,
    total_trials: int,
    completed: bool,
) -> None:
    s2_ratio._write_points_csv(out_dir / "A3_stage2_approx_ratio_bound.csv", rows)
    plot_info = s2_ratio._plot_scatter(rows, out_dir / "A3_stage2_approx_ratio_bound.png", transform=str(args.plot_transform))
    valid_rows = [row for row in rows if bool(row["valid"])]
    violation_count = sum(1 for row in valid_rows if bool(row["violated"]))
    _write_summary(
        out_dir / "A3_stage2_approx_ratio_bound_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {seed}",
            f"pE = {float(args.pE):.10g}",
            f"pN = {float(args.pN):.10g}",
            f"inner_solver = {args.inner_solver}",
            f"centralized_solver = {args.centralized_solver}",
            f"plot_transform = {args.plot_transform}",
            f"n_users_list = {args.n_users_list}",
            f"trials_per_n = {args.trials}",
            f"total_instances = {len(rows)}",
            f"valid_instances = {len(valid_rows)}",
            f"plot_valid_instances = {plot_info['valid_rows_used']}",
            f"plot_dropped_for_log = {plot_info['dropped_for_log']}",
            f"violations = {violation_count}",
            f"progress_completed_trials = {completed_trials}",
            f"progress_total_trials = {total_trials}",
            f"progress_status = {'completed' if completed else 'running'}",
        ],
    )


def main_A1() -> None:
    parser = argparse.ArgumentParser(description="Figure A1: Stage II social-cost traces across user scales.")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--n-users-list", type=str, default="50,100,150,200")
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--inner-solver", type=str, default="primal_dual", choices=["hybrid", "exact", "primal_dual"])
    parser.add_argument("--centralized-solver", type=str, default="pyomo_scip", choices=["enum", "gekko", "pyomo_scip", "skip"])
    parser.add_argument("--centralized-max-n", type=int, default=None)
    parser.add_argument("--max-iter-plot", type=int, default=12)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="Primary image language; the other language is emitted as an extra image.")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--xlabel", type=str, default=None)
    parser.add_argument("--ylabel", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_A1_stage2_social_cost_trace", args.out_dir)
    cfg = _load_cfg(args.config)
    n_list = [int(args.n_users)] if args.n_users is not None else _parse_int_list(args.n_users_list)
    centralized_max_n = max(n_list) if args.centralized_max_n is None else int(args.centralized_max_n)
    traces_by_n, centralized_by_n, rows, summary_lines = _run_stage2_trace_bundle(
        config_path=str(args.config),
        cfg=cfg,
        seed=int(args.seed),
        n_list=n_list,
        pE=float(args.pE),
        pN=float(args.pN),
        inner_solver=str(args.inner_solver),
        centralized_solver=str(args.centralized_solver),
        centralized_max_n=centralized_max_n,
    )
    primary_lang = str(args.lang)
    secondary_lang = "zh" if primary_lang == "en" else "en"
    _plot_A1_multiscale_trace(
        traces_by_n=traces_by_n,
        centralized_by_n=centralized_by_n,
        out_path=out_dir / "A1_stage2_social_cost_trace.png",
        max_iter_plot=args.max_iter_plot,
        lang=primary_lang,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    _plot_A1_multiscale_trace(
        traces_by_n=traces_by_n,
        centralized_by_n=centralized_by_n,
        out_path=out_dir / f"A1_stage2_social_cost_trace_{secondary_lang}.png",
        max_iter_plot=args.max_iter_plot,
        lang=secondary_lang,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    write_csv_rows(
        out_dir / "A1_stage2_social_cost_trace.csv",
        ["n_users", "iteration", "social_cost", "centralized_social_cost"],
        rows,
    )
    summary_lines.extend(
        [
            f"primary_lang = {primary_lang}",
            f"secondary_lang = {secondary_lang}",
            f"secondary_image = A1_stage2_social_cost_trace_{secondary_lang}.png",
            f"max_iter_plot = {'' if args.max_iter_plot is None else int(args.max_iter_plot)}",
        ]
    )
    _write_summary(out_dir / "A1_stage2_social_cost_trace_summary.txt", summary_lines)


def main_A2() -> None:
    parser = argparse.ArgumentParser(description="Figure A2: Stage II social-cost trace across user scales.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="8,12,16,20,40,60,80,100")
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--inner-solver", type=str, default="primal_dual", choices=["hybrid", "exact", "primal_dual"])
    parser.add_argument("--centralized-solver", type=str, default="enum", choices=["enum", "gekko", "pyomo_scip", "skip"])
    parser.add_argument("--centralized-max-n", type=int, default=16)
    parser.add_argument("--max-iter-plot", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_A2_stage2_social_cost_multiscale", args.out_dir)
    cfg = _load_cfg(args.config)
    n_list = _parse_int_list(args.n_users_list)
    traces_by_n, centralized_by_n, rows, summary_lines = _run_stage2_trace_bundle(
        config_path=str(args.config),
        cfg=cfg,
        seed=int(args.seed),
        n_list=n_list,
        pE=float(args.pE),
        pN=float(args.pN),
        inner_solver=str(args.inner_solver),
        centralized_solver=str(args.centralized_solver),
        centralized_max_n=int(args.centralized_max_n),
    )
    s2_compare._plot_multi_n_trace(
        traces_by_n=traces_by_n,
        centralized_by_n=centralized_by_n,
        out_path=out_dir / "A2_stage2_social_cost_multiscale.png",
        max_iter_plot=args.max_iter_plot,
    )
    write_csv_rows(
        out_dir / "A2_stage2_social_cost_multiscale.csv",
        ["n_users", "iteration", "social_cost", "centralized_social_cost"],
        rows,
    )
    _write_summary(out_dir / "A2_stage2_social_cost_multiscale_summary.txt", summary_lines)


def main_A3() -> None:
    parser = argparse.ArgumentParser(description="Figure A3: empirical approximation ratio versus theorem bound.")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="50,100,150,200")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--inner-solver", type=str, default="primal_dual", choices=["hybrid", "exact", "primal_dual"])
    parser.add_argument("--centralized-solver", type=str, default="pyomo_scip", choices=["gekko", "enum", "pyomo_scip"])
    parser.add_argument("--plot-transform", type=str, default="linear", choices=["linear", "logy", "loglog", "normalized", "margin", "margin_logfit"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_A3_stage2_approx_ratio_bound", args.out_dir)
    cfg = load_config(args.config)
    seed = int(args.seed)
    csv_path = out_dir / "A3_stage2_approx_ratio_bound.csv"
    rows: list[dict[str, object]] = list(s2_ratio._read_points_csv(csv_path)) if csv_path.exists() else []
    completed_pairs = {(int(row["n_users"]), int(row["trial"])) for row in rows}
    n_list = _parse_int_list(args.n_users_list)
    total_trials = len(n_list) * int(args.trials)
    completed_trials = len(completed_pairs)
    _flush_A3_progress(
        out_dir=out_dir,
        rows=rows,
        args=args,
        seed=seed,
        completed_trials=completed_trials,
        total_trials=total_trials,
        completed=False,
    )
    for n in n_list:
        cfg_n = replace(cfg, n_users=int(n))
        rng = np.random.default_rng(seed + 1009 * int(n))
        for trial in range(1, int(args.trials) + 1):
            if (int(n), int(trial)) in completed_pairs:
                sample_users(cfg_n, rng)
                continue
            users = sample_users(cfg_n, rng)
            V0 = float(np.sum(s2_ratio.local_cost(users)))
            stage2_out = solve_stage2_scm(users, float(args.pE), float(args.pN), cfg.system, cfg.stackelberg, inner_solver_mode=str(args.inner_solver))
            V_DG = float(stage2_out.social_cost)
            cs_out, cs_success = s2_ratio._run_centralized(users, float(args.pE), float(args.pN), cfg.system, cfg, str(args.centralized_solver))
            V_Xstar = float(cs_out.social_cost) if cs_success else float("nan")
            Xstar_size = int(len(cs_out.offloading_set)) if cs_success else -1
            delta_hat_star = float(s2_ratio._compute_delta_hat_star(users, float(args.pE), float(args.pN), cfg.system))
            bound = float("nan")
            ratio = float("nan")
            slack = float("nan")
            valid = False
            violated = False
            if cs_success and np.isfinite(V_Xstar) and V_Xstar > 0.0:
                denom = V0 - Xstar_size * delta_hat_star
                if denom > 0.0 and np.isfinite(denom):
                    bound = float((V0 - delta_hat_star) / denom)
                    ratio = float(V_DG / V_Xstar)
                    slack = float(bound - ratio)
                    if np.isfinite(bound) and np.isfinite(ratio):
                        valid = True
                        violated = bool(ratio > bound)
            rows.append(
                {
                    "n_users": int(n),
                    "trial": int(trial),
                    "V0": float(V0),
                    "V_DG": float(V_DG),
                    "V_Xstar": float(V_Xstar),
                    "X_DG_size": int(len(stage2_out.offloading_set)),
                    "Xstar_size": int(Xstar_size),
                    "delta_hat_star": float(delta_hat_star),
                    "bound": float(bound),
                    "ratio": float(ratio),
                    "slack": float(slack),
                    "violated": bool(violated),
                    "valid": bool(valid),
                    "rollback_count": int(stage2_out.rollback_count),
                    "accepted_admissions": int(stage2_out.accepted_admissions),
                    "inner_call_count": int(stage2_out.inner_call_count),
                    "runtime_sec": float(stage2_out.runtime_sec),
                    "used_exact_inner": bool(stage2_out.used_exact_inner),
                    "stage2_method": str(stage2_out.stage2_method),
                    "inner_solver_mode": str(stage2_out.inner_solver_mode),
                    "centralized_success": bool(cs_success),
                    "centralized_solver": str(cs_out.meta.get("solver", args.centralized_solver)),
                }
            )
            completed_trials += 1
            _flush_A3_progress(
                out_dir=out_dir,
                rows=rows,
                args=args,
                seed=seed,
                completed_trials=completed_trials,
                total_trials=total_trials,
                completed=False,
            )
    _flush_A3_progress(
        out_dir=out_dir,
        rows=rows,
        args=args,
        seed=seed,
        completed_trials=completed_trials,
        total_trials=total_trials,
        completed=True,
    )


def main_A6() -> None:
    parser = argparse.ArgumentParser(description="Figure A6: supplementary Stage II exploitability diagnostic.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users-list", type=str, default="20,40,60,80,100")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pE", type=float, default=0.5)
    parser.add_argument("--pN", type=float, default=0.5)
    parser.add_argument("--inner-solver", type=str, default="primal_dual", choices=["hybrid", "exact", "primal_dual"])
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_A6_stage2_exploitability_supp", args.out_dir)
    cfg = load_config(args.config)
    trial_rows: list[dict[str, object]] = []
    for n in _parse_int_list(args.n_users_list):
        cfg_n = replace(cfg, n_users=int(n))
        for trial in range(1, int(args.trials) + 1):
            trial_seed = int(args.seed) + 100000 * int(n) + int(trial)
            users = sample_users(cfg_n, np.random.default_rng(trial_seed))
            inner, iters = s2_exploit._algorithm2_profile(
                users=users,
                pE=float(args.pE),
                pN=float(args.pN),
                system=cfg.system,
                cfg=cfg.stackelberg,
                inner_solver=str(args.inner_solver),
            )
            data = s2_exploit._build_data(users)
            social_cost = float(s2_exploit._social_cost_from_inner(data.cl, users.n, inner))
            exploit_avg, exploit_max = s2_exploit._compute_trial_exploitability(users, float(args.pE), float(args.pN), cfg.system, inner)
            trial_rows.append(
                {
                    "n_users": int(n),
                    "trial": int(trial),
                    "exploitability_avg": float(exploit_avg),
                    "exploitability_max": float(exploit_max),
                    "algorithm2_social_cost": float(social_cost),
                    "offloading_size": int(len(inner.offloading_set)),
                    "algorithm2_iterations": int(iters),
                    "inner_solver": str(args.inner_solver),
                    "pE": float(args.pE),
                    "pN": float(args.pN),
                    "seed": int(trial_seed),
                }
            )
    summary_rows = s2_exploit._summarize_by_n(trial_rows, _parse_int_list(args.n_users_list))
    s2_exploit._write_summary_csv(out_dir / "A6_stage2_exploitability_supp.csv", summary_rows)
    title = f"Algorithm 2 exploitability vs number of users (pE={float(args.pE):.4g}, pN={float(args.pN):.4g})"
    s2_exploit._plot_mean_std(summary_rows, out_dir / "A6_stage2_exploitability_supp.png", title)
    _write_summary(
        out_dir / "A6_stage2_exploitability_supp_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"pE = {float(args.pE):.10g}",
            f"pN = {float(args.pN):.10g}",
            f"inner_solver = {args.inner_solver}",
            f"n_users_list = {args.n_users_list}",
            f"trials_per_n = {args.trials}",
            "status = supplementary_only",
        ],
    )


def main_B1() -> None:
    parser = argparse.ArgumentParser(description="Figure B1: joint revenue heatmap.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--pE-points", type=int, default=81)
    parser.add_argument("--pN-points", type=int, default=81)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B1_stage1_joint_revenue_heatmap", args.out_dir)
    cfg, _users, grid, joint_rev, legacy_gain_proxy, eq_mask, representative = _grid_eval_bundle(
        cfg=_load_cfg(args.config),
        n_users=int(args.n_users),
        seed=int(args.seed),
        pEmax=float(args.pEmax),
        pNmax=float(args.pNmax),
        pE_points=int(args.pE_points),
        pN_points=int(args.pN_points),
    )
    s1_heatmaps._plot_heatmap(
        values=joint_rev,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="Joint Revenue (ESP+NSP) Heatmap",
        cbar_label="ESP+NSP Revenue",
        out_path=out_dir / "B1_stage1_joint_revenue_heatmap.png",
        cmap="cividis",
        eq_mask=eq_mask,
        representative=representative,
    )
    alias_rows = _grid_alias_rows(grid, legacy_gain_proxy)
    write_csv_rows(
        out_dir / "B1_stage1_joint_revenue_heatmap.csv",
        ["pE", "pN", "esp_revenue", "nsp_revenue", "joint_revenue", "grid_ne_gap", "legacy_gain_proxy"],
        alias_rows,
    )
    _write_summary(
        out_dir / "B1_stage1_joint_revenue_heatmap_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"pEmax = {args.pEmax}",
            f"pNmax = {args.pNmax}",
            f"pE_points = {args.pE_points}",
            f"pN_points = {args.pN_points}",
        ],
    )


def main_B2() -> None:
    parser = argparse.ArgumentParser(description="Figure B2: restricted-gap heatmap.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--pE-points", type=int, default=81)
    parser.add_argument("--pN-points", type=int, default=81)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B2_stage1_restricted_gap_heatmap", args.out_dir)
    _cfg, _users, grid, _joint_rev, legacy_gain_proxy, eq_mask, representative = _grid_eval_bundle(
        cfg=_load_cfg(args.config),
        n_users=int(args.n_users),
        seed=int(args.seed),
        pEmax=float(args.pEmax),
        pNmax=float(args.pNmax),
        pE_points=int(args.pE_points),
        pN_points=int(args.pN_points),
    )
    s1_heatmaps._plot_heatmap(
        values=grid.grid_ne_gap,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        title="Grid-NE-Gap Heatmap",
        cbar_label="grid_ne_gap",
        out_path=out_dir / "B2_stage1_restricted_gap_heatmap.png",
        cmap="magma",
        eq_mask=eq_mask,
        representative=representative,
    )
    rows = [
        {
            "pE": row["pE"],
            "pN": row["pN"],
            "grid_ne_gap": row["grid_ne_gap"],
            "legacy_gain_proxy": row["legacy_gain_proxy"],
            "joint_revenue": row["joint_revenue"],
        }
        for row in _grid_alias_rows(grid, legacy_gain_proxy)
    ]
    write_csv_rows(
        out_dir / "B2_stage1_restricted_gap_heatmap.csv",
        ["pE", "pN", "grid_ne_gap", "legacy_gain_proxy", "joint_revenue"],
        rows,
    )
    _write_summary(
        out_dir / "B2_stage1_restricted_gap_heatmap_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"pEmax = {args.pEmax}",
            f"pNmax = {args.pNmax}",
            f"pE_points = {args.pE_points}",
            f"pN_points = {args.pN_points}",
        ],
    )


def main_B3() -> None:
    parser = argparse.ArgumentParser(description="Figure B3: ESP slice boundary comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--start-pE", type=float, default=0.5)
    parser.add_argument("--start-pN", type=float, default=0.5)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B3_esp_slice_boundary_comparison", args.out_dir)
    bundle = _boundary_bundle(
        cfg=_load_cfg(args.config),
        n_users=int(args.n_users),
        seed=int(args.seed),
        start_pE=float(args.start_pE),
        start_pN=float(args.start_pN),
        pEmax=float(args.pEmax),
        pNmax=float(args.pNmax),
    )
    boundary_diag._plot_slice_comparison(
        "E",
        bundle["esp_slice"],
        bundle["exact_esp"],
        bundle["old_unique"]["E"],
        bundle["hypothesis_esp"],
        bundle["start_pE"],
        out_dir / "B3_esp_slice_boundary_comparison.png",
    )
    write_csv_rows(
        out_dir / "B3_esp_slice_boundary_comparison.csv",
        ["provider", "source_type", "price_value", "pE", "pN", "fixed_price", "event_type", "point_group"],
        _boundary_alias_rows(bundle, provider="E"),
    )
    _write_summary(
        out_dir / "B3_esp_slice_boundary_comparison_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"start_pE = {bundle['start_pE']}",
            f"start_pN = {bundle['start_pN']}",
            f"exact_count = {len(bundle['exact_esp'])}",
            f"hypothesis_count = {len(bundle['hypothesis_esp'])}",
            f"old_unique_count = {len(bundle['old_unique']['E'])}",
        ],
    )


def main_B4() -> None:
    parser = argparse.ArgumentParser(description="Figure B4: NSP slice boundary comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--start-pE", type=float, default=0.5)
    parser.add_argument("--start-pN", type=float, default=0.5)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B4_nsp_slice_boundary_comparison", args.out_dir)
    bundle = _boundary_bundle(
        cfg=_load_cfg(args.config),
        n_users=int(args.n_users),
        seed=int(args.seed),
        start_pE=float(args.start_pE),
        start_pN=float(args.start_pN),
        pEmax=float(args.pEmax),
        pNmax=float(args.pNmax),
    )
    boundary_diag._plot_slice_comparison(
        "N",
        bundle["nsp_slice"],
        bundle["exact_nsp"],
        bundle["old_unique"]["N"],
        bundle["hypothesis_nsp"],
        bundle["start_pN"],
        out_dir / "B4_nsp_slice_boundary_comparison.png",
    )
    write_csv_rows(
        out_dir / "B4_nsp_slice_boundary_comparison.csv",
        ["provider", "source_type", "price_value", "pE", "pN", "fixed_price", "event_type", "point_group"],
        _boundary_alias_rows(bundle, provider="N"),
    )
    _write_summary(
        out_dir / "B4_nsp_slice_boundary_comparison_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"start_pE = {bundle['start_pE']}",
            f"start_pN = {bundle['start_pN']}",
            f"exact_count = {len(bundle['exact_nsp'])}",
            f"hypothesis_count = {len(bundle['hypothesis_nsp'])}",
            f"old_unique_count = {len(bundle['old_unique']['N'])}",
        ],
    )


def main_B5() -> None:
    parser = argparse.ArgumentParser(description="Figure B5: joint revenue boundary overlay.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--start-pE", type=float, default=0.5)
    parser.add_argument("--start-pN", type=float, default=0.5)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_B5_joint_revenue_boundary_overlay", args.out_dir)
    bundle = _boundary_bundle(
        cfg=_load_cfg(args.config),
        n_users=int(args.n_users),
        seed=int(args.seed),
        start_pE=float(args.start_pE),
        start_pN=float(args.start_pN),
        pEmax=float(args.pEmax),
        pNmax=float(args.pNmax),
    )
    boundary_diag._plot_revenue_overlay(
        bundle["joint_grid"],
        bundle["grid"].pE_grid,
        bundle["grid"].pN_grid,
        title="Joint revenue with old/exact/hypothesis boundaries",
        cbar_label="joint_revenue",
        out_path=out_dir / "B5_joint_revenue_boundary_overlay.png",
        start_pE=bundle["start_pE"],
        start_pN=bundle["start_pN"],
        old_esp=bundle["old_unique"]["E"],
        exact_esp=[event.boundary_price for event in bundle["exact_esp"]],
        hyp_esp=[event.predicted_price for event in bundle["hypothesis_esp"]],
        old_nsp=bundle["old_unique"]["N"],
        exact_nsp=[event.boundary_price for event in bundle["exact_nsp"]],
        hyp_nsp=[event.predicted_price for event in bundle["hypothesis_nsp"]],
    )
    write_csv_rows(
        out_dir / "B5_joint_revenue_boundary_overlay.csv",
        ["provider", "source_type", "price_value", "pE", "pN", "fixed_price", "event_type", "point_group"],
        _boundary_alias_rows(bundle, provider=None),
    )
    _write_summary(
        out_dir / "B5_joint_revenue_boundary_overlay_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"start_pE = {bundle['start_pE']}",
            f"start_pN = {bundle['start_pN']}",
            f"match_tol = {bundle['match_tol']}",
        ],
    )


def main_C3() -> None:
    parser = argparse.ArgumentParser(description="Figure C3: price trajectory on restricted-gap heatmap.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--pE-points", type=int, default=81)
    parser.add_argument("--pN-points", type=int, default=81)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C3_price_trajectory_on_gap_heatmap", args.out_dir)
    cfg = _load_cfg(args.config, n_users=int(args.n_users))
    users = _sample_once(cfg, int(args.n_users), int(args.seed))
    grid = evaluate_stage1_price_grid(
        users=users,
        system=cfg.system,
        stack_cfg=cfg.stackelberg,
        base_cfg=cfg.baselines,
        pE_min=0.0,
        pE_max=float(args.pEmax),
        pN_min=0.0,
        pN_max=float(args.pNmax),
        pE_points=int(args.pE_points),
        pN_points=int(args.pN_points),
        stage2_method=None,
    )
    stack_cfg = replace(cfg.stackelberg, stage1_solver_variant="vbbr_brd")
    result = solve_stage1_pricing(users, cfg.system, stack_cfg)
    trajectory = s1_traj._trajectory_points(result)
    s1_traj._plot_eps_with_trajectory(
        eps=grid.grid_ne_gap,
        pE_grid=grid.pE_grid,
        pN_grid=grid.pN_grid,
        trajectory=trajectory,
        eps_tol=1e-12,
        out_path=out_dir / "C3_price_trajectory_on_gap_heatmap.png",
    )
    write_csv_rows(
        out_dir / "C3_price_trajectory_on_gap_heatmap.csv",
        ["step", "pE", "pN", "nearest_grid_pE", "nearest_grid_pN", "nearest_grid_grid_ne_gap"],
        _trajectory_alias_rows(trajectory, grid.pE_grid, grid.pN_grid, grid.grid_ne_gap),
    )
    _write_summary(
        out_dir / "C3_price_trajectory_on_gap_heatmap_summary.txt",
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"n_users = {args.n_users}",
            f"pEmax = {args.pEmax}",
            f"pNmax = {args.pNmax}",
            f"trajectory_points = {len(trajectory)}",
            f"final_restricted_gap = {float(result.restricted_gap):.10g}",
        ],
    )


def main_C5() -> None:
    parser = argparse.ArgumentParser(description="Figure C5: supplementary trajectory comparison.")
    parser.add_argument("--config", type=str, default="configs/figures/paper_base.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--grid-csv", type=str, default=None)
    parser.add_argument("--proposed-solver-variant", type=str, default="vbbr_brd")
    parser.add_argument("--proposed-name", type=str, default=None)
    parser.add_argument("--vbbr-csv", type=str, default=None)
    parser.add_argument("--ga-csv", type=str, default=None)
    parser.add_argument("--bo-csv", type=str, default=None)
    parser.add_argument("--marl-csv", type=str, default=None)
    parser.add_argument("--common-start-pE", type=float, default=None)
    parser.add_argument("--common-start-pN", type=float, default=None)
    parser.add_argument("--pEmax", type=float, default=6.0)
    parser.add_argument("--pNmax", type=float, default=6.0)
    parser.add_argument("--pE-points", type=int, default=81)
    parser.add_argument("--pN-points", type=int, default=81)
    parser.add_argument("--baselines", type=str, default="BO,GA,MARL")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir("run_figure_C5_trajectory_compare_supp", args.out_dir)
    cfg = _load_cfg(args.config, n_users=int(args.n_users))
    users = _sample_once(cfg, int(args.n_users), int(args.seed))
    proposed_solver_variant = str(args.proposed_solver_variant).strip().lower()
    proposed_label = str(args.proposed_name).strip() if args.proposed_name else (
        "VBBR" if proposed_solver_variant == "vbbr_brd" else proposed_solver_variant
    )

    def _resolve_optional_path(raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        path = Path(raw_path)
        if not path.is_absolute() and not path.exists():
            path = ROOT / path
        return path

    def _prepend_common_start(
        trajectory: list[tuple[float, float]],
        start_point: tuple[float, float] | None,
    ) -> list[tuple[float, float]]:
        if start_point is None:
            return trajectory
        if not trajectory:
            return [start_point]
        if abs(float(trajectory[0][0]) - float(start_point[0])) + abs(float(trajectory[0][1]) - float(start_point[1])) <= 1e-12:
            return trajectory
        return [start_point, *trajectory]

    common_start: tuple[float, float] | None = None
    if args.common_start_pE is not None or args.common_start_pN is not None:
        if args.common_start_pE is None or args.common_start_pN is None:
            raise ValueError("--common-start-pE and --common-start-pN must be provided together.")
        common_start = (float(args.common_start_pE), float(args.common_start_pN))

    if args.grid_csv:
        grid_csv = _resolve_optional_path(args.grid_csv)
        assert grid_csv is not None
        pE_grid, pN_grid, esp_rev, nsp_rev, grid_ne_gap, legacy_gain_proxy = s1_compare._load_grid_csv(grid_csv)
    else:
        grid = evaluate_stage1_price_grid(
            users=users,
            system=cfg.system,
            stack_cfg=cfg.stackelberg,
            base_cfg=cfg.baselines,
            pE_min=0.0,
            pE_max=float(args.pEmax),
            pN_min=0.0,
            pN_max=float(args.pNmax),
            pE_points=int(args.pE_points),
            pN_points=int(args.pN_points),
            stage2_method=None,
        )
        pE_grid = grid.pE_grid
        pN_grid = grid.pN_grid
        esp_rev = grid.esp_rev
        nsp_rev = grid.nsp_rev
        grid_ne_gap = grid.grid_ne_gap
        legacy_gain_proxy = np.asarray([[float(out.legacy_gain_proxy) for out in row] for row in grid.outcomes], dtype=float)
    trajectories: dict[str, list[tuple[float, float]]] = {}
    meta_lines = [
        f"config = {args.config}",
        f"seed = {args.seed}",
        f"n_users = {args.n_users}",
        f"baselines = {args.baselines}",
        f"proposed_solver_variant = {proposed_solver_variant}",
        f"proposed_label = {proposed_label}",
        "status = supplementary_only",
    ]
    if args.grid_csv:
        meta_lines.extend(
            [
                "heatmap_source = precomputed_csv",
                f"heatmap_csv = {grid_csv}",
                f"pE_points = {int(pE_grid.size)}",
                f"pN_points = {int(pN_grid.size)}",
            ]
        )
    else:
        meta_lines.extend(
            [
                "heatmap_source = recomputed",
                f"pE_points = {int(args.pE_points)}",
                f"pN_points = {int(args.pN_points)}",
            ]
        )
    if common_start is not None:
        meta_lines.extend(
            [
                f"common_start_pE = {float(common_start[0]):.10g}",
                f"common_start_pN = {float(common_start[1]):.10g}",
                "common_start_mode = prepended_if_missing",
            ]
        )

    vbbr_csv = _resolve_optional_path(args.vbbr_csv)
    if vbbr_csv is not None:
        trajectories[proposed_label] = s1_compare._load_trajectory_csv(vbbr_csv)
        meta_lines.extend(
            [
                f"proposed_source = csv:{vbbr_csv}",
                f"pricing_points = {len(trajectories[proposed_label])}",
            ]
        )
    else:
        proposed_cfg = replace(cfg.stackelberg, stage1_solver_variant=proposed_solver_variant)
        proposed_out = solve_stage1_pricing(users, cfg.system, proposed_cfg)
        trajectories[proposed_label] = s1_compare._trajectory_points(proposed_out)
        meta_lines.extend(
            [
                "proposed_source = recomputed",
                f"pricing_outer_iterations = {proposed_out.outer_iterations}",
                f"pricing_stage2_calls = {proposed_out.stage2_oracle_calls}",
                f"pricing_restricted_gap = {float(proposed_out.restricted_gap):.10g}",
            ]
        )
    trajectories[proposed_label] = _prepend_common_start(trajectories[proposed_label], common_start)

    baselines = s1_compare._parse_baselines(args.baselines)
    bo_seed = int(cfg.baselines.random_seed + 101)
    marl_seed = int(cfg.baselines.random_seed + 303)
    bo_csv = _resolve_optional_path(args.bo_csv)
    ga_csv = _resolve_optional_path(args.ga_csv)
    marl_csv = _resolve_optional_path(args.marl_csv)
    if "BO" in baselines:
        if bo_csv is not None:
            bo_traj = s1_compare._load_trajectory_csv(bo_csv)
            trajectories["BO"] = bo_traj
            meta_lines.append(f"bo_source = csv:{bo_csv}")
            meta_lines.append(f"bo_points = {len(bo_traj)}")
        else:
            bo_traj, bo_meta = s1_compare._simulate_bo_trajectory(
                pE_grid=pE_grid,
                pN_grid=pN_grid,
                surface=grid_ne_gap,
                bo_init_points=int(cfg.baselines.bo_init_points),
                bo_iters=int(cfg.baselines.bo_iters),
                bo_candidate_pool=int(cfg.baselines.bo_candidate_pool),
                bo_kernel_bandwidth=float(cfg.baselines.bo_kernel_bandwidth),
                bo_ucb_beta=float(cfg.baselines.bo_ucb_beta),
                seed=bo_seed,
                trace_mode="best",
                start_pE=float(cfg.stackelberg.initial_pE),
                start_pN=float(cfg.stackelberg.initial_pN),
            )
            trajectories["BO"] = bo_traj
            meta_lines.append("bo_source = simulated")
            meta_lines.append(f"bo_points = {len(bo_traj)}")
            meta_lines.append(f"bo_best_grid_ne_gap = {float(bo_meta['best_score']):.10g}")
        trajectories["BO"] = _prepend_common_start(trajectories["BO"], common_start)
    if "GA" in baselines:
        if ga_csv is not None:
            ga_traj = s1_compare._load_trajectory_csv(ga_csv)
            trajectories["GA"] = ga_traj
            meta_lines.append(f"ga_source = csv:{ga_csv}")
            meta_lines.append(f"ga_points = {len(ga_traj)}")
        else:
            ga_out, ga_traj = s1_compare.run_stage1_genetic_algorithm(users, cfg.system, cfg.stackelberg, cfg.baselines, outcome_name="GA")
            trajectories["GA"] = ga_traj
            meta_lines.append("ga_source = recomputed")
            meta_lines.append(f"ga_points = {len(ga_traj)}")
            meta_lines.append(f"ga_final_grid_ne_gap = {float(ga_out.grid_ne_gap):.10g}")
        trajectories["GA"] = _prepend_common_start(trajectories["GA"], common_start)
    if "MARL" in baselines:
        if marl_csv is not None:
            marl_traj = s1_compare._load_trajectory_csv(marl_csv)
            trajectories["MARL"] = marl_traj
            meta_lines.append(f"marl_source = csv:{marl_csv}")
            meta_lines.append(f"marl_points = {len(marl_traj)}")
        else:
            marl_traj, marl_meta = s1_compare._simulate_marl_trajectory(
                pE_grid=pE_grid,
                pN_grid=pN_grid,
                esp_revenue=esp_rev,
                nsp_revenue=nsp_rev,
                marl_price_levels=int(cfg.baselines.marl_price_levels),
                marl_episodes=int(cfg.baselines.marl_episodes),
                marl_steps_per_episode=int(cfg.baselines.marl_steps_per_episode),
                marl_alpha=float(cfg.baselines.marl_alpha),
                marl_gamma=float(cfg.baselines.marl_gamma),
                marl_epsilon=float(cfg.baselines.marl_epsilon),
                seed=marl_seed,
            )
            trajectories["MARL"] = marl_traj
            meta_lines.append("marl_source = simulated")
            meta_lines.append(f"marl_points = {len(marl_traj)}")
            meta_lines.append(f"marl_final_joint_revenue = {float(marl_meta['final_joint_revenue']):.10g}")
        trajectories["MARL"] = _prepend_common_start(trajectories["MARL"], common_start)

    s1_compare._plot_compare(
        surface=grid_ne_gap,
        pE_grid=pE_grid,
        pN_grid=pN_grid,
        trajectories=trajectories,
        eps_tol=1e-12,
        out_path=out_dir / "C5_trajectory_compare_supp.png",
        cbar_label="grid_ne_gap",
        title="Stage-I trajectories on grid-NE-gap heatmap",
    )
    method_map = {proposed_label: "proposed", "BO": "BO", "GA": "GA", "MARL": "MARL"}
    rows: list[dict[str, object]] = []
    for name, traj in trajectories.items():
        for row in _trajectory_alias_rows(traj, pE_grid, pN_grid, grid_ne_gap):
            rows.append({"method": method_map[name], **row})
    write_csv_rows(
        out_dir / "C5_trajectory_compare_supp.csv",
        ["method", "step", "pE", "pN", "nearest_grid_pE", "nearest_grid_pN", "nearest_grid_grid_ne_gap"],
        rows,
    )
    _write_summary(out_dir / "C5_trajectory_compare_supp_summary.txt", meta_lines)

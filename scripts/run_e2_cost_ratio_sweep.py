"""E2-style strategic-setting comparison over provider-cost ratio cE/cN."""

from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
from dataclasses import is_dataclass, replace
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
SRC = ROOT / "src"
for path in (THIS_DIR, ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from _figure_missing_impl import _run_stage1_method, _sample_users, _write_summary
from _figure_wrapper_utils import load_csv_rows, write_csv_rows
from tmc26_exp.config import ExperimentConfig, SystemConfig, load_config
from tmc26_exp.stackelberg import solve_stage2_scm


METHOD_MAP = {
    "Full model": "Proposed",
    "ME": "ME",
    "SingleSP": "SingleSP",
    "Coop": "Coop",
    "Rand": "Rand",
}
METHOD_ORDER = list(METHOD_MAP.keys())
DISPLAY_METHOD_LABELS = {
    "Full model": "Stackelberg",
}
PLOT_FONT_SIZES = {
    "axis_label": 26,
    "tick_label": 22,
    "legend": 13,
    "legend_small": 10,
}
SINGLE_PLOT_LEGEND_LOC = "best"
PROVIDER_PLOT_LEGEND_LOC = "best"
LEGEND_FRAME_ALPHA = 0.82
BAND_ALPHA = 0.14


def display_method_label(method: str) -> str:
    return DISPLAY_METHOD_LABELS.get(method, method)


def single_plot_legend_loc(y_key: str) -> str:
    if y_key == "social_cost":
        return "lower left"
    return SINGLE_PLOT_LEGEND_LOC


def single_plot_legend_font_size(y_key: str) -> int:
    if y_key in {"joint_revenue", "social_cost"}:
        return PLOT_FONT_SIZES["legend_small"]
    return PLOT_FONT_SIZES["legend"]


def single_plot_axis_label_size(y_key: str) -> int:
    if y_key in {"joint_revenue", "social_cost"}:
        return PLOT_FONT_SIZES["axis_label"] + 5
    return PLOT_FONT_SIZES["axis_label"]


def single_plot_tick_label_size(y_key: str) -> int:
    if y_key in {"joint_revenue", "social_cost"}:
        return PLOT_FONT_SIZES["tick_label"] + 5
    return PLOT_FONT_SIZES["tick_label"]


@dataclass(frozen=True)
class SinglePlotSpec:
    y_key: str
    ylabel: str
    title: str
    out_name: str


SINGLE_PLOT_SPECS = [
    SinglePlotSpec(
        y_key="esp_revenue",
        ylabel="ESP revenue",
        title="ESP revenue over provider-cost ratio",
        out_name="E2_esp_revenue_cost_ratio_compare.png",
    ),
    SinglePlotSpec(
        y_key="nsp_revenue",
        ylabel="NSP revenue",
        title="NSP revenue over provider-cost ratio",
        out_name="E2_nsp_revenue_cost_ratio_compare.png",
    ),
    SinglePlotSpec(
        y_key="joint_revenue",
        ylabel="Joint revenue",
        title="Joint provider revenue over provider-cost ratio",
        out_name="E2_joint_revenue_cost_ratio_compare.png",
    ),
    SinglePlotSpec(
        y_key="social_cost",
        ylabel="User social cost",
        title="User social cost over provider-cost ratio",
        out_name="E2_user_social_cost_ratio_compare.png",
    ),
]
FIELDNAMES = [
    "figure_id",
    "block",
    "method",
    "n_users",
    "trial",
    "ratio",
    "cE",
    "cN",
    "social_cost",
    "esp_revenue",
    "nsp_revenue",
    "joint_revenue",
    "comp_utilization",
    "band_utilization",
    "final_pE",
    "final_pN",
    "offloading_size",
    "restricted_gap",
    "runtime_sec",
    "stage2_calls",
]


def parse_ratio_list(raw: str) -> list[float]:
    ratios = sorted({float(item) for item in raw.split(",") if item.strip()})
    if not ratios:
        raise ValueError("ratio-list must contain at least one positive value.")
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("All cE/cN ratios must be positive.")
    return ratios


def system_for_ratio(cfg: ExperimentConfig, *, fixed_cE: float, ratio: float) -> SystemConfig:
    return replace(cfg.system, cE=float(fixed_cE), cN=float(fixed_cE) / float(ratio))


def baselines_for_system(base_cfg, *, cfg_system_cE: float, cfg_system_cN: float, system: SystemConfig):
    e_width = max(float(base_cfg.max_price_E) - float(cfg_system_cE), 1e-9)
    n_width = max(float(base_cfg.max_price_N) - float(cfg_system_cN), 1e-9)
    max_price_E = max(float(base_cfg.max_price_E), float(system.cE) + e_width)
    max_price_N = max(float(base_cfg.max_price_N), float(system.cN) + n_width)
    if is_dataclass(base_cfg):
        return replace(base_cfg, max_price_E=max_price_E, max_price_N=max_price_N)
    adjusted = copy(base_cfg)
    adjusted.max_price_E = max_price_E
    adjusted.max_price_N = max_price_N
    return adjusted


def completed_key(row: dict[str, object]) -> tuple[str, float, int]:
    return (str(row["method"]), float(row["ratio"]), int(row["trial"]))


def _load_existing_rows(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for row in load_csv_rows(csv_path):
        rows.append(
            {
                "figure_id": str(row.get("figure_id", "E2-cost-ratio")),
                "block": str(row.get("block", "E")),
                "method": str(row["method"]),
                "n_users": int(row["n_users"]),
                "trial": int(row["trial"]),
                "ratio": float(row["ratio"]),
                "cE": float(row["cE"]),
                "cN": float(row["cN"]),
                "social_cost": float(row["social_cost"]),
                "esp_revenue": float(row["esp_revenue"]),
                "nsp_revenue": float(row["nsp_revenue"]),
                "joint_revenue": float(row["joint_revenue"]),
                "comp_utilization": float(row["comp_utilization"]),
                "band_utilization": float(row["band_utilization"]),
                "final_pE": float(row["final_pE"]),
                "final_pN": float(row["final_pN"]),
                "offloading_size": int(float(row["offloading_size"])),
                "restricted_gap": float(row["restricted_gap"]),
                "runtime_sec": float(row.get("runtime_sec", 0.0)),
                "stage2_calls": int(float(row.get("stage2_calls", 0))),
            }
        )
    return rows


def _mean_std_by_method(
    rows: list[dict[str, object]],
    *,
    y_key: str,
) -> dict[str, list[tuple[float, float, float]]]:
    out: dict[str, list[tuple[float, float, float]]] = {}
    for method in METHOD_ORDER:
        ratios = sorted({float(row["ratio"]) for row in rows if str(row["method"]) == method})
        stats: list[tuple[float, float, float]] = []
        for ratio in ratios:
            vals = np.asarray(
                [
                    float(row[y_key])
                    for row in rows
                    if str(row["method"]) == method
                    and float(row["ratio"]) == float(ratio)
                    and np.isfinite(float(row[y_key]))
                ],
                dtype=float,
            )
            if vals.size:
                stats.append((float(ratio), float(np.mean(vals)), float(np.std(vals))))
        if stats:
            out[method] = stats
    return out


def _ratio_ticks(ratios: list[float]) -> list[float]:
    start = int(np.floor(np.log10(min(ratios))))
    end = int(np.ceil(np.log10(max(ratios))))
    return [10.0**exp for exp in range(start, end + 1)]


def _plot_provider_revenue(rows: list[dict[str, object]], *, out_path: Path, ratios: list[float]) -> None:
    panels = [
        ("esp_revenue", "ESP revenue"),
        ("nsp_revenue", "NSP revenue"),
        ("joint_revenue", "Joint revenue"),
    ]
    with plt.rc_context():
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, len(panels), figsize=(17.2, 5.8), dpi=150)
        cmap = plt.get_cmap("tab10")
        ticks = _ratio_ticks(ratios)
        tick_labels = [rf"$10^{{{int(round(np.log10(tick)))}}}$" for tick in ticks]
        for panel_idx, (y_key, ylabel) in enumerate(panels):
            ax = axes[panel_idx]
            grouped = _mean_std_by_method(rows, y_key=y_key)
            for idx, method in enumerate(METHOD_ORDER):
                if method not in grouped:
                    continue
                stats = grouped[method]
                x = np.asarray([item[0] for item in stats], dtype=float)
                y = np.asarray([item[1] for item in stats], dtype=float)
                e = np.asarray([item[2] for item in stats], dtype=float)
                color = cmap(idx % 10)
                ax.plot(
                    x,
                    y,
                    linewidth=1.6,
                    markersize=4.8,
                    marker="o",
                    color=color,
                    label=display_method_label(method),
                )
                if np.any(np.isfinite(e) & (e > 0.0)):
                    ax.fill_between(x, y - e, y + e, color=color, alpha=BAND_ALPHA)
            ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_xscale("log")
            ax.set_xlim(min(ratios) / 1.35, max(ratios) * 1.35)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(r"cost ratio $c_E/c_N$", fontsize=PLOT_FONT_SIZES["axis_label"])
            ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick_label"])
            ax.grid(alpha=0.25)
            ax.legend(
                loc=PROVIDER_PLOT_LEGEND_LOC,
                fontsize=PLOT_FONT_SIZES["legend"],
                frameon=True,
                framealpha=LEGEND_FRAME_ALPHA,
            )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _plot_single_metric(
    rows: list[dict[str, object]],
    *,
    spec: SinglePlotSpec,
    out_path: Path,
    ratios: list[float],
) -> None:
    with plt.rc_context():
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=180)
        cmap = plt.get_cmap("tab10")
        ticks = _ratio_ticks(ratios)
        tick_labels = [rf"$10^{{{int(round(np.log10(tick)))}}}$" for tick in ticks]
        grouped = _mean_std_by_method(rows, y_key=spec.y_key)
        for idx, method in enumerate(METHOD_ORDER):
            if method not in grouped:
                continue
            stats = grouped[method]
            x = np.asarray([item[0] for item in stats], dtype=float)
            y = np.asarray([item[1] for item in stats], dtype=float)
            e = np.asarray([item[2] for item in stats], dtype=float)
            color = cmap(idx % 10)
            ax.plot(
                x,
                y,
                linewidth=1.8,
                markersize=5.2,
                marker="o",
                color=color,
                label=display_method_label(method),
            )
            if np.any(np.isfinite(e) & (e > 0.0)):
                ax.fill_between(x, y - e, y + e, color=color, alpha=BAND_ALPHA)
        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlim(min(ratios) / 1.35, max(ratios) * 1.35)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r"cost ratio $c_E/c_N$", fontsize=single_plot_axis_label_size(spec.y_key))
        ax.set_ylabel(spec.ylabel, fontsize=single_plot_axis_label_size(spec.y_key))
        ax.tick_params(axis="both", labelsize=single_plot_tick_label_size(spec.y_key))
        ax.grid(alpha=0.25)
        ax.legend(
            loc=single_plot_legend_loc(spec.y_key),
            fontsize=single_plot_legend_font_size(spec.y_key),
            frameon=True,
            framealpha=LEGEND_FRAME_ALPHA,
        )
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)


def _write_manifest(out_dir: Path, *, args: argparse.Namespace, ratios: list[float], runtime_sec: float) -> None:
    split_images = [spec.out_name for spec in SINGLE_PLOT_SPECS]
    manifest = {
        "schema_version": 1,
        "figure_id": "E2-cost-ratio",
        "block": "E",
        "base_stem": "E2_provider_revenue_cost_ratio_compare",
        "script": "run_e2_cost_ratio_sweep.py",
        "output_dir": str(out_dir),
        "summary_file": "E2_provider_revenue_cost_ratio_compare_summary.txt",
        "primary_image": "E2_provider_revenue_cost_ratio_compare.png",
        "split_images": split_images,
        "primary_csv": "E2_provider_revenue_cost_ratio_compare.csv",
        "config": str(args.config),
        "seed": str(args.seed),
        "n_users": str(args.n_users),
        "n_trials": str(args.trials),
        "fixed_cE": f"{float(args.fixed_cE):.10g}",
        "ratio_list": ",".join(f"{ratio:.10g}" for ratio in ratios),
        "price_cap_rule": "preserve_base_search_width_above_cost_floor",
        "runtime_sec": f"{runtime_sec:.3f}",
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def flush_progress(
    *,
    out_dir: Path,
    rows: list[dict[str, object]],
    args: argparse.Namespace,
    ratios: list[float],
    completed_points: int,
    total_points: int,
    runtime_sec: float,
) -> None:
    csv_path = out_dir / "E2_provider_revenue_cost_ratio_compare.csv"
    png_path = out_dir / "E2_provider_revenue_cost_ratio_compare.png"
    summary_path = out_dir / "E2_provider_revenue_cost_ratio_compare_summary.txt"
    write_csv_rows(csv_path, FIELDNAMES, rows)
    if rows:
        _plot_provider_revenue(rows, out_path=png_path, ratios=ratios)
        for spec in SINGLE_PLOT_SPECS:
            _plot_single_metric(rows, spec=spec, out_path=out_dir / spec.out_name, ratios=ratios)
    _write_summary(
        summary_path,
        [
            f"config = {args.config}",
            f"seed = {args.seed}",
            f"trials = {args.trials}",
            f"n_users = {args.n_users}",
            "scan_mode = fixed_cE_scan_cN",
            f"fixed_cE = {float(args.fixed_cE):.10g}",
            f"ratio_list = {','.join(f'{ratio:.10g}' for ratio in ratios)}",
            "price_cap_rule = preserve_base_search_width_above_cost_floor",
            "methods = " + ",".join(METHOD_ORDER),
            f"progress_completed_points = {completed_points}",
            f"progress_total_points = {total_points}",
            f"progress_status = {'completed' if completed_points >= total_points else 'running'}",
            f"runtime_sec = {runtime_sec:.3f}",
        ],
    )
    _write_manifest(out_dir, args=args, ratios=ratios, runtime_sec=runtime_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="E2-style provider revenue comparison over cE/cN.")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--fixed-cE", type=float, default=0.01)
    parser.add_argument("--ratio-list", type=str, default="1e-3,1e-2,1e-1,1,1e1,1e2,1e3")
    parser.add_argument("--out-dir", type=str, default="outputs/3. strategic_settings/e2_cost_ratio_n100_trial10_default")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ratios = parse_ratio_list(args.ratio_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_existing_rows(out_dir / "E2_provider_revenue_cost_ratio_compare.csv")
    completed = {completed_key(row) for row in rows}
    total_points = len(ratios) * int(args.trials) * len(METHOD_ORDER)
    start_time = time.perf_counter()

    if rows:
        flush_progress(
            out_dir=out_dir,
            rows=rows,
            args=args,
            ratios=ratios,
            completed_points=len(completed),
            total_points=total_points,
            runtime_sec=0.0,
        )

    for ratio in ratios:
        system = system_for_ratio(cfg, fixed_cE=float(args.fixed_cE), ratio=float(ratio))
        baselines = baselines_for_system(
            cfg.baselines,
            cfg_system_cE=float(cfg.system.cE),
            cfg_system_cN=float(cfg.system.cN),
            system=system,
        )
        for trial in range(1, int(args.trials) + 1):
            users = _sample_users(cfg, int(args.n_users), int(args.seed), trial)
            for method, internal_method in METHOD_MAP.items():
                key = (method, float(ratio), int(trial))
                if key in completed:
                    continue
                price, offloading_set, gap, esp_rev, nsp_rev, meta = _run_stage1_method(
                    users,
                    system,
                    cfg.stackelberg,
                    baselines,
                    internal_method,
                )
                stage2 = solve_stage2_scm(
                    users,
                    price[0],
                    price[1],
                    system,
                    cfg.stackelberg,
                    inner_solver_mode="primal_dual",
                )
                rows.append(
                    {
                        "figure_id": "E2-cost-ratio",
                        "block": "E",
                        "method": method,
                        "n_users": int(args.n_users),
                        "trial": int(trial),
                        "ratio": float(ratio),
                        "cE": float(system.cE),
                        "cN": float(system.cN),
                        "social_cost": float(meta["social_cost"]),
                        "esp_revenue": float(esp_rev),
                        "nsp_revenue": float(nsp_rev),
                        "joint_revenue": float(esp_rev + nsp_rev),
                        "comp_utilization": float(np.sum(stage2.inner_result.f) / system.F),
                        "band_utilization": float(np.sum(stage2.inner_result.b) / system.B),
                        "final_pE": float(price[0]),
                        "final_pN": float(price[1]),
                        "offloading_size": int(len(offloading_set)),
                        "restricted_gap": float(gap),
                        "runtime_sec": float(meta.get("runtime_sec", 0.0)),
                        "stage2_calls": int(meta.get("stage2_calls", 0)),
                    }
                )
                completed.add(key)
                flush_progress(
                    out_dir=out_dir,
                    rows=rows,
                    args=args,
                    ratios=ratios,
                    completed_points=len(completed),
                    total_points=total_points,
                    runtime_sec=time.perf_counter() - start_time,
                )


if __name__ == "__main__":
    main()

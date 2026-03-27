from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
SRC = ROOT / "src"
for path in (THIS_DIR, ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from _figure_wrapper_utils import resolve_out_dir
from tmc26_exp.config import load_config


METHOD_ORDER = ["Full model", "ME", "SingleSP", "Coop", "Rand"]
METHOD_LABELS = {
    "Full model": "Stackelberg",
}
METHOD_LABELS_ZH = {
    "Full model": "本文模型",
    "ME": "市场均衡",
    "SingleSP": "单服务商",
    "Coop": "合作定价",
    "Rand": "随机卸载集",
}
METHOD_COLORS = {
    "Full model": "tab:blue",
    "ME": "tab:orange",
    "SingleSP": "tab:green",
    "Coop": "tab:red",
    "Rand": "tab:purple",
}
FIXED_X_TICKS = [20, 40, 60, 80, 100]
X_AXIS_MARGIN = 4.0

METRIC_SPECS = {
    "user_social_cost_compare": {
        "y_key": "social_cost",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Total user social cost",
                "title": "User Social Cost Comparison",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "用户总社会成本",
                "title": "用户社会成本对比",
            },
        },
    },
    "esp_revenue_compare": {
        "y_key": "esp_revenue",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "ESP revenue",
                "title": "ESP Revenue vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "ESP 收益",
                "title": "ESP 收益与用户数量的关系",
            },
        },
        "method_order": ["Full model", "ME", "Coop", "Rand"],
        "method_colors": {
            "Full model": "tab:blue",
            "ME": "tab:orange",
            "Coop": "tab:red",
            "Rand": "tab:purple",
        },
    },
    "nsp_revenue_compare": {
        "y_key": "nsp_revenue",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "NSP revenue",
                "title": "NSP Revenue vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "NSP 收益",
                "title": "NSP 收益与用户数量的关系",
            },
        },
        "method_order": ["Full model", "ME", "Coop", "Rand"],
        "method_colors": {
            "Full model": "tab:blue",
            "ME": "tab:orange",
            "Coop": "tab:red",
            "Rand": "tab:purple",
        },
    },
    "joint_revenue_compare": {
        "y_key": "joint_revenue",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Joint revenue",
                "title": "Joint Revenue vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "联合收益",
                "title": "联合收益与用户数量的关系",
            },
        },
    },
    "comp_utilization_compare": {
        "y_key": "comp_utilization",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Computation utilization",
                "title": "Computation Utilization vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "计算资源利用率",
                "title": "计算资源利用率与用户数量的关系",
            },
        },
    },
    "band_utilization_compare": {
        "y_key": "band_utilization",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Bandwidth utilization",
                "title": "Bandwidth Utilization vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "带宽利用率",
                "title": "带宽利用率与用户数量的关系",
            },
        },
    },
    "offloading_ratio_compare": {
        "y_key": "offloading_ratio",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Offloading ratio",
                "title": "Offloading Ratio vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "卸载比例",
                "title": "卸载比例与用户数量的关系",
            },
        },
    },
    "offloading_users_compare": {
        "y_key": "offloading_size",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Mean number of offloading users",
                "title": "Mean Number of Offloading Users vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "平均卸载用户数",
                "title": "平均卸载用户数与用户数量的关系",
            },
        },
    },
    "final_pE_compare": {
        "y_key": "final_pE",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Final pE",
                "title": "Final pE vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "最终 pE",
                "title": "最终 pE 与用户数量的关系",
            },
        },
    },
    "final_pN_compare": {
        "y_key": "final_pN",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Final pN",
                "title": "Final pN vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "最终 pN",
                "title": "最终 pN 与用户数量的关系",
            },
        },
    },
    "total_user_payment_compare": {
        "y_key": "total_user_payment",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Total user payment",
                "title": "Total User Payment vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "用户总支付",
                "title": "用户总支付与用户数量的关系",
            },
        },
        "requires_payment_metrics": True,
    },
    "avg_payment_per_offloading_user_compare": {
        "y_key": "avg_payment_per_offloading_user",
        "labels": {
            "en": {
                "xlabel": "Number of users",
                "ylabel": "Average payment per offloading user",
                "title": "Average Payment per Offloading User vs. Number of Users",
            },
            "zh": {
                "xlabel": "用户数量",
                "ylabel": "每个卸载用户的平均支付",
                "title": "每个卸载用户的平均支付与用户数量的关系",
            },
        },
        "requires_payment_metrics": True,
    },
}


def _load_rows(path: Path, *, F_total: float | None = None, B_total: float | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict[str, object] = {
                "method": raw["method"],
                "n_users": int(raw["n_users"]),
                "trial": int(raw["trial"]),
                "social_cost": float(raw["social_cost"]),
                "esp_revenue": float(raw["esp_revenue"]),
                "nsp_revenue": float(raw["nsp_revenue"]),
                "joint_revenue": float(raw["joint_revenue"]),
                "comp_utilization": float(raw["comp_utilization"]),
                "band_utilization": float(raw["band_utilization"]),
                "final_pE": float(raw["final_pE"]),
                "final_pN": float(raw["final_pN"]),
                "offloading_size": float(raw["offloading_size"]),
            }
            row["offloading_ratio"] = (
                float(row["offloading_size"]) / float(row["n_users"]) if float(row["n_users"]) > 0 else float("nan")
            )
            if F_total is not None and B_total is not None:
                total_payment = (
                    float(row["final_pE"]) * float(row["comp_utilization"]) * float(F_total)
                    + float(row["final_pN"]) * float(row["band_utilization"]) * float(B_total)
                )
                row["total_user_payment"] = float(total_payment)
                row["avg_payment_per_offloading_user"] = (
                    float(total_payment) / float(row["offloading_size"])
                    if float(row["offloading_size"]) > 0.0
                    else float("nan")
                )
            rows.append(row)
    return rows


def _mean_std_by_method(rows: list[dict[str, object]], x_key: str, y_key: str) -> dict[str, list[tuple[float, float, float]]]:
    grouped: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        method = str(row["method"])
        x = float(row[x_key])
        y = float(row[y_key])
        grouped.setdefault(method, {}).setdefault(x, []).append(y)
    out: dict[str, list[tuple[float, float, float]]] = {}
    for method, stats in grouped.items():
        seq: list[tuple[float, float, float]] = []
        for x in sorted(stats):
            vals = np.asarray(stats[x], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                seq.append((x, float("nan"), float("nan")))
            else:
                seq.append((x, float(np.mean(vals)), float(np.std(vals))))
        out[method] = seq
    return out


def _display_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _display_method_localized(method: str, language: str) -> str:
    if language == "zh":
        return METHOD_LABELS_ZH.get(method, METHOD_LABELS.get(method, method))
    return METHOD_LABELS.get(method, method)


def _display_method_with_legend_language(method: str, plot_language: str, legend_language: str) -> str:
    if legend_language == "en":
        return METHOD_LABELS.get(method, method)
    if legend_language == "zh":
        return METHOD_LABELS_ZH.get(method, METHOD_LABELS.get(method, method))
    return _display_method_localized(method, plot_language)


def _configure_fonts(language: str, font_scale: float) -> None:
    if language == "zh":
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Arial Unicode MS",
        ]
        available = {entry.name for entry in font_manager.fontManager.ttflist}
        chosen = [name for name in candidates if name in available]
        if chosen:
            existing = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = chosen + [name for name in existing if name not in chosen]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 13.5 * font_scale
    plt.rcParams["axes.titlesize"] = 17.0 * font_scale
    plt.rcParams["axes.labelsize"] = 15.0 * font_scale
    plt.rcParams["xtick.labelsize"] = 12.8 * font_scale
    plt.rcParams["ytick.labelsize"] = 12.8 * font_scale
    plt.rcParams["legend.fontsize"] = 12.0 * font_scale
    plt.rcParams["axes.linewidth"] = 1.25
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1


def _plot_single_metric(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    method_order: list[str] | None = None,
    method_colors: dict[str, str] | None = None,
    language: str = "en",
    legend_language: str = "match",
    legend_loc: str = "best",
    legend_fontsize: float | None = None,
    title_fontsize: float | None = None,
    ylabel_fontsize: float | None = None,
    zoom_inset: dict[str, object] | None = None,
) -> None:
    grouped = _mean_std_by_method(rows, x_key, y_key)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9.6, 6.3), dpi=180)
    order = method_order or METHOD_ORDER
    plotted_series: list[tuple[np.ndarray, np.ndarray, np.ndarray, object]] = []
    for idx, method in enumerate(order):
        if method not in grouped:
            continue
        stats = grouped[method]
        x = np.asarray([item[0] for item in stats], dtype=float)
        y = np.asarray([item[1] for item in stats], dtype=float)
        e = np.asarray([item[2] for item in stats], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        x_plot = x[mask]
        y_plot = y[mask]
        e_plot = np.where(np.isfinite(e[mask]), e[mask], 0.0)
        color = (method_colors or {}).get(method, METHOD_COLORS.get(method, cmap(idx % 10)))
        ax.plot(
            x_plot,
            y_plot,
            "-o",
            linewidth=1.9,
            markersize=6.0,
            color=color,
            label=_display_method_with_legend_language(method, language, legend_language),
        )
        ax.fill_between(
            x_plot,
            y_plot - e_plot,
            y_plot + e_plot,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        plotted_series.append((x_plot, y_plot, e_plot, color))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks(FIXED_X_TICKS)
    ax.set_xlim(min(FIXED_X_TICKS) - X_AXIS_MARGIN, max(FIXED_X_TICKS) + X_AXIS_MARGIN)
    ax.grid(alpha=0.25)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
    if zoom_inset:
        inset_ax = inset_axes(
            ax,
            width=str(zoom_inset.get("width", "34%")),
            height=str(zoom_inset.get("height", "34%")),
            loc=str(zoom_inset.get("loc", "lower right")),
            borderpad=float(zoom_inset.get("borderpad", 1.0)),
        )
        x_min = float(zoom_inset.get("x_min", min(FIXED_X_TICKS)))
        x_max = float(zoom_inset.get("x_max", max(FIXED_X_TICKS)))
        y_lows: list[float] = []
        y_highs: list[float] = []
        y_mean_lows: list[float] = []
        y_mean_highs: list[float] = []
        for x_plot, y_plot, e_plot, color in plotted_series:
            zoom_mask = (x_plot >= x_min) & (x_plot <= x_max)
            if not np.any(zoom_mask):
                continue
            inset_ax.plot(
                x_plot[zoom_mask],
                y_plot[zoom_mask],
                "-o",
                linewidth=1.6,
                markersize=4.2,
                color=color,
            )
            inset_ax.fill_between(
                x_plot[zoom_mask],
                y_plot[zoom_mask] - e_plot[zoom_mask],
                y_plot[zoom_mask] + e_plot[zoom_mask],
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
            y_lows.append(float(np.min(y_plot[zoom_mask] - e_plot[zoom_mask])))
            y_highs.append(float(np.max(y_plot[zoom_mask] + e_plot[zoom_mask])))
            y_mean_lows.append(float(np.min(y_plot[zoom_mask])))
            y_mean_highs.append(float(np.max(y_plot[zoom_mask])))
        if y_lows and y_highs:
            y_min_override = zoom_inset.get("y_min")
            y_max_override = zoom_inset.get("y_max")
            if y_min_override is not None and y_max_override is not None:
                y_min = float(y_min_override)
                y_max = float(y_max_override)
            else:
                y_limit_source = str(zoom_inset.get("y_limit_source", "band"))
                if y_limit_source == "mean" and y_mean_lows and y_mean_highs:
                    y_min = min(y_mean_lows)
                    y_max = max(y_mean_highs)
                else:
                    y_min = min(y_lows)
                    y_max = max(y_highs)
                y_span = max(y_max - y_min, 1.0)
                y_pad = y_span * float(zoom_inset.get("y_pad_ratio", 0.12))
                y_min -= y_pad
                y_max += y_pad
            inset_ax.set_xlim(x_min, x_max)
            inset_ax.set_ylim(y_min, y_max)
        inset_ticks = zoom_inset.get("x_ticks", [40, 60, 80])
        inset_ax.set_xticks([float(x) for x in inset_ticks])
        inset_ax.grid(alpha=0.20)
        inset_tick_size = float(plt.rcParams["xtick.labelsize"]) * float(zoom_inset.get("tick_scale", 0.70))
        inset_ax.tick_params(axis="both", labelsize=inset_tick_size, width=0.9, length=3.0)
        for spine in inset_ax.spines.values():
            spine.set_linewidth(1.0)
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _metric_output_name(metric_name: str, suffix: str) -> str:
    return f"{metric_name}{suffix}.png"


def _parse_only_artifacts(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        return None
    unknown = [name for name in names if name not in METRIC_SPECS]
    if unknown:
        raise ValueError(f"Unknown artifact names: {', '.join(unknown)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot strategic-setting metrics from an existing E-series CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to an existing E1/E2/E3/E4 CSV with strategic rows.")
    parser.add_argument("--config", type=str, default=None, help="Config path used to compute derived payment metrics from F and B.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory under outputs/.")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="Plot language.")
    parser.add_argument(
        "--legend-language",
        type=str,
        default="match",
        choices=["match", "en", "zh"],
        help="Legend label language; 'match' follows --lang.",
    )
    parser.add_argument("--filename-suffix", type=str, default="", help="Suffix appended before .png, e.g. _zh.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Global font scale multiplier.")
    parser.add_argument(
        "--only-artifacts",
        type=str,
        default=None,
        help="Comma-separated artifact stems to render; defaults to all supported metrics.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    F_total: float | None = None
    B_total: float | None = None
    if args.config:
        cfg = load_config(args.config)
        F_total = float(cfg.system.F)
        B_total = float(cfg.system.B)
    rows = _load_rows(csv_path, F_total=F_total, B_total=B_total)
    out_dir = resolve_out_dir("replot_strategic_metrics_from_csv", args.out_dir)
    _configure_fonts(args.lang, args.font_scale)
    selected_artifacts = _parse_only_artifacts(args.only_artifacts) or list(METRIC_SPECS.keys())

    artifact_names: list[str] = []
    for metric_name in selected_artifacts:
        spec = METRIC_SPECS[metric_name]
        if spec.get("requires_payment_metrics") and (F_total is None or B_total is None):
            continue
        label_pack = spec["labels"][args.lang]
        out_name = _metric_output_name(metric_name, args.filename_suffix)
        legend_loc = "best"
        legend_fontsize: float | None = None
        title_fontsize: float | None = None
        ylabel_fontsize: float | None = None
        zoom_inset: dict[str, object] | None = None
        base_legend_fontsize = float(plt.rcParams["legend.fontsize"])
        base_title_fontsize = float(plt.rcParams["axes.titlesize"])
        base_label_fontsize = float(plt.rcParams["axes.labelsize"])
        if metric_name == "user_social_cost_compare":
            zoom_inset = {
                "x_min": 40.0,
                "x_max": 80.0,
                "x_ticks": [40, 60, 80],
                "width": "35%",
                "height": "35%",
                "loc": "lower right",
                "borderpad": 1.1,
                "tick_scale": 0.70,
                "y_min": 1000.0,
                "y_max": 1500.0,
            }
        if metric_name == "joint_revenue_compare" and args.lang == "en":
            legend_loc = "upper left"
            legend_fontsize = base_legend_fontsize * 0.88
        if metric_name == "joint_revenue_compare" and args.lang == "zh":
            legend_fontsize = base_legend_fontsize * 0.88
        if metric_name == "offloading_users_compare" and args.lang == "en":
            title_fontsize = base_title_fontsize * 0.92
            ylabel_fontsize = base_label_fontsize * 0.92
            legend_fontsize = base_legend_fontsize * 0.84
        if metric_name == "offloading_users_compare" and args.lang == "zh":
            legend_loc = "upper left"
            legend_fontsize = base_legend_fontsize * 0.78
        _plot_single_metric(
            rows,
            x_key="n_users",
            y_key=str(spec["y_key"]),
            xlabel=str(label_pack["xlabel"]),
            ylabel=str(label_pack["ylabel"]),
            title=str(label_pack["title"]),
            out_path=out_dir / out_name,
            method_order=spec.get("method_order"),
            method_colors=spec.get("method_colors"),
            language=args.lang,
            legend_language=args.legend_language,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            title_fontsize=title_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            zoom_inset=zoom_inset,
        )
        artifact_names.append(out_name)

    # Remove stale multi-panel outputs so the directory contains only the redrawn single-panel versions.
    for stale_name in [
        "E2_provider_revenue_compare.png",
        "resource_utilization_compare.png",
        "offloading_outcomes_compare.png",
        "price_outcomes_compare.png",
    ]:
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    summary_lines = [
        f"source_csv = {csv_path}",
        f"rows = {len(rows)}",
        f"methods = {','.join(sorted({str(row['method']) for row in rows}))}",
        f"n_users = {','.join(str(int(x)) for x in sorted({int(row['n_users']) for row in rows}))}",
        f"legend_alias_full_model = {METHOD_LABELS['Full model']}",
        f"language = {args.lang}",
        f"legend_language = {args.legend_language}",
        f"filename_suffix = {args.filename_suffix or '(none)'}",
        f"font_scale = {args.font_scale}",
        "plot_style = line_plot_with_error_band",
        "error_band = mean_plus_minus_std",
        f"x_ticks = {','.join(str(x) for x in FIXED_X_TICKS)}",
        f"x_limits = {min(FIXED_X_TICKS) - X_AXIS_MARGIN},{max(FIXED_X_TICKS) + X_AXIS_MARGIN}",
        f"artifacts = {','.join(artifact_names)}",
    ]
    if args.config:
        summary_lines.extend(
            [
                f"config = {args.config}",
                f"system_F = {F_total}",
                f"system_B = {B_total}",
            ]
        )
    (out_dir / "replot_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

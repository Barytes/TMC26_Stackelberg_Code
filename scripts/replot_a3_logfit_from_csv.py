"""Replot the customized A3 log-fit scatter figure from an existing CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_stage2_approximation_ratio as s2_ratio


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replot the customized A3 log-fit scatter figure from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to A3_stage2_approx_ratio_bound.csv.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for the re-rendered PNG files.")
    parser.add_argument("--font-scale", type=float, default=1.25, help="Global font scale factor.")
    parser.add_argument("--legend-loc", type=str, default="upper left", help="Matplotlib legend location.")
    return parser


def _compute_fit(rows: list[dict[str, float | int | str | bool]]) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    valid_rows = [
        r
        for r in rows
        if bool(r["valid"])
        and np.isfinite(float(r["bound"]))
        and np.isfinite(float(r["ratio"]))
        and float(r["bound"]) > 0.0
        and float(r["ratio"]) > 0.0
    ]
    if len(valid_rows) < 2:
        raise ValueError("Need at least two valid rows with positive bound and ratio.")

    x_all = np.asarray([float(r["bound"]) for r in valid_rows], dtype=float)
    y_all = np.log(x_all / np.asarray([float(r["ratio"]) for r in valid_rows], dtype=float))
    log_x = np.log(x_all)
    coeffs = np.polyfit(log_x, y_all, deg=1)
    b = float(coeffs[0])
    a = float(coeffs[1])
    y_hat = a + b * log_x
    ss_res = float(np.sum((y_all - y_hat) ** 2))
    ss_tot = float(np.sum((y_all - float(np.mean(y_all))) ** 2))
    r_squared = 1.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return x_all, y_all, a, b, r_squared


def _plot_single(
    *,
    rows: list[dict[str, float | int | str | bool]],
    out_path: Path,
    font_scale: float,
    legend_loc: str,
    title: str,
    xlabel: str,
    ylabel: str,
    fit_label: str,
    font_family: str | None = None,
) -> tuple[float, float, float, int]:
    x_all, y_all, a, b, r_squared = _compute_fit(rows)
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=150)
    cmap = plt.get_cmap("tab10")

    ns = sorted({int(r["n_users"]) for r in rows})
    for idx, n in enumerate(ns):
        n_rows = [
            r
            for r in rows
            if bool(r["valid"])
            and int(r["n_users"]) == n
            and np.isfinite(float(r["bound"]))
            and np.isfinite(float(r["ratio"]))
            and float(r["bound"]) > 0.0
            and float(r["ratio"]) > 0.0
        ]
        if not n_rows:
            continue
        bounds = np.asarray([float(r["bound"]) for r in n_rows], dtype=float)
        ratios = np.asarray([float(r["ratio"]) for r in n_rows], dtype=float)
        ax.scatter(
            bounds,
            np.log(bounds / ratios),
            s=34,
            alpha=0.9,
            color=cmap(idx % 10),
            label=rf"$\left|\mathcal{{I}}\right|$={n}",
        )

    x_low = float(np.min(x_all))
    x_high = float(np.max(x_all))
    y_low = float(np.min(y_all))
    y_high = float(np.max(y_all))
    x_pad = 0.03 * max(x_high - x_low, 1e-6)
    y_pad = 0.08 * max(y_high - y_low, 1e-6)
    x_low -= x_pad
    x_high += x_pad
    y_low = min(y_low - y_pad, 0.0)
    y_high += y_pad

    xs = np.linspace(x_low, x_high, 400)
    ax.plot(xs, np.zeros_like(xs), linestyle="--", color="black", linewidth=1.2, label="_nolegend_")
    ax.plot(xs, a + b * np.log(xs), color="tab:brown", linewidth=1.6, label=fit_label)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    text_kwargs = {} if font_family is None else {"fontfamily": font_family}
    ax.set_title(title, fontsize=16.0 * font_scale, **text_kwargs)
    ax.set_xlabel(xlabel, fontsize=13.0 * font_scale, **text_kwargs)
    ax.set_ylabel(ylabel, fontsize=13.0 * font_scale, **text_kwargs)
    ax.tick_params(axis="both", labelsize=11.0 * font_scale)
    if font_family is not None:
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontfamily(font_family)
    ax.grid(alpha=0.25)
    if font_family is None:
        ax.legend(loc=legend_loc, fontsize=11.0 * font_scale, frameon=True)
    else:
        ax.legend(
            loc=legend_loc,
            frameon=True,
            prop={"family": font_family, "size": 11.0 * font_scale},
        )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return a, b, r_squared, int(x_all.size)


def _write_summary(
    *,
    summary_path: Path,
    csv_path: Path,
    font_scale: float,
    legend_loc: str,
    a: float,
    b: float,
    r_squared: float,
    valid_rows_used: int,
) -> None:
    lines = [
        f"source_csv = {csv_path.as_posix()}",
        "transform = s(B_theo) = log(B_theo / empirical ratio)",
        "fit_model = s(B_theo) = a + b ln B_theo",
        f"a = {a:.12g}",
        f"b = {b:.12g}",
        f"equation = s(B_theo) = {a:.6g} + {b:.6g} ln B_theo",
        f"r_squared = {r_squared:.12g}",
        f"valid_rows_used = {valid_rows_used}",
        f"legend_loc = {legend_loc}",
        f"font_scale = {font_scale:.4g}",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = s2_ratio._read_points_csv(csv_path)

    en_title = "Empirical Approximation Ratio of Algorithm 2"
    en_xlabel = "Theoretical Approximation Bound $B_{theo}$"
    en_ylabel = "Log gap $s(B_{theo})=\\log(B_{theo}/\\rho_{emp})$"
    zh_title = "\u7b97\u6cd54.2\u7684\u7ecf\u9a8c\u8fd1\u4f3c\u6bd4"
    zh_xlabel = "\u7406\u8bba\u8fd1\u4f3c\u754c $B_{theo}$"
    zh_ylabel = "\u5bf9\u6570\u95f4\u9699 $s(B_{theo})=\\log(B_{theo}/\\rho_{emp})$"

    a, b, r_squared, valid_rows_used = _plot_single(
        rows=rows,
        out_path=out_dir / "A3_stage2_approx_ratio_bound_logfit.png",
        font_scale=float(args.font_scale),
        legend_loc=str(args.legend_loc),
        title=en_title,
        xlabel=en_xlabel,
        ylabel=en_ylabel,
        fit_label="Log fit",
        font_family=None,
    )
    _plot_single(
        rows=rows,
        out_path=out_dir / "A3_stage2_approx_ratio_bound_logfit_zh.png",
        font_scale=float(args.font_scale),
        legend_loc=str(args.legend_loc),
        title=zh_title,
        xlabel=zh_xlabel,
        ylabel=zh_ylabel,
        fit_label="\u5bf9\u6570\u62df\u5408",
        font_family="Microsoft YaHei",
    )
    _write_summary(
        summary_path=out_dir / "A3_stage2_approx_ratio_bound_logfit_summary.txt",
        csv_path=csv_path,
        font_scale=float(args.font_scale),
        legend_loc=str(args.legend_loc),
        a=a,
        b=b,
        r_squared=r_squared,
        valid_rows_used=valid_rows_used,
    )


if __name__ == "__main__":
    main()

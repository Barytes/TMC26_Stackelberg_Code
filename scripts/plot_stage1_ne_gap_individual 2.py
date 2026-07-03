"""Draw single-panel Stage-I NE-gap rerun figures without matplotlib.

This is a lightweight fallback for environments where matplotlib is not
available or its Agg backend is broken. It reads the completed rerun CSV files
and writes individual PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


METHOD_ORDER = ["Proposed", "GA", "BO-online", "MARL"]
METHOD_COLORS = {
    "Proposed": (31, 119, 180),
    "GA": (214, 39, 40),
    "BO-online": (44, 160, 44),
    "MARL": (148, 103, 189),
}


@dataclass(frozen=True)
class Layout:
    width: int = 1500
    height: int = 960
    left: int = 155
    right: int = 55
    top: int = 88
    bottom: int = 135

    @property
    def plot_w(self) -> int:
        return self.width - self.left - self.right

    @property
    def plot_h(self) -> int:
        return self.height - self.top - self.bottom


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = _font(34, bold=True)
LABEL_FONT = _font(24)
TICK_FONT = _font(20)
LEGEND_FONT = _font(21)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str | float | int | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def _nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if abs(vmax - vmin) < 1e-12:
        pad = max(1.0, abs(vmax) * 0.1)
        vmin -= pad
        vmax += pad
    span = vmax - vmin
    raw_step = span / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(raw_step))
    norm = raw_step / mag
    if norm <= 1:
        step = mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks: list[float] = []
    cur = start
    while cur <= end + step * 0.5:
        ticks.append(0.0 if abs(cur) < step * 1e-9 else cur)
        cur += step
    return ticks


def _format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0] + 8, box[3] - box[1] + 8
    layer = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((4, 4), text, font=font, fill=fill)
    rotated = layer.rotate(90, expand=True)
    image.alpha_composite(rotated, (xy[0] - rotated.width // 2, xy[1] - rotated.height // 2))


def _series_from_summary(
    rows: list[dict[str, str]],
    metric: str,
    methods: Iterable[str],
) -> dict[str, list[tuple[float, float, float, float]]]:
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for method in methods:
        vals = []
        for row in rows:
            if row.get("method") != method:
                continue
            x = _finite_float(row.get("n_users"))
            center = _finite_float(row.get(f"{metric}_median"))
            low = _finite_float(row.get(f"{metric}_q25"))
            high = _finite_float(row.get(f"{metric}_q75"))
            if x is None or center is None or low is None or high is None:
                continue
            vals.append((x, center, low, high))
        vals.sort(key=lambda item: item[0])
        if vals:
            out[method] = vals
    return out


def _blend(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], int(255 * alpha))


def draw_summary_plot(
    summary_rows: list[dict[str, str]],
    out_path: Path,
    *,
    metric: str,
    title: str,
    ylabel: str,
    methods: list[str] = METHOD_ORDER,
) -> None:
    layout = Layout()
    series = _series_from_summary(summary_rows, metric, methods)
    xs = [x for vals in series.values() for x, _, _, _ in vals]
    ys = [v for vals in series.values() for _, center, low, high in vals for v in (center, low, high)]
    if not xs or not ys:
        raise ValueError(f"No finite data for metric {metric}")

    x_ticks = sorted({int(x) for x in xs})
    y_min = min(ys)
    y_max = max(ys)
    pad = (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, abs(y_max) * 0.1)
    y_min = min(0.0, y_min - pad) if y_min >= 0 else y_min - pad
    y_max += pad
    y_ticks = _nice_ticks(y_min, y_max)
    y_min = min(y_ticks)
    y_max = max(y_ticks)

    def sx(x: float) -> float:
        if len(x_ticks) == 1:
            return layout.left + layout.plot_w / 2
        return layout.left + (x - min(x_ticks)) / (max(x_ticks) - min(x_ticks)) * layout.plot_w

    def sy(y: float) -> float:
        return layout.top + (y_max - y) / (y_max - y_min) * layout.plot_h

    image = Image.new("RGBA", (layout.width, layout.height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    for tick in y_ticks:
        y = sy(tick)
        draw.line((layout.left, y, layout.width - layout.right, y), fill=(220, 224, 228, 255), width=1)
        label = _format_tick(tick)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((layout.left - 16 - (bbox[2] - bbox[0]), y - 11), label, font=TICK_FONT, fill=(55, 65, 81))

    draw.line((layout.left, layout.top, layout.left, layout.height - layout.bottom), fill=(55, 65, 81), width=2)
    draw.line((layout.left, layout.height - layout.bottom, layout.width - layout.right, layout.height - layout.bottom), fill=(55, 65, 81), width=2)

    for tick in x_ticks:
        x = sx(tick)
        draw.line((x, layout.height - layout.bottom, x, layout.height - layout.bottom + 7), fill=(55, 65, 81), width=2)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x - (bbox[2] - bbox[0]) / 2, layout.height - layout.bottom + 14), label, font=TICK_FONT, fill=(55, 65, 81))

    for method in methods:
        vals = series.get(method)
        if not vals:
            continue
        color = METHOD_COLORS.get(method, (31, 41, 55))
        upper = [(sx(x), sy(high)) for x, _, _, high in vals]
        lower = [(sx(x), sy(low)) for x, _, low, _ in reversed(vals)]
        if len(vals) >= 2:
            draw.polygon(upper + lower, fill=_blend(color, 0.14))
        pts = [(sx(x), sy(center)) for x, center, _, _ in vals]
        if len(pts) >= 2:
            draw.line(pts, fill=color + (255,), width=4, joint="curve")
        for x, y in pts:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color + (255,), outline=(255, 255, 255, 255), width=2)

    draw.text((layout.left, 28), title, font=TITLE_FONT, fill=(17, 24, 39))
    x_label = "Number of users"
    bbox = draw.textbbox((0, 0), x_label, font=LABEL_FONT)
    draw.text((layout.left + layout.plot_w / 2 - (bbox[2] - bbox[0]) / 2, layout.height - 55), x_label, font=LABEL_FONT, fill=(31, 41, 55))
    _draw_rotated_text(image, (45, layout.top + layout.plot_h // 2), ylabel, LABEL_FONT, (31, 41, 55))

    lx = layout.left + 18
    ly = layout.top + 18
    for idx, method in enumerate(methods):
        if method not in series:
            continue
        y = ly + idx * 31
        color = METHOD_COLORS.get(method, (31, 41, 55))
        draw.line((lx, y + 11, lx + 34, y + 11), fill=color + (255,), width=4)
        draw.ellipse((lx + 13, y + 5, lx + 25, y + 17), fill=color + (255,))
        draw.text((lx + 46, y), method, font=LEGEND_FONT, fill=(31, 41, 55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def draw_scatter_plot(
    trial_rows: list[dict[str, str]],
    out_path: Path,
    *,
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    methods: list[str] = METHOD_ORDER,
) -> None:
    layout = Layout()
    points_by_method: dict[str, list[tuple[float, float]]] = {method: [] for method in methods}
    for row in trial_rows:
        method = row.get("method", "")
        if method not in points_by_method:
            continue
        if row.get("success") not in {"1", "1.0", 1, 1.0}:
            continue
        x = _finite_float(row.get(x_metric))
        y = _finite_float(row.get(y_metric))
        if x is not None and y is not None:
            points_by_method[method].append((x, y))
    xs = [x for pts in points_by_method.values() for x, _ in pts]
    ys = [y for pts in points_by_method.values() for _, y in pts]
    if not xs or not ys:
        raise ValueError(f"No finite scatter data for {x_metric}/{y_metric}")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    x_ticks = _nice_ticks(max(0.0, x_min - x_pad), x_max + x_pad)
    y_ticks = _nice_ticks(min(0.0, y_min - y_pad), y_max + y_pad)
    x_min, x_max = min(x_ticks), max(x_ticks)
    y_min, y_max = min(y_ticks), max(y_ticks)

    def sx(x: float) -> float:
        return layout.left + (x - x_min) / (x_max - x_min) * layout.plot_w

    def sy(y: float) -> float:
        return layout.top + (y_max - y) / (y_max - y_min) * layout.plot_h

    image = Image.new("RGBA", (layout.width, layout.height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    for tick in y_ticks:
        y = sy(tick)
        draw.line((layout.left, y, layout.width - layout.right, y), fill=(220, 224, 228, 255), width=1)
        label = _format_tick(tick)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((layout.left - 16 - (bbox[2] - bbox[0]), y - 11), label, font=TICK_FONT, fill=(55, 65, 81))
    for tick in x_ticks:
        x = sx(tick)
        draw.line((x, layout.height - layout.bottom, x, layout.height - layout.bottom + 7), fill=(55, 65, 81), width=2)
        label = _format_tick(tick)
        bbox = draw.textbbox((0, 0), label, font=TICK_FONT)
        draw.text((x - (bbox[2] - bbox[0]) / 2, layout.height - layout.bottom + 14), label, font=TICK_FONT, fill=(55, 65, 81))

    draw.line((layout.left, layout.top, layout.left, layout.height - layout.bottom), fill=(55, 65, 81), width=2)
    draw.line((layout.left, layout.height - layout.bottom, layout.width - layout.right, layout.height - layout.bottom), fill=(55, 65, 81), width=2)

    for method in methods:
        color = METHOD_COLORS.get(method, (31, 41, 55))
        for x, y in points_by_method.get(method, []):
            px, py = sx(x), sy(y)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color + (160,), outline=color + (230,), width=1)

    draw.text((layout.left, 28), title, font=TITLE_FONT, fill=(17, 24, 39))
    bbox = draw.textbbox((0, 0), xlabel, font=LABEL_FONT)
    draw.text((layout.left + layout.plot_w / 2 - (bbox[2] - bbox[0]) / 2, layout.height - 55), xlabel, font=LABEL_FONT, fill=(31, 41, 55))
    _draw_rotated_text(image, (45, layout.top + layout.plot_h // 2), ylabel, LABEL_FONT, (31, 41, 55))

    lx = layout.left + 18
    ly = layout.top + 18
    for idx, method in enumerate(methods):
        if not points_by_method.get(method):
            continue
        y = ly + idx * 31
        color = METHOD_COLORS.get(method, (31, 41, 55))
        draw.ellipse((lx + 10, y + 5, lx + 22, y + 17), fill=color + (180,))
        draw.text((lx + 46, y), method, font=LEGEND_FONT, fill=(31, 41, 55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_rows = _read_csv(out_dir / "stage1_final_grid_ne_gap_vs_users_stats.csv")
    trial_rows = _read_csv(out_dir / "stage1_final_grid_ne_gap_vs_users.csv")

    figures_dir = out_dir / "individual_figures"
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_final_grid_ne_gap_vs_users.png",
        metric="final_grid_ne_gap",
        title="Final grid-evaluated NE gap vs. number of users",
        ylabel="Final grid-evaluated NE gap",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_joint_revenue_vs_users.png",
        metric="joint_revenue",
        title="Joint revenue vs. number of users",
        ylabel="Joint revenue",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_runtime_vs_users.png",
        metric="runtime_sec",
        title="Runtime vs. number of users",
        ylabel="Runtime (sec)",
    )
    draw_summary_plot(
        summary_rows,
        figures_dir / "individual_total_stage2_calls_vs_users.png",
        metric="total_stage2_solver_calls",
        title="Total Stage-II solver calls vs. number of users",
        ylabel="Total Stage-II solver calls",
    )
    draw_scatter_plot(
        trial_rows,
        figures_dir / "individual_final_gap_vs_total_stage2_calls_trials.png",
        x_metric="total_stage2_solver_calls",
        y_metric="final_grid_ne_gap",
        title="Trial points: final gap vs. total Stage-II calls",
        xlabel="Total Stage-II solver calls",
        ylabel="Final grid-evaluated NE gap",
    )
    draw_scatter_plot(
        trial_rows,
        figures_dir / "individual_joint_revenue_vs_total_stage2_calls_trials.png",
        x_metric="total_stage2_solver_calls",
        y_metric="joint_revenue",
        title="Trial points: joint revenue vs. total Stage-II calls",
        xlabel="Total Stage-II solver calls",
        ylabel="Joint revenue",
    )
    print(f"Wrote individual figures to {figures_dir}")


if __name__ == "__main__":
    main()

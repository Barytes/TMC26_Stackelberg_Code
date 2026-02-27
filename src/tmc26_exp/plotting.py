from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .simulator import MetricSurface


def plot_surface_heatmap(surface: MetricSurface, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

    im = ax.imshow(
        surface.mean_values,
        origin="lower",
        aspect="auto",
        extent=[surface.pE_values[0], surface.pE_values[-1], surface.pN_values[0], surface.pN_values[-1]],
        cmap="viridis",
    )
    ax.set_title(f"{surface.metric_label} on (pE, pN) plane")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(surface.metric_label)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_surface_contour(surface: MetricSurface, out_path: Path) -> None:
    pE_grid, pN_grid = np.meshgrid(surface.pE_values, surface.pN_values)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    contour = ax.contourf(pE_grid, pN_grid, surface.mean_values, levels=16, cmap="plasma")
    line = ax.contour(pE_grid, pN_grid, surface.mean_values, levels=8, colors="black", linewidths=0.5)
    ax.clabel(line, inline=True, fontsize=7)
    ax.set_title(f"{surface.metric_label} contour on (pE, pN)")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(surface.metric_label)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

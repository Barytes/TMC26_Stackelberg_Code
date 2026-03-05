"""
Compute ESP/NSP deviation gaps on the pE-pN plane from existing revenue contours.

This script loads pre-computed revenue contours and computes the deviation gap at each
point. The deviation gap measures how much a provider could gain by unilaterally
deviating to a better price while the other provider's price remains fixed.

- ESP deviation gap: max_{pE'} revenue_ESP(pE', pN) - revenue_ESP(pE, pN)
- NSP deviation gap: max_{pN'} revenue_NSP(pE, pN') - revenue_NSP(pE, pN)

Additionally computes:
- Combined gap: max(ESP_gap, NSP_gap) - epsilon proxy for RNE
- Plots equilibrium point from Stackelberg trajectory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tmc26_exp.plotting import plot_surface_heatmap


def load_revenue_surface(csv_path: Path):
    """
    Load revenue surface from CSV file.

    CSV format: pE,pN,value_mean,value_std
    Returns: (pE_values, pN_values, revenue_values)
    """
    # Load data
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

    # Extract unique pE and pN values (sorted)
    pE_values = np.unique(data[:, 0])
    pN_values = np.unique(data[:, 1])

    # Reshape revenue into 2D array (pN x pE)
    revenue = data[:, 2].reshape(pN_values.size, pE_values.size)

    return pE_values, pN_values, revenue


def load_equilibrium_price(trajectory_path: Path) -> tuple[float, float] | None:
    """
    Load equilibrium price from Stackelberg trajectory CSV.

    Returns the final (pE, pN) from the trajectory (last row).
    """
    if not trajectory_path.exists():
        return None

    try:
        data = np.loadtxt(trajectory_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            # Single row
            return float(data[1]), float(data[2])
        else:
            # Multiple rows - take the last one
            return float(data[-1, 1]), float(data[-1, 2])
    except Exception:
        return None


def compute_deviation_gaps(
    esp_csv: Path,
    nsp_csv: Path,
) -> tuple:
    """
    Compute deviation gaps from existing revenue contours.

    Returns:
        (pE_values, pN_values, esp_gap, nsp_gap)
    """
    # Load revenue surfaces
    pE_values, pN_values, esp_revenue = load_revenue_surface(esp_csv)
    _, _, nsp_revenue = load_revenue_surface(nsp_csv)

    # Compute ESP deviation gap
    # For each pN, find max ESP revenue over all pE values
    esp_max_per_pN = np.max(esp_revenue, axis=1, keepdims=True)  # Shape: (n_pN, 1)
    esp_gap = esp_max_per_pN - esp_revenue  # Shape: (n_pN, n_pE)

    # Compute NSP deviation gap
    # For each pE, find max NSP revenue over all pN values
    nsp_max_per_pE = np.max(nsp_revenue, axis=0, keepdims=True)  # Shape: (1, n_pE)
    nsp_gap = nsp_max_per_pE - nsp_revenue  # Shape: (n_pN, n_pE)

    return pE_values, pN_values, esp_gap, nsp_gap


def save_gap_csv(
    pE_values: np.ndarray,
    pN_values: np.ndarray,
    gap_values: np.ndarray,
    out_path: Path,
) -> None:
    """Save deviation gap surface to CSV."""
    rows: list[str] = ["pE,pN,gap_value"]
    for n_idx, pN in enumerate(pN_values):
        for e_idx, pE in enumerate(pE_values):
            rows.append(
                f"{pE:.10g},{pN:.10g},{gap_values[n_idx, e_idx]:.10g}"
            )
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def create_gap_surface(
    pE_values: np.ndarray,
    pN_values: np.ndarray,
    gap_values: np.ndarray,
    name: str,
    label: str,
):
    """Create a surface object compatible with plotting functions."""
    class GapSurface:
        def __init__(self):
            self.metric_name = name
            self.metric_label = label
            self.pE_values = pE_values
            self.pN_values = pN_values
            self.mean_values = gap_values
            self.std_values = np.zeros_like(gap_values)  # No std for derived data

    return GapSurface()


def plot_contour_with_equilibrium(
    surface,
    out_path: Path,
    equilibrium: tuple[float, float] | None,
    title_suffix: str = "",
) -> None:
    """
    Plot contour with optional equilibrium marker.

    Args:
        surface: Surface object with metric data
        out_path: Output file path
        equilibrium: Optional (pE, pN) equilibrium point to mark
        title_suffix: Optional suffix for plot title
    """
    pE_grid, pN_grid = np.meshgrid(surface.pE_values, surface.pN_values)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    contour = ax.contourf(pE_grid, pN_grid, surface.mean_values, levels=16, cmap="plasma")
    line = ax.contour(pE_grid, pN_grid, surface.mean_values, levels=8, colors="black", linewidths=0.5)
    ax.clabel(line, inline=True, fontsize=7)

    # Mark equilibrium point if provided
    if equilibrium is not None:
        pE_eq, pN_eq = equilibrium
        ax.plot(pE_eq, pN_eq, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=1,
                label='Equilibrium', zorder=10)
        ax.legend(loc='upper right')

    title = f"{surface.metric_label} contour on (pE, pN)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(surface.metric_label)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_gap_heatmap(
    pE_values: np.ndarray,
    pN_values: np.ndarray,
    combined_gap: np.ndarray,
    out_path: Path,
    equilibrium: tuple[float, float] | None,
) -> None:
    """Plot combined max gap heatmap with equilibrium marker."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)

    im = ax.imshow(
        combined_gap,
        origin="lower",
        aspect="auto",
        extent=[pE_values[0], pE_values[-1], pN_values[0], pN_values[-1]],
        cmap="hot",
    )

    # Mark equilibrium point if provided
    if equilibrium is not None:
        pE_eq, pN_eq = equilibrium
        ax.plot(pE_eq, pN_eq, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1,
                label='Equilibrium', zorder=10)
        ax.legend(loc='upper right')

    ax.set_title("Combined Deviation Gap (max(ESP_gap, NSP_gap)) on (pE, pN) plane\n"
                 "= Epsilon proxy for RNE")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Combined Gap (max)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_gap_contour(
    pE_values: np.ndarray,
    pN_values: np.ndarray,
    combined_gap: np.ndarray,
    out_path: Path,
    equilibrium: tuple[float, float] | None,
) -> None:
    """Plot combined max gap contour with equilibrium marker."""
    pE_grid, pN_grid = np.meshgrid(pE_values, pN_values)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    contour = ax.contourf(pE_grid, pN_grid, combined_gap, levels=16, cmap="hot")
    line = ax.contour(pE_grid, pN_grid, combined_gap, levels=8, colors="black", linewidths=0.5)
    ax.clabel(line, inline=True, fontsize=7)

    # Mark equilibrium point if provided
    if equilibrium is not None:
        pE_eq, pN_eq = equilibrium
        ax.plot(pE_eq, pN_eq, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1,
                label='Equilibrium', zorder=10)
        ax.legend(loc='upper right')

    ax.set_title("Combined Deviation Gap (max(ESP_gap, NSP_gap)) contour\n"
                 "= Epsilon proxy for RNE")
    ax.set_xlabel("pE")
    ax.set_ylabel("pN")
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Combined Gap (max)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute deviation gaps from existing revenue contours"
    )
    parser.add_argument(
        "--esp-csv",
        type=str,
        default="outputs/real_revenue_contours/esp_real_revenue.csv",
        help="Path to ESP revenue contour CSV",
    )
    parser.add_argument(
        "--nsp-csv",
        type=str,
        default="outputs/real_revenue_contours/nsp_real_revenue.csv",
        help="Path to NSP revenue contour CSV",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="outputs/real_revenue_contours/stackelberg_trajectory.csv",
        help="Path to Stackelberg trajectory CSV (for equilibrium marker)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/deviation_gaps",
        help="Output directory (default: outputs/deviation_gaps)",
    )
    args = parser.parse_args()

    esp_csv = Path(args.esp_csv)
    nsp_csv = Path(args.nsp_csv)
    trajectory_path = Path(args.trajectory)

    if not esp_csv.exists():
        print(f"Error: ESP revenue file not found: {esp_csv}")
        print("Please run compute_real_revenue_contours.py first to generate revenue contours.")
        return 1

    if not nsp_csv.exists():
        print(f"Error: NSP revenue file not found: {nsp_csv}")
        print("Please run compute_real_revenue_contours.py first to generate revenue contours.")
        return 1

    # Load equilibrium price if available
    equilibrium = load_equilibrium_price(trajectory_path)
    if equilibrium is not None:
        print(f"Loaded equilibrium price: pE={equilibrium[0]:.4f}, pN={equilibrium[1]:.4f}")
    else:
        print("No equilibrium trajectory found - plots will not include equilibrium marker")
        print(f"  (expected at: {trajectory_path})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading revenue contours and computing deviation gaps...")
    print(f"  ESP revenue: {esp_csv}")
    print(f"  NSP revenue: {nsp_csv}")
    print()

    # Compute deviation gaps
    pE_values, pN_values, esp_gap, nsp_gap = compute_deviation_gaps(esp_csv, nsp_csv)

    # Compute combined gap (max of both)
    combined_gap = np.maximum(esp_gap, nsp_gap)

    print(f"Grid size: {len(pE_values)} x {len(pN_values)}")
    print()

    # ESP deviation gap
    print("ESP Deviation Gap:")
    print(f"  Range: [{esp_gap.min():.4f}, {esp_gap.max():.4f}]")
    print(f"  Mean: {esp_gap.mean():.4f}")

    esp_surface = create_gap_surface(pE_values, pN_values, esp_gap, "esp_deviation_gap", "ESP Deviation Gap")
    save_gap_csv(pE_values, pN_values, esp_gap, out_dir / "esp_deviation_gap.csv")
    plot_surface_heatmap(esp_surface, out_dir / "esp_deviation_gap_heatmap.png")
    plot_contour_with_equilibrium(esp_surface, out_dir / "esp_deviation_gap_contour.png", equilibrium)
    print()

    # NSP deviation gap
    print("NSP Deviation Gap:")
    print(f"  Range: [{nsp_gap.min():.4f}, {nsp_gap.max():.4f}]")
    print(f"  Mean: {nsp_gap.mean():.4f}")

    nsp_surface = create_gap_surface(pE_values, pN_values, nsp_gap, "nsp_deviation_gap", "NSP Deviation Gap")
    save_gap_csv(pE_values, pN_values, nsp_gap, out_dir / "nsp_deviation_gap.csv")
    plot_surface_heatmap(nsp_surface, out_dir / "nsp_deviation_gap_heatmap.png")
    plot_contour_with_equilibrium(nsp_surface, out_dir / "nsp_deviation_gap_contour.png", equilibrium)
    print()

    # Combined deviation gap
    print("Combined Deviation Gap (max(ESP_gap, NSP_gap)):")
    print(f"  Range: [{combined_gap.min():.4f}, {combined_gap.max():.4f}]")
    print(f"  Mean: {combined_gap.mean():.4f}")
    print(f"  Epsilon at equilibrium: ", end="")
    if equilibrium is not None:
        # Find closest grid point
        pE_idx = np.argmin(np.abs(pE_values - equilibrium[0]))
        pN_idx = np.argmin(np.abs(pN_values - equilibrium[1]))
        epsilon_at_eq = combined_gap[pN_idx, pE_idx]
        print(f"{epsilon_at_eq:.6f}")
    else:
        print("N/A")

    combined_surface = create_gap_surface(pE_values, pN_values, combined_gap,
                                          "combined_deviation_gap", "Combined Deviation Gap")
    save_gap_csv(pE_values, pN_values, combined_gap, out_dir / "combined_deviation_gap.csv")
    plot_combined_gap_heatmap(pE_values, pN_values, combined_gap,
                              out_dir / "combined_deviation_gap_heatmap.png", equilibrium)
    plot_combined_gap_contour(pE_values, pN_values, combined_gap,
                              out_dir / "combined_deviation_gap_contour.png", equilibrium)
    print()

    # Write summary
    summary_text = f"""Deviation Gap Analysis
======================

Input Files:
  ESP revenue: {esp_csv}
  NSP revenue: {nsp_csv}
  Trajectory: {trajectory_path} ({"found" if equilibrium else "not found"})

Output Directory: {out_dir}

Grid:
  pE values: {len(pE_values)} points, range [{pE_values.min():.4f}, {pE_values.max():.4f}]
  pN values: {len(pN_values)} points, range [{pN_values.min():.4f}, {pN_values.max():.4f}]

Equilibrium Price: {"pE={:.4f}, pN={:.4f}".format(*equilibrium) if equilibrium else "N/A"}

ESP Deviation Gap:
  Range: [{esp_gap.min():.6f}, {esp_gap.max():.6f}]
  Mean: {esp_gap.mean():.6f}
  Std: {esp_gap.std():.6f}
  Zero-gap points: {np.sum(esp_gap < 1e-10)} / {esp_gap.size}
    (These are ESP's local optima where unilateral deviation doesn't improve revenue)

NSP Deviation Gap:
  Range: [{nsp_gap.min():.6f}, {nsp_gap.max():.6f}]
  Mean: {nsp_gap.mean():.6f}
  Std: {nsp_gap.std():.6f}
  Zero-gap points: {np.sum(nsp_gap < 1e-10)} / {nsp_gap.size}
    (These are NSP's local optima where unilateral deviation doesn't improve revenue)

Combined Deviation Gap (max(ESP_gap, NSP_gap)):
  Range: [{combined_gap.min():.6f}, {combined_gap.max():.6f}]
  Mean: {combined_gap.mean():.6f}
  Std: {combined_gap.std():.6f}
  Epsilon proxy for RNE: max deviation gap at each point
"""
    if equilibrium is not None:
        pE_idx = np.argmin(np.abs(pE_values - equilibrium[0]))
        pN_idx = np.argmin(np.abs(pN_values - equilibrium[1]))
        epsilon_at_eq = combined_gap[pN_idx, pE_idx]
        summary_text += f"  Epsilon at equilibrium point: {epsilon_at_eq:.6f}\n"

    summary_text += """
Interpretation:
- The deviation gap measures how much a provider could gain by unilaterally changing
  their price while the other provider's price remains fixed.
- Zero gap indicates a local optimum (best response) for that provider.
- The intersection of zero-gap contours for both providers approximates the Nash
  equilibrium or RNE (Robust Nash Equilibrium) region.
- The combined gap (max of both) represents the epsilon value at each point,
  with the minimum combined gap indicating the RNE.

Files generated:
  - esp_deviation_gap.csv
  - esp_deviation_gap_heatmap.png
  - esp_deviation_gap_contour.png (with equilibrium marker)
  - nsp_deviation_gap.csv
  - nsp_deviation_gap_heatmap.png
  - nsp_deviation_gap_contour.png (with equilibrium marker)
  - combined_deviation_gap.csv
  - combined_deviation_gap_heatmap.png (with equilibrium marker)
  - combined_deviation_gap_contour.png (with equilibrium marker)
"""

    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"Done. Results written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

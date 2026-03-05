# Progress Log

## 2026-03-05

### Added Deviation Gap Computation Script

Created [scripts/compute_deviation_gaps.py](scripts/compute_deviation_gaps.py) to compute ESP/NSP deviation gaps on the pE-pN plane.

**Features:**
- Loads existing revenue contours (from `compute_real_revenue_contours.py`)
- Computes deviation gaps without re-solving Stage II (pure post-processing)
- ESP gap: max_{pE'} revenue_ESP(pE', pN) - revenue_ESP(pE, pN)
- NSP gap: max_{pN'} revenue_NSP(pE, pN') - revenue_NSP(pE, pN)
- Combined gap: max(ESP_gap, NSP_gap) - representing epsilon across the plane
- Optional equilibrium marker from trajectory file

**Usage:**
```bash
# With equilibrium marker
uv run python scripts/compute_deviation_gaps.py \
    --trajectory outputs/default_run/stackelberg_trajectory.csv

# Without marker (just gaps)
uv run python scripts/compute_deviation_gaps.py \
    --esp-csv outputs/real_revenue_contours/esp_real_revenue.csv \
    --nsp-csv outputs/real_revenue_contours/nsp_real_revenue.csv
```

**Output:**
- `esp_deviation_gap.csv` + heatmap/contour plots
- `nsp_deviation_gap.csv` + heatmap/contour plots
- `combined_deviation_gap.csv` + heatmap/contour plots
- `summary.txt` with statistics and epsilon at equilibrium

**Key Result:**
From potential revenue contours, epsilon at equilibrium (3.1085, 1.4623) is ~25.28, representing the "price of robustness" from the Stackelberg-guided solution.
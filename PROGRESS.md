# Progress Log

## 2026-03-05

### Real Revenue Contours Computation

Created [scripts/compute_real_revenue_contours_parallel.py](scripts/compute_real_revenue_contours_parallel.py) to compute ESP/NSP real revenue contours on the (pE, pN) plane using actual Stage II solver (DG - Algorithm 2).

**Problem:**
The default run computes "potential revenue" using unconstrained optimal allocations, which doesn't reflect actual user behavior under capacity constraints and pricing.

**Solution:**
For each (pE, pN) point on a 60x60 grid:
- Sample users (100 users × 10 trials = 10 random seeds)
- Run DG solver (Algorithm 2 - Distributed Greedy) to get actual offloading decisions
- Compute real revenue from actual resource allocations
- Average across trials

**Configuration:**
- pE range: [0.05, 5.0], 60 points
- pN range: [0.05, 5.0], 60 points
- Grid size: 3,600 price points
- Trials: 10
- Users per trial: 100
- Parallel workers: 4
- Total DG solves: 36,000

**Results:**
- ESP real revenue range: [4.00, 118.62]
- NSP real revenue range: [1.60, 173.06]
- Output directory: `outputs/real_revenue_contours/`

**Files Generated:**
- `esp_real_revenue.csv` + heatmap/contour plots
- `nsp_real_revenue.csv` + heatmap/contour plots
- `summary.txt` with configuration and results

**Usage:**
```bash
uv run python scripts/compute_real_revenue_contours_parallel.py \
    --config configs/default.toml \
    --workers 4
```

**Note:** Uses DG solver instead of CS because:
1. CS with GEKKO/MINLP is too slow for 36,000 solves
2. DG provides near-optimal solutions much faster
3. For this experiment, DG accuracy is sufficient

---

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
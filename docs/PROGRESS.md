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

---

## 2026-03-08

### 完成所有实验脚本开发 (17 个图像脚本)

根据 `docs/SPEC.md` 和 `docs/PLAN_2026-03-07.md`，完成了论文所需的全部 17 个实验图像脚本开发。

**开发计划详见**: `docs/PLAN_2026-03-07.md`

---

### Phase 1: Stage I 核心实验 (5 个脚本)

| 脚本 | Figure | 功能 |
|------|--------|------|
| `run_stage1_deviation_gap_convergence.py` | Figure 5 | Algorithm 5 epsilon 收敛曲线 |
| `run_stage1_boundary_visualization.py` | Figure 6 | 价格空间边界 + 轨迹叠加 |
| `run_stage1_gain_approximation_accuracy.py` | Figure 7 | Algorithm 3 gain 近似精度 |
| `run_stage1_candidate_family_hit_rate.py` | Figure 8 | 候选族 N(p) hit rate |
| `run_stage1_scalability.py` | Figure 9 | Stage I 可扩展性 (runtime + oracle calls) |

**技术亮点**:
- Figure 7/8: 对小规模实例使用穷举枚举计算精确 gain
- Figure 9: 使用 monkey-patch 技术追踪 Algorithm 5 中的 Stage-II 调用次数

---

### Phase 2: Stage II 补全 (4 个脚本)

| 脚本 | Figure | 功能 |
|------|--------|------|
| `run_stage2_convergence_plot.py` | Figure 1 | 迭代收敛图 |
| `run_stage2_approximation_ratio.py` | Figure 2 | Theorem 2 近似比验证 |
| `run_stage2_communication_rounds.py` | Figure 3 | 通信轮数对比 |
| `run_stage2_exploitability_comparison.py` | Figure 4 | 可剥削性对比 (DG vs CS vs Random) |

**技术亮点**:
- Figure 1: 实现了带轨迹追踪的 Algorithm 2 变体
- Figure 2: 计算 Theorem 2 理论上界并与实际比率对比

---

### Phase 3: Strategic Settings (4 个脚本)

| 脚本 | Figure | 功能 |
|------|--------|------|
| `run_strategic_social_cost.py` | Figure 10 | 用户社会成本 vs 用户数 |
| `run_strategic_joint_revenue.py` | Figure 11 | 供应商联合收益 vs 用户数 |
| `run_strategic_pareto_tradeoff.py` | Figure 12 | Pareto tradeoff 散点图 |
| `run_strategic_fb_sensitivity.py` | Figure 13 | F/B 容量敏感性分析 |

**对比方法**:
- GSSE (Proposed): Algorithm 5 Stackelberg-guided search
- MarketEquilibrium: Tatonnement-style price adjustment
- SingleSP: Single leader (ESP only)
- RandomOffloading: Random baseline

---

### Phase 4: Ablation Studies (2 个脚本)

| 脚本 | Figure | 功能 |
|------|--------|------|
| `run_ablation_L_sensitivity.py` | Figure 14 | 采样密度 L 敏感性 |
| `run_ablation_guided_search.py` | Figure 15 | Guided-search 消融对比 |

**技术亮点**:
- Figure 14: 变化 `rne_directions` 参数，观察 accuracy-complexity tradeoff
- Figure 15: 比较 GSSE vs Random search vs Exhaustive search

---

### Phase 5: Appendix (2 个脚本)

| 脚本 | Figure | 功能 |
|------|--------|------|
| `run_appendix_final_epsilon_vs_users.py` | Appendix A1 | Final epsilon vs 用户数 |
| `run_appendix_exploitability_vs_users.py` | Appendix A2 | Stage II exploitability 完整对比 |

---

### 文档更新

- 更新了 `docs/experiment_figures.md`，包含所有 17 个脚本的使用说明
- 创建了 `docs/PLAN_2026-03-07.md` 作为开发计划文档

---

### 快速测试命令

```bash
# Stage I
uv run python scripts/run_stage1_deviation_gap_convergence.py --trials 3
uv run python scripts/run_stage1_scalability.py --trials 3

# Stage II
uv run python scripts/run_stage2_convergence_plot.py --trials 3
uv run python scripts/run_stage2_communication_rounds.py --trials 3

# Strategic
uv run python scripts/run_strategic_social_cost.py --trials 5

# Ablation
uv run python scripts/run_ablation_L_sensitivity.py --trials 3

# Appendix
uv run python scripts/run_appendix_final_epsilon_vs_users.py --trials 5
```
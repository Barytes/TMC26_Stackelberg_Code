# 实验代码与论文理论对齐修复报告

## 一、发现的问题

### 1. Algorithm 5 实现与论文不一致

**当前实现** (`src/tmc26_exp/stackelberg.py` lines 702-771):
- 只使用了 `gain_E.best_set` 和 `gain_N.best_set` 两个 exact deviation targets
- 缺少 **neighborhood fallback 机制**
- 终止条件仅基于候选集是否在 visited 中

**论文要求** (SPEC.md lines 427-448):
1. 先优先评估 exact deviation targets `Y_E^*(p)` 和 `Y_N^*(p)`
2. 若无 improvement，再进入 neighborhood fallback
3. neighborhood 由 `N(p)` 诱导
4. 终止条件和复杂度分析也依赖该 neighborhood stage

---

### 2. `run_ablation_guided_search.py` 使用了错误指标

**问题**:
- 比较的是 GSSE, Random search, Exhaustive search
- 部分地方直接把 social cost 当作 epsilon 或 deviation gap 的 proxy

**SPEC.md 要求** (lines 427-448):
比较以下 variants:
1. Full Algorithm 5 (exact deviation-target prioritization + neighborhood fallback)
2. No deviation-target prioritization
3. Neighborhood-only
4. Deviation-target-only
5. Optional: random-restart local search

---

### 3. Stage I 缺少统一的 baseline comparison runner

**问题**: 没有形成论文需要的统一 baseline 对照链条

**SPEC.md 要求**:
- Algorithm 5 vs PBRD vs BO vs DRL vs GSO
- 统一 budget（相同 max Stage-II calls）
- 输出：epsilon-gap vs iteration, final epsilon vs |I|, runtime vs |I|, Stage-II calls vs |I|

---

### 4. `run_cpu_parallel_baselines.py` 中的 runtime 统计问题

**问题**: 把多个方法一起跑完，再用总时间均摊到每个方法，得到 `runtime_sec_est`

**修复**: 重命名字段为 `runtime_sec_est_shared`，明确标注不能用于 strict runtime comparison

---

## 二、已完成的修复

### 修复 1: Algorithm 5 neighborhood fallback

**文件**: `src/tmc26_exp/stackelberg.py`

**修改内容**:
1. 重写 `algorithm_5_stackelberg_guided_search` 函数
2. 添加 neighborhood fallback 机制：
   - 先评估 exact deviation targets `Y_E^*(p)` 和 `Y_N^*(p)`
   - 若无 improvement，再在 `N(p)` 中搜索
3. 添加追踪指标到 `StackelbergResult`:
   - `outer_iterations`
   - `stage2_oracle_calls`
   - `evaluated_candidates`
   - `evaluated_boundary_points`
   - `esp_revenue`
   - `nsp_revenue`

**解决的 SPEC 条目**:
- Section V.E: Algorithm 5 理论一致性
- Figure 5, 9: 追踪指标支持

---

### 修复 2: Guided search ablation 脚本重写

**文件**: `scripts/run_ablation_guided_search.py`

**修改内容**:
1. 完全重写，实现正确的 variants 比较:
   - `Full`: Full Algorithm 5
   - `NoPrio`: No deviation-target prioritization
   - `NeighborhoodOnly`: Neighborhood-only search
   - `DevTargetOnly`: Deviation-target-only search
   - `RandomRestart`: Random restart baseline
2. 使用 **REAL deviation gap (epsilon)**，而非 social cost proxy
3. 追踪真实 runtime 和 Stage-II oracle calls

**解决的 SPEC 条目**:
- Figure 15: Guided-search ablation

---

### 修复 3: 新增 Stage I baseline comparison runner

**文件**: `scripts/run_stage1_baseline_comparison.py` (新增)

**功能**:
1. 统一比较 Algorithm 5 vs PBRD vs BO vs DRL vs GSO
2. Fair budget comparison (same max Stage-II calls)
3. 真实 wall-clock runtime per method
4. REAL deviation gap (epsilon) 指标
5. 输出多张图表:
   - epsilon vs iteration curves
   - final epsilon vs |I|
   - runtime vs |I|
   - Stage-II calls vs |I|

**解决的 SPEC 条目**:
- Figure 5: epsilon-gap vs iteration
- Figure 9: runtime and Stage-II calls vs system size
- Appendix A1: final epsilon vs |I|

---

### 修复 4: Runtime 统计逻辑修复

**文件**: `scripts/run_cpu_parallel_baselines.py`

**修改内容**:
1. 字段重命名: `runtime_sec_est` → `runtime_sec_est_shared`
2. 添加警告注释：明确不能用于 strict runtime comparison
3. 更新 CSV 字段名

---

## 三、尚未完成的部分

### 1. Boundary visualization 增强

**需要**:
- 添加 baseline trajectories (PBRD, BO, DRL)
- 添加 restricted-pricing-space boundary overlay
- 添加 quantitative summary (distance to boundary)

**优先级**: 中

---

### 2. Stage II approximation ratio 图检查

**需要**:
- 确认 y-axis 是 ratio (V_DG / V_optimal) 而非 gap
- 确认显示 theorem upper bound

**优先级**: 低

---

## 四、修改文件列表

| 文件 | 修改类型 | 修改内容 |
|-----|---------|---------|
| `src/tmc26_exp/stackelberg.py` | 修改 | Algorithm 5 neighborhood fallback + tracking metrics |
| `scripts/run_ablation_guided_search.py` | 重写 | 正确的 variants 比较 + real epsilon |
| `scripts/run_stage1_baseline_comparison.py` | 新增 | 统一 Stage I baseline runner |
| `scripts/run_cpu_parallel_baselines.py` | 修改 | Runtime 字段重命名 + 警告注释 |
| `docs/EXPERIMENT_GAP_FIX_REPORT.md` | 新增 | 本报告 |

---

## 五、推荐验证脚本

完成修复后，建议按以下顺序验证：

```bash
# 1. 测试 Algorithm 5 neighborhood fallback
uv run python scripts/run_stage1_deviation_gap_convergence.py --trials 3

# 2. 测试 Stage I baseline comparison (新增)
uv run python scripts/run_stage1_baseline_comparison.py --trials 3

# 3. 测试 ablation (修复后)
uv run python scripts/run_ablation_guided_search.py --trials 3
```

---

## 六、SPEC 条目覆盖状态

| Figure | 脚本 | 状态 |
|--------|-----|------|
| Figure 1 | `run_stage2_convergence_plot.py` | 已有 |
| Figure 2 | `run_stage2_approximation_ratio.py` | 需检查 |
| Figure 3 | `run_stage2_communication_rounds.py` | 已有 |
| Figure 4 | `run_stage2_exploitability_comparison.py` | 已有 |
| **Figure 5** | `run_stage1_baseline_comparison.py` | ✅ 新增 |
| Figure 6 | `run_stage1_boundary_visualization.py` | 需增强 |
| Figure 7 | `run_stage1_gain_approximation_accuracy.py` | 已有 |
| Figure 8 | `run_stage1_candidate_family_hit_rate.py` | 已有 |
| **Figure 9** | `run_stage1_baseline_comparison.py` | ✅ 新增 |
| Figure 10 | `run_strategic_social_cost.py` | 已有 |
| Figure 11 | `run_strategic_joint_revenue.py` | 已有 |
| Figure 12 | `run_strategic_pareto_tradeoff.py` | 已有 |
| Figure 13 | `run_strategic_fb_sensitivity.py` | 已有 |
| Figure 14 | `run_ablation_L_sensitivity.py` | 已有 |
| **Figure 15** | `run_ablation_guided_search.py` | ✅ 重写 |
| Appendix A1 | `run_stage1_baseline_comparison.py` | ✅ 新增 |
| Appendix A2 | `run_appendix_exploitability_vs_users.py` | 已有 |

---

*报告完成时间: 2026-03-08*
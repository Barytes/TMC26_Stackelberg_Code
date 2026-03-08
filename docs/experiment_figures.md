# 实验图像产出目录

本文档汇总了当前实验代码库可以生成的所有实验图像，按 Phase 组织。

**总计: 17 个图像脚本，覆盖论文所有实验图表**

---

## Phase 1: Stage I 核心实验 (Figures 5-9)

Stage I 是论文的主要贡献，包含 Algorithm 5 (Stackelberg-Guided Search) 的核心实验。

| Figure | 脚本文件 | 描述 |
|--------|---------|------|
| **Figure 5** | `run_stage1_deviation_gap_convergence.py` | Algorithm 5 epsilon 收敛曲线 |
| **Figure 6** | `run_stage1_boundary_visualization.py` | 价格空间边界 + 轨迹叠加 |
| **Figure 7** | `run_stage1_gain_approximation_accuracy.py` | Algorithm 3 gain 近似精度散点图 |
| **Figure 8** | `run_stage1_candidate_family_hit_rate.py` | 候选族 N(p) hit rate 验证 |
| **Figure 9** | `run_stage1_scalability.py` | Stage I 可扩展性 (runtime + oracle calls) |

### 使用方式

```bash
# Figure 5: Deviation gap convergence
uv run python scripts/run_stage1_deviation_gap_convergence.py \
    --n-users 50,100,200 --trials 10

# Figure 6: Boundary visualization (需要先生成 revenue contours)
uv run python scripts/compute_real_revenue_contours.py
uv run python scripts/run_stage1_boundary_visualization.py --n-users 100

# Figure 7: Gain approximation accuracy (小规模实例)
uv run python scripts/run_stage1_gain_approximation_accuracy.py \
    --n-users 6,8,10,12 --trials 20

# Figure 8: Candidate family hit rate
uv run python scripts/run_stage1_candidate_family_hit_rate.py \
    --n-users 6,8,10,12,14 --trials 50

# Figure 9: Scalability analysis
uv run python scripts/run_stage1_scalability.py \
    --n-users 20,50,100,200,500 --trials 5
```

**输出路径**: `outputs/stage1_*/`

---

## Phase 2: Stage II 补全 (Figures 1-4)

Stage II 算法性能对比和收敛分析。

| Figure | 脚本文件 | 描述 |
|--------|---------|------|
| **Figure 1** | `run_stage2_convergence_plot.py` | 迭代收敛图 (social cost vs iteration) |
| **Figure 2** | `run_stage2_approximation_ratio.py` | Theorem 2 近似比验证 |
| **Figure 3** | `run_stage2_communication_rounds.py` | 通信轮数对比 |
| **Figure 4** | `run_stage2_exploitability_comparison.py` | 可剥削性对比 (DG vs CS vs Random) |

### 使用方式

```bash
# Figure 1: Convergence plot
uv run python scripts/run_stage2_convergence_plot.py \
    --n-users 50,100,200 --trials 10

# Figure 2: Approximation ratio (小规模实例，需要 CS)
uv run python scripts/run_stage2_approximation_ratio.py \
    --n-users 6,8,10,12 --trials 20

# Figure 3: Communication rounds
uv run python scripts/run_stage2_communication_rounds.py \
    --n-users 50,100,200 --trials 10

# Figure 4: Exploitability comparison
uv run python scripts/run_stage2_exploitability_comparison.py \
    --n-users 6,8,10,12 --trials 20
```

**输出路径**: `outputs/stage2_*/`

---

## Phase 3: Strategic Settings (Figures 10-13)

不同策略设置下的系统性能对比。

| Figure | 脚本文件 | 描述 |
|--------|---------|------|
| **Figure 10** | `run_strategic_social_cost.py` | 用户社会成本 vs 用户数 |
| **Figure 11** | `run_strategic_joint_revenue.py` | 供应商联合收益 vs 用户数 |
| **Figure 12** | `run_strategic_pareto_tradeoff.py` | Pareto tradeoff 散点图 |
| **Figure 13** | `run_strategic_fb_sensitivity.py` | F/B 容量敏感性分析 |

### 使用方式

```bash
# Figure 10: User social cost comparison
uv run python scripts/run_strategic_social_cost.py \
    --n-users 20,50,100,200 --trials 20

# Figure 11: Joint provider revenue
uv run python scripts/run_strategic_joint_revenue.py \
    --n-users 20,50,100,200 --trials 20

# Figure 12: Pareto tradeoff scatter
uv run python scripts/run_strategic_pareto_tradeoff.py \
    --n-users 100 --trials 50

# Figure 13: F/B sensitivity
uv run python scripts/run_strategic_fb_sensitivity.py \
    --F-values 50,100,200 --B-values 20,40,80 --trials 20
```

**输出路径**: `outputs/strategic_*/`

### 对比方法

| 方法 | 描述 |
|-----|------|
| **GSSE** | Proposed - Algorithm 5 Stackelberg-guided search |
| **MarketEquilibrium** | Tatonnement-style price adjustment |
| **SingleSP** | Single leader (ESP only) |
| **RandomOffloading** | Random baseline |

---

## Phase 4: Ablation Studies (Figures 14-15)

算法组件的消融实验。

| Figure | 脚本文件 | 描述 |
|--------|---------|------|
| **Figure 14** | `run_ablation_L_sensitivity.py` | 采样密度 L 敏感性分析 |
| **Figure 15** | `run_ablation_guided_search.py` | Guided-search 消融对比 |

### 使用方式

```bash
# Figure 14: L sensitivity
uv run python scripts/run_ablation_L_sensitivity.py \
    --L-values 4,8,12,20,32,48 --n-users 50,100,200 --trials 10

# Figure 15: Guided search ablation
uv run python scripts/run_ablation_guided_search.py \
    --n-users 20,50,100 --trials 10
```

**输出路径**: `outputs/ablation_*/`

---

## Phase 5: Appendix (A1-A2)

附录补充实验。

| Figure | 脚本文件 | 描述 |
|--------|---------|------|
| **Appendix A1** | `run_appendix_final_epsilon_vs_users.py` | Final deviation gap vs 用户数 |
| **Appendix A2** | `run_appendix_exploitability_vs_users.py` | Stage II exploitability 完整对比 |

### 使用方式

```bash
# Appendix A1: Final epsilon scaling
uv run python scripts/run_appendix_final_epsilon_vs_users.py \
    --n-users 20,50,100,200,500 --trials 20

# Appendix A2: Full exploitability comparison
uv run python scripts/run_appendix_exploitability_vs_users.py \
    --n-users 6,8,10,12,14,16 --trials 20
```

**输出路径**: `outputs/appendix_*/`

---

## 其他实验脚本

### Stage II 算法对比 (已有)

| 脚本文件 | 描述 |
|---------|------|
| `run_stage2_social_cost_vs_users.py` | Social cost vs 用户数 |
| `run_stage2_wall_time_vs_users.py` | Runtime vs 用户数 |
| `run_stage2_offloading_size_vs_users.py` | Offloading size vs 用户数 |
| `run_algorithm2_social_cost_gap_vs_users.py` | Algorithm 2 vs CS gap |
| `run_algorithm2_exploitability_vs_users.py` | Algorithm 2 exploitability |

### 价格平面分析 (已有)

| 脚本文件 | 描述 |
|---------|------|
| `compute_real_revenue_contours.py` | ESP/NSP 真实收益等高线 |
| `compute_deviation_gaps.py` | Deviation gap 分析 |

### Suite 实验

```bash
uv run python scripts/run_suite.py --config configs/default.toml --suite suites/<name>.toml
```

---

## 图例说明

### Stage II 方法颜色

| 方法 | 颜色 | 标记 |
|-----|------|------|
| Algorithm 2 (DG) | 深绿 (#1b5e20) | 圆圈 (o) |
| UBRD | 深红 (#b71c1c) | 方块 (s) |
| VI | 深蓝 (#0d47a1) | 三角 (^) |
| PEN | 橙色 (#e65100) | 倒三角 (v) |
| CS | 紫色 (#4a148c) | 菱形 (D) |
| Random | 灰色 (#757575) | 叉号 (x) |

### Strategic Settings 颜色

| 方法 | 颜色 | 标记 |
|-----|------|------|
| GSSE (Proposed) | 深绿 (#1b5e20) | 圆圈 (o) |
| MarketEquilibrium | 深蓝 (#0d47a1) | 三角 (^) |
| SingleSP | 橙色 (#e65100) | 方块 (s) |
| RandomOffloading | 灰色 (#757575) | 叉号 (x) |

---

## 快速测试

运行所有脚本的快速测试 (小规模):

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

---

*最后更新: 2026-03-08*
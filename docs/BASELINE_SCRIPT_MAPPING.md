# Baseline And Script Mapping

## 1. 目的

本文件是工作流 4 的正式交付物。

它用于冻结两类映射：

- 论文 baseline taxonomy 与当前代码实现之间的对应关系；
- `scripts/` 与论文 block A-F 之间的对应关系。

---

## 2. 论文正式 baseline taxonomy

当前论文把 baseline 分成两组：

- Stackelberg-equilibrium baselines:
  - `GSO`
  - `GA`
  - `BO`
  - `MARL`
- strategic-setting baselines:
  - `ME`
  - `SingleSP`
  - `Coop`
  - `Rand`

冻结规则：

- README、SPEC、图注、summary 优先使用上面这组 canonical label；
- 代码里的旧函数名、旧输出名、辅助方法名，不能反过来替代这组 paper label。

---

## 3. 论文 baseline 到代码实现映射

| 论文标签 | 当前代码入口 | 代码状态 | 对外处置 |
| --- | --- | --- | --- |
| `GSO` | `baseline_stage1_grid_search_oracle` | 已直接实现 | 正式 paper-facing baseline |
| `GA` | `baseline_stage1_ga` | 已直接实现 | 正式 paper-facing baseline |
| `BO` | `baseline_stage1_bo` | 已直接实现 | 正式 paper-facing baseline |
| `MARL` | `baseline_stage1_drl` | 仅有 legacy RL proxy，不是明确的 MARL public wrapper | 对外必须写成 `MARL (current code proxy: DRL)` |
| `ME` | `baseline_market_equilibrium` | 已直接实现，但输出名是 `MarketEquilibrium` | 对外统一写 `ME` |
| `SingleSP` | `baseline_single_sp` | 已直接实现 | 正式 paper-facing baseline |
| `Coop` | 无 dedicated public wrapper | 当前缺口 | 文档中必须明确“尚未有单独公共实现” |
| `Rand` | `baseline_random_offloading` | 已直接实现，但输出名是 `RandomOffloading` | 对外统一写 `Rand` |

重要结论：

- 当前代码并没有一个真正独立、公开、稳定的 `Coop` 入口；
- 当前代码里的 `DRL` 不能无说明地直接等同为论文里的 `MARL`；
- 因此，工作流 4 之后对外叙述必须显式承认这两个实现缺口。

---

## 4. Auxiliary / Legacy 方法冻结表

下列方法可以继续保留，但不应再作为论文正式 baseline taxonomy 的主成员：

| 代码标签 | 当前定位 | 说明 |
| --- | --- | --- |
| `GSSE` / `proposed_gsse` | proposal/internal | 属于方法本身，不是 baseline |
| `CS` / `baseline_stage2_centralized_solver` | Stage II reference baseline | 用于 Block A exact reference，不属于 Stage I 主 baseline 组 |
| `UBRD` | auxiliary Stage II comparison | supplementary / archive 倾向 |
| `VI` | auxiliary Stage II comparison | supplementary / archive 倾向 |
| `PEN` | auxiliary Stage II comparison | supplementary / archive 倾向 |
| `PBRD` | legacy Stage I comparison | 仅 auxiliary/legacy |
| `PBRD_DISCRETE` | legacy plotting/comparison helper | 仅 auxiliary/legacy |
| `EPEC-DIAG` | auxiliary diagnostic | 仅 auxiliary/legacy |
| `BO-online` | auxiliary ablation / online variant | 仅 supplementary |
| `DRL` | legacy RL code path | 若继续保留，只能作为 `MARL` 的临时代码 proxy 注记 |

冻结规则：

- `run_all_baselines()` 当前混合了 formal baseline、proposal、auxiliary 和 legacy 方法；
- 因此它不能被解释成“论文 baseline 的官方总入口”；
- 在后续工作流中，如果需要正式 baseline runner，应单独拆出 paper-facing baseline launcher。

---

## 5. Script-To-Block 映射

说明：

- `run_figure_*.py` 是 workflow 5 之后的公开 figure-level 入口；
- legacy `run_*` / `plot_*` 脚本作为共享实现参考或辅助工具继续保留；
- 当前所有 blueprint 图号都已经有直接可运行的 figure-level script。

### Canonical Figure-Level Entries

| 脚本 | 当前定位 | 状态 |
| --- | --- | --- |
| `scripts/run_figure_A1_stage2_social_cost_trace.py` | Figure A1 canonical entry | direct runner |
| `scripts/run_figure_A2_stage2_social_cost_multiscale.py` | Figure A2 canonical entry | direct runner |
| `scripts/run_figure_A3_stage2_approx_ratio_bound.py` | Figure A3 canonical entry | direct runner |
| `scripts/run_figure_A4_stage2_runtime_vs_users.py` | Figure A4 canonical entry | direct runner |
| `scripts/run_figure_A5_stage2_rollback_diagnostics.py` | Figure A5 canonical entry | direct runner |
| `scripts/run_figure_A6_stage2_exploitability_supp.py` | Figure A6 canonical entry | direct runner |
| `scripts/run_figure_B1_stage1_joint_revenue_heatmap.py` | Figure B1 canonical entry | direct runner |
| `scripts/run_figure_B2_stage1_restricted_gap_heatmap.py` | Figure B2 canonical entry | direct runner |
| `scripts/run_figure_B3_esp_slice_boundary_comparison.py` | Figure B3 canonical entry | direct runner |
| `scripts/run_figure_B4_nsp_slice_boundary_comparison.py` | Figure B4 canonical entry | direct runner |
| `scripts/run_figure_B5_joint_revenue_boundary_overlay.py` | Figure B5 canonical entry | direct runner |
| `scripts/run_figure_B6_candidate_family_diagnostics.py` | Figure B6 canonical entry | direct runner |
| `scripts/run_figure_C1_restricted_gap_trajectory.py` | Figure C1 canonical entry | direct runner |
| `scripts/run_figure_C2_best_response_gain_trajectory.py` | Figure C2 canonical entry | direct runner |
| `scripts/run_figure_C3_price_trajectory_on_gap_heatmap.py` | Figure C3 canonical entry | direct runner |
| `scripts/run_figure_C4_final_gap_vs_budget.py` | Figure C4 canonical entry | direct runner |
| `scripts/run_figure_C5_trajectory_compare_supp.py` | Figure C5 canonical entry | direct runner |
| `scripts/run_figure_D1_stage2_runtime_vs_users.py` | Figure D1 canonical entry | direct runner |
| `scripts/run_figure_D2_stage1_runtime_vs_users.py` | Figure D2 canonical entry | direct runner |
| `scripts/run_figure_D3_stage2_calls_inside_stage1.py` | Figure D3 canonical entry | direct runner |
| `scripts/run_figure_D4_exact_runtime_feasibility.py` | Figure D4 canonical entry | direct runner |
| `scripts/run_figure_E1_user_social_cost_compare.py` | Figure E1 canonical entry | direct runner |
| `scripts/run_figure_E2_provider_revenue_compare.py` | Figure E2 canonical entry | direct runner |
| `scripts/run_figure_E3_resource_utilization_compare.py` | Figure E3 canonical entry | direct runner |
| `scripts/run_figure_E4_price_and_offloading_compare.py` | Figure E4 canonical entry | direct runner |
| `scripts/run_figure_F1_q_sensitivity.py` | Figure F1 canonical entry | direct runner |
| `scripts/run_figure_F2_resource_asymmetry_sensitivity.py` | Figure F2 canonical entry | direct runner |
| `scripts/run_figure_F3_provider_cost_sensitivity.py` | Figure F3 canonical entry | direct runner |
| `scripts/run_figure_F4_user_distribution_sensitivity.py` | Figure F4 canonical entry | direct runner |

### Block A: Stage II SCM quality

| 脚本 | 当前定位 | 状态 |
| --- | --- | --- |
| `scripts/run_stage2_social_cost_compare.py` | integrated Stage II SCM solver 的 social-cost trace 与 centralized reference 对比 | primary |
| `scripts/run_stage2_approximation_ratio.py` | `V(X_DG)/V(X*)` 与 theorem bound | primary |
| `scripts/run_algorithm2_exploitability_vs_users.py` | supplementary exploitability diagnostic | supplementary |

### Block B: Stage I geometry and boundary-price diagnostics

| 脚本 | 当前定位 | 状态 |
| --- | --- | --- |
| `scripts/run_boundary_hypothesis_check.py` | switching / boundary-price / candidate-family diagnostics | primary |
| `scripts/run_stage1_price_heatmaps.py` | revenue / gap heatmap diagnostics | auxiliary |
| `scripts/run_stage1_price_heatmaps_cs_gekko.py` | exact-CS-assisted heatmap diagnostics | auxiliary |
| `scripts/reprint_stage1_selected_figures.py` | 基于既有输出重绘 boundary/slice 图 | helper |

### Block C: Stage I iterative-pricing convergence

| 脚本 | 当前定位 | 状态 |
| --- | --- | --- |
| `scripts/run_stage1_vbbr_trajectory_on_heatmap.py` | Stage I trajectory on heatmap | primary |
| `scripts/plot_stage1_trajectories_on_heatmap.py` | proposal / baseline trajectory overlay on precomputed heatmap | auxiliary |
| `scripts/plot_pbdr_trajectory_from_heatmap_csv.py` | discrete PBRD legacy trajectory plot | legacy helper |

### Block D/E/F

当前仓库已经有覆盖 Block A-F 的 figure-level public scripts，后续工作重点不再是补 public runner，而是校准统计设定、baseline 完整性和正式论文产物质量。

---

## 6. 当前脚本命名的冻结解释

现有脚本名里仍有多处 legacy 命名：

- `vbbr`
- `pbdr`
- `algorithm2`
- `exploitability`

工作流 4 之后的解释规则是：

- 文件名可以暂时保留；
- 但脚本头注释、README、文档映射、summary 都必须按论文 block 和当前术语解释；
- 不允许因为脚本文件名旧，就在文档里回退到旧叙事。

---

## 7. 待收敛缺口

工作流 4 完成后，仍明确存在以下缺口：

- 需要 dedicated `Coop` public baseline；
- 需要一个更名或重写后的 `MARL` public baseline，而不是继续直接暴露 `DRL`；
- 需要为 Block D/E/F 增加 dedicated public scripts；
- 需要在后续工作流中决定 `run_all_baselines()` 是否拆分成：
  - formal paper baselines
  - auxiliary diagnostics
  - archived legacy methods

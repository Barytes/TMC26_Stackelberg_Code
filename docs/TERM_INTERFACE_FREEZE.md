# 术语与接口冻结

## 1. 目的

本文件是工作流 1 的正式交付物。

它用于冻结三类内容：

- 论文当前版本允许使用的正式术语；
- 仓库中已经存在但不再作为主叙事的 legacy 命名；
- `src/tmc26_exp/` 与 `scripts/` 的稳定接口边界。

从本文件开始，后续所有文档、脚本、图表标题、结果汇总，都应优先使用这里定义的 canonical terminology。

---

## 2. Source of Truth 顺序

发生冲突时，优先级固定为：

1. `TMC26_Stackelberg.tex`
2. 本文件
3. `docs/SPEC.md`
4. `docs/DEV.md`
5. 现有代码命名、脚本命名、配置项命名

含义很直接：

- 代码里出现旧名字，不代表论文术语已经变回旧版；
- 新增文档和新脚本，不得再把旧名字抬升成“正式词汇”。

---

## 3. 正式术语冻结表

| 论文正式术语 | 冻结定义 | 文档/图表中允许的简称 | 代码映射锚点 |
| --- | --- | --- | --- |
| Stage II integrated SCM solver | `Algorithm 2 + Algorithm 1` 的整体 follower-side SCM / SOE 求解路线 | `Stage II SCM solver` | `stackelberg.py` |
| Algorithm 1 | Stage II inner resource-allocation solver | `Alg. 1` | `algorithm_1_distributed_primal_dual` |
| Algorithm 2 | Stage II outer offloading-user selection with rollback | `Alg. 2` | `algorithm_2_heuristic_user_selection` |
| SOE | socially optimal equilibrium | `SOE` | `GreedySelectionResult.social_cost` 等 Stage II 输出 |
| SCM | social cost minimization problem in Stage II | `SCM` | Stage II route overall |
| restricted pricing space | 固定 offloading set 下的定价区域 | `restricted pricing space` | Stage I structural analysis |
| switching set | Stage I 中切换不同 offloading regime 的边界集合 | `switching set` | Stage I structural diagnostics |
| local candidate family | `N_Q(p)`，当前价格附近的局部候选集合族 | `candidate family` | `_candidate_family` |
| candidate boundary price | 固定对手价格和候选集合下的边界价格 | `boundary price` | `_boundary_price_for_provider` |
| Algorithm 3 | 论文当前的 boundary-price-based best-response estimation | `Alg. 3` | `algorithm_3_boundary_price_br_estimation` |
| paper iterative pricing | 论文当前的 Stage I 主线 iterative pricing pipeline | `paper_iterative_pricing` | `algorithm_paper_iterative_pricing_stage1` |
| NE gap | Stage I pricing game 的正式 NE gap | `NE gap` | 论文定义；代码中存在近似实现 |
| boundary-price-restricted NE gap | Stage I 在候选边界价格集合上的 restricted NE gap | `restricted NE gap` | 论文当前主口径 |
| Stackelberg-equilibrium baselines | `GSO / GA / BO / MARL` | `equilibrium baselines` | `baselines.py` |
| strategic-setting baselines | `ME / SingleSP / Coop / Rand` | `strategic-setting baselines` | `baselines.py` |

冻结规则：

- 文档、图表、summary、README 一律优先使用上表左列术语；
- 如果需要兼容旧代码名字，只能在括号里说明，不可反过来以旧名为主；
- 新增代码对象命名，应尽量贴合上表而不是继续扩张旧前缀。

---

## 4. Legacy 命名处置表

| Legacy 名称 | 当前应解释为 | 处置决策 | 说明 |
| --- | --- | --- | --- |
| `vbbr` | 旧版 Stage I solver 命名；现在只能解释为 boundary-price / iterative-pricing 历史实现名 | `仅文档映射，逐步移除` | 不再作为论文主术语 |
| `pbdr` | 旧版 price-BRD / plotting helper 命名 | `仅内部保留` | 只允许出现在 legacy helper 或旧输出文件名 |
| `verified epsilon` | 旧版 gap 语言 | `禁止在新文档中使用` | 统一替换为 `NE gap` 或 `restricted NE gap` |
| `epsilon_proxy` | baseline 中的近似 gap 字段 | `内部保留，后续重命名` | 对外说明时必须写明其是 proxy |
| `RNE` | 旧版结果对象命名 | `仅内部保留` | 文档中不再使用 |
| `VBBROracleResult` | 旧版 Stage I oracle 结果对象 | `仅内部保留` | paper-facing oracle 已新增独立对象 |
| `stage1_solver_variant = vbbr_brd` | 保留的备选 Stage I pipeline 配置值 | `允许保留，但必须标注 backup` | 不能再当成论文主线默认叙事 |
| `stage1_solver_variant = paper_iterative_pricing` | 论文主线 Stage I pipeline 配置值 | `推荐公开使用` | 论文对齐实验优先采用 |
| `run_stage1_vbbr_trajectory_on_heatmap.py` | Stage I pricing trajectory figure 的 legacy 脚本名 | `保留兼容层` | 公共入口改用 `run_figure_C3_price_trajectory_on_gap_heatmap.py` |
| `run_boundary_hypothesis_check.py` | Stage I boundary diagnostics 的 legacy 脚本名 | `保留兼容层` | 公共入口改用 `run_figure_B3_esp_slice_boundary_comparison.py` / `run_figure_B4_nsp_slice_boundary_comparison.py` / `run_figure_B5_joint_revenue_boundary_overlay.py` |
| `tmc26-exp` / `cli.py` | internal umbrella runner | `降级为内部工具` | 不再作为 README 主入口 |
| exploitability figure/script | supplementary diagnostic | `保留但降级` | 不能替代论文主指标 |

冻结规则：

- 旧名字可以继续存在于代码中，但只能作为兼容层；
- 新文件名、README 主入口、图标题、结果汇总，不得再直接采用旧名字；
- 如果某旧名字必须保留，应在注释或 README 映射中明确说明其 canonical meaning。

---

## 5. 模块接口冻结表

### `src/tmc26_exp/model.py`

职责：

- 定义 `UserBatch`
- 提供单用户/逐用户静态公式：
  - `local_cost`
  - `theta`
  - unconstrained proxy 系列

冻结接口定位：

- 这是“模型原语层”，不负责 Stage I / Stage II 求解流程；
- `unconstrained_*` 只属于 proxy diagnostics，不得被写成论文正式结果。

### `src/tmc26_exp/simulator.py`

职责：

- `sample_users`
- `evaluate_metric_surface`

冻结接口定位：

- 这是“采样与 surface diagnostics 层”；
- 它生成 `MetricSurface`，但不直接代表论文主求解结果。

### `src/tmc26_exp/metrics.py`

职责：

- 注册 proxy metric；
- 暴露 `Metric`、`METRICS`、`get_metric`。

冻结接口定位：

- 当前 `metrics.py` 是 proxy / diagnostic metric 层；
- 不应把这里的 surface 指标当成论文的主 social cost、NE gap、provider revenue 结果。

### `src/tmc26_exp/plotting.py`

职责：

- 画 generic metric surface heatmap / contour。

冻结接口定位：

- 这是 generic plotting helper；
- 不是论文 figure 的唯一入口；
- 真正论文图入口应优先来自 `scripts/run_figure_*.py`。

### `src/tmc26_exp/stackelberg.py`

职责：

- Stage II 核心求解逻辑；
- Stage I 核心求解逻辑；
- 结构诊断辅助函数。

冻结接口定位：

- 核心结果对象：
  - `InnerSolveResult`
  - `GreedySelectionResult`
  - `StackelbergResult`
- 核心求解入口：
  - `solve_stage2_scm`
  - `solve_stage1_pricing`
  - `algorithm_1_distributed_primal_dual`
  - `algorithm_2_heuristic_user_selection`
  - `run_stage1_solver`
- 核心结构辅助：
  - `_candidate_family`
  - `_boundary_price_for_provider`
  - `_provider_revenue_from_stage2_result`

冻结规则：

- `solve_stage2_scm(..., inner_solver_mode="primal_dual")` 是 Stage II 的 canonical paper-facing 入口；
- `solve_stage1_pricing(...)` 是 Stage I 的 canonical paper-facing 入口；
- 文档优先解释 `GreedySelectionResult` 为 Stage II integrated SCM solver result；
- `GreedySelectionResult` 应稳定提供：
  - `social_cost`
  - `rollback_count`
  - `accepted_admissions`
  - `inner_call_count`
  - `runtime_sec`
  - `social_cost_trace`
- 文档优先解释 `StackelbergResult` 为 Stage I iterative-pricing result；
- `StackelbergResult` 应稳定提供：
  - `price`
  - `restricted_gap`
  - `gain_E`
  - `gain_N`
  - `trajectory`
  - `final_stage2_result`
- `GainApproxResult`、`VBBROracleResult`、`BoundaryPriceBROracleResult` 属于内部 / auxiliary 结果对象，不作为对外主接口。

### `src/tmc26_exp/baselines.py`

职责：

- baseline 运行；
- exact/grid evaluation；
- `BaselineOutcome` / `Stage1GridEvaluation`。

冻结接口定位：

- baseline 按两大组解释：
  - equilibrium baselines
  - strategic-setting baselines
- 当前代码与论文 baseline taxonomy 的冻结映射是：
  - direct paper-facing: `GSO / GA / BO / ME / SingleSP / Rand`
  - legacy proxy only: `DRL -> MARL proxy`
  - missing dedicated public wrapper: `Coop`
  - auxiliary / legacy only: `UBRD / VI / PEN / PBRD / PBRD_DISCRETE / EPEC-DIAG / BO-online / GSSE`
- 当前文件内部若继续调用旧命名函数，不影响外部术语冻结；
- 对外说明时必须以 baseline 类型和论文分组为主，而不是内部函数名。

### `src/tmc26_exp/config.py`

职责：

- 解析 TOML；
- 定义实验配置 dataclass。

冻结接口定位：

- 稳定的对外配置层级是：
  - top-level run controls
  - `[system]`
  - `[stackelberg]`
  - `[baselines]`
  - `[detailed_experiment]`
  - `[price_grid]`
  - `[user_distributions]`
- `StackelbergConfig` 和 `BaselineConfig` 中仍有大量 legacy 字段；
- 在文档里应描述“语义分组”，不必逐字镜像所有旧配置键。

### `src/tmc26_exp/cli.py`

职责：

- internal umbrella runner；
- metric surfaces + Stage I run + baseline run + detailed plan emission。

冻结接口定位：

- 定位为 internal tool；
- 不是论文主实验入口；
- README 中只能作为 secondary / internal path 出现。

---

## 6. 脚本入口冻结表

### Canonical figure-level public entry points

这些脚本是当前冻结后的公共入口：

- `scripts/run_figure_A1_stage2_social_cost_trace.py`
- `scripts/run_figure_A2_stage2_social_cost_multiscale.py`
- `scripts/run_figure_A3_stage2_approx_ratio_bound.py`
- `scripts/run_figure_A6_stage2_exploitability_supp.py`
- `scripts/run_figure_B1_stage1_joint_revenue_heatmap.py`
- `scripts/run_figure_B2_stage1_restricted_gap_heatmap.py`
- `scripts/run_figure_B3_esp_slice_boundary_comparison.py`
- `scripts/run_figure_B4_nsp_slice_boundary_comparison.py`
- `scripts/run_figure_B5_joint_revenue_boundary_overlay.py`
- `scripts/run_figure_C3_price_trajectory_on_gap_heatmap.py`

### Legacy script aliases

这些脚本暂时保留，但只作为兼容层：

- `scripts/run_stage2_social_cost_compare.py`
- `scripts/run_stage2_approximation_ratio.py`
- `scripts/run_algorithm2_exploitability_vs_users.py`
- `scripts/run_boundary_hypothesis_check.py`
- `scripts/run_stage1_price_heatmaps.py`
- `scripts/run_stage1_price_heatmaps_cs_gekko.py`
- `scripts/run_stage1_vbbr_trajectory_on_heatmap.py`

### Plotting / helper scripts

这些不作为公共 figure 主入口：

- `scripts/plot_stage1_trajectories_on_heatmap.py`
- `scripts/plot_pbdr_trajectory_from_heatmap_csv.py`
- `scripts/reprint_stage1_selected_figures.py`

冻结规则：

- README、说明文档、后续实验记录一律优先引用 canonical figure scripts；
- legacy scripts 可以继续作为共享实现参考被调用，但 public `run_figure_*.py` 入口不再走 wrapper/subprocess 层。

---

## 7. 配置接口冻结规则

当前配置层存在“语义正确但名字陈旧”的问题，因此冻结规则如下：

- 文档中按语义描述配置，不按旧键名逐个教学；
- 如需在脚本帮助信息中展示 legacy 配置项，应同时给出 canonical explanation；
- 新增配置项不得继续使用 `vbbr_*`、`pbdr_*`、`verified_*` 一类前缀；
- 对现有旧配置项，短期内采用：
  - 代码兼容
  - 文档重解释
  - 后续再重命名

---

## 8. 对后续工作流的硬约束

从本文件生效后，工作流 2-7 必须遵守以下约束：

- 不再新增新的 legacy 前缀命名；
- Stage II 始终写成 integrated SCM solver；
- Stage I 始终写成 boundary-price / iterative-pricing 路线；
- 新脚本优先采用 `run_figure_*` 命名；
- 新图表标题不得出现 `VBBR`、`PBDR`、`verified epsilon` 等旧主术语；
- 若某内部字段仍叫旧名字，对外导出层必须补 canonical label。

---

## 9. 本工作流完成定义

当以下条件满足时，认为“工作流 1：术语与接口冻结”完成：

- 有一份稳定的术语冻结文档；
- 有一份稳定的模块接口冻结表；
- 有一份 stable public entry points 列表；
- README / SPEC / DEV 与本文件之间不再互相冲突；
- 后续改动可以明确判断“某名字是否允许继续扩散”。

# 更改计划

## 1. 目的

本计划用于指导当前仓库从“旧实现命名与旧实验组织”逐步收敛到 `TMC26_Stackelberg.tex`、`docs/SPEC.md` 和 `docs/DEV.md` 所定义的最新版论文口径。

计划目标有三项：

- 让代码结构、脚本命名、实验指标、baseline 分组与论文当前叙事一致；
- 让 Stage II 按 `Algorithm 2 + Algorithm 1` 的整体 SCM 求解路线组织，而不是拆成两条独立实验线；
- 让 Stage I 围绕 restricted pricing space、boundary price、best-response estimation、iterative pricing 展开，而不是继续沿用旧版 VBBR / verified-gap 语言。

---

## 2. 当前状态

已完成：

- `README.md` 已按当前论文口径重写；
- `docs/SPEC.md` 已按当前论文结构重写；
- `docs/DEV.md` 已压缩 Stage II 口径；
- `CLAUDE.md` 已改为 `TMC26_Stackelberg.tex` 优先；
- 工作流 1 的冻结文档已落地为 `docs/TERM_INTERFACE_FREEZE.md`；
- 工作流 2 的 Stage II 审计文档已落地为 `docs/STAGE2_SCM_AUDIT.md`；
- Stage II 的 canonical 代码入口已冻结为 `solve_stage2_scm`，Block A 主脚本已改用统一结果结构；
- 工作流 3 的 Stage I 审计文档已落地为 `docs/STAGE1_PRICING_AUDIT.md`；
- Stage I 的 canonical 代码入口已冻结为 `solve_stage1_pricing`，`StackelbergResult` 已携带 final Stage II outcome；
- 工作流 4 的 baseline/script 映射文档已落地为 `docs/BASELINE_SCRIPT_MAPPING.md`；
- 工作流 5 前的逐图蓝图已落地为 `docs/FIGURE_SCRIPT_BLUEPRINT.md`；
- 工作流 5 已完成：
  - `configs/figures/paper_base.toml` 已提供论文对齐的 figure-level 默认配置；
  - A-F 全部 blueprint 图号均已落成 `scripts/run_figure_*.py` canonical 入口；
  - 全部 figure-level 入口均为直接可运行脚本，不再通过 subprocess wrapper 或 placeholder 占位；
- 旧的 `docs/stage2_vi_penalty_implementation_guide.md` 和 `review.md` 已删除。

仍待完成：

- 代码、脚本、配置项里仍存在旧命名；
- Stage I / Stage II 求解逻辑是否完全贴合当前论文，仍需逐模块核对；
- 论文当前实验 block A-F 已经有公开脚本入口，后续重点转为统计质量、参数校准和产物验收；
- 部分脚本名仍保留旧语义，尚未统一清理或建立稳定映射。

---

## 3. 约束与原则

执行过程中遵守以下原则：

- `TMC26_Stackelberg.tex` 是最高优先级来源；
- `docs/SPEC.md` 定义实验主线；
- `docs/DEV.md` 提供实现映射；
- 如果脚本名、配置项名与论文冲突，优先修正文档映射，再决定是否改代码命名；
- 优先做“语义对齐”和“实验可复现”，再做命名洁癖式清理。

---

## 4. 工作流总览

建议按 7 个工作流推进：

1. 术语与接口冻结
2. Stage II SCM 路线核对与补强
3. Stage I 定价求解路线核对与补强
4. Baseline 体系与脚本映射清理
5. 实验 Block A-F 落地
6. 图表、输出与结果汇总统一
7. 最终一致性验收

---

## 5. 工作流 1：术语与接口冻结

### 目标

- 固定当前项目允许使用的正式术语；
- 建立代码对象与论文对象的一一映射。

### 任务

- 列出论文正式术语表：
  - SOE
  - SCM
  - Algorithm 1 / Algorithm 2
  - restricted pricing space
  - switching set
  - candidate boundary price
  - NE gap / boundary-price-restricted NE gap
- 列出当前代码中的旧命名表：
  - `vbbr`
  - `pbdr`
  - `verified epsilon`
  - 其他 legacy 名称
- 对每个旧命名建立“保留 / 重命名 / 仅文档映射”决策。

### 交付物

- 一份术语映射表；
- 一份接口映射表，覆盖 `model.py`、`stackelberg.py`、`baselines.py`、`simulator.py`、`plotting.py`。

### 验收标准

- 后续文档、脚本注释、图表标题都不再混用新旧术语；
- 所有人都能明确 Stage II 是 integrated SCM solver，而不是两条独立算法故事。

---

## 6. 工作流 2：Stage II SCM 路线核对与补强

### 目标

- 确认当前实现能按论文定义求解 Stage II 的 SCM / SOE；
- 让 Stage II 的输出与实验 block A 对齐。

### 任务

- 核对 `Algorithm 2 + Algorithm 1` 的整体调用链是否成立：
  - outer offloading-set selection
  - inner resource allocation
  - rollback 逻辑
- 核对固定集合 closed form 的使用位置：
  - 只能作为 Stage I 结构分析与辅助验证；
  - 不能偷换成 Stage II 主实验叙事。
- 统一 Stage II 返回对象，至少包含：
  - offloading set
  - `f`, `b`
  - `lambda_F`, `lambda_B`
  - social cost
  - runtime
  - rollback count
- 补充小规模 exact 对照入口，用于 approximation ratio 和 social cost 对比。

### 交付物

- Stage II 统一结果结构；
- Block A 所需数据字段；
- 一份 Stage II sanity-check 清单。

### 验收标准

- 可以稳定输出 `V(X_DG)`、`V(X*)`、rollback 次数、offloading-set size；
- Stage II 不再被描述成 Algorithm 1 单独实验。

---

## 7. 工作流 3：Stage I 定价求解路线核对与补强

### 目标

- 让 Stage I 实现与论文当前结构一致；
- 为 block B、C、D、F 提供统一数据接口。

### 任务

- 核对 restricted pricing space 与 switching set 的实现映射；
- 核对 local candidate family `N_Q(p)` 的构造逻辑；
- 核对 candidate boundary price 的计算与分段条件；
- 核对 iterative pricing 的更新顺序与停止条件；
- 统一 Stage I 结果对象，至少包含：
  - price trajectory
  - current / final restricted NE gap
  - ESP / NSP best-response gains
  - candidate family size diagnostics
  - final price pair
  - final Stage II outcome

### 交付物

- Stage I 统一结果结构；
- Block B/C/D/F 所需数据字段；
- 一份 Stage I structural diagnostics 清单。

### 验收标准

- 可以稳定生成价格轨迹、best-response gain、restricted NE gap；
- Heatmap、slice、boundary overlay、trajectory 可复用同一组底层结果。

---

## 8. 工作流 4：Baseline 体系与脚本映射清理

### 目标

- 让 baseline 体系与论文当前分组一致；
- 清理或重新解释旧脚本名。

### 任务

- 核对 baseline 分组是否统一为：
  - Stackelberg-equilibrium baselines: GSO / GA / BO / MARL
  - strategic-setting baselines: ME / SingleSP / Coop / Rand
- 检查 `baselines.py` 是否包含旧版非主线方法，并决定：
  - 删除
  - 保留但降级为 auxiliary
  - 仅内部调试使用
- 为现有脚本建立论文口径映射：
  - 哪些属于 Stage II SCM quality
  - 哪些属于 Stage I geometry
  - 哪些属于 trajectory / plotting helper
- 如果短期不改文件名，至少在脚本头注释里说明其论文对应关系。

### 交付物

- baseline 映射表；
- script-to-block 映射表；
- 待重命名脚本列表。

### 验收标准

- 不再出现 README / SPEC / 实验输出说一套、脚本文件名暗示另一套的情况。

---

## 9. 工作流 5：实验 Block A-F 落地

### Block A：Stage II SCM quality

任务：

- social cost 对比；
- approximation ratio；
- theorem upper bound；
- rollback diagnostics；
- Stage II runtime。

### Block B：Stage I geometry and boundary-price diagnostics

任务：

- price heatmap / contour；
- fixed-opponent revenue slices；
- boundary overlay；
- `N_Q(p)` 局部候选族诊断。

### Block C：Stage I iterative-pricing convergence

任务：

- restricted NE gap trajectory；
- best-response gain trajectory；
- final price pair；
- Stage I 与 GSO / GA / BO / MARL 对比。

### Block D：Runtime and scalability

任务：

- Stage II SCM runtime vs. users；
- Stage I runtime vs. users；
- exact solver runtime on feasible small instances；
- Stage II solve count inside Stage I。

### Block E：Strategic-setting comparisons

任务：

- full model vs. ME / SingleSP / Coop / Rand；
- user social cost；
- provider revenues；
- utilization。

### Block F：Robustness and sensitivity

任务：

- `Q` 扫描；
- `F`、`B`、`F/B` 扫描；
- `c_E`、`c_N` 扫描；
- 用户分布 / 异质性扫描。

### 每个 Block 的统一要求

- 明确输入配置；
- 明确输出 CSV；
- 明确图表文件名；
- 明确汇总指标；
- 明确随机种子和 trial 数。
- 具体逐图脚本矩阵以 `docs/FIGURE_SCRIPT_BLUEPRINT.md` 为准；
- workflow 5 以后采用 one-figure-one-script 公开入口。

### 交付物

- 每个 block 至少 1 个主脚本；
- 配套配置文件；
- 输出目录命名规范；
- 一页 block-level README 或注释说明。

### 验收标准

- A-F 六个 block 都能从脚本入口明确落地；
- 每个 block 都能直接对应 `docs/SPEC.md` 中的目标与指标。

---

## 10. 工作流 6：图表、输出与结果汇总统一

### 目标

- 统一图表命名、输出目录结构、CSV schema、summary 文本格式；
- 为后续论文排图提供稳定入口。

### 任务

- 统一 CSV 列名；
- 统一图表标题和坐标轴术语；
- 统一输出目录命名；
- 统一 summary 模板，至少包含：
  - config snapshot
  - seed
  - n_users / n_trials
  - primary metrics
  - runtime
- 统一 plotting helper 的输入格式。

### 交付物

- 一套输出 schema 约定；
- 一套 plotting I/O 约定；
- 一套 summary 模板。

### 验收标准

- 不同 block 的结果可被统一读取、汇总和重画；
- 图表标题不再混用旧术语。

当前状态：

- 已完成 figure-level 输出目录统一，全部写入项目 `outputs/`；
- 已完成 figure CSV 标准列补充：`figure_id`、`block`；
- 已完成统一 summary 模板与 `figure_manifest.json` 生成；
- 已新增 `scripts/collect_figure_manifests.py` 作为跨输出目录的统一聚合入口；
- 输出契约已落地为 `docs/OUTPUT_SCHEMA.md`。

---

## 11. 工作流 7：最终一致性验收

### 目标

- 在交付前做一次文档、代码、脚本、输出的总对齐检查。

### 任务

- 检查文档：
  - `README.md`
  - `docs/SPEC.md`
  - `docs/DEV.md`
  - 本计划文件
- 检查代码术语与结果字段；

当前状态：

- 已完成文档、figure script、输出 schema 和聚合入口的一致性复核；
- 已修正文档中残留的过时 canonical script 名称；
- 已新增最终审计文档 `docs/FINAL_CONSISTENCY_AUDIT.md`；
- workflow 7 当前判定为完成，剩余 legacy 项仅限内部兼容层，不再影响公共接口和论文主叙事。
- 检查脚本命名与 block 映射；
- 检查 baseline 分组；
- 检查输出目录和图表标题；
- 检查是否仍残留旧版 VBBR / verified-gap 主叙事。

### 验收清单

- `TMC26_Stackelberg.tex`、文档、脚本、输出术语一致；
- Stage II 被一致描述为 integrated SCM solver；
- Stage I 被一致描述为 boundary-price / iterative-pricing 路线；
- Block A-F 全部有落地入口；
- baseline 体系与论文当前版本一致。

---

## 12. 推荐执行顺序

建议按以下顺序推进：

1. 工作流 1：冻结术语与接口
2. 工作流 2：Stage II SCM 路线
3. 工作流 3：Stage I 路线
4. 工作流 4：baseline 与脚本清理
5. 工作流 5：A-F block 落地
6. 工作流 6：图表与输出统一
7. 工作流 7：最终验收

原因：

- 先统一语义，再动实现；
- 先保证求解逻辑，再铺实验；
- 先让数据结构稳定，再统一作图与排版。

---

## 13. 近期最优先事项

如果只做下一轮最有价值的工作，优先级建议如下：

1. 核对 `stackelberg.py` 中 Stage II / Stage I 的当前实现是否真正贴合论文；
2. 建立 script-to-block 映射表；
3. 先把 Block A、B、C 跑通；
4. 再扩展 Block D、E、F；
5. 最后处理脚本重命名和输出目录美化。

---

## 14. 完成定义

当以下条件同时满足时，认为本轮更改计划完成：

- 文档、代码、脚本的论文口径一致；
- 主实验 A-E 和稳健性实验 F 都有明确入口；
- 任一实验结果都能回溯到具体配置、脚本和输出；
- 不再需要依赖旧版命名来解释当前项目。

# Stage I Pricing Audit

## 1. 目的

本文件是工作流 3 的正式交付物。

它用于冻结 Stage I 的三件事：

- canonical 调用入口；
- `StackelbergResult` 与 `SearchStep` 的统一语义；
- block B/C/D/F 所需的结构诊断字段与 sanity-check 清单。

---

## 2. Canonical 调用链

当前 Stage I 的 paper-facing 入口冻结为：

```text
solve_stage1_pricing
  -> run_stage1_solver
    -> algorithm_paper_iterative_pricing_stage1
    -> algorithm_vbbr_brd_stage1
    -> algorithm_topk_brd_stage1
```

冻结解释：

- `solve_stage1_pricing(...)` 是对外统一入口；
- `run_stage1_solver(...)` 继续保留为 legacy/internal dispatch；
- 具体后端实现仍可随 `stage1_solver_variant` 切换，但对外结果对象必须统一；
- `algorithm_paper_iterative_pricing_stage1` 是当前论文主线；
- `algorithm_vbbr_brd_stage1` 是保留的备选 / 回退实现；
- `algorithm_topk_brd_stage1` 仍属于辅助 / legacy 路线，不应作为论文主结果入口。

---

## 3. 统一结果结构

### `StackelbergResult`

Stage I 的统一结果对象仍为 `StackelbergResult`，但其语义已固定为：

- 最终价格点上的 Stage I pricing result；
- 同时携带该最终价格点上的真实 final Stage II outcome。

必备字段如下：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `price` | 最终价格对 `(p_E, p_N)` | price trajectory / terminal point |
| `restricted_gap` | 当前论文口径下的 boundary-price-restricted NE gap | Stage I 主指标 |
| `epsilon` | legacy 兼容字段，语义应解释为 `restricted_gap` | 兼容旧脚本 |
| `gain_E`, `gain_N` | ESP / NSP unilateral gain diagnostics | best-response gap |
| `trajectory` | `SearchStep` 序列 | block B/C/D/F |
| `final_stage2_result` | 最终价格下重新求解得到的 Stage II integrated outcome | final follower outcome |
| `offloading_set` | `final_stage2_result.offloading_set` | final set |
| `inner_result` | `final_stage2_result.inner_result` | final allocation |
| `social_cost` | `final_stage2_result.social_cost` | final SCM quality |
| `esp_revenue`, `nsp_revenue` | 最终价格下的双边收益 | pricing result |
| `stage2_oracle_calls` | Stage I 过程中调用 Stage II 的次数 | runtime / complexity |
| `evaluated_candidates` | 已评估候选数 | candidate diagnostics |
| `evaluated_boundary_points` | 已评估 boundary points 数 | structural diagnostics |
| `stopping_reason` | 停止原因 | convergence diagnostics |
| `stage1_method` | 具体实现后端名 | output attribution |

冻结规则：

- `StackelbergResult.offloading_set`、`inner_result`、`social_cost` 必须与 `final_stage2_result` 自洽；
- 不能再返回“按固定集合拼出来的末态”，却不重新在最终价格点求一次 Stage II；
- 文档和 summary 中优先写 `restricted_gap`，`epsilon` 只作为兼容字段保留。

### `SearchStep`

Stage I 轨迹中的 `SearchStep` 现在至少应承载：

| 字段 | 含义 |
| --- | --- |
| `pE`, `pN` | 当前价格点 |
| `restricted_gap` | 当前 restricted gap |
| `restricted_gap_delta` | 相邻两步 gap 改变量 |
| `esp_gain`, `nsp_gain` | 双边 gain diagnostics |
| `candidate_family_size` | 当前候选族规模 |
| `esp_candidate_family_size` | ESP 方向候选族规模 |
| `nsp_candidate_family_size` | NSP 方向候选族规模 |
| `stage2_offloading_size` | 当前关联 Stage II 集合大小 |
| `stage2_social_cost` | 当前关联 Stage II social cost（若可得） |

---

## 4. 结构对象与代码映射

Stage I 的关键结构映射保持如下：

- restricted pricing space / switching-set analysis
  - `_candidate_family`
  - `_boundary_price_for_provider`
- local candidate family `N_Q(p)`
  - `_candidate_family`
  - `_vbbr_local_candidate_family`
- boundary-price-based best-response estimation
  - `algorithm_3_boundary_price_br_estimation`
  - `BoundaryPriceBROracleResult`
  - `_paper_boundary_price_br_oracle`
  - `_vbbr_verified_br_oracle`
- paper-facing iterative pricing pipeline
  - `algorithm_paper_iterative_pricing_stage1`
- backup iterative pricing pipeline
  - `_vbbr_verified_br_oracle`
  - `algorithm_vbbr_brd_stage1`
- iterative pricing result
  - `solve_stage1_pricing`
  - `StackelbergResult`

冻结解释：

- 这些结构函数依然是实现内核；
- `algorithm_paper_iterative_pricing_stage1` 应被视为当前论文口径的 canonical Stage I implementation；
- `algorithm_vbbr_brd_stage1` 可以公开保留，但文档里必须标成 backup / alternative implementation；
- 对外脚本和文档不应再把 Stage I 主结果解释成 `vbbr trajectory` 或 `verified epsilon` 本身；
- 它们应被解释成 boundary-price-restricted iterative pricing diagnostics。

---

## 5. Final Stage II Outcome 规则

工作流 3 之后，Stage I 的最终返回必须满足：

- 在最终价格 `(p_E, p_N)` 上重新求解一次 Stage II；
- `StackelbergResult.final_stage2_result` 保存这次真实 follower-side outcome；
- `StackelbergResult.offloading_set`、`inner_result`、`social_cost` 都必须来自这次 final Stage II solve。

这条规则的目的很直接：

- 避免把 Stage I 搜索过程中维护的固定集合近似，误当成最终 follower reaction；
- 让主图、summary、runtime、social cost 都落在同一个 terminal point semantics 上。

---

## 6. Block B/C/D/F 映射

当前 Stage I 结果对象已足够服务这些 block：

- Block B: geometry / boundary diagnostics
  - `candidate_family_size`
  - `esp_candidate_family_size`
  - `nsp_candidate_family_size`
  - trajectory 上的价格点
- Block C: iterative pricing convergence
  - `restricted_gap`
  - `restricted_gap_delta`
  - `esp_gain`, `nsp_gain`
  - `stopping_reason`
- Block D: runtime / scalability
  - `outer_iterations`
  - `stage2_oracle_calls`
  - `evaluated_candidates`
  - `evaluated_boundary_points`
- Block F: robustness / sensitivity
  - `price`
  - `final_stage2_result`
  - `social_cost`
  - `offloading_set`

---

## 7. Sanity-Check 清单

工作流 3 之后，Stage I 至少应满足以下检查项：

- `StackelbergResult.restricted_gap` 与 `StackelbergResult.epsilon` 语义一致。
- `StackelbergResult.final_stage2_result is not None`。
- `StackelbergResult.offloading_set == StackelbergResult.final_stage2_result.offloading_set`。
- `StackelbergResult.social_cost == StackelbergResult.final_stage2_result.social_cost`。
- `StackelbergResult.inner_result is StackelbergResult.final_stage2_result.inner_result` 或数值一致。
- `trajectory[-1]` 的价格点应与 `result.price` 一致，或者仅差一个“最终价格补点”。
- VBBR 路径下，`candidate_family_size` 应能反映 oracle 内部本地候选族规模。
- summary / trajectory 脚本优先报告 `restricted_gap`，不再只报告 `epsilon`。

---

## 8. 当前遗留项

工作流 3 之后留下的两项问题，现在状态如下：

- 许多脚本文件名仍带有 `vbbr`、`epsilon heatmap` 等 legacy 命名，但公开 figure 入口、图标题和输出字段已经统一到当前口径；
- heatmap / slice / overlay 脚本已统一到 `grid_ne_gap` / `legacy_gain_proxy` 口径，但仍保留少量 legacy alias 兼容分支，不应再作为公开命名使用。

# Stage II SCM Audit

## 1. 目的

本文件是工作流 2 的正式交付物。

它用于冻结 Stage II 的三件事：

- 论文口径下的 canonical 调用链；
- `GreedySelectionResult` 的统一结果字段；
- Block A 所需输出与 sanity-check 清单。

---

## 2. Canonical 调用链

当前 Stage II 的 paper-facing 入口冻结为：

```text
solve_stage2_scm
  -> algorithm_2_heuristic_user_selection
    -> Algorithm 2 outer user-selection with rollback
    -> _solve_stage2_inner
      -> algorithm_1_distributed_primal_dual   (canonical)
      -> _solve_fixed_set_inner_exact          (auxiliary exact/hybrid backend)
```

冻结解释：

- `solve_stage2_scm(..., inner_solver_mode="primal_dual")` 是论文主口径下的 Stage II integrated SCM solver；
- `inner_solver_mode="hybrid"` 允许 fixed-set exact inner 作为辅助加速/验证后端；
- `inner_solver_mode="exact"` 仅用于小规模验证，不作为主实验默认路线；
- `algorithm_2_heuristic_user_selection` 继续保留为低层接口，但对外不再是首选入口。

---

## 3. 统一结果结构

Stage II 的统一结果对象仍为 `GreedySelectionResult`，但现在必须被解释为 integrated SCM result，而不是单纯的 greedy set-selection result。

必备字段如下：

| 字段 | 含义 | Block A 用途 |
| --- | --- | --- |
| `offloading_set` | 最终卸载用户集合 | offloading-set size |
| `inner_result.f` | 最终计算资源分配 | feasibility / allocation |
| `inner_result.b` | 最终带宽分配 | feasibility / allocation |
| `inner_result.lambda_F` | 计算资源 shadow price | outer heuristic diagnostics |
| `inner_result.lambda_B` | 带宽 shadow price | outer heuristic diagnostics |
| `social_cost` | `V(X_DG)` | social-cost 主指标 |
| `iterations` | Stage II 外层迭代次数 | runtime diagnostics |
| `rollback_count` | rollback 触发次数 | rollback diagnostics |
| `accepted_admissions` | 外层启发式接受过的加入次数 | admission diagnostics |
| `inner_call_count` | inner solver 调用次数 | complexity diagnostics |
| `runtime_sec` | Stage II 总耗时 | runtime diagnostics |
| `stage2_method` | 统一标记，当前固定为 `algorithm_2_plus_algorithm_1` | 输出归档 |
| `inner_solver_mode` | `primal_dual / hybrid / exact` | 区分主路线与辅助验证 |
| `used_exact_inner` | 本次运行是否实际走过 exact inner | supplementary diagnostics |
| `social_cost_trace` | 每轮外层迭代对应的 social cost 序列 | social-cost trace 图 |

冻结规则：

- Block A 需要的主要统计，必须优先从 `GreedySelectionResult` 直接读取；
- 不再允许各个 Stage II 脚本各自复制一份 Algorithm 2 逻辑、各自拼接诊断字段；
- 如果未来继续扩字段，应优先加在统一结果对象上，而不是散落在脚本局部变量里。

---

## 4. Block A 脚本映射

当前 Stage II / Block A 的主脚本映射如下：

- `scripts/run_stage2_social_cost_compare.py`
  - 读取 `GreedySelectionResult.social_cost_trace`
  - 输出 social-cost trace、runtime、rollback、admission、final set size
- `scripts/run_stage2_approximation_ratio.py`
  - 读取 `GreedySelectionResult.social_cost`
  - 输出 `V(X_DG)/V(X*)`、theorem bound
  - 同时把 `rollback_count`、`inner_call_count`、`runtime_sec` 等 Block A 诊断字段写入 CSV

小规模 exact 对照入口：

- `baseline_stage2_centralized_solver`
- `_solve_centralized_minlp`
- `_solve_centralized_pyomo_scip`

冻结解释：

- exact / centralized 对照用于 Stage II quality validation；
- 这些 exact solver 不是 follower-side deployed solver，而是 Block A 的 reference baseline。

---

## 5. Fixed-Set Closed Form 的使用边界

当前仓库仍保留 `_solve_fixed_set_inner_exact`。

工作流 2 对它的边界定义如下：

- 可以作为 `hybrid` / `exact` 模式中的辅助验证后端；
- 可以服务于小规模对照、数值一致性检查、Stage I 结构分析；
- 不能在文档中替代 `Algorithm 2 + Algorithm 1`，被描述成 Stage II 的唯一主算法。

所以后续文档和图注必须明确：

- 论文主口径的 Stage II solver 是 `solve_stage2_scm(..., inner_solver_mode="primal_dual")`；
- exact/hybrid 只是在工程上提供 supplementary verification 或加速路径。

---

## 6. Sanity-Check 清单

工作流 2 之后，Stage II 至少应满足以下检查项：

- `GreedySelectionResult.stage2_method == "algorithm_2_plus_algorithm_1"`。
- `GreedySelectionResult.social_cost_trace[-1] == GreedySelectionResult.social_cost`。
- `rollback_count <= accepted_admissions`。
- `len(offloading_set) <= accepted_admissions`。
- `sum(f) <= F` 且 `sum(b) <= B`。
- 对 `offloading_set` 中每个用户，offloading cost 不高于 local cost。
- `inner_call_count <= 2 * n_users + 1`。
- `solve_stage2_scm(..., inner_solver_mode="primal_dual")` 不依赖 exact inner。
- Stage II social-cost / approximation-ratio 脚本都从统一结果对象取字段，而不是自己重写 Algorithm 2。

---

## 7. 当前遗留项

工作流 2 之后识别出的两项遗留，现在状态如下：

- Stage I 路线里的 `allow_exact_inner` legacy 开关已不再影响 paper-facing 主路径，但兼容配置项仍然存在；
- `algorithm_2_heuristic_user_selection` 仍保留 legacy 兼容行为，但公共脚本不再直接依赖它的旧默认语义。

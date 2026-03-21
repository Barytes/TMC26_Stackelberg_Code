# TMC26 开发备忘

这篇论文把 MEC 场景建成一个两阶段 Stackelberg game：

- Stage I：ESP 和 NSP 分别对计算资源与带宽定价。
- Stage II：用户在给定价格下决定是否卸载，以及买多少计算资源 `f_i` 和带宽 `b_i`。

对实验代码最关键的理解是：`Stage II` 不是“任意一个用户均衡”，而是要取 `SOE`，即 social cost 最小的那个 GNE。`Stage I` 的收益、边界价、近似 NE 都建立在这个选择之上。

## 1. 先固定的核心量

建议在 `model.py` 层统一预计算：

- `C_i^l = alpha_i * (w_i / f_i^l) + beta_i * kappa_i * w_i * (f_i^l)^2`
- `theta_i = d_i * (alpha_i + beta_i * rho_i * varpi_i) / sigma_i`
- `a_i = sqrt(alpha_i * w_i)`
- `s_i = sqrt(theta_i)`

这四组量基本贯穿全文。现有 [model.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/model.py) 已经有 `local_cost()` 和 `theta()`，继续围绕它们扩展是对的。

另外，一个非常有用的 proxy 是单用户无容量耦合时的卸载成本：

```math
C_{i,\mathrm{unc}}^e = 2 \sqrt{\alpha_i w_i p_E} + 2 \sqrt{\theta_i p_N}
```

它适合做快速诊断，但不能代替真正的 `Stage II SOE`。

## 2. Stage II 的正确实现对象

### 2.1 用户博弈的真正求解目标

论文先证明用户博弈是 exact generalized potential game，potential 就是总 social cost。所以代码里最好把 `Stage II` 封装成：

`(users, system, pE, pN) -> SOE outcome`

而不是只做“用户各自 best response 到稳定”为止。前者对应论文，后者不一定对应 SOE。

### 2.2 固定卸载集合 `X` 的内层资源分配

给定非空卸载集合 `X`，内层问题是：

```math
\min \sum_{i \in X}\left(\frac{\alpha_i w_i}{f_i} + \frac{\theta_i}{b_i} + p_E f_i + p_N b_i\right)
```

约束是：

- `sum_i f_i <= F`
- `sum_i b_i <= B`
- 对 `i in X` 满足 individual rationality：`C_i^e <= C_i^l`

论文主文给了 primal-dual 算法，但附录更适合实验代码：固定集合时有闭式解。

定义：

```math
\bar p_E(X)=\left(\frac{\sum_{i\in X} a_i}{F}\right)^2,\quad
\bar p_N(X)=\left(\frac{\sum_{i\in X} s_i}{B}\right)^2
```

```math
\tilde p_E=\max\{p_E,\bar p_E(X)\},\quad
\tilde p_N=\max\{p_N,\bar p_N(X)\}
```

则最优资源分配与 shadow price 为：

```math
f_i^*=\sqrt{\frac{\alpha_i w_i}{\tilde p_E}},\quad
b_i^*=\sqrt{\frac{\theta_i}{\tilde p_N}}
```

```math
\lambda_F^*=\tilde p_E-p_E,\quad
\lambda_B^*=\tilde p_N-p_N
```

实现含义很直接：

- `p < bar_p` 时资源拥塞，shadow price 为正。
- `p >= bar_p` 时资源不拥塞，shadow price 为零。
- 固定集合的资源分配不应默认走数值优化，闭式解更快，尤其适合价格热图、边界价覆盖图和大批量 Monte Carlo。

但这里有一个重要限制：

- 闭式解本身只解决“固定集合下的资源分配”。
- 某个集合在当前价格下是否真的是合法 offloading set，还要结合 individual rationality margin 检查，不能只看容量。

### 2.3 固定集合下最值得缓存的量

对一个固定集合 `X`，建议缓存：

- `S_E(X) = sum_{i in X} a_i`
- `S_N(X) = sum_{i in X} s_i`
- `bar_p_E(X), bar_p_N(X)`
- `f^*(X; p), b^*(X; p)`
- `lambda_F^*(X; p), lambda_B^*(X; p)`

如果后续要扫大量价格点，这些量能显著减少重复计算。

### 2.4 外层卸载集合选择

外层目标是：

```math
V(X)=\sum_{i\in X} C_i^e(f_i^*(X), b_i^*(X)) + \sum_{j \notin X} C_j^l
```

论文证明这个集合函数一般：

- 不是 submodular
- 不是 monotone
- 整个 SCM 是 NP-hard

所以论文采用 greedy-like heuristic，而不是精确全搜索。

核心局部打分：

```math
h_j(\lambda_F,\lambda_B)=
\min_{0<f_j\le F,\ 0<b_j\le B}
\left\{ C_j^e(f_j,b_j) + \lambda_F f_j + \lambda_B b_j - C_j^l \right\}
```

这个量是 `Delta V(X, {j})` 的下界，不是精确边际收益。因此：

- `h_j >= 0` 时，用户 `j` 一定不该加入。
- `h_j < 0` 时，只是“可能有利”，必须做 rollback check。

这点非常关键，不能把 `h_j < 0` 直接当成真实改善。

### 2.5 `h_j` 的可编码闭式

由定义可直接拆成两个 1D 问题。令：

- `tE = p_E + lambda_F`
- `tN = p_N + lambda_B`

则

```math
h_j = \min_{0<f\le F}\left(\frac{\alpha_j w_j}{f}+t_E f\right)
     + \min_{0<b\le B}\left(\frac{\theta_j}{b}+t_N b\right)
     - C_j^l
```

每个 1D 子问题都可写成 bounded minimum：

- 若 `sqrt(a / t) <= upper`，最小值是 `2 sqrt(a t)`
- 否则最小值是 `a / upper + t * upper`

这和现有 [stackelberg.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/stackelberg.py) 里的 `_bounded_1d_min()` 思路一致，适合保留。

### 2.6 外层算法的正确停止/回滚逻辑

论文算法不是“每次选最小 `h_j` 一直加到加不动”为止，而是：

1. 当前集合 `X^(t)` 先解一次 inner problem。
2. 对上一步刚加入的 `j_last` 做真实 `Delta V_true` 检查。
3. 若 `Delta V_true >= 0`，回滚并将其从候选池移除。
4. 再用当前 shadow prices 给所有非卸载用户计算 `h_j`。
5. 若最小 `h_j < 0`，加入该用户；否则停止。

这意味着：

- rollback 不是优化细节，而是算法定义的一部分。
- 外层最多加入 `|I|` 次用户。
- inner problem 最多被调用 `2|I| + 1` 次。

如果复现论文实验，`Stage II` 返回结果里最好保留：

- `offloading_set`
- `f, b`
- `lambda_F, lambda_B`
- `social_cost`
- `iterations`
- `rollback_count` 或等价诊断信息

## 3. Stage I 的关键结构

### 3.1 Restricted pricing space

对固定卸载集合 `X`，论文定义：

```math
P_X = \{ p : V(X; p) \le V(Y; p),\ \forall Y \subseteq I \}
```

意思是：在这个价格区域里，`X` 是 Stage II 的 SOE 卸载集合。

这带来一个非常重要的结论：

- 一旦 `X` 固定，ESP 和 NSP 的收益都能写成闭式。
- 他们在该区域内都总想继续涨自己的价格。

闭式收益：

```math
U_E(p, X) = (p_E - c_E)\frac{S_E(X)}{\sqrt{\tilde p_E}}
```

```math
U_N(p, X) = (p_N - c_N)\frac{S_N(X)}{\sqrt{\tilde p_N}}
```

这里 `S_E(X) = sum a_i`，`S_N(X) = sum s_i`。

### 3.2 最重要的 Stage I 结论

论文证明：

- 在非空卸载区域 `P^+` 内，Stage I 的任何 NE 都必须落在 switching set `Sigma` 上。
- 换句话说，不需要在整个连续价格空间盲搜。

对实验代码的直接含义是：

- 价格搜索应尽量围绕“集合切换边界”展开。
- 热图可以 dense grid 做可视化，但求解器不应靠 dense grid 找最优响应。

## 4. Boundary price 才是 Stage I 的主角

### 4.1 本地候选集合族

给定当前价格 `p` 下的 SOE 集合 `X*(p)`：

- 当前卸载用户按 realized margin `m_i(p) = C_i^l - C_i^e(f_i^*(p), b_i^*(p); p)` 升序排序。
- 当前非卸载用户按 heuristic score `h_j(p)` 升序排序。

然后定义局部候选族 `N_Q(p)`：

- 从当前集合里删前 `r` 个最脆弱的用户
- 再加前 `s` 个最有希望加入的用户
- `0 <= r <= Q, 0 <= s <= Q`

论文给出的候选族大小上界是：

```math
|N_Q(p)| \le (Q + 1)^2
```

这就是 Stage I 算法可跑的关键原因。

### 4.2 Candidate boundary price 的正确定义

对候选集合 `Y` 和固定对手价格 `p_{-k}`，定义固定集合下的 margin：

```math
m_i((p_k, p_{-k}), Y)
=
C_i^l - C_i^e(f_i^*((p_k,p_{-k}),Y), b_i^*((p_k,p_{-k}),Y); (p_k,p_{-k}))
```

boundary price `\hat p_k(Y | p_{-k})` 是：

```math
\min_{i \in Y} m_i((p_k, p_{-k}), Y) = 0
```

对应“候选集合 `Y` 第一次失去内部稳定性”的价格。

### 4.3 Appendix 给了真正该实现的闭式 boundary price

这部分对代码非常重要，因为它是分段的，不是单一公式。

先定义：

- `R_i^E(p_N, Y) = C_i^l - chi_i^N(p_N, Y)`
- `R_i^N(p_E, Y) = C_i^l - chi_i^E(p_E, Y)`

其中 `chi` 本身也是按拥塞/非拥塞分段的。

然后：

```math
\hat p_E(Y | p_N) = \min_{i \in Y} \tau_{E,i}(p_N, Y)
```

```math
\hat p_N(Y | p_E) = \min_{i \in Y} \tau_{N,i}(p_E, Y)
```

而 `tau` 也要分两段：

- 若解落在拥塞区，则是线性表达式
- 若解落在非拥塞区，则是平方表达式

也就是说，boundary price 不是永远都等于某个 `((...)/(2a_i))^2`。实现时一定要保留 congested / non-congested 两个分支。

这是论文附录中最值得落地成独立函数的一部分，建议单独封装，不要散落在 `stackelberg.py` 的搜索循环里。

### 4.4 Best-response estimation

给服务商 `k` 的一轮 best response 估计流程是：

1. 在当前价格先求一次 `Stage II SOE`
2. 由当前 SOE 构造 `N_Q(p)`
3. 对每个候选集合算一个 boundary price
4. 对每个 boundary price 再求一次 `Stage II`
5. 选择收益最大的那个价格

所以 `Stage I` 的实际计算瓶颈不是 boundary formula，而是反复调用 `Stage II`。开发时应优先做：

- `Stage II` 结果缓存
- 同一实例上的重复价格评估缓存
- 候选集合和边界价的诊断输出

## 5. 对现有代码结构最有用的映射

结合当前仓库，比较自然的职责划分是：

- [model.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/model.py)
  - 静态公式、预计算量、单用户闭式成本
- [stackelberg.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/stackelberg.py)
  - `Stage II` inner/outer solver
  - candidate family
  - boundary price
  - restricted BR / iterative pricing dynamics
- [metrics.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/metrics.py)
  - proxy 指标和真正基于 `Stage II` 的指标应分开
- [baselines.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/baselines.py)
  - GSO / GA / BO / MARL / ME / SingleSP / Coop / Rand
- [simulator.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/simulator.py)
  - 随机实例生成和试验循环
- [plotting.py](/Users/beiyanliu/Desktop/TMC26_Stackelberg_Code/src/tmc26_exp/plotting.py)
  - heatmap、trajectory、boundary overlay

一个很重要的开发原则是：

- `proxy` 可以放在 `metrics.py`
- 论文主结果必须以 `Stage II SOE` 评估为准

不能把 unconstrained offload proxy 误当成论文里的真正 social cost / revenue。

## 6. 论文实验设置的最小复现清单

默认仿真参数：

- 用户数最多到 `100`
- 用户均匀分布在半径 `500m` 圆内
- 路损：`L(r)=128.1 + 37.6 log10(r)`，`r` 用 km
- 小尺度衰落：单位均值指数分布
- `N0 = -114 dBm`
- `rho_i ~ U[0.1, 2] W`
- `d_i ~ U[1, 5] Mbits`
- `w_i ~ U[0.1, 1.0] Gcycles`
- `f_i^l ∈ {0.5, 0.8, 1.0, 1.2} GHz`
- `alpha_i ~ U[1, 2]`
- `beta_i ~ U[0.1, 0.5]`
- `kappa_i = 1e-27`
- `varpi_i = 1 / 0.35`
- `F = 20 GHz`
- `B = 50 MHz`
- `c_E = c_N = 0.1`

比较对象分两类：

- 求解 Stage I 的 baseline：`GSO`、`GA`、`BO`、`MARL`
- 比较战略设定的 baseline：`ME`、`SingleSP`、`Coop`、`Rand`

## 7. 建议优先验证的结果

如果要判断代码是否和论文逻辑一致，我会优先看这几类实验：

- `Stage II`：greedy 外层 social cost 是否始终不高于全本地 `V(emptyset)`
- `Stage II`：rollback 是否确实阻止了错误加入用户
- `Stage II`：inner closed form 与数值解是否一致
- `Stage I`：固定对手价格时，收益曲线是否呈分段严格递增
- `Stage I`：boundary overlay 是否接近真实 switching points
- `Stage I`：最终价格轨迹是否收敛到小 restricted NE gap

现有仓库里的 `price heatmap`、`boundary overlay`、`vbbr trajectory`、`approximation ratio`、`exploitability` 这些输出目录，基本都能对应到论文里的关键验证点。

## 8. 开发时最容易犯错的地方

- 把 `Stage II` 的任意 GNE 当成 SOE。
- 固定集合时只检查容量，不检查 `C_i^e <= C_i^l`。
- 把 `h_j < 0` 当成真实边际收益改善，省略 rollback。
- boundary price 只写非拥塞分支，漏掉拥塞分支。
- 用 proxy revenue / proxy offload ratio 代替真正的 Stage I 收益比较。
- 在价格搜索中做过密 grid brute force，而没有利用 switching/boundary 结构。

## 9. 我对这篇论文的简短判断

如果只看“可实现性”，论文最值钱的不是 Stackelberg 的概念框架，而是三件事：

- 固定卸载集合时的 closed-form inner solution
- `h_j + rollback` 的外层近似求解
- “NE 只会出现在 switching set 上”带来的 boundary-price 搜索框架

实验代码是否高效、是否像论文，基本取决于这三点是否被干净地实现出来。

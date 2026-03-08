# Strategic Setting Baselines 改进计划

**日期**: 2026-03-07

## Context

根据对以下资料的分析，需要改进 Market Equilibrium (ME) 和 Single SP 两个 baseline 的实现：
1. ICASSP26_Strategic 代码库 (`previous_work.py`)
2. TMC26_Stackelberg_Code 代码库 (`baselines.py`)
3. Tutuncuoglu et al. 2024 论文 (`tutuncuoglu2024joint.pdf`)

---

## 论文 SingleSP 机制深入分析

### 论文核心模型 (Tutuncuoglu et al. 2024)

论文研究的是**单一服务提供商 (Single SP)** 场景：
- 一个 SP 同时提供 edge computing 和 network bandwidth 资源
- SP 为每个用户设定**个性化价格** (personalized pricing)
- 用户根据价格决定是否 offload

### Single SP 的定价机制

**关键洞察**: SP 可以对每个用户收取不同的价格，实现**价格歧视**。

**用户行为**: 用户 i 选择 offload 当且仅当：
```
offload_cost_i + price_i ≤ local_cost_i
```

**SP 效用函数**:
```
U_SP = Σ_{i∈X} price_i - operational_cost(X)
```

**最优定价策略**:
对于每个选择 offload 的用户 i，SP 设定：
```
price_i = local_cost_i - offload_cost_i
```

这样：
1. 用户刚好 indifferent (offload cost + price = local cost)
2. SP 提取了用户的**全部剩余** (entire surplus)
3. 用户仍有 offload 的动力 (不等式取等号时仍满足)

### 与 ICASSP26 代码的一致性

ICASSP26 的 `tutuncuoglu_greedy_search` 实现完全符合论文机制：

```python
# ICASSP26: tutuncuoglu_greedy_search
def utility(off_set):
    ...
    user_payment = sum(cl[i] - ce[i] for i in off_set)  # price_i = cl_i - ce_i
    op_cost = cE * sum(f) + cN * sum(b)
    return user_payment - op_cost
```

---

## 现有实现对比分析

### Market Equilibrium (ME)

| 特性 | ICASSP26 (`epf_baseline`) | TMC26 (`baseline_market_equilibrium`) |
|------|---------------------------|---------------------------------------|
| **核心机制** | Tatonnement 价格调整 | 类似 Tatonnement |
| **Offloader 选择** | `greedy_scm` | Stage-II solver (DG) |
| **价格更新** | `p *= (1 + step * excess_ratio)` | `p *= exp(step * excess_ratio)` |
| **收敛条件** | 价格变化 < tolerance | 固定迭代次数 |

### Single SP

| 特性 | ICASSP26 (`tutuncuoglu_greedy_search`) | TMC26 (`baseline_single_sp`) |
|------|----------------------------------------|------------------------------|
| **定价机制** | **个性化定价**: `price_i = cl_i - ce_i` | **统一成本价**: `pE=cE, pN=cN` |
| **SP 效用** | `Σ(cl_i - ce_i) - op_cost` **(非零)** | `(cE-cE)*Σf + (cN-cN)*Σb` **(恒为零)** |
| **用户剩余** | 0 (SP 提取全部剩余) | cl_i - ce_i > 0 (用户获得剩余) |
| **经济学含义** | 垄断者价格歧视 | 社会最优运营 (零利润) |

**关键问题**: TMC26 的 Single SP 实现与论文机制不一致，无法作为 "centralized provider coordination" 的 benchmark。

---

## SPEC.md Strategic-Setting Conclusions 验证需求

根据 SPEC.md 第57-74行，实验需要验证：

| 结论 | 描述 | 需要的比较 |
|------|------|-----------|
| 1 | Provider-side selfish pricing matters | Full model utility > ME utility |
| 2 | Centralized provider coordination stronger | **SingleSP utility ≥ Full model utility** |
| 3 | User-side selfish offloading matters | Full model user cost < SingleSP user cost |
| 4 | Provider-side non-coordination hurts users | ME user cost < Full model user cost |
| 5 | Resource asymmetry effects | F/B ratio 变化时差异更明显 |

**问题**: 当前 TMC26 的 Single SP utility 恒为零，**无法验证结论 2**。

---

## 改进方案

### 改进 1: 重构 Single SP 定价机制 (核心)

**目标**: 实现论文中的个性化定价机制，使 Single SP 成为有效的 "centralized provider" benchmark。

**方案: 激励兼容定价 (Incentive-Compatible Pricing)**

```python
def baseline_single_sp(users, system, stack_cfg, base_cfg):
    """
    Single SP with personalized incentive-compatible pricing.

    Mechanism (from Tutuncuoglu et al. 2024):
    1. SP selects a subset of users to serve
    2. For each selected user i, sets price_i = cl_i - ce_i
    3. User i is indifferent: offload_cost + price = local_cost
    4. SP extracts all user surplus as utility

    SP Utility = Σ_{i∈X} (cl_i - ce_i) - operational_cost
    """
    data = _build_data(users)

    # Step 1: Greedy user selection
    # Add users one by one that increase SP utility
    off: set[int] = set()

    def sp_utility(off_set):
        if not off_set:
            return 0.0
        f, b = _set_allocations(data, off_set, system.cE, system.cN, system)
        if np.sum(f) > system.F or np.sum(b) > system.B:
            return -1e18  # Infeasible
        ce = _offload_costs(data, f, b, system.cE, system.cN)
        idx = list(off_set)
        # Check incentive constraint: ce[i] <= cl[i] for all i
        if any(ce[i] >= data.cl[i] for i in idx):
            return -1e18
        # SP utility = sum of prices - operational cost
        # price_i = cl_i - ce_i (extracts all surplus)
        user_payments = sum(data.cl[i] - ce[i] for i in idx)
        op_cost = system.cE * np.sum(f[idx]) + system.cN * np.sum(b[idx])
        return user_payments - op_cost

    current_utility = 0.0
    while True:
        best_gain = 0.0
        best_user = None
        for j in range(users.n):
            if j in off:
                continue
            cand = tuple(sorted(list(off | {j})))
            gain = sp_utility(cand) - current_utility
            if gain > best_gain:
                best_gain = gain
                best_user = j
        if best_user is None or best_gain <= 0:
            break
        off.add(best_user)
        current_utility = sp_utility(tuple(sorted(list(off))))

    # Step 2: Calculate final outcome with incentive-compatible pricing
    offloading_set = tuple(sorted(list(off)))
    f, b = _set_allocations(data, offloading_set, system.cE, system.cN, system)
    ce = _offload_costs(data, f, b, system.cE, system.cN)

    idx = list(offloading_set) if offloading_set else []
    loc_idx = [i for i in range(users.n) if i not in off]

    # User payments (personalized prices)
    esp_revenue = sum(data.cl[i] - ce[i] for i in idx) if idx else 0.0
    # Operational cost deduction
    op_cost = system.cE * sum(f[idx]) + system.cN * sum(b[idx]) if idx else 0.0
    sp_utility_final = esp_revenue - op_cost

    # Social cost
    social_cost = sum(ce[i] for i in idx) + sum(data.cl[i] for i in loc_idx)

    return BaselineOutcome(
        name="SingleSP",
        price=(system.cE, system.cN),  # Base prices for allocation
        offloading_set=offloading_set,
        social_cost=social_cost,
        esp_revenue=sp_utility_final,  # SP utility (non-zero!)
        nsp_revenue=0.0,  # No separate NSP
        epsilon_proxy=0.0,  # Centralized, no deviation incentive
        meta={"sp_utility": sp_utility_final, "num_offloaders": len(offloading_set)},
    )
```

### 改进 2: ME 收敛检测

**方案**: 添加收敛条件而非固定迭代次数

```python
def baseline_market_equilibrium(...):
    ...
    for t in range(base_cfg.market_max_iters):
        old_pE, old_pN = pE, pN
        # ... price update ...

        # Convergence check
        price_change = abs(pE - old_pE) + abs(pN - old_pN)
        if price_change < base_cfg.market_tol:
            break
    ...
```

### 改进 3: 配置参数

在 `BaselineConfig` 中添加：
```python
market_tol: float = 1e-4  # Market equilibrium price convergence tolerance
```

---

## 预期效果

改进后，各 baseline 的经济学含义：

| Baseline | 定价方式 | Provider Utility | User Surplus |
|----------|----------|------------------|--------------|
| **Full Model** | 竞争均衡 (Nash) | 中等 (竞争结果) | 中等 |
| **ME** | 市场出清 | 低 (接近零) | 高 |
| **SingleSP** | 个性化定价 (垄断) | **最高** (提取全部剩余) | **零** |
| **Random** | 随机 | 低 | 不确定 |

实验预期结果 (符合 SPEC.md)：

| 比较 | 预期结果 |
|------|----------|
| Full model utility vs ME | Full > ME (结论1) ✓ |
| SingleSP utility vs Full model | SingleSP > Full (结论2) ✓ |
| Full model user cost vs SingleSP | Full < SingleSP (结论3) ✓ |
| ME user cost vs Full model | ME < Full (结论4) ✓ |

---

## 实现步骤

1. **修改** `src/tmc26_exp/baselines.py` 中的 `baseline_single_sp` 函数
2. **改进** `baseline_market_equilibrium` 添加收敛检测
3. **添加** `src/tmc26_exp/config.py` 中的配置参数
4. **编写** 单元测试验证 SP utility > 0

## 关键文件

- [src/tmc26_exp/baselines.py](src/tmc26_exp/baselines.py) - 主要修改
- [src/tmc26_exp/config.py](src/tmc26_exp/config.py) - 配置参数
- [ref_papers/tutuncuoglu2024joint.pdf](ref_papers/tutuncuoglu2024joint.pdf) - 理论依据
- [docs/SPEC.md](docs/SPEC.md) - 实验规范
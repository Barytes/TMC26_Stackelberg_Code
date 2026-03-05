# Stage-II GNEP Baselines Implementation Guide

This document summarizes two Stage-II baselines for the user offloading game defined in `bare_jrnl_new_sample4.tex`:

- **VI baseline**: a shared-multiplier **variational-equilibrium-inspired** baseline.
- **PEN baseline**: a **penalized sequential best-response** baseline.

These two methods are intended as **baselines**, not exact solvers for the original binary GNEP. In particular:

- the VI baseline solves a shared-multiplier simplification of the user-side equilibrium conditions and then applies a feasibility repair;
- the PEN baseline solves a sequence of penalized user games and then applies a feasibility repair if needed.

The goal of this note is to provide an implementation-oriented specification that can be translated directly into `src/tmc26_exp/baselines.py`.

---

## 1. Original Stage-II problem

For fixed provider prices `(p_E, p_N)`, user `i` chooses

- binary offloading decision `o_i in {0, 1}`,
- computation resource demand `f_i >= 0`,
- bandwidth demand `b_i >= 0`.

The user cost is

```text
C_i(o_i, f_i, b_i) = o_i C_i^e(f_i, b_i) + (1 - o_i) C_i^l,
```

subject to the coupled capacity constraints

```text
sum_i f_i <= F,
sum_i b_i <= B.
```

From the system model,

```text
C_i^e(f_i, b_i)
= alpha_i * (w_i / f_i + d_i / (b_i * sigma_i))
+ beta_i  * (rho_i * varpi_i * d_i / (b_i * sigma_i))
+ p_E * f_i + p_N * b_i.
```

Equivalently, define

```text
a_i = alpha_i * w_i,
t_i = ((alpha_i + beta_i * rho_i * varpi_i) * d_i) / sigma_i,
```

so that the offloading branch becomes

```text
C_i^e(f_i, b_i) = a_i / f_i + t_i / b_i + p_E f_i + p_N b_i.
```

If the existing code already stores the bandwidth-side coefficient as `theta_i`, then simply use

```text
t_i = theta_i.
```

The local branch is the scalar `C_i^l`.

---

## 2. Shared data model and helper functions

It is recommended to standardize the per-user fields as follows.

```python
user = {
    "alpha": float,
    "beta": float,
    "w": float,
    "d": float,
    "sigma": float,
    "rho_tx": float,
    "varpi": float,
    "C_local": float,
}
```

Precompute the reduced coefficients:

```python
a_i = alpha_i * w_i
t_i = ((alpha_i + beta_i * rho_i * varpi_i) * d_i) / sigma_i
C_l_i = C_local
```

Recommended common helper functions:

### 2.1 Offloading cost

```python
def offloading_cost(a_i, t_i, p_E, p_N, f_i, b_i):
    return a_i / f_i + t_i / b_i + p_E * f_i + p_N * b_i
```

### 2.2 Social cost of a realized profile

```python
def realized_social_cost(users, decisions, p_E, p_N):
    # decisions[i] = {"o": 0/1, "f": float, "b": float}
    # always compute the true Stage-II cost, never the surrogate objective
```

### 2.3 Capacity statistics

```python
def total_demands(decisions):
    F_used = sum(x["f"] for x in decisions)
    B_used = sum(x["b"] for x in decisions)
    return F_used, B_used
```

### 2.4 Repair to capacity

A single repair function should be shared by both baselines.

```python
def repair_to_capacity(decisions, users, p_E, p_N, F_cap, B_cap):
    """
    Repeatedly remove the currently weakest offloader until
    both capacities are satisfied.

    "weakest" = smallest positive realized cost reduction
                C_i^l - C_i^e(f_i, b_i).
    """
```

Repair rule:

1. compute total computation and bandwidth demand;
2. if both are feasible, stop;
3. among current offloaders, compute
   ```text
   gain_i = C_i^l - C_i^e(f_i, b_i);
   ```
4. remove the offloader with the smallest `gain_i`;
5. set `(o_i, f_i, b_i) = (0, 0, 0)` and repeat.

If several users tie, break ties deterministically, for example by user index.

---

## 3. VI baseline

## 3.1 Positioning

This baseline is **not** an exact VI reformulation of the original binary GNEP.
It is a **shared-multiplier variational-equilibrium-inspired approximation**:

- users share a common multiplier `lambda_F` for the computation capacity;
- users share a common multiplier `lambda_B` for the bandwidth capacity;
- binary offloading is recovered by comparing the best offloading branch against local execution;
- a final repair step enforces feasibility.

This is the correct way to present and implement it as a baseline.

---

## 3.2 User response under common multipliers

For fixed multipliers `(lambda_F, lambda_B)`, user `i` solves the augmented offloading branch

```text
min_{f_i > 0, b_i > 0}
    a_i / f_i + t_i / b_i + (p_E + lambda_F) f_i + (p_N + lambda_B) b_i.
```

The first-order conditions are

```text
-a_i / f_i^2 + p_E + lambda_F = 0,
-t_i / b_i^2 + p_N + lambda_B = 0.
```

Hence the unique minimizer is

```text
f_i^*(lambda) = sqrt(a_i / (p_E + lambda_F)),
b_i^*(lambda) = sqrt(t_i / (p_N + lambda_B)).
```

The minimized augmented offloading value is

```text
psi_i(lambda)
= 2 * sqrt(a_i * (p_E + lambda_F))
+ 2 * sqrt(t_i * (p_N + lambda_B)).
```

Decision rule:

- if `psi_i(lambda) < C_i^l`, user `i` is treated as an offloader under the VI baseline;
- otherwise user `i` stays local.

Implementation helper:

```python
def vi_response_from_multipliers(users, p_E, p_N, lambda_F, lambda_B):
    """
    Returns
        decisions: list of {"o": 0/1, "f": float, "b": float}
        D_F: total computation demand
        D_B: total bandwidth demand
        psi: list of augmented offloading values
    """
```

Suggested logic:

```python
q_E = p_E + lambda_F
q_N = p_N + lambda_B

for each user i:
    f_star = sqrt(a_i / q_E)
    b_star = sqrt(t_i / q_N)
    psi_i = 2 * sqrt(a_i * q_E) + 2 * sqrt(t_i * q_N)

    if psi_i < C_l_i:
        o_i = 1
        f_i = f_star
        b_i = b_star
    else:
        o_i = 0
        f_i = 0.0
        b_i = 0.0
```

---

## 3.3 Multiplier update

The aggregate demands induced by a multiplier pair are

```text
D_F(lambda) = sum_i f_i^*(lambda) * 1{psi_i(lambda) < C_i^l},
D_B(lambda) = sum_i b_i^*(lambda) * 1{psi_i(lambda) < C_i^l}.
```

The shared-multiplier equilibrium condition is approximated by the complementarity system

```text
lambda_F >= 0,
D_F(lambda) <= F,
lambda_F * (D_F(lambda) - F) = 0,

lambda_B >= 0,
D_B(lambda) <= B,
lambda_B * (D_B(lambda) - B) = 0.
```

Recommended update rule: **projected dual ascent**.

```python
lambda_F = max(0.0, lambda_F + eta_t * (D_F - F_cap))
lambda_B = max(0.0, lambda_B + eta_t * (D_B - B_cap))
```

with, for example,

```python
eta_t = eta_0 / math.sqrt(t + 1)
```

Recommended hyperparameters:

- `lambda_init = 0.0`
- `eta_0 = 0.5` or `1.0`
- `max_iter = 300` to `1000`
- `tol = 1e-6` to `1e-4`

Stopping rule:

stop if all of the following hold:

```text
abs(lambda_F_new - lambda_F) <= tol,
abs(lambda_B_new - lambda_B) <= tol,
max(0, D_F - F) <= tol,
max(0, D_B - B) <= tol.
```

Because the offloading threshold introduces discontinuities through `1{psi_i < C_i^l}`, the dual ascent should be treated as a **numerical fixed-point search**, not as a theorem-backed exact VI solver.

---

## 3.4 VI algorithm summary

```python
def baseline_stage2_vi(users, p_E, p_N, F_cap, B_cap,
                       eta_0=0.5, max_iter=500, tol=1e-6,
                       do_repair=True):
    """
    Shared-multiplier VE-inspired baseline.
    """
```

Recommended algorithm:

1. initialize `lambda_F = 0`, `lambda_B = 0`;
2. repeat until convergence or `max_iter`:
   - call `vi_response_from_multipliers(...)`;
   - compute `D_F`, `D_B`;
   - update multipliers by projected dual ascent;
3. build the realized binary profile from the final multipliers;
4. if `do_repair` is true, call `repair_to_capacity(...)`;
5. compute the true Stage-II social cost and return the profile.

Recommended return fields:

```python
{
    "name": "VI",
    "decisions": decisions,
    "social_cost": float,
    "F_used": float,
    "B_used": float,
    "lambda_F": float,
    "lambda_B": float,
    "iterations": int,
    "repaired": bool,
}
```

---

## 3.5 Numerical notes for VI

### Use strict positivity guards

Avoid division by zero by ensuring

```python
q_E = max(p_E + lambda_F, eps)
q_N = max(p_N + lambda_B, eps)
```

with `eps = 1e-12`.

### Tie handling

When `psi_i(lambda)` is extremely close to `C_i^l`, use a small margin to avoid chattering:

```python
if psi_i <= C_l_i - tie_tol:
    offload
else:
    local
```

with `tie_tol = 1e-10` or `1e-8`.

### Optional damping

If multipliers oscillate, use damped updates:

```python
lambda_F = (1 - gamma) * lambda_F + gamma * lambda_F_new
lambda_B = (1 - gamma) * lambda_B + gamma * lambda_B_new
```

with `gamma in (0, 1]`.

---

## 4. Penalty baseline

## 4.1 Positioning

This baseline is a **penalized sequential best-response method**.
It does **not** exactly solve the original GNEP for any finite penalty coefficient `rho`.
Instead, it solves a sequence of penalized user games and uses increasing penalties to push the profile toward feasibility.

This is the correct interpretation and implementation target.

---

## 4.2 Penalized user objective

Fix the other users' current total demands seen by user `i`:

```text
R_F^{-i} = sum_{j != i} f_j,
R_B^{-i} = sum_{j != i} b_j.
```

Define the penalized user objective

```text
J_i^rho(o_i, f_i, b_i)
= o_i C_i^e(f_i, b_i) + (1 - o_i) C_i^l
+ (rho / 2) * [R_F^{-i} + f_i - F]_+^2
+ (rho / 2) * [R_B^{-i} + b_i - B]_+^2.
```

The local branch is

```text
J_i^rho(local)
= C_i^l
+ (rho / 2) * [R_F^{-i} - F]_+^2
+ (rho / 2) * [R_B^{-i} - B]_+^2.
```

The offloading branch is

```text
min_{f_i > 0, b_i > 0}
    a_i / f_i + t_i / b_i + p_E f_i + p_N b_i
  + (rho / 2) * [R_F^{-i} + f_i - F]_+^2
  + (rho / 2) * [R_B^{-i} + b_i - B]_+^2.
```

Because this objective is separable in `f_i` and `b_i`, the offloading branch reduces to two one-dimensional convex subproblems:

```text
min_{f_i > 0}
    a_i / f_i + p_E f_i + (rho / 2) * [R_F^{-i} + f_i - F]_+^2,

min_{b_i > 0}
    t_i / b_i + p_N b_i + (rho / 2) * [R_B^{-i} + b_i - B]_+^2.
```

---

## 4.3 One-dimensional penalized subproblem solver

Recommended helper:

```python
def penalty_axis_best_response(a, p, R_minus, cap, rho,
                               upper_bound=None, eps=1e-12):
    """
    Solves
        min_{x > 0} a/x + p*x + (rho/2) * [R_minus + x - cap]_+^2
    using bounded scalar minimization.
    """
```

### Objective structure

For an axis variable `x`, define

```text
phi(x) = a / x + p * x + (rho / 2) * [R_minus + x - cap]_+^2,
```

for `x > 0`.

This is convex on `(0, +inf)` because:

- `a / x` is convex for `x > 0`;
- `p * x` is linear;
- `[R_minus + x - cap]_+^2` is convex.

### Practical solver choice

Use one of the following:

1. `scipy.optimize.minimize_scalar(method="bounded")`, or
2. a custom golden-section search over a bounded interval.

### Choosing the upper search bound

A safe practical default is

```python
x_free = math.sqrt(a / max(p, eps))
upper = max(2.0 * x_free, cap + 1.0, 1.0)
```

If the unconstrained optimum already lies far below the capacity threshold, the penalty term is inactive and the minimizer stays near `x_free`.
If the penalty is active, the optimizer will not grow arbitrarily because both the linear term and the quadratic penalty increase with `x`.

---

## 4.4 User best response under penalty

Recommended helper:

```python
def penalty_user_best_response(user_i, current_decisions, i,
                               p_E, p_N, F_cap, B_cap, rho):
    """
    Returns the best penalized response of user i:
        {"o": 0/1, "f": float, "b": float}
    """
```

Algorithm:

1. compute `R_F_minus_i` and `R_B_minus_i` from all users except `i`;
2. compute the local penalized value;
3. solve the 1-D `f_i` subproblem;
4. solve the 1-D `b_i` subproblem;
5. evaluate the offloading penalized value;
6. choose the branch with the smaller penalized objective.

Decision rule:

```python
if J_off <= J_local - tie_tol:
    return {"o": 1, "f": f_star, "b": b_star}
else:
    return {"o": 0, "f": 0.0, "b": 0.0}
```

---

## 4.5 Sequential penalty dynamics

Recommended main entry point:

```python
def baseline_stage2_penalty(users, p_E, p_N, F_cap, B_cap,
                            rho0=1.0, rho_multiplier=5.0,
                            inner_max_iter=50, outer_max_iter=8,
                            tol=1e-6, random_order=True,
                            seed=None, do_repair=True):
    """
    Penalized sequential best-response baseline.
    """
```

Recommended algorithm:

1. initialize every user as local:
   ```python
   decisions[i] = {"o": 0, "f": 0.0, "b": 0.0}
   ```
2. set `rho = rho0`;
3. for each outer iteration:
   - repeat up to `inner_max_iter` times:
     - choose user order, random or deterministic;
     - for each user, replace its action by `penalty_user_best_response(...)`;
     - if no user's action changes by more than `tol`, stop the inner loop;
   - compute total capacity violations;
   - if both violations are below tolerance, stop the outer loop;
   - else set `rho = rho * rho_multiplier` and continue;
4. if `do_repair` is true, call `repair_to_capacity(...)`;
5. compute the true Stage-II social cost and return the profile.

Recommended return fields:

```python
{
    "name": "PEN",
    "decisions": decisions,
    "social_cost": float,
    "F_used": float,
    "B_used": float,
    "rho_final": float,
    "inner_iterations": int,
    "outer_iterations": int,
    "repaired": bool,
}
```

---

## 4.6 Numerical notes for PEN

### Random-order updates are preferable

Sequential best-response with a random user order per sweep is usually more stable than a fixed order.
Use a controllable random seed for reproducibility.

### Penalty growth can become stiff

Large `rho` may cause numerical stiffness and make the 1-D subproblems more sharply curved near the capacity boundary.
This is expected and should be documented in experiments.

### Inner convergence test

A practical stopping rule for the inner loop is

```text
max_i |f_i^{new} - f_i^{old}| <= tol
and
max_i |b_i^{new} - b_i^{old}| <= tol
and
all binary decisions unchanged.
```

### Optional warm start

For each larger `rho`, use the previous profile as initialization.
This usually improves stability.

---

## 5. What must be reported as the output objective

Both baselines may use surrogate objectives internally:

- the VI baseline uses augmented offloading values with shared multipliers;
- the PEN baseline uses penalized user objectives.

However, the final reported objective must always be the **true Stage-II social cost**:

```text
SC = sum_i C_i(o_i, f_i, b_i).
```

Never report the augmented VI objective or the penalized surrogate objective as the final social cost.

---

## 6. Minimal unit tests

The following tests should be added before integrating the baselines into experiments.

## 6.1 Cost primitive tests

- verify that `offloading_cost(a, t, p_E, p_N, f, b)` matches the expanded model formula;
- verify that `a_i > 0`, `t_i > 0`, `C_i^l > 0` under normal parameter settings.

## 6.2 VI response tests

For fixed positive multipliers, check that

```text
f_i^* = sqrt(a_i / (p_E + lambda_F)),
b_i^* = sqrt(t_i / (p_N + lambda_B))
```

indeed minimize the augmented offloading branch, for example by comparing against numerical scalar minimization.

## 6.3 Penalty axis solver tests

For random `(a, p, R_minus, cap, rho)`, compare the returned `x_star` against a dense-grid reference or numerical derivative sign changes.

## 6.4 Repair tests

Create an intentionally infeasible profile and verify that:

- after repair, `sum_i f_i <= F` and `sum_i b_i <= B`;
- only current offloaders are removed;
- users removed by repair are exactly those with the smallest realized gains under deterministic tie-breaking.

## 6.5 End-to-end sanity tests

For a very small instance, for example `I = 2` or `3`:

- run `baseline_stage2_vi` and `baseline_stage2_penalty`;
- verify the returned profile is feasible after repair;
- verify the returned `social_cost` equals the direct sum of realized user costs.

---

## 7. Suggested code organization

Recommended functions inside `src/tmc26_exp/baselines.py`:

```python
def preprocess_stage2_coefficients(users):
    ...

def offloading_cost(a_i, t_i, p_E, p_N, f_i, b_i):
    ...

def realized_social_cost(users, decisions, p_E, p_N):
    ...

def total_demands(decisions):
    ...

def repair_to_capacity(decisions, users, p_E, p_N, F_cap, B_cap):
    ...

def vi_response_from_multipliers(users, p_E, p_N, lambda_F, lambda_B):
    ...

def baseline_stage2_vi(...):
    ...

def penalty_axis_best_response(a, p, R_minus, cap, rho, ...):
    ...

def penalty_user_best_response(user_i, current_decisions, i, p_E, p_N, F_cap, B_cap, rho):
    ...

def baseline_stage2_penalty(...):
    ...
```

If desired, `repair_to_capacity(...)` and cost utilities can be moved to a shared utility module.

---

## 8. Final implementation notes

1. **Document them honestly as baselines.**
   - VI: shared-multiplier VE-inspired baseline with repair.
   - PEN: penalized sequential best-response baseline with repair.

2. **Always return a feasible profile.**
   The repair step is part of the baseline definition.

3. **Keep the final evaluation consistent.**
   Experimental plots should compare the true realized Stage-II social cost, not surrogate objectives.

4. **Use deterministic tie-breaking where possible.**
   This improves reproducibility.

5. **Log internal solver metadata.**
   It is useful to store multiplier values, penalty values, iteration counts, and whether repair was triggered.


# Stage-II GNEP Baselines: VI and Penalty

This note derives the two added Stage-II baselines directly from the Stage-II user offloading game in `TMC26_Stackelberg.tex`.

## 1. Starting point

For fixed provider prices `(p_E, p_N)`, user `i` solves

`min_{o_i, f_i, b_i} o_i C_i^e(f_i, b_i) + (1 - o_i) C_i^l`

subject to

- `o_i in {0, 1}`
- `f_i >= 0`, `b_i >= 0`
- `f_i + sum_{j != i} f_j <= F`
- `b_i + sum_{j != i} b_j <= B`

Using the notation already used in the code,

- `a_i = alpha_i * w_i`
- `t_i = theta_i`

the offloading branch is

`C_i^e(f_i, b_i) = a_i / f_i + t_i / b_i + p_E f_i + p_N b_i`.

The local branch is the scalar `C_i^l`.

The binary variable `o_i` only chooses between:

1. local execution: `(o_i, f_i, b_i) = (0, 0, 0)`
2. offloading: `(o_i, f_i, b_i) = (1, f_i, b_i)` with `f_i > 0`, `b_i > 0`

This makes both baselines below naturally split into "evaluate local branch" vs "solve continuous offloading branch".

## 2. VI baseline (shared-multiplier variational equilibrium)

### 2.1 Simplification from GNEP/QVI to VI

The coupling comes from the shared capacities `F` and `B`. In the original GNEP, each user's feasible set depends on the others, which can be written as a QVI.

To obtain a tractable baseline, we impose the standard variational-equilibrium simplification:

- all users share the same multiplier `lambda_F` for the computation capacity
- all users share the same multiplier `lambda_B` for the bandwidth capacity

This turns the user-side equilibrium condition into a standard VI / KKT system with common prices for the shared constraints.

### 2.2 User-level best response under common multipliers

For fixed `(lambda_F, lambda_B)`, the offloading branch of user `i` becomes

`min_{f_i > 0, b_i > 0} C_i^e(f_i, b_i) + lambda_F f_i + lambda_B b_i`.

Because the objective is separable, the first-order conditions are

- `-a_i / f_i^2 + p_E + lambda_F = 0`
- `-t_i / b_i^2 + p_N + lambda_B = 0`

Hence the unique minimizer is

- `f_i^*(lambda) = sqrt(a_i / (p_E + lambda_F))`
- `b_i^*(lambda) = sqrt(t_i / (p_N + lambda_B))`

and the minimized augmented offloading value is

`psi_i(lambda) = 2 sqrt(a_i (p_E + lambda_F)) + 2 sqrt(t_i (p_N + lambda_B))`.

The local branch has value `C_i^l`, so user `i` offloads iff

`psi_i(lambda) < C_i^l`.

### 2.3 Aggregate VI condition

The induced aggregate demands are

- `D_F(lambda) = sum_i f_i^*(lambda) * 1{psi_i(lambda) < C_i^l}`
- `D_B(lambda) = sum_i b_i^*(lambda) * 1{psi_i(lambda) < C_i^l}`

The variational-equilibrium multipliers satisfy the complementarity system

- `lambda_F >= 0`, `D_F(lambda) <= F`, `lambda_F (D_F(lambda) - F) = 0`
- `lambda_B >= 0`, `D_B(lambda) <= B`, `lambda_B (D_B(lambda) - B) = 0`

The implementation solves this with projected dual ascent:

- `lambda_F^{t+1} = [lambda_F^t + eta_t (D_F(lambda^t) - F)]_+`
- `lambda_B^{t+1} = [lambda_B^t + eta_t (D_B(lambda^t) - B)]_+`

with `eta_t = eta_0 / sqrt(t + 1)`.

### 2.4 Discrete repair

Because offloading remains binary, the threshold `psi_i(lambda) < C_i^l` can create a small residual capacity overshoot even near a fixed point.

To convert the VI solution into a feasible binary profile, the implementation applies a minimal repair:

- compute the actual cost reduction `C_i^l - C_i^e`
- if capacity is still violated, remove the offloading user with the smallest positive reduction
- repeat until both capacities are satisfied

This keeps the baseline faithful to the VI idea while always returning a feasible Stage-II strategy profile.

## 3. Penalty baseline (penalized NEP)

### 3.1 Penalized user objective

Instead of enforcing shared constraints through the feasible set, we move them into the objective. Given the other users' current demands

- `R_F^{-i} = sum_{j != i} f_j`
- `R_B^{-i} = sum_{j != i} b_j`

user `i` solves the penalized problem

`min_{o_i, f_i, b_i} o_i C_i^e(f_i, b_i) + (1 - o_i) C_i^l`

`+ (rho / 2) ([R_F^{-i} + f_i - F]_+^2 + [R_B^{-i} + b_i - B]_+^2)`

where `[x]_+ = max(x, 0)`.

This removes the shared feasible-set coupling and turns the GNEP into a sequence of standard NEPs parameterized by `rho`.

### 3.2 Local branch vs offloading branch

For the local branch, user `i` picks `(o_i, f_i, b_i) = (0, 0, 0)`, so its penalized value is

`C_i^l + (rho / 2) ([R_F^{-i} - F]_+^2 + [R_B^{-i} - B]_+^2)`.

For the offloading branch, the problem is

`min_{f_i > 0, b_i > 0} a_i / f_i + t_i / b_i + p_E f_i + p_N b_i`

`+ (rho / 2) ([R_F^{-i} + f_i - F]_+^2 + [R_B^{-i} + b_i - B]_+^2)`.

This objective is still separable in `f_i` and `b_i`, so the offloading branch decomposes into two one-dimensional convex minimizations:

- `min_{f_i > 0} a_i / f_i + p_E f_i + (rho / 2) [R_F^{-i} + f_i - F]_+^2`
- `min_{b_i > 0} t_i / b_i + p_N b_i + (rho / 2) [R_B^{-i} + b_i - B]_+^2`

The code solves each 1-D problem with bounded scalar minimization.

### 3.3 Sequential penalty dynamics

The implemented baseline uses:

1. initialize all users at local execution
2. for fixed `rho`, run random-order sequential best-response updates
3. if total demand still violates capacity, increase `rho` geometrically
4. repeat until capacity is nearly satisfied or the outer iteration budget is reached

This is intentionally a baseline:

- simple
- standard
- easy to interpret

but it can become numerically stiff when `rho` grows, which is exactly the well-known drawback of penalty methods.

### 3.4 Final repair

As with the VI baseline, a final capacity repair removes the weakest offloaders if the last penalized equilibrium still slightly exceeds capacity.

## 4. Code mapping

The implementation lives in `src/tmc26_exp/baselines.py`.

- `baseline_stage2_vi(...)`: shared-multiplier VI baseline
- `baseline_stage2_penalty(...)`: penalty-method baseline
- `_vi_response_from_multipliers(...)`: closed-form VI user response
- `_penalty_axis_best_response(...)`: 1-D penalized subproblem solver
- `_repair_to_capacity(...)`: converts approximate equilibrium outputs into a feasible Stage-II profile

The returned social cost is always computed from the actual Stage-II cost `sum_i C_i`, not from the augmented VI objective or the penalized surrogate objective.

# SPEC.md

## Objective

Design the experiment section so that it directly validates the paper's theoretical contributions, with **Stage I as the main focus** and **Stage II only providing the necessary oracle-side evidence**.

The experimental narrative should not be a generic benchmark suite. It should answer the following question:

> Does the proposed two-stage framework work for the reasons claimed by the theory?

In particular:

- **Stage II** should only show that Algorithms 1 and 2 form a reliable and scalable **SOE approximation oracle**.
- **Stage I** should be the main experimental focus, validating the structural insights behind the restricted pricing reformulation, the usefulness of the gain approximation, and the efficiency/equilibrium quality of the guided search.
- **Strategic-setting experiments** should verify the economic conclusions of the model rather than just reporting raw numbers.
- **Ablation experiments** should isolate the structural ingredients that make Stage I work.

---

## Core Claims to Validate

### Stage II: follower-side oracle
The experiments should validate only the following claim:

- **Algorithms 1 and 2 are a sufficiently good SOE approximation oracle**: their social cost is close to the centralized optimum, and their runtime / communication cost scales well with the number of users.

In addition, the experiment design should explicitly confirm that the Stage-II output is not merely a centrally imposed solution, but is also **strategically stable from the users' perspective**.

---

### Stage I: main contribution
The experiments should validate the following claims:

1. **Boundary structure is correct**  
   For a fixed offloading set, good RNE solutions lie on the upper boundary of the restricted pricing space, and the Stackelberg equilibrium price should also lie on the boundary of some $\partial \mathcal{P}_{\mathcal{X}^*}$.

2. **Algorithm 3 provides a useful gain approximation**  
   The approximated best-response gain $\widehat{\mathscr{G}}_k$ must be empirically close to the exact best-response gain $\mathscr{G}_k$ on small instances, otherwise Algorithms 4 and 5 are not convincing.

3. **The candidate family $\mathcal{N}(\boldsymbol p)$ is effective**  
   The construction of $\mathcal{N}(\boldsymbol p)$ should capture the true deviation targets frequently enough to justify Proposition 4 and the use of Algorithm 3.

4. **Algorithm 4 finds good RNE points on the boundary**  
   Increasing the sampling density $L$ should reduce the deviation gap, while increasing computation cost, revealing a clear accuracy--complexity tradeoff.

5. **Algorithm 5 finds low-gap $\epsilon$-NE efficiently**  
   The final output should have small deviation gap, require relatively few iterations, and incur much fewer expensive Stage-II calls than generic black-box baselines.

6. **The deviation-target guidance in Algorithm 5 is essential**  
   The exact deviation targets $\mathcal{Y}_E^*(\boldsymbol p)$ and $\mathcal{Y}_N^*(\boldsymbol p)$ should improve the search beyond random neighborhood exploration or unguided local search.

7. **Algorithm 5 is better suited to this problem than generic baselines**  
   PBRD, BO, DRL, and GSO may all be applicable, but they do not exploit the restricted pricing space, the boundary structure, or the deviation-target structure. Therefore, under comparable budgets, they should be slower, use more Stage-II calls, and/or return larger final deviation gaps.

---

### Strategic-setting conclusions
The experiments should validate the following conclusions:

1. **Provider-side selfish pricing matters**  
   Selfish pricing competition between ESP and NSP should yield significant utility gains over non-strategic provider models such as Market Equilibrium.

2. **Centralized provider coordination remains stronger on provider utility**  
   The full strategic model may still produce lower provider utility than a centralized pricing/control benchmark such as SingleSP.

3. **User-side selfish offloading matters**  
   Competitive user-side offloading should reduce user social cost compared with non-strategic user-side baselines such as SingleSP and Random Offloading.

4. **Provider-side non-coordination hurts users**  
   User social cost should still be worse than in Market Equilibrium because providers do not coordinate.

5. **Complementary-resource effects become more visible under resource asymmetry**  
   The differences among strategic settings should be more pronounced when the ratio $F/B$ varies.

---

## Main-Text Figures

### Figure 1 — Stage II convergence to the optimal social cost
**Purpose**  
Validate that Algorithms 1 and 2 converge to a near-optimal SOE objective value.

**Plot**
- x-axis: iteration index
- y-axis: total social cost
- curves:
  - Algorithm 1 + 2
  - optional: Algorithm 1 only for a fixed offloading set
- reference line:
  - centralized solver optimal social cost (dashed horizontal line)

**Baselines**
- Centralized Solver (CS), for small-scale instances only

**Expected takeaway**
- The follower-side oracle converges stably.
- The final social cost is close to the centralized optimum.

---

### Figure 2 — Stage II approximation ratio vs. system size (Theorem 2 validation)
**Purpose**  
Validate the approximation quality of Algorithm 2 and visualize the theorem-based guarantee.

**Plot**
- x-axis: number of users $|\mathcal{I}|$ (small-scale regime only)
- y-axis: approximation ratio
  $$
  \frac{V(\mathcal{X}^{\mathrm{DG}})}{V(\mathcal{X}^*)}
  $$
- curves:
  - empirical ratio achieved by Algorithm 2
  - theorem upper bound
  $$
  \frac{V(\emptyset) - \Delta \hat{C}_{i^*}}{V(\emptyset) - |\mathcal{X}^*|\Delta \hat{C}_{i^*}}
  $$

**Baselines**
- Centralized Solver to compute $V(\mathcal{X}^*)$

**Expected takeaway**
- The empirical approximation ratio stays well below the theorem upper bound.
- The theorem bound is valid and visually interpretable.

---

### Figure 3 — Stage II scalability: runtime and communication
**Purpose**  
Show that Algorithms 1 and 2 are scalable enough to serve as the Stage-I oracle.

**Plot**
- x-axis: number of users $|\mathcal{I}|$
- y-axis (left): wall-clock runtime
- y-axis (right): communication rounds / total messages
- curves:
  - Algorithm 1 + 2
  - CS
  - optional: VI / PEN if implemented for comparison

**Baselines**
- CS
- optional: VI / PEN

**Expected takeaway**
- The proposed Stage-II oracle scales much better than centralized optimization.
- Runtime and communication remain manageable as the system grows.

---

### Figure 4 — Stage II strategic stability check
**Purpose**  
Show that the Stage-II output is not merely a centrally imposed solution, but is also strategically stable for users.

**Plot**
- x-axis: number of users $|\mathcal{I}|$
- y-axis: average exploitability
  $$
  \frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}
  \bigl(C_i(\text{current}) - C_i(\text{BR}_i)\bigr)_+
  $$
- curves:
  - Algorithm 1 + 2
  - centralized optimum (if available)
  - random feasible offloading/allocation

**Baselines**
- CS
- Random feasible solution

**Expected takeaway**
- The output of Algorithms 1 and 2 is not only low-cost but also close to user-side equilibrium.
- This addresses the concern that Stage II might simply be enforcing a centrally chosen outcome.

**Placement**
- Prefer Appendix if the main text is too crowded.

---

### Figure 5 — Stage I deviation gap vs. iteration
**Purpose**  
This is the main convergence figure for Stage I. It validates that Algorithm 5 progressively reduces the approximation gap to a true NE.

**Plot**
- x-axis: outer iteration index
- y-axis:
  $$
  \epsilon^{(t)} = \max\{\widehat{\mathscr{G}}_E(\boldsymbol p^{(t)}),\widehat{\mathscr{G}}_N(\boldsymbol p^{(t)})\}
  $$
- curves:
  - Algorithm 5
  - PBRD
  - BO
  - DRL

**Baselines**
- PBRD
- BO
- DRL

**Expected takeaway**
- Algorithm 5 converges quickly to a low-gap price.
- Generic black-box methods reduce the gap more slowly and/or stall at worse final values.

---

### Figure 6 — Boundary visualization and price trajectory
**Purpose**  
Visually validate the boundary structure behind Theorem 3 and Corollary 1.

**Plot**
- x-axis: $p_E$
- y-axis: $p_N$
- overlay:
  - heatmap or contour of $\epsilon(\boldsymbol p)$ on a small instance
  - the upper boundary of the restricted pricing space
  - trajectories of Algorithm 5 and selected baselines

**Baselines**
- Algorithm 5
- PBRD
- BO
- DRL

**Expected takeaway**
- Good pricing solutions lie on or near the boundary.
- Algorithm 5 searches in a theory-guided region rather than wandering in the full 2D price space.

---

### Figure 7 — Algorithm 3 gain approximation accuracy
**Purpose**  
Directly validate that $\widehat{\mathscr{G}}_k$ is a meaningful approximation to $\mathscr{G}_k$.

**Plot option A**
- x-axis: exact gain $\mathscr{G}_k$
- y-axis: approximated gain $\widehat{\mathscr{G}}_k$
- plot type: scatter plot
- ideal reference: diagonal line $y=x$

**Plot option B**
- x-axis: number of users $|\mathcal I|$
- y-axis: relative error
  $$
  \frac{|\widehat{\mathscr{G}}_k - \mathscr{G}_k|}{\max\{\mathscr{G}_k,10^{-8}\}}
  $$
- plot type: line or box plot

**Baselines**
- exact exhaustive deviation search on small instances

**Expected takeaway**
- Algorithm 3 provides accurate gain estimates.
- This empirically supports its use inside Algorithms 4 and 5.

---

### Figure 8 — Effectiveness of the candidate family $\mathcal{N}(\boldsymbol p)$
**Purpose**  
Explicitly validate the construction of $\mathcal{N}(\boldsymbol p)$.

**Plot**
- x-axis: number of users $|\mathcal I|$ or instance index
- y-axis: one of the following:
  - hit rate:
    $$
    \Pr\bigl(\mathcal{Y}_k^* \in \mathcal{N}(\boldsymbol p)\bigr)
    $$
  - candidate recall:
    proportion of exact deviation targets covered by $\mathcal{N}(\boldsymbol p)$
  - average rank of the exact $\mathcal{Y}_k^*$ within candidates sorted by approximated gain

**Baselines**
- optional comparison to simpler candidate constructions:
  - random candidate family of the same size
  - one-hop neighborhood only
  - top-$m$ users by local score only

**Expected takeaway**
- $\mathcal{N}(\boldsymbol p)$ is not arbitrary.
- It captures the true deviation target frequently enough to justify Proposition 4 and the gain-approximation mechanism.

---

### Figure 9 — Stage I runtime and Stage-II calls vs. system size
**Purpose**  
Show that Algorithm 5 is computationally preferable to generic baselines.

**Plot**
- x-axis: number of users $|\mathcal I|$
- y-axis (left): wall-clock runtime
- y-axis (right): number of Stage-II oracle calls
- curves:
  - Algorithm 5
  - PBRD
  - BO
  - DRL
  - GSO on small instances only

**Baselines**
- PBRD
- BO
- DRL
- GSO (small-scale only)

**Expected takeaway**
- Algorithm 5 needs substantially fewer expensive Stage-II calls.
- It scales better in practice than generic black-box search.

---

## Strategic-Setting Figures

### Figure 10 — User social cost vs. $|\mathcal I|$
**Purpose**  
Validate the conclusion that user-side selfish offloading reduces user cost compared with non-strategic user-side models.

**Plot**
- x-axis: number of users $|\mathcal I|$
- y-axis: total user social cost
- curves:
  - Full model (Algorithm 5 + Algorithms 1/2)
  - Market Equilibrium (ME)
  - SingleSP
  - Random Offloading

**Expected takeaway**
- The full model beats SingleSP and Random Offloading on user cost.
- Market Equilibrium can still be more user-friendly because providers are non-strategic.

---

### Figure 11 — Joint provider revenue vs. $|\mathcal I|$
**Purpose**  
Validate the conclusion that provider-side selfish pricing improves provider utility over non-strategic models, but may remain below centralized coordination.

**Plot**
- x-axis: number of users $|\mathcal I|$
- y-axis:
  $$
  U_E + U_N
  $$
- curves:
  - Full model
  - ME
  - SingleSP
  - Random Offloading

**Expected takeaway**
- Strategic provider pricing yields large gains over ME.
- Centralized provider control (SingleSP) can still dominate in total provider utility.

---

### Figure 12 — Pareto-style tradeoff among strategic settings
**Purpose**  
Summarize the economic tradeoff of the different strategic settings.

**Plot**
- x-axis: total user social cost
- y-axis:
  $$
  U_E + U_N
  $$
- each strategic setting is shown as:
  - one point (fixed configuration), or
  - a curve (if sweeping $|\mathcal I|$ or another system parameter)

**Baselines**
- Full model
- ME
- SingleSP
- Random Offloading

**Expected takeaway**
- The full strategic model provides a distinct tradeoff between user efficiency and provider profitability.
- This figure helps explain why neither purely non-strategic nor fully centralized models are appropriate substitutes.

---

### Figure 13 — Revenue and utilization vs. $F/B$ ratio
**Purpose**  
Reveal the role of complementary computation and bandwidth resources under resource asymmetry.

**Plot**
- x-axis: resource ratio $F/B$
- y-axis (left):
  $$
  U_E + U_N
  $$
- y-axis (right):
  - computation utilization $\sum_i f_i / F$
  - bandwidth utilization $\sum_i b_i / B$

**Baselines**
- Full model
- ME
- SingleSP
- Random Offloading

**Expected takeaway**
- Resource asymmetry amplifies the difference among strategic settings.
- The complementary-resource nature of the model becomes most visible when $F/B$ varies.

---

## Ablation Figures

### Figure 14 — Sampling density $L$: accuracy--complexity tradeoff
**Purpose**  
Validate the role of Algorithm 4's boundary sampling density.

**Plot**
- subplot (a):
  - x-axis: $L$
  - y-axis: final deviation gap $\epsilon$
- subplot (b):
  - x-axis: $L$
  - y-axis: runtime and/or Stage-II calls

**Expected takeaway**
- Increasing $L$ initially reduces the final gap significantly.
- The improvement then saturates, while cost grows roughly linearly.
- This identifies a practical operating range for $L$.

---

### Figure 15 — Guided-search ablation for Algorithm 5
**Purpose**  
Validate that the exact deviation targets $\mathcal{Y}_E^*(\boldsymbol p)$ and $\mathcal{Y}_N^*(\boldsymbol p)$ are genuinely useful.

**Variants**
- Full Algorithm 5
- No deviation-target prioritization; random neighbor from $\mathcal N(\boldsymbol p)$
- Neighborhood-only local search
- Deviation-target-only search (without neighborhood fallback)
- optional: random restart local search

**Plot**
- x-axis: algorithm variant
- y-axis:
  - final deviation gap $\epsilon$
  - runtime
  - Stage-II calls

**Expected takeaway**
- The deviation-target guidance is not cosmetic.
- Full Algorithm 5 should outperform random or unguided neighborhood exploration.
- The fallback neighborhood search should also contribute beyond using only $\mathcal{Y}_E^*$ and $\mathcal{Y}_N^*$.

---

## Appendix Figures

### Appendix Figure A1 — Final deviation gap vs. $|\mathcal I|$
**Purpose**  
Provide a direct final-quality comparison across methods, especially when GSO is only feasible on small instances.

**Plot**
- x-axis: number of users $|\mathcal I|$
- y-axis: final deviation gap $\epsilon$
- curves:
  - Algorithm 5
  - PBRD
  - BO
  - DRL
  - GSO for small-scale instances

**Expected takeaway**
- Algorithm 5 consistently returns lower-gap solutions, especially as the problem grows.

---

### Appendix Figure A2 — Stage II exploitability vs. $|\mathcal I|$
**Purpose**  
Provide additional evidence that the Stage-II solution is a strategically stable user-side outcome.

**Plot**
- x-axis: number of users $|\mathcal I|$
- y-axis: exploitability
- curves:
  - Algorithm 1 + 2
  - CS
  - random feasible solution

**Expected takeaway**
- The Stage-II output is close to a user equilibrium, not merely a centrally enforced allocation.

---

## Implementation Notes

### Small-scale regime
Use small instances for:
- Centralized Solver comparisons
- exact exhaustive computation of $\mathscr{G}_k$
- hit-rate evaluation for $\mathcal N(\boldsymbol p)$
- heatmaps / price-space visualization

Recommended range:
- $|\mathcal I| \in [6, 20]$

### Medium- and large-scale regime
Use larger instances for:
- runtime/scalability
- Stage-II oracle cost
- Stage-I search efficiency
- strategic-setting comparisons

Recommended range:
- $|\mathcal I| \in [20, 200]$ or larger if runtime allows

### Common metrics
Always record:
- wall-clock runtime
- number of Stage-II solver calls
- final deviation gap $\epsilon$
- total user social cost
- provider revenues $U_E$, $U_N$, and $U_E+U_N$

### Budget normalization for Stage-I baselines
PBRD, BO, DRL, and GSO must be compared under a fair budget, preferably:
- the same maximum number of Stage-II oracle calls, and
- the same wall-clock cap

This is critical because the main cost of Stage I is repeated evaluation of Stage II.

---

## Recommended Main-Text Storyline

The final experiment section should be organized in the following order:

1. **Stage II as a reliable follower oracle**
   - Figure 1
   - Figure 2
   - Figure 3
   - optional: Figure 4 in Appendix

2. **Stage I equilibrium computation**
   - Figure 5
   - Figure 6
   - Figure 7
   - Figure 8
   - Figure 9

3. **Strategic-setting implications**
   - Figure 10
   - Figure 11
   - Figure 12
   - Figure 13

4. **Ablation studies**
   - Figure 14
   - Figure 15

This ordering ensures that Stage II is treated as a necessary oracle module, while Stage I remains the central experimental focus.
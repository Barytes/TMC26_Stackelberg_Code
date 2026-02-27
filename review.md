# Weaknesses

## Technical limitations or concerns
- Conceptual mismatch between the stated “selfish users” model and the algorithmic selection of the Socially Optimal Equilibrium (SOE): the proposed Stage II selection procedure requires a coordinator and effectively enforces a social-welfare minimum among multiple equilibria, which may not coincide with the outcome of decentralized selfish dynamics.
- External stability in the restricted pricing space relies on a heuristic score h_j that lower-bounds the social-cost marginal, not the individual user’s private incentive. Thus Px captures stability under the SOE-selection rule, not necessarily stability to unilateral individual deviations (true GNE with private costs).
- The ε-NE certificate for pricing relies on approximated best-response gains without quantifying approximation error; “ε” is with respect to the approximation, not necessarily the true game.
## Experimental gaps or methodological issues
- The experiments are incomplete in the current draft: several “TODO” markers, placeholder citations, and no reported quantitative results or figures. This leaves key performance claims (convergence, efficiency, superiority) unsupported.
- No empirical validation of approximation accuracy for best-response gains or the impact of sampling density L and candidate family size on final outcomes.
## Clarity or presentation issues
- Numerous notational inconsistencies and typos (e.g., varpi_i vs w_i; C_i^I vs C_i^l; dangling “option: [ ]” markers; minor LaTeX artifacts). These distract and can impede careful verification.
- Some assumptions deserve more explicit discussion, e.g., indivisible tasks and exclusive VM-like allocations with no queueing, and their impact on generality.
## Missing related work or comparisons
- While the related work is broad, it would benefit from tighter connections to recent advances in distributed GNE computation and equilibrium selection (e.g., variational GNE vs welfare GNE trade-offs), and multi-resource multi-provider markets in MEC-like settings.
- The paper would benefit from situating its SOE selection among known equilibrium selection concepts and distributed mechanisms (e.g., showing conditions under which selfish dynamics can be steered to SOE).

# Detailed Comments

## Technical soundness evaluation
- The exact generalized potential game statement with potential equal to total user cost is plausible because unilateral deviations change only one player’s cost, even with shared-capacity coupling (feasible sets depend on others, but when comparing (s_i’, s_-i) vs (s_i, s_-i), the difference in the potential equals the deviator’s cost difference). Existence of a global minimizer over a compact joint feasible set is reasonable.
- NP-hardness of the SOE selection (outer problem) is credible via reduction from standard subset selection/knapsack-type problems; assuming the detailed proof holds, this justifies heuristic design.
- The convexity and strict convexity of the inner allocation (sum of α w_i/f_i + α d_i/(b_i σ_i) + β ρ_i varpi_i d_i/(b_i σ_i) + linear price terms) with linear capacity constraints are correct; strong duality under Slater’s condition (Assumption 1) is standard. The resulting closed forms under capped prices are consistent with first-order conditions and resource-binding regimes.
- Monotonicity of provider revenue within Px is well-argued: for p below the congestion threshold revenue increases linearly; above threshold, (p−c)·A/√p is strictly increasing due to the derivative structure (A p^−3/2 (p/2 + c/2) > 0).
- The restricted pricing space and boundary characterization of RNE are mathematically neat; however, the “external stability” condition uses h_j(λ*) ≥ 0 (a social-cost proxy) rather than the individual-gain condition C_j^{e,*} ≥ C_j^l. This implies the RNE characterization targets stability under the specific SOE-inducing mechanism, not necessarily individual best responses in a decentralized user game.
- The BRGM reduction and boundary-price based approximation are sensible engineering choices; candidate family construction via “boundary users” is intuitive and reduces combinatorics. Yet, without bounds on approximation error, the ε-NE claim should be framed as approximate relative to the surrogate best-response oracle.
## Experimental evaluation assessment
- In the present form, the experimental section lacks quantitative results, ablations, and figures; key statements are placeholders (e.g., “TODO”). This is a critical shortcoming for TMC standards.
- To meet the bar, the paper should report: (i) pricing/game outcomes (revenues, user costs) versus strong baselines (grid search, BO/DRL), (ii) scalability with number of users and sensitivity to F, B, c_E, c_N, (iii) convergence profiles and runtime/communication overheads of Algorithms 1–5, (iv) approximation fidelity for best-response gains (compare to grid/oracle on smaller instances), and (v) robustness to heterogeneity (task sizes, channels) and to the number of “sensitive” users considered.
## Comparison with related work (using the summaries provided)
- Relative to multi-provider and Stackelberg models in MEC (e.g., [15], [18], [20], [21]), this paper’s explicit modeling of complementary resources controlled by independent providers is a meaningful extension; prior works often assume one type of provider/resource or centralized coordination.
- In relation to GNE computation literature (e.g., distributed primal–dual for coupled constraints, variational GNE selections), the paper takes a different path: it selects the SOE (a welfare optimum among equilibria) rather than variational GNE; some recent works argue trade-offs between solution concepts (fairness/normalization vs welfare). Making this distinction more explicit would strengthen positioning.
- The methodological choices (restricted domain, piecewise structural formulas, heuristic guided search) differ from polynomial/GNE decomposition approaches and moment-SOS relaxations (which have limited scalability but can compute/characterize all equilibria under specific structure). The presented approach is more scalable and tailored to the MEC economics at the cost of approximate guarantees; this trade-off should be made explicit.
## Discussion of broader impact and significance
- The setting is practically relevant (e.g., cellular providers vs edge-cloud compute providers). The boundary characterization can inform pricing heuristics in real systems.
- A key takeaway—providers tend to push prices to the “upper boundary” constrained by user indifference—may inform regulator or platform design (e.g., safeguards against excessive markups when resources are complementary bottlenecks).
- If coupled with a mechanism that aligns individual incentives toward SOE (e.g., congestion charges equal to shadow prices), the approach could reduce system-wide cost while maintaining provider revenues—an interesting policy angle. Currently, the mechanism interpretation is incomplete.

# Questions for Authors

- Stage II selection: Are users truly selfish followers, or is a coordinator enforcing the SOE via Algorithm 2? Under what mechanism would decentralized selfish dynamics converge to the SOE rather than an arbitrary GNE? Can you provide a mechanism interpretation (e.g., real congestion charges equal to λ) that aligns individual incentives with the SOE?
- External stability in Px: Why is h_j(λ_F*, λ_B*) ≥ 0 an appropriate condition for “no entry” by a non-offloading user when a selfish user’s criterion is C_j^{e,*} ≥ C_j^l? Can you either prove equivalence under your setting or revise the definition to reflect individual incentives?
- ε-NE certificate: Since best-response gains are approximated, can you bound the gap between the approximated gain and the true gain (e.g., by showing that the optimal deviation set belongs to N(p) under mild conditions), or provide empirical evidence quantifying approximation error?
- Sensitivity to candidate family and L: How do the size of N(p) (via numbers of “boundary users”) and the number of sampled directions L in Algorithm 4 affect solution quality and runtime? Please provide ablation results.
- Scalability and overhead: What are the computational and communication costs of Algorithms 1–5 for I up to, say, hundreds or thousands? Please report convergence rates, wall-clock times, and message counts per iteration.
- Robustness and generality: How sensitive are outcomes to modeling assumptions (e.g., indivisible tasks, exclusive VM allocation without queueing, negligible downlink, OFDMA)? Can your structural results extend to partial offloading or queuing models?
- Empirical validation: Please include complete experimental results against the listed baselines (GSO, BO, DRL, PBRD, Market Equilibrium, SingleSP) with statistical significance and discuss where your approach wins/loses and why.
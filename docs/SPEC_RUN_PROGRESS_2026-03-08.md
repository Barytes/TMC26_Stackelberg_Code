# SPEC figure generation progress (2026-03-08)

## Scope
Resumed full SPEC figure production (Fig.1–15 + Appendix A1/A2), reusing existing outputs as cache and generating missing outputs.

## Completion status
- **Stage II done**: Fig1–4 generated
- **Stage I done**: Fig5–9 generated
- **Strategic done**: Fig10–13 generated
- **Ablation done**: Fig14–15 generated
- **Appendix done**: A1–A2 generated

## Primary output folders (final)
- `outputs/fig1_stage2_convergence_final`
- `outputs/fig2_stage2_approx_ratio_final`
- `outputs/fig3_stage2_comm_final`
- `outputs/fig4_stage2_exploitability_final`
- `outputs/fig5_stage1_convergence_final`
- `outputs/fig6_boundary_final`
- `outputs/fig7_gain_accuracy_final`
- `outputs/fig8_candidate_hit_final`
- `outputs/fig9_scalability_final`
- `outputs/fig10_social_cost_final`
- `outputs/fig11_joint_revenue_final`
- `outputs/fig12_pareto_final`
- `outputs/fig13_fb_final`
- `outputs/fig14_L_final`
- `outputs/fig15_guided_final`
- `outputs/appA1_final`
- `outputs/appA2_final`

## Fixes applied during resume
1. `scripts/run_strategic_fb_sensitivity.py`
   - Fixed `SystemConfig` construction to match current dataclass fields.
   - Fixed GSSE runner to use `cfg.stackelberg` instead of invalid `StackelbergConfig()` no-arg construction.
2. `scripts/run_ablation_L_sensitivity.py`
   - Replaced invalid `StackelbergConfig(rne_directions=L)` with `replace(cfg.stackelberg, rne_directions=L)`.
3. `scripts/run_ablation_guided_search.py`
   - Replaced invalid `Provider.E / Provider.N` usage with literal providers `"E" / "N"` (compatible with current type alias).
4. `scripts/run_appendix_exploitability_vs_users.py`
   - Patched inner-solver config shim to include required fields (`inner_eta_*`, `inner_max_iters`, `inner_tol`) for `algorithm_1_distributed_primal_dual`.

## Notes
- Existing outputs from earlier runs were preserved and reused when present.
- A long full-default run was interrupted by runtime pressure; resumed with targeted runs and completed all required figure scripts with organized output directories above.

from __future__ import annotations

import unittest

import numpy as np

from tmc26_exp.config import StackelbergConfig, SystemConfig
from tmc26_exp.model import UserBatch, local_cost
from tmc26_exp.stackelberg import solve_stage2_user_best_response_dynamics


def _stack_cfg() -> StackelbergConfig:
    return StackelbergConfig(
        enabled=True,
        initial_pE=1.0,
        initial_pN=1.0,
        inner_eta_F0=0.5,
        inner_eta_B0=0.5,
        inner_eta_mu0=0.5,
        inner_max_iters=50,
        inner_tol=1e-6,
        greedy_max_iters=32,
        rne_directions=8,
        rne_root_tol=1e-6,
        rne_max_expand_steps=16,
        search_max_iters=4,
        search_improvement_tol=1e-9,
        stage1_neighborhood_mode="two_stage",
        stage1_neighborhood_max_candidates=32,
        gain_estimator_variant="boundary",
        gain_topk_k=2,
        stage1_solver_variant="paper_iterative_pricing",
        paper_local_Q=1,
        paper_restricted_gap_tol=1e-7,
        paper_outer_update_mode="esp_first",
        topk_brd_price_tol=1e-6,
        topk_brd_epsilon_tol=1e-7,
        topk_brd_cycle_window=4,
        vbbr_local_R=1,
        vbbr_local_S=1,
        vbbr_local_budget=2,
        vbbr_top_m=2,
        vbbr_oracle_max_rounds=2,
        vbbr_oracle_improve_tol=1e-9,
        vbbr_no_improve_patience=1,
        vbbr_outer_gain_tol=1e-7,
        vbbr_damping_alpha=1.0,
        vbbr_outer_update_mode="esp_first",
        vbbr_cycle_window=4,
        vbbr_disable_exact_inner=True,
    )


def _two_user_paper_example() -> UserBatch:
    ones = np.ones(2, dtype=float)
    return UserBatch(
        w=ones.copy(),
        d=np.full(2, 0.5),
        fl=np.full(2, 2.0),
        alpha=ones.copy(),
        beta=ones.copy(),
        rho=ones.copy(),
        varpi=ones.copy(),
        kappa=ones.copy(),
        sigma=ones.copy(),
    )


class Stage2UserBestResponseDynamicsTest(unittest.TestCase):
    def test_sequential_best_response_converges_to_feasible_profile(self) -> None:
        users = _two_user_paper_example()
        system = SystemConfig(F=1.0, B=1.0, cE=0.0, cN=0.0)

        result = solve_stage2_user_best_response_dynamics(
            users,
            pE=1.0,
            pN=1.0,
            system=system,
            cfg=_stack_cfg(),
            max_rounds=8,
            improvement_tol=1e-9,
        )

        self.assertTrue(result.converged)
        self.assertFalse(result.cycle_detected)
        self.assertEqual(len(result.offloading_set), 1)
        self.assertLessEqual(float(result.f.sum()), system.F + 1e-9)
        self.assertLessEqual(float(result.b.sum()), system.B + 1e-9)
        self.assertLess(result.social_cost, float(local_cost(users).sum()))
        self.assertLessEqual(result.trajectory[-1].max_gain, 1e-9)

    def test_social_cost_trace_is_nonincreasing_after_accepted_updates(self) -> None:
        users = _two_user_paper_example()
        system = SystemConfig(F=1.0, B=1.0, cE=0.0, cN=0.0)

        result = solve_stage2_user_best_response_dynamics(
            users,
            pE=1.0,
            pN=1.0,
            system=system,
            cfg=_stack_cfg(),
            max_rounds=8,
            improvement_tol=1e-9,
        )

        social_costs = [step.social_cost for step in result.trajectory]
        self.assertGreaterEqual(len(social_costs), 2)
        for before, after in zip(social_costs, social_costs[1:]):
            self.assertLessEqual(after, before + 1e-9)


if __name__ == "__main__":
    unittest.main()

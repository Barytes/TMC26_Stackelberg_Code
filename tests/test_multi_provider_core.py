from __future__ import annotations

import unittest

import numpy as np

from scripts.multi_provider.multi_provider_core import (
    MultiProviderProblem,
    average_pair_offloading_cost,
    contrast_text_color,
    format_pair_cost_label,
    fixed_assignment_response,
    generate_random_problem,
    solve_multi_provider_stage1,
    solve_multi_provider_stage2,
)


def _toy_problem() -> MultiProviderProblem:
    return MultiProviderProblem(
        aw=np.array([1.0, 1.2, 0.9, 1.1]),
        theta=np.array(
            [
                [0.35, 0.45],
                [0.40, 0.32],
                [0.30, 0.38],
                [0.42, 0.33],
            ]
        ),
        local_cost=np.array([4.0, 4.2, 3.8, 4.1]),
        F=np.array([1.6, 1.6]),
        B=np.array([1.4, 1.4]),
        cE=np.array([0.1, 0.1]),
        cN=np.array([0.1, 0.1]),
    )


class MultiProviderCoreTest(unittest.TestCase):
    def test_fixed_assignment_response_respects_provider_capacities(self) -> None:
        problem = _toy_problem()
        assignment_e = np.array([0, 1, -1, 0])
        assignment_n = np.array([0, 1, -1, 1])

        response = fixed_assignment_response(
            problem,
            assignment_e=assignment_e,
            assignment_n=assignment_n,
            pE=np.array([0.5, 0.55]),
            pN=np.array([0.45, 0.5]),
        )

        for e in range(problem.num_esp):
            self.assertLessEqual(float(response.f[assignment_e == e].sum()), float(problem.F[e]) + 1e-9)
        for n in range(problem.num_nsp):
            self.assertLessEqual(float(response.b[assignment_n == n].sum()), float(problem.B[n]) + 1e-9)
        self.assertGreater(response.social_cost, 0.0)

    def test_stage2_returns_individually_rational_offloading_users(self) -> None:
        problem = _toy_problem()
        result = solve_multi_provider_stage2(
            problem,
            pE=np.array([0.35, 0.35]),
            pN=np.array([0.35, 0.35]),
            max_iters=12,
        )

        self.assertGreaterEqual(result.offloading_count, 1)
        offloading = result.assignment_e >= 0
        self.assertTrue(np.all(result.margins[offloading] >= -1e-8))
        self.assertLess(result.social_cost, float(problem.local_cost.sum()))

    def test_stage1_produces_prices_and_iteration_trace(self) -> None:
        problem = _toy_problem()
        result = solve_multi_provider_stage1(
            problem,
            initial_pE=np.array([0.35, 0.36]),
            initial_pN=np.array([0.34, 0.37]),
            max_iters=3,
            q=1,
            tol=1e-9,
        )

        self.assertEqual(result.pE.shape, (problem.num_esp,))
        self.assertEqual(result.pN.shape, (problem.num_nsp,))
        self.assertGreaterEqual(len(result.trajectory), 1)
        self.assertGreaterEqual(result.restricted_gap, 0.0)

    def test_code_default_heterogeneous_setup_uses_experiment_scale(self) -> None:
        problem = generate_random_problem(
            n_users=30,
            num_esp=3,
            num_nsp=3,
            seed=2026,
            setup="code_default_heterogeneous",
        )

        np.testing.assert_allclose(problem.F, np.array([45.0, 35.0, 25.0]))
        np.testing.assert_allclose(problem.B, np.array([20.0, 14.0, 10.0]))
        np.testing.assert_allclose(problem.cE, np.array([0.008, 0.010, 0.013]))
        np.testing.assert_allclose(problem.cN, np.array([0.007, 0.010, 0.014]))
        self.assertEqual(problem.aw.shape, (30,))
        self.assertEqual(problem.theta.shape, (30, 3))

    def test_code_default_setup_can_scale_total_nsp_bandwidth(self) -> None:
        problem = generate_random_problem(
            n_users=30,
            num_esp=3,
            num_nsp=3,
            seed=2026,
            setup="code_default_heterogeneous",
            nsp_total_bandwidth=100.0,
        )

        self.assertAlmostEqual(float(problem.B.sum()), 100.0)
        np.testing.assert_allclose(
            problem.B,
            np.array([20.0, 14.0, 10.0]) * (100.0 / 44.0),
        )

    def test_average_pair_offloading_cost_uses_nan_for_empty_pairs(self) -> None:
        problem = _toy_problem()
        result = solve_multi_provider_stage1(
            problem,
            initial_pE=np.array([0.35, 0.36]),
            initial_pN=np.array([0.34, 0.37]),
            max_iters=2,
            q=1,
        )

        matrix = average_pair_offloading_cost(problem, result)

        self.assertEqual(matrix.shape, (problem.num_esp, problem.num_nsp))
        self.assertTrue(np.any(np.isfinite(matrix)))
        self.assertTrue(np.any(np.isnan(matrix)))

    def test_contrast_text_color_uses_black_on_light_and_white_on_dark(self) -> None:
        self.assertEqual(contrast_text_color((0.99, 0.90, 0.10, 1.0)), "black")
        self.assertEqual(contrast_text_color((0.12, 0.05, 0.30, 1.0)), "white")

    def test_pair_cost_label_uses_mathtext_user_count(self) -> None:
        self.assertEqual(format_pair_cost_label(25.0502, 6), "25.05\n$|\\mathcal{X}|=6$")


if __name__ == "__main__":
    unittest.main()

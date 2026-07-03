from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmc26_exp.config import SystemConfig

from scripts import run_e2_cost_ratio_sweep as sweep


class E2CostRatioSweepTest(unittest.TestCase):
    def test_parse_ratio_list_sorts_unique_positive_values(self) -> None:
        self.assertEqual(sweep.parse_ratio_list("1e1, 1e-1, 10, 1"), [0.1, 1.0, 10.0])

    def test_parse_ratio_list_rejects_nonpositive_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            sweep.parse_ratio_list("1,0")

    def test_system_for_ratio_holds_ce_fixed_and_sets_cn_from_ratio(self) -> None:
        cfg = SimpleNamespace(system=SystemConfig(F=100.0, B=40.0, cE=0.2, cN=0.5))

        system = sweep.system_for_ratio(cfg, fixed_cE=0.01, ratio=1e3)

        self.assertAlmostEqual(system.cE, 0.01)
        self.assertAlmostEqual(system.cN, 1e-5)
        self.assertAlmostEqual(system.F, cfg.system.F)
        self.assertAlmostEqual(system.B, cfg.system.B)
        self.assertAlmostEqual(cfg.system.cE, 0.2)
        self.assertAlmostEqual(cfg.system.cN, 0.5)

    def test_completed_key_uses_method_ratio_and_trial_only(self) -> None:
        row = {
            "method": "Full model",
            "ratio": "1000",
            "n_users": "100",
            "trial": "7",
        }

        self.assertEqual(sweep.completed_key(row), ("Full model", 1000.0, 7))

    def test_baselines_for_system_preserves_search_width_above_cost_floors(self) -> None:
        base_cfg = SimpleNamespace(max_price_E=6.0, max_price_N=6.0)
        system = SystemConfig(F=100.0, B=40.0, cE=0.01, cN=10.0)

        adjusted = sweep.baselines_for_system(base_cfg, cfg_system_cE=0.01, cfg_system_cN=0.01, system=system)

        self.assertAlmostEqual(adjusted.max_price_E, 6.0)
        self.assertAlmostEqual(adjusted.max_price_N, 15.99)
        self.assertAlmostEqual(base_cfg.max_price_N, 6.0)

    def test_single_plot_specs_include_split_revenues_and_social_cost(self) -> None:
        out_names = {spec.out_name for spec in sweep.SINGLE_PLOT_SPECS}

        self.assertEqual(
            out_names,
            {
                "E2_esp_revenue_cost_ratio_compare.png",
                "E2_nsp_revenue_cost_ratio_compare.png",
                "E2_joint_revenue_cost_ratio_compare.png",
                "E2_user_social_cost_ratio_compare.png",
            },
        )

    def test_plot_display_labels_and_fonts_match_publication_request(self) -> None:
        self.assertEqual(sweep.display_method_label("Full model"), "Stackelberg")
        self.assertEqual(sweep.display_method_label("ME"), "ME")
        self.assertGreaterEqual(sweep.PLOT_FONT_SIZES["axis_label"], 24)
        self.assertGreaterEqual(sweep.PLOT_FONT_SIZES["tick_label"], 20)
        self.assertLessEqual(sweep.PLOT_FONT_SIZES["legend"], 14)
        self.assertEqual(sweep.BAND_ALPHA, 0.14)
        self.assertEqual(sweep.single_plot_legend_loc("social_cost"), "lower left")
        self.assertEqual(sweep.single_plot_legend_font_size("joint_revenue"), 10)
        self.assertEqual(sweep.single_plot_legend_font_size("social_cost"), 10)
        self.assertEqual(sweep.single_plot_axis_label_size("joint_revenue"), sweep.PLOT_FONT_SIZES["axis_label"] + 5)
        self.assertEqual(sweep.single_plot_axis_label_size("social_cost"), sweep.PLOT_FONT_SIZES["axis_label"] + 5)
        self.assertEqual(sweep.single_plot_tick_label_size("joint_revenue"), sweep.PLOT_FONT_SIZES["tick_label"] + 5)
        self.assertEqual(sweep.single_plot_tick_label_size("social_cost"), sweep.PLOT_FONT_SIZES["tick_label"] + 5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage1_budget_matched_quality_vs_users as budget_matched


class _CaptureStream:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.flush_count = 0

    def write(self, text: str) -> None:
        self.messages.append(text)

    def flush(self) -> None:
        self.flush_count += 1


class Stage1BudgetMatchedProgressTest(unittest.TestCase):
    def test_progress_message_identifies_current_trial_method_and_budget(self) -> None:
        message = budget_matched._format_progress_message(
            completed=7,
            total=150,
            n_users=100,
            trial=3,
            trials=10,
            method="GA",
            budget=49,
            phase="start",
        )

        self.assertIn("[8/150]", message)
        self.assertIn("phase=start", message)
        self.assertIn("n_users=100", message)
        self.assertIn("trial=3/10", message)
        self.assertIn("method=GA", message)
        self.assertIn("budget_stage2_calls=49", message)

    def test_print_progress_flushes_stream_for_live_terminal_updates(self) -> None:
        stream = _CaptureStream()

        budget_matched._print_progress(
            completed=0,
            total=15,
            n_users=50,
            trial=1,
            trials=5,
            method="BO-online",
            budget=43,
            phase="done",
            stream=stream,
        )

        self.assertEqual(stream.flush_count, 1)
        self.assertTrue("".join(stream.messages).endswith("\n"))

    def test_load_checkpoint_rows_returns_completed_trial_method_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stage1_budget_matched_quality_vs_users.csv"
            budget_matched.write_csv_rows(
                csv_path,
                budget_matched.TRIAL_FIELDS,
                [
                    {
                        "method": "BO-online",
                        "n_users": 100,
                        "trial": 4,
                        "source": "budget_matched_rerun",
                        "success": 1,
                        "budget_stage2_calls": 49,
                        "search_budget_exhausted": 1,
                        "budget_stop_mode": "budget_after_objective",
                        "final_pE": 1.0,
                        "final_pN": 2.0,
                        "offloading_size": 3,
                        "final_grid_ne_gap": 0.1,
                        "esp_revenue": 4.0,
                        "nsp_revenue": 5.0,
                        "joint_revenue": 9.0,
                        "runtime_sec": 0.2,
                        "stage2_solver_calls": 49,
                        "audit_stage2_solver_calls": 8,
                        "total_stage2_solver_calls": 57,
                        "error": "",
                    }
                ],
            )

            rows, completed = budget_matched._load_checkpoint_rows(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(completed, {("BO-online", 100, 4)})


if __name__ == "__main__":
    unittest.main()

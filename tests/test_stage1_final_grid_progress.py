from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage1_final_grid_ne_gap_vs_users as final_grid


class _CaptureStream:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.flush_count = 0

    def write(self, text: str) -> None:
        self.messages.append(text)

    def flush(self) -> None:
        self.flush_count += 1


class Stage1FinalGridProgressTest(unittest.TestCase):
    def test_progress_message_identifies_current_trial_method_and_phase(self) -> None:
        message = final_grid._format_progress_message(
            completed=11,
            total=200,
            n_users=150,
            trial=4,
            trials=10,
            method="Proposed",
            phase="start",
        )

        self.assertIn("[12/200]", message)
        self.assertIn("phase=start", message)
        self.assertIn("n_users=150", message)
        self.assertIn("trial=4/10", message)
        self.assertIn("method=Proposed", message)

    def test_print_progress_flushes_stream_for_live_terminal_updates(self) -> None:
        stream = _CaptureStream()

        final_grid._print_progress(
            completed=2,
            total=20,
            n_users=50,
            trial=1,
            trials=5,
            method="MARL",
            phase="done",
            stream=stream,
        )

        self.assertEqual(stream.flush_count, 1)
        self.assertTrue("".join(stream.messages).endswith("\n"))

    def test_load_checkpoint_rows_returns_completed_trial_method_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stage1_final_grid_ne_gap_vs_users.csv"
            final_grid.write_csv_rows(
                csv_path,
                final_grid.TRIAL_FIELDS,
                [
                    {
                        "method": "GA",
                        "n_users": 50,
                        "trial": 2,
                        "success": 1,
                        "final_pE": 1.0,
                        "final_pN": 2.0,
                        "offloading_size": 3,
                        "restricted_gap": 0.0,
                        "final_grid_ne_gap": 0.1,
                        "final_grid_ne_gap_source": "audit_grid",
                        "esp_revenue": 4.0,
                        "nsp_revenue": 5.0,
                        "joint_revenue": 9.0,
                        "runtime_sec": 0.2,
                        "stage2_solver_calls": 7,
                        "audit_stage2_solver_calls": 8,
                        "total_stage2_solver_calls": 15,
                        "error": "",
                    }
                ],
            )

            rows, completed = final_grid._load_checkpoint_rows(csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(completed, {("GA", 50, 2)})

    def test_expected_trial_keys_are_limited_to_requested_methods(self) -> None:
        keys = final_grid._expected_trial_keys(["Proposed"], [50, 100], 2)

        self.assertEqual(
            keys,
            {
                ("Proposed", 50, 1),
                ("Proposed", 50, 2),
                ("Proposed", 100, 1),
                ("Proposed", 100, 2),
            },
        )


if __name__ == "__main__":
    unittest.main()

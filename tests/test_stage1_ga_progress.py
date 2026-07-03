from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tmc26_exp import baselines


class _CaptureStream:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.flush_count = 0

    def write(self, text: str) -> None:
        self.messages.append(text)

    def flush(self) -> None:
        self.flush_count += 1


class Stage1GaProgressTest(unittest.TestCase):
    def test_format_ga_progress_message_includes_generation_individual_and_phase(self) -> None:
        message = baselines._format_ga_progress_message(
            phase="objective_start",
            generation="3/8",
            individual_index=4,
            population_size=12,
            evals=27,
            stage2_unique_prices=19,
            best_score=0.125,
            best_price=(2.5, 3.5),
            elapsed_sec=42.25,
        )

        self.assertIn("[GA]", message)
        self.assertIn("phase=objective_start", message)
        self.assertIn("generation=3/8", message)
        self.assertIn("individual=4/12", message)
        self.assertIn("evals=27", message)
        self.assertIn("stage2_unique_prices=19", message)
        self.assertIn("best_score=0.125", message)
        self.assertIn("best_price=2.5,3.5", message)
        self.assertIn("elapsed_sec=42.25", message)

    def test_print_ga_progress_flushes_stream_for_live_updates(self) -> None:
        stream = _CaptureStream()

        baselines._print_ga_progress(
            phase="individual_start",
            generation="initial",
            individual_index=1,
            population_size=12,
            stream=stream,
        )

        self.assertEqual(stream.flush_count, 1)
        self.assertTrue("".join(stream.messages).endswith("\n"))


if __name__ == "__main__":
    unittest.main()

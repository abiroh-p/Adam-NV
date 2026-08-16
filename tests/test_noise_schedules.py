"""Deterministic unit tests for non-stationary noise schedules."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "e2_noisy_gradient" / "src" / "noise_schedule.py"
SPEC = importlib.util.spec_from_file_location("noise_schedule", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NoiseScheduleTest(unittest.TestCase):
    def test_constant(self) -> None:
        self.assertEqual(MODULE.get_sigma_schedule("constant", 4, 2.0), [2.0] * 4)

    def test_intermittent_starts_clean(self) -> None:
        values = MODULE.get_sigma_schedule(
            "intermittent", 8, 2.0, burst_epochs=2, gap_epochs=2,
            start_clean=True)
        self.assertEqual(values, [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0])

    def test_phased_has_clean_noisy_clean_blocks(self) -> None:
        values = MODULE.get_sigma_schedule(
            "phased", 6, 2.0, phase_start=1 / 3, phase_end=2 / 3)
        self.assertEqual(values, [0.0, 0.0, 2.0, 2.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

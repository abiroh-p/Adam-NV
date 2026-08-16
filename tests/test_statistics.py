"""Unit tests for the paired statistical analysis."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis" / "analyze_e13_e15.py"
SPEC = importlib.util.spec_from_file_location("analyze_e13_e15", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StatisticsTest(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_p_values(self) -> None:
        raw = [0.03, 0.001, 0.2]
        adjusted = MODULE.holm_adjust(raw)
        order = np.argsort(raw)
        sorted_adjusted = [adjusted[i] for i in order]
        self.assertEqual(sorted_adjusted, sorted(sorted_adjusted))
        self.assertTrue(all(a >= p for a, p in zip(adjusted, raw)))

    def test_sign_flip_pvalue_is_one_for_zero_mean_symmetric_pair(self) -> None:
        self.assertEqual(MODULE.exact_sign_flip_pvalue(np.array([1.0, -1.0])), 1.0)


if __name__ == "__main__":
    unittest.main()

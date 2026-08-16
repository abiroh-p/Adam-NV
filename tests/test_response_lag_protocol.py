"""Tests preventing invalid legacy lag logs from entering reported analyses."""

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "e2_noisy_gradient" / "src" / "analyze_response_lag.py"
SPEC = importlib.util.spec_from_file_location("analyze_response_lag", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResponseLagProtocolTest(unittest.TestCase):
    def test_unversioned_log_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unversioned legacy diagnostic"):
            MODULE.validate_protocol(pd.DataFrame({"global_step": [1]}), "legacy.csv")

    def test_v2_log_is_accepted(self) -> None:
        MODULE.validate_protocol(
            pd.DataFrame({"protocol_version": [2, 2], "global_step": [1, 2]}),
            "corrected.csv",
        )


if __name__ == "__main__":
    unittest.main()

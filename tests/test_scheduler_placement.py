"""Regression tests for learning-rate scheduler placement.

The response-lag diagnostic once advanced CosineAnnealingLR after every
minibatch even though T_max was expressed in epochs. These tests inspect the
training-loop structure so that the diagnostic cannot silently diverge from
the main experiment protocol again.
"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = (
    ROOT / "e2_noisy_gradient" / "src" / "diagnose_response_lag.py",
    ROOT / "e2_noisy_gradient" / "src" / "diagnose_response_lag_v1.py",
)


def _is_scheduler_step(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scheduler"
    )


def _contains_scheduler_step(node: ast.AST) -> bool:
    return any(_is_scheduler_step(child) for child in ast.walk(node))


class SchedulerPlacementTest(unittest.TestCase):
    def test_diagnostics_step_scheduler_once_per_epoch(self) -> None:
        for path in DIAGNOSTICS:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                epoch_loops = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == "epoch"
                ]
                self.assertEqual(len(epoch_loops), 1)
                epoch_loop = epoch_loops[0]

                batch_loops = [
                    node
                    for node in epoch_loop.body
                    if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Tuple)
                    and [elt.id for elt in node.target.elts if isinstance(elt, ast.Name)]
                    == ["x", "y"]
                ]
                self.assertEqual(len(batch_loops), 1)
                self.assertFalse(
                    _contains_scheduler_step(batch_loops[0]),
                    "scheduler.step() must not run inside the minibatch loop",
                )

                epoch_level_steps = [
                    node
                    for node in epoch_loop.body
                    if _contains_scheduler_step(node)
                ]
                self.assertEqual(
                    len(epoch_level_steps),
                    1,
                    "scheduler.step() must run exactly once in each epoch",
                )


if __name__ == "__main__":
    unittest.main()

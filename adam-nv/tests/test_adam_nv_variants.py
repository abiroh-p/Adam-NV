"""Tests for the bias_correction ablation option of Adam-NV and for copy consistency."""

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "e1_convergence" / "src" / "adam_nv.py"
E2 = ROOT / "e2_noisy_gradient" / "src" / "adam_nv.py"


def _norm(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


class CopyConsistencyTest(unittest.TestCase):
    def test_e1_and_e2_adam_nv_are_identical(self) -> None:
        # Two copies exist because each experiment folder imports its own;
        # they must never drift apart again (line endings ignored).
        self.assertEqual(_norm(E1), _norm(E2))


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class BiasCorrectionOptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("adam_nv_variants_under_test", E2)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.AdamNV = module.AdamNV

    def _grads(self, n, shape, seed):
        gen = torch.Generator().manual_seed(seed)
        return [(0.1 + 2.0 * (i % 50 < 25)) * torch.randn(*shape, generator=gen, dtype=torch.float64)
                for i in range(n)]

    def _run(self, **kw):
        p = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(3, 4))
        opt = self.AdamNV([p], lr=1e-3, **kw)
        for g in self._grads(300, (3, 4), seed=5):
            p.grad = g.clone()
            opt.step()
        return p.detach()

    def test_default_equals_explicit_running_product(self) -> None:
        self.assertTrue(torch.equal(self._run(), self._run(bias_correction="running_product")))

    def test_base_correction_with_gamma_zero_matches_torch_adam(self) -> None:
        p = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(3, 4))
        ref = torch.nn.Parameter(p.detach().clone())
        ours = self.AdamNV([p], lr=1e-3, gamma=0.0, bias_correction="base")
        adam = torch.optim.Adam([ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for g in self._grads(300, (3, 4), seed=5):
            p.grad = g.clone(); ref.grad = g.clone()
            ours.step(); adam.step()
        self.assertTrue(torch.allclose(p, ref, atol=1e-11, rtol=1e-9))

    def test_base_correction_differs_when_controller_is_active(self) -> None:
        diff = (self._run(bias_correction="base") - self._run()).abs().max().item()
        self.assertGreater(diff, 1e-6)

    def test_invalid_value_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.AdamNV([torch.nn.Parameter(torch.ones(1))], bias_correction="none")


if __name__ == "__main__":
    unittest.main()

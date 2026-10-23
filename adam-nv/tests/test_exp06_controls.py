"""Exp. 6 support test: with gamma=0 the default (running-product) Adam-NV must match torch Adam.

The Adam-NV docstring says gamma=0 reduces to Adam "exactly". With the default running-product
correction that holds only up to the P underflow clamp (_MIN_PRODUCT = 1e-7, reached after
about 153 steps at beta=0.9), so this test uses a tolerance of about 1e-6, not bit equality.
"""

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "e1_convergence" / "src" / "adam_nv.py"


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class GammaZeroDefaultMatchesAdamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("adam_nv_exp06_under_test", E1)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.AdamNV = module.AdamNV

    def test_gamma_zero_running_product_matches_torch_adam(self) -> None:
        gen = torch.Generator().manual_seed(5)
        p = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(3, 4))
        ref = torch.nn.Parameter(p.detach().clone())
        ours = self.AdamNV([p], lr=1e-3, gamma=0.0)  # default bias_correction="running_product"
        adam = torch.optim.Adam([ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for _ in range(500):
            g = torch.randn(3, 4, generator=gen, dtype=torch.float64)
            p.grad = g.clone()
            ref.grad = g.clone()
            ours.step()
            adam.step()
        self.assertTrue(torch.allclose(p, ref, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

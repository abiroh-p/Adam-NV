"""Exp. 7 support test: the `adam_nv_fixed` configuration used as the constant-beta1 control
(AdamNV with gamma=0, beta_base=beta_min=0.81) must match torch Adam with betas=(0.81, 0.999).

The running-product bias correction is clamped at 1e-7 (reached after about 62 steps at
beta1=0.81), so agreement is checked to about 1e-6, not bit equality.
"""

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "e2_noisy_gradient" / "src" / "adam_nv.py"


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class FixedBeta1MatchesAdamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("adam_nv_exp07_under_test", E2)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.AdamNV = module.AdamNV

    def test_fixed_beta1_matches_torch_adam(self) -> None:
        beta1 = 0.81
        gen = torch.Generator().manual_seed(11)
        p = torch.nn.Parameter(torch.linspace(-1, 1, 12, dtype=torch.float64).reshape(3, 4))
        ref = torch.nn.Parameter(p.detach().clone())
        # same construction as build_optimizer("adam_nv_fixed", ...) in part_b_resnet_v8.py
        ours = self.AdamNV([p], lr=1e-3, gamma=0.0, beta_base=beta1, beta_min=beta1)
        adam = torch.optim.Adam([ref], lr=1e-3, betas=(beta1, 0.999), eps=1e-8)
        for _ in range(500):
            g = torch.randn(3, 4, generator=gen, dtype=torch.float64)
            p.grad = g.clone()
            ref.grad = g.clone()
            ours.step()
            adam.step()
        self.assertTrue(torch.allclose(p, ref, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

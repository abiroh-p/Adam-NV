"""Numerical optimizer tests. Skipped when PyTorch is unavailable."""

import importlib.util
from pathlib import Path
import unittest


try:
    import torch
except ImportError:  # analysis-only environments need not install GPU packages
    torch = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AdamNVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "e2_noisy_gradient" / "src" / "adam_nv.py"
        spec = importlib.util.spec_from_file_location("adam_nv_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.AdamNV = module.AdamNV

    def test_gamma_zero_matches_torch_adam(self) -> None:
        initial = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
        ours = torch.nn.Parameter(initial.clone())
        reference = torch.nn.Parameter(initial.clone())
        ours_opt = self.AdamNV([ours], lr=1e-3, gamma=0.0)
        reference_opt = torch.optim.Adam(
            [reference], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        generator = torch.Generator().manual_seed(7)
        for _ in range(50):
            gradient = torch.randn(3, generator=generator, dtype=torch.float64)
            ours.grad = gradient.clone()
            reference.grad = gradient.clone()
            ours_opt.step()
            reference_opt.step()
        self.assertTrue(torch.allclose(ours, reference, atol=1e-11, rtol=1e-9))

    def test_decoupled_noise_buffer_does_not_change_update_v_decay(self) -> None:
        coupled_p = torch.nn.Parameter(torch.tensor([1.0]))
        decoupled_p = torch.nn.Parameter(torch.tensor([1.0]))
        coupled = self.AdamNV([coupled_p], beta_2=0.999)
        decoupled = self.AdamNV(
            [decoupled_p], beta_2=0.999, beta_2_noise=0.95)
        for value in (1.0, 2.0, -0.5):
            coupled_p.grad = torch.tensor([value])
            decoupled_p.grad = torch.tensor([value])
            coupled.step()
            decoupled.step()
        coupled_v = coupled.state[coupled_p]["v"]
        decoupled_v = decoupled.state[decoupled_p]["v"]
        decoupled_v_noise = decoupled.state[decoupled_p]["v_noise"]
        self.assertTrue(torch.equal(coupled_v, decoupled_v))
        self.assertFalse(torch.equal(decoupled_v, decoupled_v_noise))

    def test_response_modes_move_beta1_in_opposite_directions(self) -> None:
        distrust_p = torch.nn.Parameter(torch.tensor([1.0]))
        smooth_p = torch.nn.Parameter(torch.tensor([1.0]))
        distrust = self.AdamNV([distrust_p], response_mode="distrust", k=0)
        smooth = self.AdamNV([smooth_p], response_mode="smooth", k=0)
        for value in (1.0, -1.0, 1.0, -1.0):
            distrust_p.grad = torch.tensor([value])
            smooth_p.grad = torch.tensor([value])
            distrust.step()
            smooth.step()
        distrust_beta = distrust.get_beta1_stats()[0]
        smooth_beta = smooth.get_beta1_stats()[0]
        self.assertLess(distrust_beta, 0.9)
        self.assertGreater(smooth_beta, 0.9)


if __name__ == "__main__":
    unittest.main()

"""Numerical tests for the GONO baseline. Skipped when PyTorch is unavailable."""

import importlib.util
import math
from pathlib import Path
import unittest


try:
    import torch
except ImportError:  # analysis-only environments need not install GPU packages
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def reference_gono_tensor_scope(theta0, grads, lr=1e-3, beta1=0.9, beta2=0.999,
                                eps=1e-8, lam=0.4, b1_min=0.5, b1_max=0.99,
                                cos_eps=1e-10):
    """Plain-formula GONO for ONE tensor (Algorithm 1 of arXiv:2605.06575).

    Deliberately written differently from the optimizer under test (no
    in-place ops, explicit m_hat / v_hat) so the comparison is not circular.
    """
    theta = theta0.clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    g_prev = torch.zeros_like(theta)
    for t, g in enumerate(grads, start=1):
        cc = float((g * g_prev).sum() / (g.norm() * g_prev.norm() + cos_eps))
        b1t = min(max(beta1 * (1.0 + lam * cc), b1_min), b1_max)
        m = b1t * m + (1.0 - b1t) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - lr * m_hat / (v_hat.sqrt() + eps)
        g_prev = g.clone()
    return theta


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class GONOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "e1_convergence" / "src" / "gono.py"
        spec = importlib.util.spec_from_file_location("gono_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.GONO = module.GONO

    def _grads(self, n, shape, seed):
        gen = torch.Generator().manual_seed(seed)
        return [torch.randn(*shape, generator=gen, dtype=torch.float64)
                for _ in range(n)]

    def test_lam_zero_matches_torch_adam_both_scopes(self) -> None:
        for scope in ("tensor", "global"):
            initial = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
            ours = torch.nn.Parameter(initial.clone())
            ref = torch.nn.Parameter(initial.clone())
            ours_opt = self.GONO([ours], lr=1e-3, lam=0.0, cosine_scope=scope)
            ref_opt = torch.optim.Adam([ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
            for g in self._grads(60, (3,), seed=7):
                ours.grad = g.clone()
                ref.grad = g.clone()
                ours_opt.step()
                ref_opt.step()
            self.assertTrue(torch.allclose(ours, ref, atol=1e-11, rtol=1e-9),
                            msg=f"lam=0 differs from Adam for scope={scope}")

    def test_matches_independent_reference_tensor_scope(self) -> None:
        initial = torch.randn(4, 3, generator=torch.Generator().manual_seed(1),
                              dtype=torch.float64)
        grads = self._grads(120, (4, 3), seed=11)
        p = torch.nn.Parameter(initial.clone())
        opt = self.GONO([p], lr=1e-3)
        for g in grads:
            p.grad = g.clone()
            opt.step()
        expected = reference_gono_tensor_scope(initial, grads)
        self.assertTrue(torch.allclose(p, expected, atol=1e-11, rtol=1e-9))

    def test_first_step_uses_base_beta1(self) -> None:
        p = torch.nn.Parameter(torch.ones(5, dtype=torch.float64))
        opt = self.GONO([p])
        p.grad = torch.randn(5, dtype=torch.float64)
        opt.step()
        mean, lo, hi = opt.get_beta1_stats()
        self.assertEqual((mean, lo, hi), (0.9, 0.9, 0.9))
        self.assertEqual(opt.get_cc_stats()[0], 0.0)

    def test_oscillating_gradients_lower_beta1(self) -> None:
        p = torch.nn.Parameter(torch.ones(6, dtype=torch.float64))
        opt = self.GONO([p])
        base = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.25], dtype=torch.float64)
        for i in range(10):
            p.grad = base * (1 if i % 2 == 0 else -1)
            opt.step()
        self.assertAlmostEqual(opt.get_beta1_stats()[0], 0.9 * (1 - 0.4), places=6)

    def test_consistent_gradients_hit_upper_bound(self) -> None:
        p = torch.nn.Parameter(torch.ones(6, dtype=torch.float64))
        opt = self.GONO([p])
        base = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.25], dtype=torch.float64)
        for _ in range(10):
            p.grad = base.clone()
            opt.step()
        self.assertAlmostEqual(opt.get_beta1_stats()[0], 0.99, places=9)

    def test_beta1_always_within_bounds(self) -> None:
        p = torch.nn.Parameter(torch.randn(20, dtype=torch.float64))
        opt = self.GONO([p])
        for g in self._grads(200, (20,), seed=3):
            p.grad = g
            opt.step()
            mean, lo, hi = opt.get_beta1_stats()
            self.assertGreaterEqual(lo, 0.5 - 1e-12)
            self.assertLessEqual(hi, 0.99 + 1e-12)

    def test_global_scope_shares_one_beta1_tensor_scope_does_not(self) -> None:
        def run(scope):
            a = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
            b = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
            opt = self.GONO([a, b], cosine_scope=scope)
            va = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
            vb = torch.tensor([0.3, -0.7, 1.1, 2.0], dtype=torch.float64)
            for i in range(8):
                a.grad = va.clone()                      # consistent
                b.grad = vb * (1 if i % 2 == 0 else -1)  # oscillating
                opt.step()
            return opt.state[a]["beta1_t"].item(), opt.state[b]["beta1_t"].item()

        ta, tb = run("tensor")
        self.assertAlmostEqual(ta, 0.99, places=9)
        self.assertAlmostEqual(tb, 0.54, places=6)
        ga, gb = run("global")
        self.assertAlmostEqual(ga, gb, places=12)

    def test_zero_gradient_does_not_produce_nan(self) -> None:
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        opt = self.GONO([p])
        for _ in range(3):
            p.grad = torch.zeros(3, dtype=torch.float64)
            opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_invalid_arguments_rejected(self) -> None:
        p = torch.nn.Parameter(torch.ones(1))
        for kwargs in ({"lam": -0.1}, {"beta1_min": 0.95}, {"beta1_max": 1.0},
                       {"cosine_scope": "layer"}, {"cos_eps": 0.0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                self.GONO([p], **kwargs)

    def test_exact_correction_lam_zero_matches_torch_adam(self) -> None:
        initial = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
        ours = torch.nn.Parameter(initial.clone())
        ref = torch.nn.Parameter(initial.clone())
        ours_opt = self.GONO([ours], lr=1e-3, lam=0.0, bias_correction="exact")
        ref_opt = torch.optim.Adam([ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for g in self._grads(60, (3,), seed=7):      # 0.9**60 >> P_min: clamp inactive
            ours.grad = g.clone()
            ref.grad = g.clone()
            ours_opt.step()
            ref_opt.step()
        self.assertTrue(torch.allclose(ours, ref, atol=1e-11, rtol=1e-9))

    def test_exact_correction_recovers_constant_gradient_base_does_not(self) -> None:
        base = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        results = {}
        for mode in ("exact", "base"):
            p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
            opt = self.GONO([p], lr=1e-3, bias_correction=mode)
            for _ in range(8):
                p.grad = base.clone()
                opt.step()
            st = opt.state[p]
            if mode == "exact":
                m_hat = st["m"] / (1.0 - st["P"])
            else:
                m_hat = st["m"] / (1.0 - 0.9 ** st["t"])
            results[mode] = m_hat
        # with a constant gradient the exact normalizer returns the gradient itself
        self.assertTrue(torch.allclose(results["exact"], base, atol=1e-12))
        # the base-beta1 normalizer under-estimates it while beta1_t > beta1
        self.assertTrue(torch.all(results["base"].abs() < 0.5 * base.abs()))

    def test_invalid_bias_correction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.GONO([torch.nn.Parameter(torch.ones(1))], bias_correction="none")


if __name__ == "__main__":
    unittest.main()

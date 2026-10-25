"""Exp. 11 tests: AdamNVShuffle (beta_1 shuffle control).

Checks (CPU, small tensors):
  * shuffle="none" is bit-identical to AdamNV;
  * with gamma = 0 (uniform beta_1) any shuffle mode is bit-identical to AdamNV;
  * "step" and "fixed" permute beta_1 within each tensor (the multiset of values is preserved);
  * "fixed" reuses one permutation, "step" draws a new one;
  * a shuffled run differs from the unshuffled one when the controller is active, is
    reproducible for a given seed and differs for another seed;
  * the distribution log is valid JSON lines at the requested cadence;
  * invalid arguments are rejected.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "e2_noisy_gradient" / "src"


def _load(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    spec = importlib.util.spec_from_file_location(name + "_exp11_under_test", SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grads(steps, shape, seed=0):
    """Heteroscedastic zero-mean gradients so that the noise score is clearly active."""
    gen = torch.Generator().manual_seed(seed)
    scale = torch.ones(shape)
    scale.view(-1)[::2] = 4.0
    return [torch.randn(shape, generator=gen) * scale for _ in range(steps)]


def _run(opt_cls, steps=60, shape=(6, 8), seed=0, **kw):
    p = torch.nn.Parameter(torch.linspace(-1, 1, shape[0] * shape[1]).reshape(shape))
    opt = opt_cls([p], lr=1e-3, **kw)
    for g in _grads(steps, shape, seed):
        p.grad = g.clone()
        opt.step()
    return p.detach().clone(), opt


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AdamNVShuffleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AdamNV = _load("adam_nv").AdamNV
        cls.Shuffle = _load("adam_nv_shuffle").AdamNVShuffle

    def test_none_is_bit_identical_to_adam_nv(self):
        a, _ = _run(self.AdamNV)
        b, _ = _run(self.Shuffle, shuffle="none")
        self.assertTrue(torch.equal(a, b))

    def test_gamma_zero_makes_every_mode_identical(self):
        a, _ = _run(self.AdamNV, gamma=0.0)
        for mode in ("step", "fixed"):
            b, _ = _run(self.Shuffle, shuffle=mode, gamma=0.0)
            self.assertTrue(torch.equal(a, b), mode)

    def test_modes_preserve_the_beta1_multiset_within_a_tensor(self):
        records = []
        outer = self

        class Recording(self.Shuffle):
            def _apply_shuffle(self, p, state, beta_1):
                out = super()._apply_shuffle(p, state, beta_1)
                records.append((beta_1.detach().clone(), out.detach().clone()))
                return out

        for mode in ("step", "fixed"):
            records.clear()
            _run(Recording, shuffle=mode)
            moved = 0
            for before, after in records:
                self.assertTrue(torch.equal(torch.sort(before.reshape(-1)).values,
                                            torch.sort(after.reshape(-1)).values), mode)
                moved += int(not torch.equal(before, after))
            outer.assertGreater(moved, 0, f"{mode}: beta_1 was never actually permuted")

    def test_fixed_reuses_one_permutation_and_step_draws_new_ones(self):
        _, fixed = _run(self.Shuffle, shuffle="fixed", steps=5)
        p = fixed.param_groups[0]["params"][0]
        perm1 = fixed.state[p]["perm"].clone()
        g = _grads(1, tuple(p.shape), seed=9)[0]
        p.grad = g
        fixed.step()
        self.assertTrue(torch.equal(perm1, fixed.state[p]["perm"]))

        perms = []

        class Recording(self.Shuffle):
            def _apply_shuffle(self, p, state, beta_1):
                n = beta_1.numel()
                before = torch.Generator(device=beta_1.device)
                before.set_state(self._generator(beta_1.device).get_state())
                perms.append(torch.randperm(n, generator=before, device=beta_1.device))
                return super()._apply_shuffle(p, state, beta_1)

        _run(Recording, shuffle="step", steps=4)
        self.assertFalse(all(torch.equal(perms[0], q) for q in perms[1:]))

    def test_shuffle_changes_the_trajectory_and_is_reproducible(self):
        base, _ = _run(self.Shuffle, shuffle="none")
        s0a, _ = _run(self.Shuffle, shuffle="step", shuffle_seed=0)
        s0b, _ = _run(self.Shuffle, shuffle="step", shuffle_seed=0)
        s1, _ = _run(self.Shuffle, shuffle="step", shuffle_seed=1)
        self.assertFalse(torch.equal(base, s0a))
        self.assertTrue(torch.equal(s0a, s0b))
        self.assertFalse(torch.equal(s0a, s1))

    def test_distribution_log_is_written_at_the_requested_cadence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dist.jsonl"
            _run(self.Shuffle, shuffle="none", steps=12, dist_log_path=str(path), dist_log_every=4)
            lines = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([r["step"] for r in lines], [4, 8, 12])
            for r in lines:
                for key in ("n_elements", "mean", "min", "max", "q01", "q50", "q99",
                            "frac_below_0.5", "frac_below_0.7"):
                    self.assertIn(key, r)
                self.assertLessEqual(r["min"], r["q50"])
                self.assertLessEqual(r["q50"], r["max"])

    def test_invalid_arguments_rejected(self):
        p = torch.nn.Parameter(torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            self.Shuffle([p], shuffle="bogus")


if __name__ == "__main__":
    unittest.main()

"""GONO baseline for Adam-NV Protocol v2.

Independent re-implementation of the optimizer described in
    V. D. Gera, "Directional Consistency as a Complementary Optimization
    Signal: The GONO Framework", arXiv:2605.06575 (2026), Algorithm 1.
Written from the paper's algorithm, NOT copied from the authors' repository
(github.com/victordaniel/gono-optimizer), which ships without a license file.

Update rule (per Algorithm 1):
    cc_t    = <g_t, g_{t-1}> / (||g_t|| ||g_{t-1}|| + eps_cos)
    b1_t    = clip(beta1 * (1 + lam * cc_t), beta1_min, beta1_max)
    m_t     = b1_t * m_{t-1} + (1 - b1_t) * g_t
    v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat   = m_t / (1 - beta1^t)          <- BASE beta1, as stated in the paper
    v_hat   = v_t / (1 - beta2^t)
    theta  -= lr * m_hat / (sqrt(v_hat) + eps)
With lam = 0 this equals torch.optim.Adam (checked in tests/test_gono.py).

Choices where the paper and the authors' code disagree or are silent. Each
one is exposed or documented so the paper can state it:

1. Scope of the cosine (argument `cosine_scope`).
   * "tensor" (DEFAULT): cc_t is computed separately for every parameter
     tensor, giving one scalar beta1_t per tensor. This is what the authors'
     released code does.
   * "global": a single cc_t over the whole concatenated gradient vector,
     which is the literal reading of Algorithm 1.
2. Epsilon inside the cosine (`cos_eps`, default 1e-10). The paper writes the
   same eps (1e-8) used by Adam; the authors' code uses 1e-10. We follow the
   code, because 1e-8 would damp cc_t for tensors whose gradient-norm
   product is near 1e-8.
3. Epsilon placement in the Adam step. We use the paper / torch.optim.Adam
   convention: theta -= lr/(1-beta1^t) * m / (sqrt(v)/sqrt(1-beta2^t) + eps).
   The authors' code instead uses sqrt(v) + eps with step size
   lr*sqrt(1-beta2^t)/(1-beta1^t), i.e. a larger effective epsilon in the early
   steps. Their docstring says lam=0 is exactly Adam; measured, it is not.
   Measured against their released file (float64, 3000 steps, synthetic
   gradients, lam=0.4): max parameter difference 2.5e-6 for parameter
   movements of about 0.2, caused mostly by this epsilon placement (the
   difference arises early and persists as a small parameter offset).
   For gradient elements of scale ~1e-6 the offset reached ~1.4% of the
   parameter movement in a float32 synthetic test, so this is not always
   negligible.
3b. The authors' code also passes beta1_t through float32
   (torch.tensor(...)); a smaller effect (~7e-8 in the same test).
   With both differences matched, this implementation and the authors' file
   agree to 9e-16, so the cosine / clipping / momentum logic is identical.
4. First step: g_prev = 0, hence cc_1 = 0 and b1_1 = beta1.
5. `bias_correction` ("base" default = the paper; "exact" = ablation that
   replaces 1 - beta1^t by 1 - P_t, P_t = prod_s beta1_s, clamped below at
   1e-7 as in Adam-NV). Only "base" is the published GONO.

Logged diagnostics: for "tensor" scope, beta1_* statistics are taken over
parameter TENSORS (one scalar each), not over individual elements as for
Adam-NV, so they must not be compared numerically with Adam-NV's
element-wise beta1_mean / beta1_min / beta1_max.
"""

import math

import torch

# Same underflow floor as Adam-NV's running product (adam_nv._MIN_PRODUCT).
_MIN_PRODUCT = 1e-7


class GONO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 lam=0.4, beta1_min=0.5, beta1_max=0.99, weight_decay=0.0,
                 cosine_scope="tensor", cos_eps=1e-10, bias_correction="base"):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not lam >= 0.0:
            raise ValueError(f"Invalid lam: {lam}")
        if not 0.0 <= beta1_min <= beta1 <= beta1_max < 1.0:
            raise ValueError(
                f"Need 0 <= beta1_min <= beta1 <= beta1_max < 1, got "
                f"{beta1_min}, {beta1}, {beta1_max}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if cosine_scope not in ("tensor", "global"):
            raise ValueError(f"Invalid cosine_scope: {cosine_scope}")
        if not cos_eps > 0.0:
            raise ValueError(f"Invalid cos_eps: {cos_eps}")
        if bias_correction not in ("base", "exact"):
            raise ValueError(f"Invalid bias_correction: {bias_correction}")
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, lam=lam,
                        beta1_min=beta1_min, beta1_max=beta1_max,
                        weight_decay=weight_decay,
                        cosine_scope=cosine_scope, cos_eps=cos_eps,
                        bias_correction=bias_correction)
        super().__init__(params, defaults)

    @staticmethod
    def _cosine(dot, g_norm, prev_norm, cos_eps):
        return dot / (g_norm * prev_norm + cos_eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # ---- Pass 1: collect gradients, init state, compute cosines -------
        entries = []          # (group, p, g, state)
        g_dot = g_sq = prev_sq = None   # accumulators for the global scope
        any_global = any(gr["cosine_scope"] == "global" for gr in self.param_groups)
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                st = self.state[p]
                if len(st) == 0:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    st["g_prev"] = torch.zeros_like(p)
                    st["beta1_t"] = torch.full((), group["beta1"], dtype=p.dtype,
                                               device=p.device)
                    st["cc"] = torch.zeros((), dtype=p.dtype, device=p.device)
                    st["P"] = torch.ones((), dtype=p.dtype, device=p.device)
                entries.append((group, p, g, st))
                if any_global:
                    gp = st["g_prev"]
                    d = torch.sum(g * gp)
                    a = torch.sum(g * g)
                    b = torch.sum(gp * gp)
                    g_dot = d if g_dot is None else g_dot + d.to(g_dot.device)
                    g_sq = a if g_sq is None else g_sq + a.to(g_sq.device)
                    prev_sq = b if prev_sq is None else prev_sq + b.to(prev_sq.device)

        global_cc = None
        if any_global and g_dot is not None:
            # cos_eps taken from the first global-scope group
            ce = next(gr["cos_eps"] for gr in self.param_groups
                      if gr["cosine_scope"] == "global")
            global_cc = self._cosine(g_dot, g_sq.sqrt(), prev_sq.sqrt(), ce)

        # ---- Pass 2: update ------------------------------------------------
        for group, p, g, st in entries:
            b1 = group["beta1"]
            b2 = group["beta2"]
            eps = group["eps"]
            lam = group["lam"]

            if group["cosine_scope"] == "global":
                cc = global_cc.to(p.device, p.dtype)
            else:
                gp = st["g_prev"]
                cc = self._cosine(torch.sum(g * gp), g.norm(), gp.norm(),
                                  group["cos_eps"])

            st["t"] += 1
            t = st["t"]
            b1t = (b1 * (1.0 + lam * cc)).clamp(group["beta1_min"],
                                                 group["beta1_max"])

            m, v = st["m"], st["v"]
            m.lerp_(g, 1.0 - b1t)                  # = b1t*m + (1-b1t)*g
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            bias2 = 1.0 - b2 ** t
            denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
            if group["bias_correction"] == "base":
                bias1 = 1.0 - b1 ** t              # base beta1 (paper, Alg. 1)
                p.addcdiv_(m, denom, value=-group["lr"] / bias1)
            else:
                # ABLATION (Protocol v2, Exp. 3): exact normalizer 1 - P_t with
                # P_t = prod_s beta1_s, as used by Adam-NV.
                P = (st["P"] * b1t).clamp(min=_MIN_PRODUCT)
                st["P"] = P
                m_hat = m / (1.0 - P).clamp(min=eps)
                p.addcdiv_(m_hat, denom, value=-group["lr"])

            st["g_prev"].copy_(g)
            st["beta1_t"] = b1t.detach()
            st["cc"] = cc.detach()
        return loss

    # ---- diagnostics (call once per epoch; syncs with the device) ---------
    def _stats(self, key):
        vals = [st[key].item() for group in self.param_groups
                for p in group["params"] if (st := self.state.get(p)) and key in st]
        if not vals:
            nan = float("nan")
            return nan, nan, nan
        return sum(vals) / len(vals), min(vals), max(vals)

    def get_beta1_stats(self):
        """(mean, min, max) of the last-step beta1_t over parameter tensors."""
        return self._stats("beta1_t")

    def get_cc_stats(self):
        """(mean, min, max) of the last-step consecutive cosine over tensors."""
        return self._stats("cc")

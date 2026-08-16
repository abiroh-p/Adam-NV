"""
Adam-NV: Noise-Aware Variance Momentum Optimizer
=================================================
Drop-in replacement for torch.optim.Adam.

Usage:
    from adam_nv import AdamNV
    opt = AdamNV(model.parameters(), lr=1e-3)

Changes from v1:
    [FIX 1] Per-element beta_1 update uses explicit two-line form to avoid
            ambiguous in-place semantics with tensor alpha argument.
    [FIX 2] Added P underflow guard: P is clamped to [min_product, 1] each
            step to prevent float32 collapse after many steps with low beta_1.
            Also logs a warning (once) if underflow is detected.
    [FIX 3] Docstring clarified: gamma=0 reduces to Adam exactly because
            P_t = beta_base^t recovers standard bias correction (1 - beta_1^t).
    [FIX 4] Explicit comment that sigma^2 uses moments from t-1 (intentional).
    [FIX 5] Per-element nature of beta_1 is explicitly documented.
"""

import math
import warnings
import torch
from torch.optim import Optimizer


# Minimum value P is allowed to reach before we warn about underflow.
# float32 has ~7 decimal digits of precision; below 1e-7 bias correction
# becomes unreliable.
_MIN_PRODUCT = 1e-7


class AdamNV(Optimizer):
    """
    Adam-NV Optimizer — Noise-Aware Variance Momentum.

    Adapts beta_1 **per element** based on each weight's local gradient noise:
        - High gradient noise  → low beta_1  (trust raw gradient more)
        - Low gradient noise   → high beta_1 (trust accumulated history more)

    Because beta_1 varies per step, the standard bias correction
    (1 - beta_1^t) is invalid (it assumes a fixed beta_1). Adam-NV instead
    maintains a running product P_t = prod_{s=1}^{t} beta_1(s) per element,
    giving an exact bias correction: m_hat = m / (1 - P_t).

    When gamma=0: scale=1 everywhere, beta_1=beta_base (constant), and
    P_t = beta_base^t, so (1 - P_t) = (1 - beta_base^t), which exactly
    matches standard Adam bias correction. Adam-NV reduces exactly to Adam.

    Args:
        params       : Iterable of parameters or param groups.
        lr           : Learning rate (eta). Default: 1e-3.
        beta_base    : Maximum beta_1 (used when noise is zero). Default: 0.9.
        beta_min     : Minimum beta_1 (floor for stability). Default: 0.1.
        beta_2       : Fixed second-moment decay. Default: 0.999.
        gamma        : Noise sensitivity. gamma=0 → reduces exactly to Adam.
                       Default: 1.0.
        alpha        : EMA smoothing for noise signal rho. Default: 0.95.
        k            : Warm-up suppression constant. Default: 100.
        lam          : Denominator stabilizer (lambda). Default: 0.1.
        eps          : Numerical stability epsilon. Default: 1e-8.
        weight_decay : L2 regularization. Default: 0.
    """

    def __init__(self, params, lr=1e-3, beta_base=0.9, beta_min=0.1,
                 beta_2=0.999, gamma=1.0, alpha=0.95, k=100,
                 lam=0.1, eps=1e-8, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_base <= 1.0:
            raise ValueError(f"Invalid beta_base: {beta_base}")
        if not 0.0 <= beta_min <= beta_base:
            raise ValueError(f"Invalid beta_min: {beta_min} (must be <= beta_base)")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"Invalid beta_2: {beta_2}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= lam:
            raise ValueError(f"Invalid lam: {lam}")

        defaults = dict(lr=lr, beta_base=beta_base, beta_min=beta_min,
                        beta_2=beta_2, gamma=gamma, alpha=alpha,
                        k=k, lam=lam, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Track whether we've already warned about P underflow (warn once).
        self._underflow_warned = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']
            beta_base = group['beta_base']
            beta_min  = group['beta_min']
            beta_2    = group['beta_2']
            gamma     = group['gamma']
            alpha_ema = group['alpha']
            k         = group['k']
            lam       = group['lam']
            eps       = group['eps']
            wd        = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if wd != 0.0:
                    # L2 regularization: creates a new tensor, p.grad is NOT modified.
                    # This is coupled weight decay (not AdamW-style decoupled decay).
                    # All baseline comparisons should use the same wd mode for fairness.
                    g = p.grad.add(p.data, alpha=wd)

                state = self.state[p]

                # ── INITIALIZE STATE ──────────────────────────────────────────
                if len(state) == 0:
                    state['t']   = 0
                    state['m']   = torch.zeros_like(p)  # first moment (per-element)
                    state['v']   = torch.zeros_like(p)  # second moment (per-element)
                    state['rho'] = torch.zeros_like(p)  # smoothed noise (per-element)
                    state['P']   = torch.ones_like(p)   # running product of beta_1 (per-element)

                t   = state['t'] + 1
                m   = state['m']
                v   = state['v']
                rho = state['rho']
                P   = state['P']

                # ── STEP 1: Gradient Noise Signal ─────────────────────────────
                # NOTE: m and v here are from step t-1 (intentional).
                # Using the current gradient to compute the noise signal that
                # determines how much to trust the current gradient would be
                # circular; stale moments give a stable, causal estimate.
                #
                # sigma^2_t = max(v_{t-1} - m^2_{t-1}, 0) / (v_{t-1} + lam*|m_{t-1}| + eps)
                # sigma^2 in [0, 1]: measures gradient variance relative to mean.
                numerator   = torch.clamp(v - m.pow(2), min=0.0)
                denominator = v + lam * m.abs() + eps
                sigma2      = numerator / denominator

                # ── STEP 2: Warm-up Rectification ────────────────────────────
                # Suppresses noise signal when moment estimates are young
                # (unreliable in early steps). Approaches 1 as t → ∞.
                warmup_factor = t / (t + k)
                sigma2_tilde  = warmup_factor * sigma2

                # ── STEP 3: Smooth Noise Signal ───────────────────────────────
                # rho_t = alpha * rho_{t-1} + (1 - alpha) * sigma2_tilde
                rho.mul_(alpha_ema).add_(sigma2_tilde, alpha=(1.0 - alpha_ema))

                # ── STEP 4: Compute Adaptive beta_1 (per-element tensor) ──────
                # beta_1(t) = beta_min + (beta_base - beta_min) * max(1 - gamma*rho_t, 0)
                # Each element of each parameter gets its own beta_1 value,
                # consistent with Adam's per-element second moment v.
                scale  = torch.clamp(1.0 - gamma * rho, min=0.0)
                beta_1 = beta_min + (beta_base - beta_min) * scale  # shape == p.shape

                # ── STEP 5: Update First and Second Moments ───────────────────
                # FIX 1: Two-line form avoids ambiguous in-place semantics when
                # alpha argument is a tensor (not a scalar).
                #
                # m_t = beta_1(t) * m_{t-1} + (1 - beta_1(t)) * g_t
                m.mul_(beta_1)
                m.add_((1.0 - beta_1) * g)

                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
                v.mul_(beta_2).addcmul_(g, g, value=(1.0 - beta_2))

                # ── STEP 6: Bias Correction ───────────────────────────────────
                # Because beta_1 varies per step and per element, the standard
                # correction (1 - beta_1^t) is invalid. We track the running
                # product P_t = prod_{s=1}^{t} beta_1(s) for an exact correction.
                #
                # FIX 2: Clamp P to avoid float32 underflow after many steps
                # with low beta_1 (e.g., beta_min=0.1 → P≈0 after ~160 steps),
                # which would cause m_hat to blow up.
                P.mul_(beta_1)
                P.clamp_(min=_MIN_PRODUCT)

                # Warn once if underflow pressure is detected.
                if not self._underflow_warned and P.min().item() <= _MIN_PRODUCT * 10:
                    warnings.warn(
                        "Adam-NV: P (running product of beta_1) is approaching "
                        "the underflow floor. This is expected with low beta_min "
                        "over many steps. Bias correction is clamped and stable.",
                        RuntimeWarning, stacklevel=2
                    )
                    self._underflow_warned = True

                m_hat = m / torch.clamp(1.0 - P, min=eps)           # first moment bias-corrected
                v_hat = v / (1.0 - beta_2 ** t)        # second moment bias-corrected

                # ── STEP 7: Parameter Update ──────────────────────────────────
                # theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

                # Persist state
                state['t']   = t
                state['m']   = m
                state['v']   = v
                state['rho'] = rho
                state['P']   = P

        return loss

    # ── DIAGNOSTIC UTILITIES ──────────────────────────────────────────────────

    def get_beta1_stats(self):
        """
        Returns (mean, min, max) of current beta_1 values across all parameters.
        Each value is the per-parameter *mean* of its per-element beta_1 tensor.
        So min/max reflect the range across parameters, not across individual elements.
        Use this to verify the adaptive range is being exercised during training
        (min and max should differ clearly from each other and from beta_base).
        """
        all_beta1 = []
        for group in self.param_groups:
            beta_base = group['beta_base']
            beta_min  = group['beta_min']
            gamma     = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                scale  = torch.clamp(1.0 - gamma * state['rho'], min=0.0)
                beta_1 = beta_min + (beta_base - beta_min) * scale
                all_beta1.append(beta_1.mean().item())
        if not all_beta1:
            return None, None, None
        return (
            sum(all_beta1) / len(all_beta1),
            min(all_beta1),
            max(all_beta1),
        )

    def get_sigma2_stats(self):
        """
        Returns (mean, min, max) of current per-element sigma^2 values.
        Should stay in [0, 1] by construction. Use to verify noise signal
        is active and not collapsing to 0 or saturating at 1.
        """
        all_sigma2 = []
        for group in self.param_groups:
            lam = group['lam']
            eps = group['eps']
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                m, v   = state['m'], state['v']
                num    = torch.clamp(v - m.pow(2), min=0.0)
                denom  = v + lam * m.abs() + eps
                all_sigma2.append((num / denom).mean().item())
        if not all_sigma2:
            return None, None, None
        return (
            sum(all_sigma2) / len(all_sigma2),
            min(all_sigma2),
            max(all_sigma2),
        )

    def get_P_stats(self):
        """
        Returns (mean, min, max) of running product P across all parameters.
        Monitor this to catch underflow: if min approaches _MIN_PRODUCT (1e-7),
        the underflow clamp has engaged. This is expected behavior, not a bug.
        """
        all_P = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                all_P.append(state['P'].mean().item())
        if not all_P:
            return None, None, None
        return (
            sum(all_P) / len(all_P),
            min(all_P),
            max(all_P),
        )

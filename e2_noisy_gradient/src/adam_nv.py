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
    [FIX 6] Added optional decoupled v_noise: the Phase 1.6 beta_2 sweep
            (see project report, Section 5) found that beta_2 controls two
            things at once when v is shared between the noise estimator and
            the parameter update -- how fast the noise signal tracks a real
            change in gradient statistics, AND how much the update's own
            step-size denominator is smoothed. Shrinking beta_2 fixed a
            deficit under fast noise oscillation (intermittent schedule) but
            introduced a NEW deficit under slow, smooth noise drift (ramp-up)
            that was not there before -- a trade-off, not a free fix. Setting
            beta_2_noise (default None) gives the noise estimator its own v
            buffer with its own decay, so noise-tracking speed can be tuned
            without touching the update's step-size normalization at all.
            Default (None) reproduces every prior Adam-NV result exactly --
            this is strictly additive, not a behavior change.
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

    Adapts beta_1 **per element** based on a local gradient-noise heuristic:
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
        beta_2       : Fixed second-moment decay, used for the parameter
                       update's own step-size denominator (v_hat). Default: 0.999.
        beta_2_noise : Optional, separate second-moment decay used ONLY for
                       the noise estimator's v buffer (v_noise), isolated
                       from the parameter update. Default: None, meaning
                       v_noise = v (the original, coupled behavior --
                       exactly reproduces every prior Adam-NV result). Set
                       this to decouple noise-tracking speed from the
                       update's step-size normalization; see FIX 6 above.
        gamma        : Noise sensitivity. gamma=0 → reduces exactly to Adam.
                       Default: 1.0.
        response_mode: `distrust` preserves the original rule (higher score
                       lowers beta_1). `smooth` tests the alternative momentum
                       hypothesis (higher score raises beta_1 from beta_base
                       toward beta_max). Default: `distrust`.
        beta_max     : Upper beta_1 limit used only by response_mode=`smooth`.
                       Default: 0.999.
        alpha        : EMA smoothing for noise signal rho. Default: 0.95.
        k            : Warm-up suppression constant. Default: 100.
        lam          : Denominator stabilizer (lambda). Default: 0.1.
        eps          : Numerical stability epsilon. Default: 1e-8.
        weight_decay : L2 regularization. Default: 0.
    """

    def __init__(self, params, lr=1e-3, beta_base=0.9, beta_min=0.1,
                 beta_2=0.999, beta_2_noise=None, gamma=1.0, alpha=0.95, k=100,
                 lam=0.1, eps=1e-8, weight_decay=0.0,
                 response_mode="distrust", beta_max=0.999):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta_base <= 1.0:
            raise ValueError(f"Invalid beta_base: {beta_base}")
        if not 0.0 <= beta_min <= beta_base:
            raise ValueError(f"Invalid beta_min: {beta_min} (must be <= beta_base)")
        if not beta_base <= beta_max < 1.0:
            raise ValueError(f"Invalid beta_max: {beta_max} (must satisfy beta_base <= beta_max < 1)")
        if response_mode not in ("distrust", "smooth"):
            raise ValueError(f"Invalid response_mode: {response_mode}")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"Invalid beta_2: {beta_2}")
        if beta_2_noise is not None and not 0.0 <= beta_2_noise < 1.0:
            raise ValueError(f"Invalid beta_2_noise: {beta_2_noise}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= lam:
            raise ValueError(f"Invalid lam: {lam}")

        defaults = dict(lr=lr, beta_base=beta_base, beta_min=beta_min,
                        beta_2=beta_2, beta_2_noise=beta_2_noise, gamma=gamma,
                        alpha=alpha, k=k, lam=lam, eps=eps, weight_decay=weight_decay,
                        response_mode=response_mode, beta_max=beta_max)
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
            beta_max  = group['beta_max']
            response_mode = group['response_mode']
            beta_2    = group['beta_2']
            beta_2_noise = group['beta_2_noise']
            decoupled = beta_2_noise is not None
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
                    if decoupled:
                        # FIX 6: separate second-moment buffer dedicated to the
                        # noise estimator, isolated from the update's v above.
                        state['v_noise'] = torch.zeros_like(p)

                t   = state['t'] + 1
                m   = state['m']
                v   = state['v']
                rho = state['rho']
                P   = state['P']

                # ── STEP 1: Gradient Noise Signal ─────────────────────────────
                # NOTE: m and v_for_noise here are from step t-1 (intentional).
                # Using the current gradient to compute the noise signal that
                # determines how much to trust the current gradient would be
                # circular; stale moments give a stable, causal estimate.
                #
                # FIX 6: v_for_noise is v_noise when decoupled (its own
                # beta_2_noise-governed buffer), or v itself when not (original
                # coupled behavior, default). Either way this is the STALE
                # value from before this step's moment updates (Step 5).
                #
                # sigma^2_t = max(v_for_noise_{t-1} - m^2_{t-1}, 0) / (v_for_noise_{t-1} + lam*|m_{t-1}| + eps)
                # sigma^2 in [0, 1] is a heuristic noise score. Because m and
                # v_for_noise can use different decay rates and are read before
                # matching bias corrections, it is not a calibrated variance or
                # noise-to-signal estimator.
                v_for_noise = state['v_noise'] if decoupled else v
                numerator   = torch.clamp(v_for_noise - m.pow(2), min=0.0)
                denominator = v_for_noise + lam * m.abs() + eps
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
                # Each element of each parameter gets its own beta_1 value,
                # consistent with Adam's per-element second moment v.
                if response_mode == "distrust":
                    # Original Adam-NV rule: high score -> lower beta_1.
                    scale = torch.clamp(1.0 - gamma * rho, min=0.0)
                    beta_1 = beta_min + (beta_base - beta_min) * scale
                else:
                    # Alternative smoothing hypothesis: preserve Adam's 0.9
                    # low-noise baseline, then increase momentum as the score rises.
                    scale = torch.clamp(gamma * rho, min=0.0, max=1.0)
                    beta_1 = beta_base + (beta_max - beta_base) * scale

                # ── STEP 5: Update First and Second Moments ───────────────────
                # FIX 1: Two-line form avoids ambiguous in-place semantics when
                # alpha argument is a tensor (not a scalar).
                #
                # m_t = beta_1(t) * m_{t-1} + (1 - beta_1(t)) * g_t
                m.mul_(beta_1)
                m.add_((1.0 - beta_1) * g)

                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
                v.mul_(beta_2).addcmul_(g, g, value=(1.0 - beta_2))

                # FIX 6: v_noise advances on its own beta_2_noise decay, using
                # the SAME gradient g, completely independent of v above. This
                # is the only place v_noise is touched other than Step 1 (read)
                # and initialization -- it never feeds into m_hat/v_hat/the
                # parameter update, only back into Step 1 next step.
                if decoupled:
                    v_noise = state['v_noise']
                    v_noise.mul_(beta_2_noise).addcmul_(g, g, value=(1.0 - beta_2_noise))

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
                if decoupled:
                    state['v_noise'] = v_noise

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
            beta_max  = group['beta_max']
            response_mode = group['response_mode']
            gamma     = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                if response_mode == "distrust":
                    scale = torch.clamp(1.0 - gamma * state['rho'], min=0.0)
                    beta_1 = beta_min + (beta_base - beta_min) * scale
                else:
                    scale = torch.clamp(gamma * state['rho'], min=0.0, max=1.0)
                    beta_1 = beta_base + (beta_max - beta_base) * scale
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
        Returns (mean, min, max) of the current per-element sigma^2 heuristic.
        Should stay in [0, 1] by construction. Use to verify the heuristic
        is active and not collapsing to 0 or saturating at 1.

        FIX 6: uses v_noise (not v) when beta_2_noise is set, matching
        what step() actually feeds into the noise signal -- otherwise this
        diagnostic would silently report the wrong second moment.
        """
        all_sigma2 = []
        for group in self.param_groups:
            lam = group['lam']
            eps = group['eps']
            decoupled = group['beta_2_noise'] is not None
            for p in group['params']:
                state = self.state[p]
                if not state:
                    continue
                m = state['m']
                v_for_noise = state['v_noise'] if decoupled else state['v']
                num    = torch.clamp(v_for_noise - m.pow(2), min=0.0)
                denom  = v_for_noise + lam * m.abs() + eps
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

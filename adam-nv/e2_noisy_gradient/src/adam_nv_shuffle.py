"""Adam-NV with a beta_1 shuffle control (Exp. 11).

AdamNVShuffle is AdamNV plus one intervention. After the controller has computed its
per-element beta_1 tensor for a parameter, the tensor is permuted across the elements of
that parameter before it is applied to the first-moment update:

  shuffle="none"   no permutation; bit-identical to AdamNV (used as a regression check)
  shuffle="step"   a fresh random permutation every step (per-step distribution preserved,
                   element-wise link to the element's own noise score destroyed, and an
                   element's beta_1 fluctuates from step to step)
  shuffle="fixed"  one random permutation per parameter, drawn once (the beta_1 trajectory
                   of a source element is applied to a fixed, unrelated target element;
                   temporal structure preserved, link to the target's own gradients destroyed)

The running product P_t uses the beta_1 values actually applied, so the bias correction stays
exact. The controller state rho is computed from each element's own gradients as before.
get_beta1_stats() still reports the controller's own values (per-tensor means), which a
permutation within a tensor leaves unchanged.

Optionally the element-level beta_1 distribution (pre-shuffle) is appended to a JSON-lines
file every `dist_log_every` steps (quantiles, extremes, and fractions below thresholds).
adam_nv.py itself is NOT modified, so its recorded hash stays valid for Exp. 1-10.
"""

import json
import torch

from adam_nv import AdamNV, _MIN_PRODUCT  # noqa: F401  (_MIN_PRODUCT used by the copied step)
import warnings  # used by the copied step

_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


class AdamNVShuffle(AdamNV):
    def __init__(self, params, *args, shuffle="step", shuffle_seed=0,
                 dist_log_path=None, dist_log_every=391, **kwargs):
        if shuffle not in ("none", "step", "fixed"):
            raise ValueError(f"Invalid shuffle: {shuffle!r} (use 'none', 'step' or 'fixed')")
        super().__init__(params, *args, **kwargs)
        self._shuffle = shuffle
        self._shuffle_seed = int(shuffle_seed)
        self._gens = {}
        self._dist_log_path = dist_log_path
        self._dist_log_every = int(dist_log_every)
        self._step_count = 0

    def _generator(self, device):
        key = str(device)
        if key not in self._gens:
            g = torch.Generator(device=device)
            g.manual_seed(self._shuffle_seed)
            self._gens[key] = g
        return self._gens[key]

    def _apply_shuffle(self, p, state, beta_1):
        if self._shuffle == "none" or beta_1.dim() == 0 or beta_1.numel() < 2:
            return beta_1
        n = beta_1.numel()
        if self._shuffle == "step":
            perm = torch.randperm(n, generator=self._generator(beta_1.device), device=beta_1.device)
        else:
            if "perm" not in state:
                state["perm"] = torch.randperm(n, generator=self._generator(beta_1.device),
                                               device=beta_1.device)
            perm = state["perm"]
        return beta_1.reshape(-1)[perm].reshape(beta_1.shape)

    def _write_distribution(self, parts):
        flat = torch.cat(parts)
        srt = torch.sort(flat).values
        n = srt.numel()
        rec = {"step": self._step_count + 1, "n_elements": n, "mean": float(flat.mean()),
               "min": float(srt[0]), "max": float(srt[-1]),
               "frac_below_0.5": float((flat < 0.5).float().mean()),
               "frac_below_0.7": float((flat < 0.7).float().mean())}
        for q in _QUANTILES:
            rec[f"q{int(round(q * 100)):02d}"] = float(srt[min(n - 1, int(q * (n - 1)))])
        with open(self._dist_log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        log_now = (self._dist_log_path is not None
                   and (self._step_count + 1) % self._dist_log_every == 0)
        dist_buf = []

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
            bias_correction = group['bias_correction']
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

                # ── SHUFFLE CONTROL (Exp. 11) ─────────────────────────────────
                # beta_1_ctrl is the controller's own per-element value. The value
                # actually applied below is a permutation of it within this tensor, so
                # the per-step distribution of beta_1 is preserved but its element-wise
                # link to the element's own noise score is destroyed. shuffle="none"
                # leaves beta_1 untouched and is bit-identical to AdamNV.
                if log_now:
                    dist_buf.append(beta_1.detach().reshape(-1).float())
                beta_1 = self._apply_shuffle(p, state, beta_1)

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

                if bias_correction == "running_product":
                    m_hat = m / torch.clamp(1.0 - P, min=eps)       # first moment bias-corrected
                else:
                    # ablation only: standard Adam normalizer, ignores realized beta_1
                    m_hat = m / (1.0 - beta_base ** t)
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

        if log_now and dist_buf:
            self._write_distribution(dist_buf)
        self._step_count += 1
        return loss


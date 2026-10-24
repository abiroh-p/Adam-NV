"""
E2 · Noise Injection Module
============================
Implements norm-scaled Gaussian gradient noise:

    g_noisy = g + σ · ‖g‖_F · ε̂,   ε̂ = ε / ‖ε‖_F,  ε ~ N(0, I)

where ‖g‖_F is the per-tensor Frobenius norm.

Design decisions
----------------
1. Standalone function, not embedded in the training loop.
   Called identically for every optimizer between loss.backward()
   and optimizer.step(). Matched seeds reproduce the same Gaussian
   directions across optimizer runs. The final perturbation tensors are
   not byte-identical because each direction is scaled by that run's own
   gradient norm.

2. Per-tensor norm scaling (not global, not per-element).
   Using each tensor's own Frobenius norm makes σ a true SNR ratio
   that is comparable across layers of different sizes and magnitudes.
   A conv weight tensor and a bias tensor both get noise whose scale
   is σ × their own gradient magnitude — not a fixed additive offset
   that would dominate small tensors.

3. In-place modification of p.grad.
   After loss.backward(), PyTorch stores gradients in p.grad.
   Modifying p.grad in-place before optimizer.step() means the
   optimizer sees the noisy gradient as if it were the real gradient.
   No special handling needed inside any optimizer.

4. σ=0.0 is a no-op (explicit early return).
   This lets the same training loop handle clean and noisy runs
   without branching — just pass σ=0.0 for the clean baseline.

Usage
-----
    from noise import inject_noise, compute_snr

    loss.backward()
    inject_noise(model, sigma=0.1)   # modifies p.grad in-place
    optimizer.step()

SNR estimation
--------------
    snr = compute_snr(model)   # call BEFORE inject_noise
    # returns mean SNR across all parameter tensors
"""

import torch
import torch.nn as nn
from typing import Optional


@torch.no_grad()
def inject_noise(model: nn.Module, sigma: float, generator: Optional[torch.Generator] = None) -> None:
    """
    Inject norm-scaled Gaussian noise into all parameter gradients.

    Implements: g_noisy = g + σ · ‖g‖_F · ε̂,  ε̂ = ε / ‖ε‖_F,  ε ~ N(0, I)

    ε is normalized to unit Frobenius norm before scaling, so the
    injected noise magnitude is EXACTLY σ · ‖g‖_F for every tensor,
    regardless of tensor size. Without this normalization, a raw
    N(0,I) sample of n elements has ‖ε‖_F ≈ √n, which would inflate
    the noise magnitude by √n — differently for every tensor shape.

    Call this AFTER loss.backward() and BEFORE optimizer.step().
    Modifies p.grad in-place. Parameters with no gradient are skipped.

    Args:
        model     : The model whose parameter gradients to perturb.
        sigma     : Noise scale (SNR ratio). 0.0 is a no-op.
                    Typical values: 0.05 (mild), 0.5 (strong).
        generator : Optional torch.Generator for reproducible noise.
                    Pass a seeded generator to get matched Gaussian
                    directions across optimizer comparison runs. Actual
                    magnitudes remain optimizer-dependent because they are
                    scaled by each run's gradient norm.

    Returns:
        None (modifies p.grad in-place)

    Example:
        loss.backward()
        inject_noise(model, sigma=0.1, generator=noise_gen)
        optimizer.step()
    """
    if sigma == 0.0:
        return  # clean baseline — no-op, no overhead

    for p in model.parameters():
        if p.grad is None:
            continue

        g = p.grad

        # Per-tensor Frobenius norm: scalar, shape ()
        # Frobenius norm of a tensor = sqrt(sum of squared elements)
        # torch.norm defaults to Frobenius for matrices and tensors
        g_norm = torch.norm(g)

        if g_norm == 0.0:
            # Zero gradient — noise would be zero regardless; skip
            continue

        # Sample ε ~ N(0, I), same shape as gradient tensor
        if generator is not None:
            eps = torch.randn(g.shape, device=g.device, dtype=g.dtype,
                              generator=generator)
        else:
            eps = torch.randn_like(g)

        # Normalize ε to unit Frobenius norm: ε̂ = ε / ‖ε‖_F
        # Without this, ‖ε‖_F ≈ √n_elements (chi distribution), so the
        # noise added would be σ·‖g‖_F·√n instead of σ·‖g‖_F — off by
        # a tensor-size-dependent factor that breaks the constant-SNR
        # property this function is supposed to guarantee.
        eps_norm = torch.norm(eps)
        if eps_norm == 0.0:
            # Astronomically unlikely for continuous Gaussian, but
            # guard against div-by-zero regardless.
            continue
        eps = eps / eps_norm

        # g_noisy = g + σ · ‖g‖_F · ε̂   (‖ε̂‖_F == 1 exactly)
        # In-place addition: p.grad += sigma * g_norm * eps_normalized
        p.grad.add_(eps, alpha=sigma * g_norm.item())


@torch.no_grad()
def compute_snr(model: nn.Module) -> dict:
    """
    Estimate the gradient SNR across all parameter tensors.

    SNR here is defined as:
        SNR_tensor = ‖g‖_F² / (σ_noise² · ‖g‖_F²) = 1 / σ²

    But since we don't know the injected σ from inside this function,
    we instead return the per-tensor gradient norm statistics, which
    serve as a proxy for the signal strength. The ratio of signal to
    injected noise is then 1/σ by construction of our noise formula.

    More practically for calibration: this function returns statistics
    that let you verify whether σ=0.05 and σ=0.5 produce meaningfully
    different perturbation magnitudes relative to real gradient norms.

    Args:
        model : The model to inspect (call after loss.backward(),
                before inject_noise).

    Returns:
        dict with keys:
            'mean_grad_norm'   : mean ‖g‖_F across all param tensors
            'min_grad_norm'    : minimum ‖g‖_F
            'max_grad_norm'    : maximum ‖g‖_F
            'n_params'         : number of parameter tensors with grad
    """
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(torch.norm(p.grad).item())

    if not norms:
        return {'mean_grad_norm': 0.0, 'min_grad_norm': 0.0,
                'max_grad_norm': 0.0, 'n_params': 0}

    return {
        'mean_grad_norm': sum(norms) / len(norms),
        'min_grad_norm': min(norms),
        'max_grad_norm': max(norms),
        'n_params': len(norms),
    }


def effective_snr(grad_norm: float, sigma: float) -> float:
    """
    Compute the effective SNR for a given gradient norm and sigma.

    SNR = ‖g‖_F / (σ · ‖g‖_F) = 1 / σ

    This is constant by construction of our noise formula — the
    noise magnitude is always σ × ‖g‖_F, so the SNR is exactly 1/σ
    regardless of the actual gradient magnitude. This is the point of
    norm-scaling: SNR is controlled entirely by σ, not by the gradient.

    In practice, what varies across layers is the absolute noise
    magnitude (σ · ‖g‖_F), not the SNR. This function is provided
    for documentation clarity and use in calibration reports.

    Args:
        grad_norm : ‖g‖_F for a parameter tensor
        sigma     : injected noise scale

    Returns:
        SNR as float (= 1/sigma for norm-scaled noise)
    """
    if sigma == 0.0:
        return float('inf')
    return 1.0 / sigma

"""
Phase 1.5 · Non-Stationary Noise Schedules
============================================
E2/E7/E11/E12 and the Phase 1.4 baseline-noise suite all hold sigma
CONSTANT for the full 60-epoch run. That directly tests "does Adam-NV
help under a fixed noise level" — and the answer so far (see the E2/E11
report) is no, not measurably. It does NOT test the premise Adam-NV
was actually built on: that gradient reliability SHIFTS over the
course of training (paper Introduction — batch composition, label
quality, and network state all drift). A fixed beta_1 baseline can
settle into whatever constant works for one constant noise level;
there's no moment where "static" is structurally penalized and
"adaptive" isn't.

This module supplies per-epoch sigma(t) for schedules where the noise
level itself changes within a single run, so that question can finally
be asked directly.

Schedules
---------
constant       : sigma(t) = sigma_max for all epochs. Included only as
                 an internal sanity check (should reproduce Phase 1.4
                 behavior exactly) — not intended as a new condition.
phased         : three phases — clean / noisy / clean. Tests whether
                 Adam-NV recovers faster than fixed-beta1 optimizers
                 when noise SHUTS OFF, and degrades less when it
                 SWITCHES ON, compared to a run that never had to
                 transition.
ramp_up        : sigma rises linearly from 0 to sigma_max over the
                 full run. Tests gradual drift, e.g. a dataset that
                 becomes progressively noisier (closer to the
                 "long training schedules where noise drifts over
                 time" motivating scenario in the paper).
ramp_updown    : triangular — linear 0 -> sigma_max -> 0. Tests
                 whether Adam-NV tracks a *smooth, continuous* change
                 in noise level, not just a step change.
intermittent   : alternating fixed-length blocks of noisy / clean
                 epochs (burst_epochs on, gap_epochs off, repeating).
                 Tests repeated, fast on/off transitions — closest to
                 "noisy mini-batches interspersed with clean ones" in
                 spirit, at epoch granularity.

Design decisions
----------------
1. Schedules are precomputed once per run as a plain list[float] of
   length `total_epochs`, not recomputed inside the training loop.
   Same reasoning as noise.py's standalone-function design: compute
   once, call identically every epoch, no room for schedule logic to
   drift between optimizers being compared.

2. Epoch-granularity, not step-granularity. E2/E7/E11/E12 all log at
   epoch resolution, and Adam-NV's beta1_mean/rho_mean diagnostics are
   epoch-level too. Matching that resolution means sigma(t) can be
   directly correlated against beta1_mean per epoch in the same log
   file — the central diagnostic for this suite (does beta1 track a
   changing environment IN REAL TIME within a single run, not just
   differ ACROSS separate constant-sigma runs).

3. sigma_max plays the same role as --sigma did in v7's constant-noise
   runs, so existing sigma=2.0 results remain the reference point for
   "how strong is this schedule's peak noise."

Usage
-----
    from noise_schedule import get_sigma_schedule

    sigmas = get_sigma_schedule("phased", total_epochs=60, sigma_max=2.0,
                                 phase_start=1/3, phase_end=2/3)
    # sigmas[epoch - 1] gives sigma for that epoch (1-indexed epochs)
"""

from typing import List


def _phased(total_epochs: int, sigma_max: float,
            phase_start: float, phase_end: float) -> List[float]:
    if not 0.0 <= phase_start < phase_end <= 1.0:
        raise ValueError(
            f"phase_start ({phase_start}) must be < phase_end ({phase_end}), "
            "both in [0, 1]")
    start_epoch = round(phase_start * total_epochs)
    end_epoch = round(phase_end * total_epochs)
    return [
        sigma_max if start_epoch <= e < end_epoch else 0.0
        for e in range(total_epochs)
    ]


def _ramp_up(total_epochs: int, sigma_max: float) -> List[float]:
    if total_epochs <= 1:
        return [sigma_max] * total_epochs
    return [sigma_max * (e / (total_epochs - 1)) for e in range(total_epochs)]


def _ramp_updown(total_epochs: int, sigma_max: float) -> List[float]:
    # Triangular: rises for the first half, falls for the second half.
    # Peak sits at the midpoint epoch regardless of parity.
    if total_epochs <= 1:
        return [sigma_max] * total_epochs
    mid = (total_epochs - 1) / 2.0
    return [
        sigma_max * (1.0 - abs(e - mid) / mid) if mid > 0 else sigma_max
        for e in range(total_epochs)
    ]


def _intermittent(total_epochs: int, sigma_max: float,
                   burst_epochs: int, gap_epochs: int,
                   start_clean: bool) -> List[float]:
    if burst_epochs <= 0 or gap_epochs <= 0:
        raise ValueError("burst_epochs and gap_epochs must both be > 0")
    period = burst_epochs + gap_epochs
    sigmas = []
    for e in range(total_epochs):
        phase = e % period
        if start_clean:
            noisy = phase >= gap_epochs
        else:
            noisy = phase < burst_epochs
        sigmas.append(sigma_max if noisy else 0.0)
    return sigmas


def get_sigma_schedule(schedule: str, total_epochs: int, sigma_max: float,
                        phase_start: float = 1.0 / 3.0,
                        phase_end: float = 2.0 / 3.0,
                        burst_epochs: int = 5,
                        gap_epochs: int = 5,
                        start_clean: bool = True) -> List[float]:
    """
    Precompute sigma(t) for every epoch of a run.

    Args:
        schedule     : One of "constant", "phased", "ramp_up",
                       "ramp_updown", "intermittent".
        total_epochs : Number of epochs in the run (matches --epochs).
        sigma_max    : Peak noise level (same meaning/scale as v7's
                       --sigma for constant-noise runs).
        phase_start, phase_end : Fractional boundaries (0-1) of the
                       noisy phase, used by "phased" only.
        burst_epochs, gap_epochs : Block lengths in epochs, used by
                       "intermittent" only.
        start_clean  : If True, "intermittent" begins with a clean
                       (gap) block; if False, begins with a noisy
                       (burst) block.

    Returns:
        List[float] of length total_epochs. sigmas[e] is the sigma to
        use for epoch e+1 (0-indexed list, 1-indexed epochs — matches
        the `for epoch in range(1, args.epochs + 1)` convention used
        throughout part_b_resnet_v*.py).
    """
    schedule = schedule.lower()
    if schedule == "constant":
        return [sigma_max] * total_epochs
    elif schedule == "phased":
        return _phased(total_epochs, sigma_max, phase_start, phase_end)
    elif schedule == "ramp_up":
        return _ramp_up(total_epochs, sigma_max)
    elif schedule == "ramp_updown":
        return _ramp_updown(total_epochs, sigma_max)
    elif schedule == "intermittent":
        return _intermittent(total_epochs, sigma_max, burst_epochs,
                              gap_epochs, start_clean)
    else:
        raise ValueError(
            "Unknown schedule. Expected one of: constant, phased, "
            f"ramp_up, ramp_updown, intermittent. Got: {schedule}")


def describe_schedule(schedule: str, phase_start: float = 1.0 / 3.0,
                       phase_end: float = 2.0 / 3.0,
                       burst_epochs: int = 5, gap_epochs: int = 5,
                       start_clean: bool = True) -> str:
    """Human-readable one-liner for print statements / run names."""
    schedule = schedule.lower()
    if schedule == "constant":
        return "constant (sanity-check; should match Phase 1.4 exactly)"
    if schedule == "phased":
        return f"phased (clean -> noisy over [{phase_start:.2f}, {phase_end:.2f}] -> clean)"
    if schedule == "ramp_up":
        return "ramp_up (linear 0 -> sigma_max)"
    if schedule == "ramp_updown":
        return "ramp_updown (triangular 0 -> sigma_max -> 0)"
    if schedule == "intermittent":
        first = "clean" if start_clean else "noisy"
        return (f"intermittent ({burst_epochs} noisy / {gap_epochs} clean "
                f"epochs, repeating, starting {first})")
    return schedule

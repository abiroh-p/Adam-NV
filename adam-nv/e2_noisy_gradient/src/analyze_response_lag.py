"""
Response-Dynamics Metrics — onset, overshoot, correction, settling
========================================================================
Computes four precisely-defined metrics from a diagnose_response_lag.py
fine-grained CSV, so results are directly comparable across schedules
and beta_2/beta_2_noise settings rather than eyeballed off a plot.

Per Ms. Pradhan's review (priority #1): this is what actually tests
whether decoupling makes noise-tracking faster at the step level, or
only appears to via the accuracy trade-off in E14/E15.

Metric definitions
-------------------
All four are computed relative to a single transition step T, using
beta1_mean as the primary signal (rho_mean available via --signal rho
for a cross-check, since the mechanism story in the report runs through
rho -> beta_1).

1. ONSET (steps): first step after T where the signal deviates from its
   pre-transition baseline by more than 2x the pre-transition rolling
   std (both computed over the PRE_WINDOW steps immediately before T).
   Using a per-run adaptive threshold, not a fixed absolute one, so
   runs with different natural noise floors (e.g. different
   beta_2_noise values) are compared fairly.

2. OVERSHOOT MAGNITUDE: the peak deviation from pre-transition baseline
   in the WRONG direction (opposite of where the signal should
   eventually settle) within OVERSHOOT_WINDOW steps after T. Reported
   both in absolute beta_1 units and as a fraction of the total
   eventual swing (pre-level vs. the post-window reference level).
   Direction convention: clean->noisy transitions should eventually
   DECREASE beta_1 (more noise -> more distrust), so a wrong-direction
   overshoot there is an INCREASE. noisy->clean transitions should
   eventually INCREASE beta_1; a wrong-direction overshoot there is a
   DECREASE. Passed explicitly via --direction.

3. TIME TO CORRECT DIRECTION (steps): steps from T until the signal
   first re-crosses its own pre-transition baseline level, heading in
   the CORRECT direction -- i.e. the step where the wrong-direction
   overshoot ends and the signal enters correct-direction territory
   for the first time.

4. TIME TO STEADY STATE (steps): standard control-theory settling
   time. First step after which the signal stays within a tolerance
   band (default +/-10% of the total eventual swing, pre-level vs.
   post-window reference) for the REMAINDER of the available window.
   Because "remainder of available window" depends on how much data
   was logged, this number is a lower bound if the window ends before
   true settling -- flagged explicitly in the output when that happens.

Usage
-----
    # Single run:
    python analyze_response_lag.py run1.csv \\
        --direction clean_to_noisy --pre-window 100 --overshoot-window 1000

    # Compare multiple runs side by side (e.g. beta_2=0.999 vs
    # beta_2_noise=0.99 vs beta_2_noise=0.95, same schedule/transition
    # direction):
    python analyze_response_lag.py \\
        ramp_up_beta2_0.999.csv:beta2=0.999 \\
        ramp_up_beta2noise_0.99.csv:beta2_noise=0.99 \\
        ramp_up_beta2noise_0.95.csv:beta2_noise=0.95 \\
        --direction clean_to_noisy --schedule-name ramp_up

Each positional arg can optionally carry a ":label" suffix for the
output table; if omitted, the filename is used.
"""

import argparse
import sys

import numpy as np
import pandas as pd


MIN_VALID_PROTOCOL_VERSION = 2


def validate_protocol(df: pd.DataFrame, path: str, allow_legacy: bool = False) -> None:
    """Reject scheduler-bugged v1 diagnostics unless explicitly overridden."""
    if "protocol_version" not in df.columns:
        if allow_legacy:
            print(
                f"  WARNING: {path} has no protocol_version and may use the invalid "
                "per-minibatch scheduler trajectory.",
                file=sys.stderr,
            )
            return
        raise ValueError(
            f"{path} is an unversioned legacy diagnostic. Those logs were produced "
            "before the scheduler-placement fix and must not support response-time "
            "claims. Rerun diagnose_response_lag.py to create a v2 log. Use "
            "--allow-legacy only for debugging, never for reported results."
        )

    versions = set(pd.to_numeric(df["protocol_version"], errors="coerce").dropna())
    if not versions or min(versions) < MIN_VALID_PROTOCOL_VERSION:
        raise ValueError(
            f"{path} has invalid diagnostic protocol version(s): {sorted(versions)}; "
            f"version {MIN_VALID_PROTOCOL_VERSION} or newer is required."
        )


def find_transition(df: pd.DataFrame) -> int:
    """Returns the global_step of the first sigma_t change in the log."""
    prev = df["sigma_t"].shift(1)
    changed = df[(df["sigma_t"] != prev) & prev.notna()]
    if len(changed) == 0:
        raise ValueError(
            "No sigma_t transition found in this file. For ramp_up, "
            "sigma_t changes every epoch (not every step) -- pick an "
            "epoch boundary inside your --fine-log-start/--fine-log-end "
            "window with --transition-step to force one explicitly.")
    return int(changed.iloc[0]["global_step"])


def compute_metrics(df: pd.DataFrame, transition_step: int, direction: str,
                     signal_col: str, pre_window: int, overshoot_window: int,
                     settle_tol_frac: float):
    df = df.sort_values("global_step").reset_index(drop=True)
    pre = df[df.global_step < transition_step]
    post = df[df.global_step >= transition_step].reset_index(drop=True)

    if len(pre) < pre_window:
        print(f"  WARNING: only {len(pre)} pre-transition steps available "
              f"(wanted {pre_window}) -- baseline/threshold may be noisy.",
              file=sys.stderr)
    pre_tail = pre.tail(pre_window)
    baseline = pre_tail[signal_col].mean()
    pre_std = pre_tail[signal_col].std()
    if pre_std == 0 or np.isnan(pre_std):
        pre_std = 1e-8  # avoid a zero threshold on a perfectly flat pre-window

    post_window_df = post.head(overshoot_window)
    if len(post_window_df) < overshoot_window:
        print(f"  WARNING: only {len(post_window_df)} post-transition steps "
              f"available (wanted {overshoot_window}) -- overshoot/settling "
              f"metrics are lower bounds, not confirmed final values.",
              file=sys.stderr)

    # "wrong" direction per transition type
    # clean_to_noisy: signal should eventually DECREASE -> wrong = increase
    # noisy_to_clean: signal should eventually INCREASE -> wrong = decrease
    wrong_sign = +1 if direction == "clean_to_noisy" else -1

    deviation = (post_window_df[signal_col] - baseline) * wrong_sign
    # deviation > 0 means moving in the WRONG direction

    # --- (1) Onset: first step where |deviation from baseline| > 2*pre_std ---
    onset_mask = (post_window_df[signal_col] - baseline).abs() > 2 * pre_std
    onset_steps = post_window_df.loc[onset_mask, "global_step"]
    onset = int(onset_steps.iloc[0] - transition_step) if len(onset_steps) else None

    # --- (2) Overshoot magnitude: peak wrong-direction deviation ---
    if deviation.max() > 0:
        peak_idx = deviation.idxmax()
        overshoot_abs = float(deviation.loc[peak_idx])
        overshoot_step = int(post_window_df.loc[peak_idx, "global_step"] - transition_step)
    else:
        overshoot_abs = 0.0
        overshoot_step = None

    # reference "post" level for computing total eventual swing (last 10% of window)
    tail_n = max(10, len(post_window_df) // 10)
    post_ref = post_window_df.tail(tail_n)[signal_col].mean()
    total_swing = abs(post_ref - baseline)
    overshoot_frac = (overshoot_abs / total_swing) if total_swing > 1e-8 else float("nan")

    # --- (3) Time to correct direction: first re-cross of baseline in correct direction ---
    correct_mask = deviation < 0  # deviation<0 means now on the CORRECT side of baseline
    # only look after the overshoot peak, so we don't catch pre-onset noise
    if overshoot_step is not None:
        after_peak = post_window_df[post_window_df.global_step - transition_step > overshoot_step]
        after_peak_dev = (after_peak[signal_col] - baseline) * wrong_sign
        crossing = after_peak[after_peak_dev < 0]
        time_to_correct = int(crossing.iloc[0].global_step - transition_step) if len(crossing) else None
    else:
        time_to_correct = 0  # never moved wrong-direction at all

    # --- (4) Time to steady state: settling time within +/- tol of post_ref ---
    tol = settle_tol_frac * total_swing if total_swing > 1e-8 else settle_tol_frac * pre_std
    within_band = (post_window_df[signal_col] - post_ref).abs() <= tol
    # find the LAST step where it's still outside the band; settling = step after that
    outside = post_window_df[~within_band]
    if len(outside) == 0:
        settle_step = 0
    else:
        last_outside_step = outside.iloc[-1].global_step
        settle_step = int(last_outside_step - transition_step)
        if last_outside_step == post_window_df.iloc[-1].global_step:
            settle_step = None  # never actually settled within the logged window

    return {
        "baseline": baseline,
        "pre_std": pre_std,
        "onset_steps": onset,
        "overshoot_abs": overshoot_abs,
        "overshoot_frac_of_swing": overshoot_frac,
        "overshoot_step": overshoot_step,
        "time_to_correct_steps": time_to_correct,
        "time_to_steady_state_steps": settle_step,
        "post_reference_level": post_ref,
        "total_swing": total_swing,
        "n_post_steps_available": len(post_window_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute response-dynamics metrics from fine-grained lag-diagnostic CSVs")
    parser.add_argument("files", nargs="+", help="CSV[:label] ... one or more fine-grained logs to compare")
    parser.add_argument("--direction", required=True, choices=["clean_to_noisy", "noisy_to_clean"],
                        help="What kind of transition this is, determines which direction counts as 'wrong'.")
    parser.add_argument("--signal", default="beta1_mean", choices=["beta1_mean", "rho_mean"],
                        help="Which logged column to analyze. Default beta1_mean.")
    parser.add_argument("--transition-step", type=int, default=None,
                        help="Force a specific global_step as the transition, instead of "
                             "auto-detecting the first sigma_t change. Required for ramp_up "
                             "(sigma changes every epoch, not at a single discrete jump) -- "
                             "pick the global_step at the epoch boundary you want to analyze.")
    parser.add_argument("--pre-window", type=int, default=100,
                        help="Steps before transition used to compute baseline/threshold. Default 100.")
    parser.add_argument("--overshoot-window", type=int, default=1000,
                        help="Steps after transition searched for overshoot/settling. Default 1000.")
    parser.add_argument("--settle-tol-frac", type=float, default=0.10,
                        help="Settling tolerance as a fraction of total swing. Default 0.10 (10%%).")
    parser.add_argument("--schedule-name", default="", help="Label for the printed table header only.")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Allow unversioned pre-fix logs for debugging only. Never use them for reported results.",
    )
    args = parser.parse_args()

    rows = []
    for spec in args.files:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, spec
        df = pd.read_csv(path)
        validate_protocol(df, path, allow_legacy=args.allow_legacy)
        transition_step = args.transition_step or find_transition(df)
        print(f"[{label}] transition at global_step={transition_step}")
        m = compute_metrics(df, transition_step, args.direction, args.signal,
                            args.pre_window, args.overshoot_window, args.settle_tol_frac)
        m["label"] = label
        m["file"] = path
        rows.append(m)

    result = pd.DataFrame(rows).set_index("label")
    cols = ["onset_steps", "overshoot_abs", "overshoot_frac_of_swing",
            "time_to_correct_steps", "time_to_steady_state_steps",
            "n_post_steps_available"]
    pd.set_option("display.width", 160)
    title = f"Response dynamics ({args.schedule_name})" if args.schedule_name else "Response dynamics"
    print(f"\n=== {title} — signal={args.signal}, direction={args.direction} ===")
    print(result[cols].to_string())

    for label, row in result.iterrows():
        if row["time_to_steady_state_steps"] is None or pd.isna(row["time_to_steady_state_steps"]):
            print(f"\nNOTE [{label}]: did not settle within the logged window "
                  f"({row['n_post_steps_available']} steps) -- widen "
                  f"--fine-log-end or --overshoot-window and re-run for a "
                  f"confirmed (not lower-bound) settling time.")


if __name__ == "__main__":
    main()

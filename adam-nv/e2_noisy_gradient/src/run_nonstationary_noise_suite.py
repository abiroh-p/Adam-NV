"""
Phase 1.5 · Non-Stationary Noise Suite Runner
====================================================
Runs Adam and Adam-NV (the core ablation — see part_b_resnet_v8.py
docstring) through non-stationary noise schedules, so the "does
real-time beta1 adaptation help when noise CHANGES during training"
question from the E2/E11 report can finally be tested directly.

Default schedules: phased, ramp_up, intermittent. ramp_updown is
available via --schedules but not run by default (adds runtime without
being obviously more informative than phased+ramp_up+intermittent
together — add it explicitly if you want it).

adamw/adabelief are NOT included by default — the primary question
here is Adam vs Adam-NV (does adaptivity help under drift). Pass
--optimizers adam adam_nv adamw adabelief to extend to the full set
once the core comparison is in.

Uses part_b_resnet_v8.py. v7-produced logs (constant-sigma conditions)
are untouched — this suite writes to the same --log-dir but under
*_schedule{name}_* filenames, so nothing collides with existing runs.

Default: 2 optimizers x 3 schedules x 4 seeds = 24 runs.

Crash-safe / resumable, same pattern as run_baseline_noise_suite.py:
if one run fails, the suite logs it and continues; re-run with
--skip-existing to pick up where it left off after a disconnect.

Usage
-----
    # Full Phase 1.5 suite (adam, adam_nv x phased/ramp_up/intermittent x 4 seeds):
    python run_nonstationary_noise_suite.py --epochs 60 --batch-size 128

    # Resume after disconnect:
    python run_nonstationary_noise_suite.py --epochs 60 --skip-existing

    # Smoke test (CPU, synthetic, fast):
    python run_nonstationary_noise_suite.py --epochs 6 --smoke-test --synthetic-data

    # Just phased, just Adam-NV, single seed:
    python run_nonstationary_noise_suite.py --optimizers adam_nv \
        --schedules phased --seeds 0

    # Extend to all four optimizers, include the triangular ramp:
    python run_nonstationary_noise_suite.py \
        --optimizers adam adam_nv adamw adabelief \
        --schedules phased ramp_up ramp_updown intermittent
"""

import argparse
import csv
import os
import subprocess
import sys
import time

OPTIMIZERS = ["adam", "adam_nv"]
SCHEDULES  = ["phased", "ramp_up", "intermittent"]
SEEDS      = [0, 1, 2, 3]
SIGMA_MAX  = 2.0  # matches the sigma level used throughout E2/E7/E11/E12
                  # and the Phase 1.4 baseline suite, so peak-noise
                  # severity is comparable across constant vs. scheduled runs


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.5 non-stationary noise suite runner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=SIGMA_MAX,
                        help="Peak noise level (sigma_max) for every "
                             f"schedule in this suite. Default {SIGMA_MAX} "
                             "(matches E2/E7/E11/E12 and Phase 1.4).")
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose log CSV already exists "
                             "(use to resume after disconnect)")
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZERS,
                        choices=["adam", "adam_nv", "adamw", "adabelief"])
    parser.add_argument("--schedules", nargs="+", default=SCHEDULES,
                        choices=["phased", "ramp_up", "ramp_updown",
                                "intermittent"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    # schedule-parameter passthrough — same defaults as v8
    parser.add_argument("--phase-start", type=float, default=1.0 / 3.0)
    parser.add_argument("--phase-end", type=float, default=2.0 / 3.0)
    parser.add_argument("--burst-epochs", type=int, default=5)
    parser.add_argument("--gap-epochs", type=int, default=5)
    args = parser.parse_args()

    combos = [
        (opt, sched, seed)
        for opt   in args.optimizers
        for sched in args.schedules
        for seed  in args.seeds
    ]

    print(f"[Nonstationary-Noise Suite] {len(combos)} runs planned")
    for opt, sched, seed in combos:
        print(f"  optimizer={opt}  schedule={sched}  "
              f"sigma_max={args.sigma}  seed={seed}")
    print(f"[Nonstationary-Noise Suite] epochs={args.epochs} "
          f"batch_size={args.batch_size} smoke={args.smoke_test} "
          f"synthetic={args.synthetic_data}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log = os.path.join(args.log_dir, "_nonstationary_noise_suite_status.csv")
    suite_rows = []
    suite_start = time.time()

    for i, (opt, sched, seed) in enumerate(combos, start=1):
        run_name = f"{opt}_schedule{sched}_sigma{args.sigma}_seed{seed}"
        if args.smoke_test:
            run_name += "_smoke"
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            # Same partial-CSV guard as run_baseline_noise_suite.py — a
            # disconnect mid-run leaves an incomplete file that must not
            # be mistaken for a finished one.
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1  # minus header
            if n_data_rows >= args.epochs:
                print(f"[Nonstationary-Noise Suite] ({i}/{len(combos)}) "
                      f"SKIPPING {run_name} — log complete "
                      f"({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({
                    "optimizer": opt, "schedule": sched, "sigma_max": args.sigma,
                    "seed": seed, "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue
            else:
                print(f"[Nonstationary-Noise Suite] ({i}/{len(combos)}) "
                      f"{run_name} log is INCOMPLETE "
                      f"({n_data_rows}/{args.epochs} epochs) "
                      f"— re-running from scratch")

        elapsed_so_far = time.time() - suite_start
        print(f"\n[Nonstationary-Noise Suite] ({i}/{len(combos)}) "
              f"STARTING {run_name} | suite elapsed: {elapsed_so_far:.0f}s")

        cmd = [
            sys.executable, "part_b_resnet_v8.py",
            "--optimizer", opt,
            "--schedule", sched,
            "--sigma", str(args.sigma),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        if sched == "phased":
            cmd += ["--phase-start", str(args.phase_start),
                    "--phase-end", str(args.phase_end)]
        if sched == "intermittent":
            cmd += ["--burst-epochs", str(args.burst_epochs),
                    "--gap-epochs", str(args.gap_epochs)]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.synthetic_data:
            cmd.append("--synthetic-data")

        run_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            status, error_msg = "success", ""
        except subprocess.CalledProcessError as e:
            status    = "FAILED"
            error_msg = f"exit code {e.returncode}"
            print(f"[Nonstationary-Noise Suite] *** {run_name} FAILED "
                  f"({error_msg}) — continuing ***")

        run_elapsed = time.time() - run_start
        suite_rows.append({
            "optimizer": opt, "schedule": sched, "sigma_max": args.sigma,
            "seed": seed, "status": status, "elapsed_sec": round(run_elapsed, 1),
            "error": error_msg,
        })

        # Write suite status after every run so progress survives disconnects
        with open(suite_log, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["optimizer", "schedule", "sigma_max",
                               "seed", "status", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_ok   = sum(1 for r in suite_rows if r["status"] == "success")
    n_fail = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skip = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[Nonstationary-Noise Suite] DONE — {n_ok} succeeded, {n_fail} "
          f"failed, {n_skip} skipped | total: {total_elapsed/60:.1f} min")
    print(f"[Nonstationary-Noise Suite] Status saved to {suite_log}")

    if n_fail:
        print(f"[Nonstationary-Noise Suite] {n_fail} run(s) failed — fix and "
              f"rerun with --skip-existing to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()

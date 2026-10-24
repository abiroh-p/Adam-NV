"""
Phase 1.7 · Decoupled v_noise Suite Runner
================================================
The Phase 1.6 beta_2 sweep (project report, Section 5) found a genuine
trade-off: shrinking beta_2 fixed Adam-NV's deficit under intermittent
noise, but broke ramp-up, which had been fine before. Since beta_2 was
shared between the noise estimator and the parameter update's own v_hat
denominator, that result could not distinguish two explanations:

  (a) noise-tracking speed itself has an inherent cost under smooth,
      slow-changing noise, OR
  (b) the cost came from changing v_hat's smoothing (a generic Adam
      effect having nothing to do with noise-adaptivity), which just
      happened to move in lockstep with beta_2 because v was shared.

adam_nv.py now supports beta_2_noise (FIX 6): a separate v buffer
dedicated to the noise estimator, isolated from v_hat. beta_2 itself
stays at its class default (0.999) for every run in this suite -- ONLY
the noise-tracking speed changes. If explanation (b) is correct, this
suite should show intermittent's deficit resolve WITHOUT ramp-up
breaking, since v_hat (and therefore Adam's own step-size behavior)
never changes here. If ramp-up still breaks even with v_hat untouched,
explanation (a) is the real one, and slow-changing noise is
inherently harder to track quickly without cost, regardless of coupling.

Default: 2 optimizers... no -- adam_nv only (this is a mechanism-
isolation experiment, not a baseline comparison) x 2 beta_2_noise
values x 4 schedules x 4 seeds = 32 runs. Schedules include phased and
constant as regression checks (should stay unaffected, matching the
Phase 1.6 pattern) alongside the two schedules that actually showed
effects (ramp_up, intermittent).

Uses part_b_resnet_v8.py. Crash-safe / resumable, same pattern as
prior suite runners.

Usage
-----
    # Full Phase 1.7 suite:
    python run_vnoise_suite.py --epochs 60

    # Resume after disconnect:
    python run_vnoise_suite.py --epochs 60 --skip-existing

    # Just the two schedules that matter, single beta_2_noise value:
    python run_vnoise_suite.py --schedules ramp_up intermittent \\
        --beta2-noise-values 0.95 --seeds 0

    # Smoke test:
    python run_vnoise_suite.py --epochs 6 --smoke-test --synthetic-data
"""

import argparse
import csv
import os
import subprocess
import sys
import time

BETA2_NOISE_VALUES = [0.99, 0.95]  # same values tested in the coupled Phase 1.6 sweep
SCHEDULES           = ["constant", "phased", "ramp_up", "intermittent"]
SEEDS                = [0, 1, 2, 3]
SIGMA_MAX            = 2.0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.7 decoupled v_noise suite runner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=SIGMA_MAX)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--beta2-noise-values", nargs="+", type=float,
                        default=BETA2_NOISE_VALUES)
    parser.add_argument("--schedules", nargs="+", default=SCHEDULES,
                        choices=["constant", "phased", "ramp_up",
                                "ramp_updown", "intermittent"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--phase-start", type=float, default=1.0 / 3.0)
    parser.add_argument("--phase-end", type=float, default=2.0 / 3.0)
    parser.add_argument("--burst-epochs", type=int, default=5)
    parser.add_argument("--gap-epochs", type=int, default=5)
    args = parser.parse_args()

    combos = [
        (b2n, sched, seed)
        for b2n   in args.beta2_noise_values
        for sched in args.schedules
        for seed  in args.seeds
    ]

    print(f"[vnoise Suite] {len(combos)} runs planned (optimizer=adam_nv, "
          f"beta_2=0.999 fixed for all runs, only beta_2_noise varies)")
    for b2n, sched, seed in combos:
        print(f"  beta2_noise={b2n}  schedule={sched}  sigma_max={args.sigma}  seed={seed}")
    print(f"[vnoise Suite] epochs={args.epochs} batch_size={args.batch_size} "
          f"smoke={args.smoke_test} synthetic={args.synthetic_data}\n")
    print("[vnoise Suite] Compare these results against your existing Phase "
          "1.6 beta2-sweep table: if ramp_up stays clean here while "
          "intermittent still improves, the trade-off was a v_hat-coupling "
          "artifact, not an inherent cost of noise-tracking speed.\n")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log = os.path.join(args.log_dir, "_vnoise_suite_status.csv")
    suite_rows = []
    suite_start = time.time()

    for i, (b2n, sched, seed) in enumerate(combos, start=1):
        run_name = f"adam_nv_schedule{sched}_sigma{args.sigma}_beta2noise{b2n}_seed{seed}"
        if args.smoke_test:
            run_name += "_smoke"
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1
            if n_data_rows >= args.epochs:
                print(f"[vnoise Suite] ({i}/{len(combos)}) SKIPPING {run_name} "
                      f"— log complete ({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({
                    "beta2_noise": b2n, "schedule": sched, "sigma_max": args.sigma,
                    "seed": seed, "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue
            else:
                print(f"[vnoise Suite] ({i}/{len(combos)}) {run_name} log is "
                      f"INCOMPLETE ({n_data_rows}/{args.epochs} epochs) "
                      f"— re-running from scratch")

        elapsed_so_far = time.time() - suite_start
        print(f"\n[vnoise Suite] ({i}/{len(combos)}) STARTING {run_name} "
              f"| suite elapsed: {elapsed_so_far:.0f}s")

        cmd = [
            sys.executable, "part_b_resnet_v8.py",
            "--optimizer", "adam_nv",
            "--schedule", sched,
            "--sigma", str(args.sigma),
            "--beta2-noise", str(b2n),
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
            print(f"[vnoise Suite] *** {run_name} FAILED ({error_msg}) — continuing ***")

        run_elapsed = time.time() - run_start
        suite_rows.append({
            "beta2_noise": b2n, "schedule": sched, "sigma_max": args.sigma,
            "seed": seed, "status": status, "elapsed_sec": round(run_elapsed, 1),
            "error": error_msg,
        })

        with open(suite_log, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["beta2_noise", "schedule", "sigma_max", "seed",
                               "status", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_ok   = sum(1 for r in suite_rows if r["status"] == "success")
    n_fail = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skip = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[vnoise Suite] DONE — {n_ok} succeeded, {n_fail} failed, "
          f"{n_skip} skipped | total: {total_elapsed/60:.1f} min")
    print(f"[vnoise Suite] Status saved to {suite_log}")

    if n_fail:
        print(f"[vnoise Suite] {n_fail} run(s) failed — fix and rerun "
              f"with --skip-existing to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()

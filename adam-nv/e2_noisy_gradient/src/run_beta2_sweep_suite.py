"""
Phase 1.6 · Beta-2 Sweep Suite Runner (Adam-NV noise-tracking speed)
========================================================================
Screening experiment for the lag hypothesis in the Phase 1.4/1.5 report
(§4.2): if v's beta_2=0.999 (~2.5-epoch memory) is what makes Adam-NV's
beta_1 lag ~3-15 epochs behind a changing noise environment, then a
shorter-memory beta_2 should shrink that lag -- and possibly recover the
small, consistent accuracy deficit found under the intermittent schedule.

This sweep only needs beta_2 in {0.99, 0.95}. beta_2=0.999 is the
AdamNV class default and is NOT re-run here -- your existing Phase 1.5
logs (adam_nv_schedule*_sigma2.0_seed*.csv, no beta2 suffix) already
ARE the beta_2=0.999 arm of this sweep. Re-running them would just
burn compute reproducing data you already have.

Default: 2 beta_2 values x 4 schedules x 4 seeds = 32 runs. 'constant'
is included alongside phased/ramp_up/intermittent as a regression
check -- shrinking beta_2 must not break the "no significant difference
from Adam" result under constant noise (Phase 1.4), or the fix isn't
free.

CONFOUND WARNING (see part_b_resnet_v8.py docstring, "beta2-sweep
experiment"): beta_2 is shared between the noise estimator and Adam's
own step-size denominator. A result here screens whether beta_2 speed
matters at all -- it does NOT by itself prove the effect is due to
faster noise-tracking specifically, since general Adam behavior also
changes with beta_2. If this sweep shows a clear effect, the next step
is the decoupled v_noise (separate beta_2_noise, isolated from the
parameter update) version of adam_nv.py to confirm it -- not to run
more of this sweep.

Uses part_b_resnet_v8.py. Crash-safe / resumable, same pattern as
run_baseline_noise_suite.py / run_nonstationary_noise_suite.py.

Usage
-----
    # Full Phase 1.6 sweep (adam_nv, beta2 in {0.99, 0.95}, 4 schedules, 4 seeds):
    python run_beta2_sweep_suite.py --epochs 60

    # Resume after disconnect:
    python run_beta2_sweep_suite.py --epochs 60 --skip-existing

    # Smoke test:
    python run_beta2_sweep_suite.py --epochs 6 --smoke-test --synthetic-data

    # Just the schedule that actually showed a deficit, single beta_2:
    python run_beta2_sweep_suite.py --schedules intermittent --beta2-values 0.95 --seeds 0

    # Include beta_2=0.999 explicitly anyway (e.g. re-verifying on a new
    # machine, or if you no longer trust the existing Phase 1.5 logs):
    python run_beta2_sweep_suite.py --beta2-values 0.999 0.99 0.95
"""

import argparse
import csv
import os
import subprocess
import sys
import time

BETA2_VALUES = [0.99, 0.95]  # 0.999 deliberately excluded -- see module docstring
SCHEDULES    = ["constant", "phased", "ramp_up", "intermittent"]
SEEDS        = [0, 1, 2, 3]
SIGMA_MAX    = 2.0  # matches Phase 1.4/1.5 for direct comparability


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.6 beta_2 sweep suite runner (Adam-NV noise-tracking speed)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=SIGMA_MAX)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose log CSV already exists "
                             "(use to resume after disconnect)")
    parser.add_argument("--beta2-values", nargs="+", type=float, default=BETA2_VALUES,
                        help=f"beta_2 values to sweep. Default {BETA2_VALUES} -- "
                             "0.999 is deliberately excluded since it's already "
                             "covered by your existing Phase 1.5 logs (pass it "
                             "explicitly if you want it re-run anyway).")
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
        (b2, sched, seed)
        for b2    in args.beta2_values
        for sched in args.schedules
        for seed  in args.seeds
    ]

    print(f"[Beta2-Sweep Suite] {len(combos)} runs planned (optimizer=adam_nv only)")
    for b2, sched, seed in combos:
        print(f"  beta2={b2}  schedule={sched}  sigma_max={args.sigma}  seed={seed}")
    print(f"[Beta2-Sweep Suite] epochs={args.epochs} batch_size={args.batch_size} "
          f"smoke={args.smoke_test} synthetic={args.synthetic_data}\n")
    print("[Beta2-Sweep Suite] NOTE: beta2=0.999 is not included by default -- "
          "your existing Phase 1.5 logs already cover it. See module docstring.\n")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log = os.path.join(args.log_dir, "_beta2_sweep_suite_status.csv")
    suite_rows = []
    suite_start = time.time()

    for i, (b2, sched, seed) in enumerate(combos, start=1):
        run_name = f"adam_nv_schedule{sched}_sigma{args.sigma}_beta2{b2}_seed{seed}"
        if args.smoke_test:
            run_name += "_smoke"
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1
            if n_data_rows >= args.epochs:
                print(f"[Beta2-Sweep Suite] ({i}/{len(combos)}) SKIPPING {run_name} "
                      f"— log complete ({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({
                    "beta2": b2, "schedule": sched, "sigma_max": args.sigma,
                    "seed": seed, "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue
            else:
                print(f"[Beta2-Sweep Suite] ({i}/{len(combos)}) {run_name} log is "
                      f"INCOMPLETE ({n_data_rows}/{args.epochs} epochs) "
                      f"— re-running from scratch")

        elapsed_so_far = time.time() - suite_start
        print(f"\n[Beta2-Sweep Suite] ({i}/{len(combos)}) STARTING {run_name} "
              f"| suite elapsed: {elapsed_so_far:.0f}s")

        cmd = [
            sys.executable, "part_b_resnet_v8.py",
            "--optimizer", "adam_nv",
            "--schedule", sched,
            "--sigma", str(args.sigma),
            "--beta2", str(b2),
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
            print(f"[Beta2-Sweep Suite] *** {run_name} FAILED ({error_msg}) "
                  f"— continuing ***")

        run_elapsed = time.time() - run_start
        suite_rows.append({
            "beta2": b2, "schedule": sched, "sigma_max": args.sigma,
            "seed": seed, "status": status, "elapsed_sec": round(run_elapsed, 1),
            "error": error_msg,
        })

        with open(suite_log, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["beta2", "schedule", "sigma_max", "seed",
                               "status", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_ok   = sum(1 for r in suite_rows if r["status"] == "success")
    n_fail = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skip = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[Beta2-Sweep Suite] DONE — {n_ok} succeeded, {n_fail} failed, "
          f"{n_skip} skipped | total: {total_elapsed/60:.1f} min")
    print(f"[Beta2-Sweep Suite] Status saved to {suite_log}")
    print("[Beta2-Sweep Suite] Remember: combine with your existing beta2=0.999 "
          "Phase 1.5 logs for the full 3-value comparison during analysis.")

    if n_fail:
        print(f"[Beta2-Sweep Suite] {n_fail} run(s) failed — fix and rerun "
              f"with --skip-existing to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()

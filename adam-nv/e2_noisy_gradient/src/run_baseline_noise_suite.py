"""
Phase 1.4 · Baseline-Optimizer Noise Suite Runner
====================================================
Runs AdamW and AdaBelief through the two highest-signal noise
conditions already established in the paper, so Adam-NV's Δacc
can be benchmarked against them, not just against plain Adam:

    Condition A (matches E2/E7, "gradient"): sigma=2.0, label_noise=0.0
    Condition B (matches E11, "label"):       sigma=0.0, label_noise=0.2

adam and adam_nv are NOT re-run here — their logs for both
conditions already exist from E2/E7-A/E11. This suite only fills
the adamw/adabelief gap identified in review.

Uses part_b_resnet_v7.py (adds adamw/adabelief to the v6 script).
v6-produced logs for adam/adam_nv are untouched — this suite
writes to the same --log-dir but under adamw_*/adabelief_*
filenames, so nothing collides.

Total: 2 optimizers x 2 conditions x 4 seeds = 16 runs.
Default seeds {0,1,2,3} match the seed values used for E2/E11's
adam/adam_nv runs (same run_name convention), so the two seed sets
line up as matched comparisons rather than independent samples.

Crash-safe / resumable, same as run_e2_suite.py: if one run fails,
the suite logs it and continues; re-run with --skip-existing to
pick up where it left off after a Colab disconnect.

Usage
-----
    # Full Phase 1.4 suite:
    python run_baseline_noise_suite.py --epochs 60 --batch-size 128

    # Resume after disconnect:
    python run_baseline_noise_suite.py --epochs 60 --skip-existing

    # Smoke test (CPU, synthetic, fast):
    python run_baseline_noise_suite.py --epochs 2 --smoke-test --synthetic-data

    # Just the label-noise condition, just AdaBelief:
    python run_baseline_noise_suite.py --optimizers adabelief --conditions label
"""

import argparse
import csv
import os
import subprocess
import sys
import time

OPTIMIZERS = ["adamw", "adabelief"]
SEEDS      = [0, 1, 2, 3]

# name -> (sigma, label_noise), matching E2(sigma=2.0)/E11(label_noise=0.2)
CONDITIONS = {
    "gradient": (2.0, 0.0),
    "label":    (0.0, 0.2),
}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1.4 baseline-optimizer noise suite runner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose log CSV already exists "
                             "(use to resume after disconnect)")
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZERS,
                        choices=["adamw", "adabelief"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS),
                        help="Which noise condition(s) to run: "
                             "'gradient' (sigma=2.0) and/or "
                             "'label' (label_noise=0.2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    combos = [
        (opt, cond, seed)
        for opt  in args.optimizers
        for cond in args.conditions
        for seed in args.seeds
    ]

    print(f"[Baseline-Noise Suite] {len(combos)} runs planned")
    for opt, cond, seed in combos:
        sigma, label_noise = CONDITIONS[cond]
        print(f"  optimizer={opt}  condition={cond} "
              f"(sigma={sigma}, label_noise={label_noise})  seed={seed}")
    print(f"[Baseline-Noise Suite] epochs={args.epochs} "
          f"batch_size={args.batch_size} smoke={args.smoke_test} "
          f"synthetic={args.synthetic_data}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log = os.path.join(args.log_dir, "_baseline_noise_suite_status.csv")
    suite_rows = []
    suite_start = time.time()

    for i, (opt, cond, seed) in enumerate(combos, start=1):
        sigma, label_noise = CONDITIONS[cond]

        suffix = "_smoke" if args.smoke_test else ""
        ln_suffix = f"_ln{label_noise}" if label_noise > 0.0 else ""
        run_name = f"{opt}_sigma{sigma}{ln_suffix}_seed{seed}{suffix}"
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            # Same partial-CSV guard as run_e2_suite.py — a disconnect
            # mid-run leaves an incomplete file that must not be mistaken
            # for a finished one.
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1  # minus header
            if n_data_rows >= args.epochs:
                print(f"[Baseline-Noise Suite] ({i}/{len(combos)}) "
                      f"SKIPPING {run_name} — log complete "
                      f"({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({
                    "optimizer": opt, "condition": cond, "sigma": sigma,
                    "label_noise": label_noise, "seed": seed,
                    "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue
            else:
                print(f"[Baseline-Noise Suite] ({i}/{len(combos)}) "
                      f"{run_name} log is INCOMPLETE "
                      f"({n_data_rows}/{args.epochs} epochs) "
                      f"— re-running from scratch")

        elapsed_so_far = time.time() - suite_start
        print(f"\n[Baseline-Noise Suite] ({i}/{len(combos)}) "
              f"STARTING {run_name} | suite elapsed: {elapsed_so_far:.0f}s")

        cmd = [
            sys.executable, "part_b_resnet_v7.py",
            "--optimizer", opt,
            "--sigma", str(sigma),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        if label_noise > 0.0:
            cmd += ["--label-noise", str(label_noise)]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.synthetic_data:
            if label_noise > 0.0:
                # part_b_resnet_v7.py rejects --label-noise +
                # --synthetic-data (synthetic labels are already random,
                # so corrupting them is meaningless — same guard as v6).
                # Fall back to real CIFAR-10 under --smoke-test's small
                # subset (512 train / 256 test images) instead, which is
                # still fast and lets the label-noise condition actually
                # get smoke-tested rather than skipped.
                print(f"[Baseline-Noise Suite] {run_name}: label_noise>0, "
                      f"so using the real-data --smoke-test subset "
                      f"instead of --synthetic-data for this run.")
            else:
                cmd.append("--synthetic-data")

        run_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            status, error_msg = "success", ""
        except subprocess.CalledProcessError as e:
            status    = "FAILED"
            error_msg = f"exit code {e.returncode}"
            print(f"[Baseline-Noise Suite] *** {run_name} FAILED "
                  f"({error_msg}) — continuing ***")

        run_elapsed = time.time() - run_start
        suite_rows.append({
            "optimizer": opt, "condition": cond, "sigma": sigma,
            "label_noise": label_noise, "seed": seed,
            "status": status, "elapsed_sec": round(run_elapsed, 1),
            "error": error_msg,
        })

        # Write suite status after every run so progress survives disconnects
        with open(suite_log, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["optimizer", "condition", "sigma",
                               "label_noise", "seed", "status",
                               "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_ok   = sum(1 for r in suite_rows if r["status"] == "success")
    n_fail = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skip = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[Baseline-Noise Suite] DONE — {n_ok} succeeded, {n_fail} "
          f"failed, {n_skip} skipped | total: {total_elapsed/60:.1f} min")
    print(f"[Baseline-Noise Suite] Status saved to {suite_log}")

    if n_fail:
        print(f"[Baseline-Noise Suite] {n_fail} run(s) failed — fix and "
              f"rerun with --skip-existing to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()

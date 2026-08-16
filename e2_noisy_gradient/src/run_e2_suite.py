"""
E2 · Part B Suite Runner
=========================
Runs all (optimizer × sigma × seed) combinations for Part B.

Tier-1 plan:
    optimizers = [adam_nv, adam]
    sigmas     = [0.05, 0.5]      ← σ=0.0 reused from E1, not re-run
    seeds      = [0, 1]

Total: 2 × 2 × 2 = 8 runs  (~12 GPU-hours on Colab T4)

Crash-safe: if one run fails, the suite logs the failure and
continues. Re-run with --skip-existing to resume after a disconnect.

Usage
-----
    # Full Tier-1 suite:
    python run_e2_suite.py --epochs 60 --batch-size 128

    # Resume after disconnect:
    python run_e2_suite.py --epochs 60 --batch-size 128 --skip-existing

    # Smoke test (CPU, synthetic, fast):
    python run_e2_suite.py --epochs 2 --smoke-test --synthetic-data

    # Single sigma only (e.g., strong noise first):
    python run_e2_suite.py --epochs 60 --sigmas 0.5
"""

import argparse
import csv
import os
import subprocess
import sys
import time

OPTIMIZERS = ["adam_nv", "adam"]
SIGMAS     = [0.05, 0.5]
SEEDS      = [0, 1]


def main():
    parser = argparse.ArgumentParser(description="E2 Part B suite runner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs whose log CSV already exists "
                             "(use to resume after disconnect)")
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZERS)
    parser.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    combos = [
        (opt, sigma, seed)
        for opt   in args.optimizers
        for sigma in args.sigmas
        for seed  in args.seeds
    ]

    print(f"[E2 Suite] {len(combos)} runs planned")
    for c in combos:
        print(f"  optimizer={c[0]}  sigma={c[1]}  seed={c[2]}")
    print(f"[E2 Suite] epochs={args.epochs} batch_size={args.batch_size} "
          f"smoke={args.smoke_test} synthetic={args.synthetic_data}\n")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log = os.path.join(args.log_dir, "_e2_suite_status.csv")
    suite_rows = []
    suite_start = time.time()

    for i, (opt, sigma, seed) in enumerate(combos, start=1):
        suffix   = "_smoke" if args.smoke_test else ""
        run_name = f"{opt}_sigma{sigma}_seed{seed}{suffix}"
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            # Check completion, not just existence — a partial CSV from an
            # interrupted run (e.g. Colab disconnect) would otherwise be
            # silently treated as done, contaminating downstream analysis
            # with an under-trained run masquerading as complete.
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1  # minus header
            if n_data_rows >= args.epochs:
                print(f"[E2 Suite] ({i}/{len(combos)}) SKIPPING {run_name} "
                      f"— log complete ({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({"optimizer": opt, "sigma": sigma, "seed": seed,
                                    "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue
            else:
                print(f"[E2 Suite] ({i}/{len(combos)}) {run_name} log is "
                      f"INCOMPLETE ({n_data_rows}/{args.epochs} epochs) "
                      f"— re-running from scratch")

        elapsed_so_far = time.time() - suite_start
        print(f"\n[E2 Suite] ({i}/{len(combos)}) STARTING {run_name} "
              f"| suite elapsed: {elapsed_so_far:.0f}s")

        cmd = [
            sys.executable, "part_b_resnet.py",
            "--optimizer", opt,
            "--sigma", str(sigma),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        if args.smoke_test:    cmd.append("--smoke-test")
        if args.synthetic_data: cmd.append("--synthetic-data")

        run_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            status, error_msg = "success", ""
        except subprocess.CalledProcessError as e:
            status    = "FAILED"
            error_msg = f"exit code {e.returncode}"
            print(f"[E2 Suite] *** {run_name} FAILED ({error_msg}) "
                  f"— continuing ***")

        run_elapsed = time.time() - run_start
        suite_rows.append({
            "optimizer": opt, "sigma": sigma, "seed": seed,
            "status": status, "elapsed_sec": round(run_elapsed, 1),
            "error": error_msg,
        })

        # Write suite status after every run so progress survives disconnects
        with open(suite_log, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["optimizer", "sigma", "seed",
                               "status", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_ok   = sum(1 for r in suite_rows if r["status"] == "success")
    n_fail = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skip = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[E2 Suite] DONE — {n_ok} succeeded, {n_fail} failed, "
          f"{n_skip} skipped | total: {total_elapsed/60:.1f} min")
    print(f"[E2 Suite] Status saved to {suite_log}")

    if n_fail:
        print(f"[E2 Suite] {n_fail} run(s) failed — fix and rerun with "
              f"--skip-existing to resume")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
E1 · Convergence Benchmarks — orchestration script.

Runs all Tier-1 (optimizer x seed) combinations back-to-back:
    optimizers = [adam, adamw, adabelief, adam_nv]   (adam_nv last so the
                  baselines are already on disk before the long pole runs)
    seeds      = [0, 1, 2, 3]

Total: 4 optimizers x 4 seeds = 16 runs.
(Note: Tier-1 budget in the design doc assumed 3 optimizers; AdaBelief is
included here as a baseline alongside Adam/AdamW since it was confirmed
available. Drop it from OPTIMIZERS below to fall back to the 3-optimizer,
~12 GPU-hr budget if you want to stay strictly within the original Tier-1 plan.)

If a single run crashes (e.g. OOM, transient CUDA error), the suite logs
the failure and continues to the next combination rather than aborting
the whole batch — you don't want to lose 5 successful runs because the
6th hit an out-of-memory error.

Usage:
    python run_e1_suite.py --epochs 60
    python run_e1_suite.py --epochs 60 --skip-existing   # resume after interruption
    python run_e1_suite.py --epochs 2 --smoke-test --synthetic-data  # dry run
"""

import argparse
import csv
import os
import subprocess
import sys
import time

OPTIMIZERS = ["adam", "adamw", "adabelief", "adam_nv"]
SEEDS = [0, 1, 2, 3]


def main():
    parser = argparse.ArgumentParser(description="E1 Tier-1 suite runner")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="../logs")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Skip a (optimizer, seed) combo if its log CSV "
                              "already exists — useful for resuming after a "
                              "crashed or interrupted session.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the learning rate (default: per-optimizer default).")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Suffix for run/file names, e.g. _lr3e-4 (use with --lr).")
    parser.add_argument("--status-file", type=str, default="_suite_status.csv",
                         help="Name of the per-invocation status CSV written inside "
                              "--log-dir. The default is OVERWRITTEN by every "
                              "invocation, so give concurrent or follow-up "
                              "invocations distinct names.")
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZERS,
                         help=f"Subset of optimizers to run. Default: {OPTIMIZERS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                         help=f"Subset of seeds to run. Default: {SEEDS}")
    args = parser.parse_args()

    combos = [(opt, seed) for opt in args.optimizers for seed in args.seeds]
    print(f"[E1 Suite] {len(combos)} runs planned: {combos}")
    print(f"[E1 Suite] epochs={args.epochs} batch_size={args.batch_size} "
          f"smoke_test={args.smoke_test} synthetic_data={args.synthetic_data}")

    os.makedirs(args.log_dir, exist_ok=True)
    suite_log_path = os.path.join(args.log_dir, args.status_file)
    suite_rows = []

    suite_start = time.time()

    for i, (opt, seed) in enumerate(combos, start=1):
        run_name = f"{opt}{args.run_tag}_seed{seed}" + ("_smoke" if args.smoke_test else "")
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")

        if args.skip_existing and os.path.exists(log_path):
            with open(log_path, newline="") as lf:
                n_data_rows = sum(1 for _ in lf) - 1
            if n_data_rows >= args.epochs:
                print(f"\n[E1 Suite] ({i}/{len(combos)}) SKIPPING {run_name} "
                      f"— complete log ({n_data_rows}/{args.epochs} epochs)")
                suite_rows.append({"optimizer": opt, "seed": seed, "status": "skipped",
                                    "elapsed_sec": 0, "error": ""})
                continue
            print(f"\n[E1 Suite] ({i}/{len(combos)}) {run_name} is incomplete "
                  f"({n_data_rows}/{args.epochs} epochs); re-running it")

        print(f"\n[E1 Suite] ({i}/{len(combos)}) STARTING {run_name} "
              f"| elapsed so far: {time.time() - suite_start:.0f}s")

        cmd = [
            sys.executable, "train_resnet.py",
            "--optimizer", opt,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        if args.run_tag:
            cmd += ["--run-tag", args.run_tag]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.synthetic_data:
            cmd.append("--synthetic-data")
        if args.deterministic:
            cmd.append("--deterministic")

        run_start = time.time()
        try:
            result = subprocess.run(cmd, check=True)
            status = "success"
            error_msg = ""
        except subprocess.CalledProcessError as e:
            status = "FAILED"
            error_msg = f"exit code {e.returncode}"
            print(f"[E1 Suite] *** {run_name} FAILED (exit code {e.returncode}) "
                  f"— continuing to next run ***")
        run_elapsed = time.time() - run_start

        suite_rows.append({"optimizer": opt, "seed": seed, "status": status,
                            "elapsed_sec": round(run_elapsed, 1), "error": error_msg})

        # Write suite status after every run so partial progress survives
        # a crash or session timeout (e.g. Colab disconnect).
        with open(suite_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["optimizer", "seed", "status",
                                                     "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(suite_rows)

    # Final write: when every combination was skipped the in-loop write above
    # never runs, which would leave no status file for this invocation.
    with open(suite_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "seed", "status",
                                                 "elapsed_sec", "error"])
        writer.writeheader()
        writer.writerows(suite_rows)

    total_elapsed = time.time() - suite_start
    n_success = sum(1 for r in suite_rows if r["status"] == "success")
    n_failed = sum(1 for r in suite_rows if r["status"] == "FAILED")
    n_skipped = sum(1 for r in suite_rows if r["status"] == "skipped")

    print(f"\n[E1 Suite] DONE — {n_success} succeeded, {n_failed} failed, "
          f"{n_skipped} skipped. Total time: {total_elapsed/60:.1f} min")
    print(f"[E1 Suite] Status table saved to {suite_log_path}")

    if n_failed > 0:
        print(f"[E1 Suite] WARNING: {n_failed} run(s) failed. Check the printed "
              f"errors above, fix the issue, then re-run with --skip-existing "
              f"to resume only the missing combinations.")
        sys.exit(1)


if __name__ == "__main__":
    main()

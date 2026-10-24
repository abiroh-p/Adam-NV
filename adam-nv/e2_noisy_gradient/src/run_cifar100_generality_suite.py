"""E16 CIFAR-100 generality suite.

Runs matched Adam and Adam-NV seeds on CIFAR-100 under a clean condition and
the intermittent-noise condition that most directly tests adaptation.

Default: 2 optimizers x 2 conditions x 4 seeds = 16 runs.
"""

import argparse
import csv
import os
import subprocess
import sys
import time


OPTIMIZERS = ["adam", "adam_nv"]
SEEDS = [0, 1, 2, 3]
CONDITIONS = {
    "clean": {"schedule": "constant", "sigma": 0.0},
    "intermittent": {"schedule": "intermittent", "sigma": 2.0},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="E16 CIFAR-100 generality suite")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--log-dir", default="../logs/cifar100")
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZERS,
                        choices=OPTIMIZERS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    combos = [
        (optimizer, condition, seed)
        for optimizer in args.optimizers
        for condition in args.conditions
        for seed in args.seeds
    ]
    os.makedirs(args.log_dir, exist_ok=True)
    status_path = os.path.join(args.log_dir, "_cifar100_suite_status.csv")
    rows = []

    for index, (optimizer, condition, seed) in enumerate(combos, start=1):
        config = CONDITIONS[condition]
        schedule = config["schedule"]
        sigma = config["sigma"]
        run_name = (
            f"{optimizer}_schedule{schedule}_sigma{sigma}_cifar100_seed{seed}"
            + ("_smoke" if args.smoke_test else "")
        )
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")
        if args.skip_existing and os.path.exists(log_path):
            with open(log_path, newline="") as handle:
                n_rows = sum(1 for _ in handle) - 1
            if n_rows >= args.epochs:
                print(f"[E16] ({index}/{len(combos)}) skipping complete {run_name}")
                rows.append({"optimizer": optimizer, "condition": condition,
                             "seed": seed, "status": "skipped", "elapsed_sec": 0,
                             "error": ""})
                continue

        command = [
            sys.executable, "part_b_resnet_v8.py",
            "--optimizer", optimizer,
            "--schedule", schedule,
            "--sigma", str(sigma),
            "--dataset", "cifar100",
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
        ]
        if args.smoke_test:
            command.append("--smoke-test")
        if args.synthetic_data:
            command.append("--synthetic-data")
        if args.deterministic:
            command.append("--deterministic")

        print(f"[E16] ({index}/{len(combos)}) starting {run_name}")
        started = time.time()
        try:
            subprocess.run(command, check=True)
            status, error = "success", ""
        except subprocess.CalledProcessError as exc:
            status, error = "FAILED", f"exit code {exc.returncode}"
        rows.append({"optimizer": optimizer, "condition": condition,
                     "seed": seed, "status": status,
                     "elapsed_sec": round(time.time() - started, 1),
                     "error": error})
        with open(status_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["optimizer", "condition", "seed", "status",
                            "elapsed_sec", "error"],
            )
            writer.writeheader()
            writer.writerows(rows)

    failures = sum(row["status"] == "FAILED" for row in rows)
    print(f"[E16] finished: {len(rows) - failures} complete/skipped, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

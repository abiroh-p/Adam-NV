"""Controller-direction ablation for the Adam-NV beta_1 rule.

Compares the original `distrust` response (higher heuristic -> lower beta_1)
against a `smooth` response (higher heuristic -> beta_1 rises from 0.9 toward
0.999). Clean, constant-noise, and intermittent-noise conditions distinguish
baseline effects from noise-specific behavior.

Default: 2 response modes x 3 conditions x 4 seeds = 24 runs.
"""

import argparse
import csv
import os
import subprocess
import sys
import time


SEEDS = [0, 1, 2, 3]
MODES = ["distrust", "smooth"]
CONDITIONS = {
    "clean": ("constant", 0.0),
    "constant_noise": ("constant", 2.0),
    "intermittent": ("intermittent", 2.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--log-dir", default="../logs/controller_direction")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    combos = [
        (mode, condition, seed)
        for mode in args.modes
        for condition in args.conditions
        for seed in args.seeds
    ]
    os.makedirs(args.log_dir, exist_ok=True)
    status_path = os.path.join(args.log_dir, "_controller_direction_status.csv")
    rows = []

    for index, (mode, condition, seed) in enumerate(combos, start=1):
        schedule, sigma = CONDITIONS[condition]
        mode_suffix = "" if mode == "distrust" else "_responsesmooth_betamax0.999"
        run_name = (
            f"adam_nv_schedule{schedule}_sigma{sigma}{mode_suffix}_seed{seed}"
            + ("_smoke" if args.smoke_test else "")
        )
        log_path = os.path.join(args.log_dir, f"{run_name}.csv")
        if args.skip_existing and os.path.exists(log_path):
            with open(log_path, newline="") as handle:
                n_rows = sum(1 for _ in handle) - 1
            if n_rows >= args.epochs:
                print(f"[Direction] ({index}/{len(combos)}) skipping {run_name}")
                rows.append({"mode": mode, "condition": condition, "seed": seed,
                             "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue

        command = [
            sys.executable, "part_b_resnet_v8.py",
            "--optimizer", "adam_nv",
            "--response-mode", mode,
            "--schedule", schedule,
            "--sigma", str(sigma),
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

        print(f"[Direction] ({index}/{len(combos)}) starting {run_name}")
        started = time.time()
        try:
            subprocess.run(command, check=True)
            status, error = "success", ""
        except subprocess.CalledProcessError as exc:
            status, error = "FAILED", f"exit code {exc.returncode}"
        rows.append({"mode": mode, "condition": condition, "seed": seed,
                     "status": status,
                     "elapsed_sec": round(time.time() - started, 1),
                     "error": error})
        with open(status_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["mode", "condition", "seed", "status",
                            "elapsed_sec", "error"],
            )
            writer.writeheader()
            writer.writerows(rows)

    failures = sum(row["status"] == "FAILED" for row in rows)
    print(f"[Direction] finished: {len(rows) - failures} complete/skipped, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

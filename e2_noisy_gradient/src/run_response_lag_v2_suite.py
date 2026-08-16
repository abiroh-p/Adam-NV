"""Multi-seed runner for corrected protocol-v2 response-lag diagnostics.

Runs the coupled default and both decoupled noise-tracking settings on the
ramp-up and intermittent schedules. This replaces the invalid seed-0-only
diagnostics produced before the scheduler fix.

Default: 2 schedules x 3 designs x 4 seeds = 24 runs.
"""

import argparse
import csv
import os
import subprocess
import sys
import time


SEEDS = [0, 1, 2, 3]
SCHEDULES = ["ramp_up", "intermittent"]
DESIGNS = ["coupled_default", "decoupled_0.99", "decoupled_0.95"]
FINE_WINDOWS = {
    "ramp_up": (18, 30),
    "intermittent": (37, 45),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--log-dir", default="../logs/diagnose_v2")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--schedules", nargs="+", default=SCHEDULES,
                        choices=SCHEDULES)
    parser.add_argument("--designs", nargs="+", default=DESIGNS,
                        choices=DESIGNS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    combos = [
        (schedule, design, seed)
        for schedule in args.schedules
        for design in args.designs
        for seed in args.seeds
    ]
    os.makedirs(args.log_dir, exist_ok=True)
    status_path = os.path.join(args.log_dir, "_response_lag_v2_suite_status.csv")
    rows = []

    for index, (schedule, design, seed) in enumerate(combos, start=1):
        beta_suffix = ""
        beta_args = []
        if design == "decoupled_0.99":
            beta_suffix = "_beta2noise0.99"
            beta_args = ["--beta2-noise", "0.99"]
        elif design == "decoupled_0.95":
            beta_suffix = "_beta2noise0.95"
            beta_args = ["--beta2-noise", "0.95"]

        run_name = (
            f"adam_nv_schedule{schedule}_sigma{args.sigma}{beta_suffix}"
            f"_seed{seed}_lagdiag_v2"
            + ("_smoke" if args.smoke_test else "")
        )
        epoch_log = os.path.join(args.log_dir, f"{run_name}.csv")
        if args.skip_existing and os.path.exists(epoch_log):
            with open(epoch_log, newline="") as handle:
                reader = csv.DictReader(handle)
                data = list(reader)
            valid_version = data and data[-1].get("protocol_version") == "2"
            if len(data) >= args.epochs and valid_version:
                print(f"[Lag v2] ({index}/{len(combos)}) skipping complete {run_name}")
                rows.append({"schedule": schedule, "design": design, "seed": seed,
                             "status": "skipped", "elapsed_sec": 0, "error": ""})
                continue

        fine_start, fine_end = FINE_WINDOWS[schedule]
        command = [
            sys.executable, "diagnose_response_lag.py",
            "--schedule", schedule,
            "--sigma", str(args.sigma),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--data-dir", args.data_dir,
            "--log-dir", args.log_dir,
            "--fine-log-start", str(fine_start),
            "--fine-log-end", str(fine_end),
            *beta_args,
        ]
        if args.smoke_test:
            command.append("--smoke-test")
        if args.synthetic_data:
            command.append("--synthetic-data")
        if args.deterministic:
            command.append("--deterministic")

        print(f"[Lag v2] ({index}/{len(combos)}) starting {run_name}")
        started = time.time()
        try:
            subprocess.run(command, check=True)
            status, error = "success", ""
        except subprocess.CalledProcessError as exc:
            status, error = "FAILED", f"exit code {exc.returncode}"
        rows.append({"schedule": schedule, "design": design, "seed": seed,
                     "status": status,
                     "elapsed_sec": round(time.time() - started, 1),
                     "error": error})
        with open(status_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["schedule", "design", "seed", "status",
                            "elapsed_sec", "error"],
            )
            writer.writeheader()
            writer.writerows(rows)

    failures = sum(row["status"] == "FAILED" for row in rows)
    print(f"[Lag v2] finished: {len(rows) - failures} complete/skipped, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

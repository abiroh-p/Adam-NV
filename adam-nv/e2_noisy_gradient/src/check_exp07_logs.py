"""Verify completeness and provenance of Exp. 7 logs (label noise 0.2, part_b_resnet_v8).

For every requested "<optimizer>:<seed>" job it checks that
  * the CSV exists and has exactly --epochs data rows,
  * the metadata JSON exists and its arguments match the protocol
    (optimizer, seed, constant schedule, sigma 0, label_noise 0.2, batch 128, deterministic,
     real data, not a smoke test, default lr, fixed_beta1 as requested for adam_nv_fixed),
  * the software stack equals the one used for the earlier label-diagnostic runs
    (Python 3.12.13, torch 2.10.0+cu128, torchvision 0.25.0+cu128).
Exit code is 1 if any check fails.

Example:
    python3 check_exp07_logs.py --log-dir ../logs_exp07 --fixed-beta1 0.81 \
        --jobs adam_nv_fixed:0 adam_nv_fixed:1 adam:4
"""

import argparse
import csv
import json
import os
import sys

STACK = {"python_version": "3.12.13", "torch_version": "2.10.0+cu128",
         "torchvision_version": "0.25.0+cu128"}


def check_job(log_dir, opt, seed, epochs, fixed_beta1):
    base = f"{opt}_scheduleconstant_sigma0.0_ln0.2_seed{seed}"
    csv_path = os.path.join(log_dir, base + ".csv")
    meta_path = os.path.join(log_dir, base + "_metadata.json")
    problems = []
    if not os.path.isfile(csv_path):
        return ["missing CSV"]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != epochs:
        problems.append(f"{len(rows)}/{epochs} epochs")
    if not os.path.isfile(meta_path):
        problems.append("missing metadata")
        return problems
    with open(meta_path) as f:
        meta = json.load(f)
    a = meta.get("arguments", {})
    expect = {"optimizer": opt, "seed": seed, "schedule": "constant", "sigma": 0.0,
              "label_noise": 0.2, "batch_size": 128, "epochs": epochs,
              "deterministic": True, "smoke_test": False, "synthetic_data": False,
              "lr": None, "dataset": "cifar10"}
    if opt == "adam_nv_fixed":
        expect["fixed_beta1"] = fixed_beta1
    else:
        expect["fixed_beta1"] = None
    for k, v in expect.items():
        got = a.get(k)
        if isinstance(v, float) and isinstance(got, (int, float)):
            same = abs(got - v) < 1e-12
        else:
            same = got == v
        if not same:
            problems.append(f"argument {k}: got {got!r}, expected {v!r}")
    for k, v in STACK.items():
        if meta.get(k) != v:
            problems.append(f"{k}: got {meta.get(k)!r}, expected {v!r}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fixed-beta1", type=float, default=0.81)
    ap.add_argument("--jobs", nargs="+", required=True, help="<optimizer>:<seed> ...")
    args = ap.parse_args()

    failed = 0
    print(f"{'job':24s} result")
    for job in args.jobs:
        opt, seed = job.split(":")
        problems = check_job(args.log_dir, opt, int(seed), args.epochs, args.fixed_beta1)
        print(f"{job:24s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
        failed += bool(problems)
    print(f"\n{len(args.jobs) - failed}/{len(args.jobs)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

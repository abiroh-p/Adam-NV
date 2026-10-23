"""Pre-upload check for E1-style run logs ({optimizer}_seed{N}.csv + metadata).

For every expected (optimizer, seed) pair it verifies:
  * the CSV exists and has exactly --epochs data rows;
  * no NaN/inf in any numeric column and val_acc is a valid fraction;
  * the metadata JSON exists and records the same optimizer and seed,
    cuda_available == True, and the expected torch / torchvision versions;
  * metadata contains code_sha256 hashes (added in Protocol v2).

Usage:
    python3 check_logs.py --log-dir ../logs --optimizers gono --seeds 0 1 2 3 4 5 6 7 8 9

Exit code 0 only if every check passes.
"""

import argparse
import csv
import json
import math
import os
import sys


def check_one(log_dir, opt, seed, epochs, expect_torch, expect_tv, tag="", lr=None):
    problems = []
    base = os.path.join(log_dir, f"{opt}{tag}_seed{seed}")
    csv_path, meta_path = base + ".csv", base + "_metadata.json"
    if not os.path.exists(csv_path):
        return [f"missing CSV {csv_path}"]
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != epochs:
        problems.append(f"{len(rows)} data rows, expected {epochs}")
    for r in rows:
        for k, v in r.items():
            try:
                x = float(v)
            except (TypeError, ValueError):
                problems.append(f"non-numeric {k}={v!r} at epoch {r.get('epoch')}")
                continue
            if not math.isfinite(x):
                problems.append(f"non-finite {k} at epoch {r.get('epoch')}")
        if rows and not 0.0 <= float(r["val_acc"]) <= 1.0:
            problems.append(f"val_acc out of range at epoch {r['epoch']}")
    if not os.path.exists(meta_path):
        problems.append("missing metadata JSON")
        return problems
    with open(meta_path) as f:
        meta = json.load(f)
    args = meta.get("arguments", {})
    if args.get("optimizer") != opt:
        problems.append(f"metadata optimizer={args.get('optimizer')!r}")
    if lr is not None and args.get("lr") != lr:
        problems.append(f"metadata lr={args.get('lr')!r}, expected {lr}")
    if args.get("seed") != seed:
        problems.append(f"metadata seed={args.get('seed')!r}")
    if meta.get("cuda_available") is not True:
        problems.append("cuda_available is not True")
    if expect_torch and meta.get("torch_version") != expect_torch:
        problems.append(f"torch_version={meta.get('torch_version')} (expected {expect_torch})")
    if expect_tv and meta.get("torchvision_version") != expect_tv:
        problems.append(f"torchvision_version={meta.get('torchvision_version')} (expected {expect_tv})")
    if "code_sha256" not in meta:
        problems.append("metadata has no code_sha256 (run with the patched scripts)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--optimizers", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--run-tag", default="", help="file-name suffix, e.g. _lr3e-4")
    ap.add_argument("--lr", type=float, default=None, help="expected learning rate recorded in metadata")
    ap.add_argument("--expect-torch", default="2.10.0+cu128")
    ap.add_argument("--expect-torchvision", default="0.25.0+cu128")
    args = ap.parse_args()

    bad = 0
    for opt in args.optimizers:
        for seed in args.seeds:
            problems = check_one(args.log_dir, opt, seed, args.epochs,
                                 args.expect_torch, args.expect_torchvision,
                                 args.run_tag, args.lr)
            tag = "PASS" if not problems else "FAIL"
            print(f"[{tag}] {opt}{args.run_tag}_seed{seed}" + ("" if not problems else ": " + "; ".join(problems)))
            bad += bool(problems)
    print(f"\n{bad} problem run(s)" if bad else "\nALL CHECKS PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

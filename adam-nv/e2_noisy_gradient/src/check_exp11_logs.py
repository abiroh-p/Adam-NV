"""Verify completeness and provenance of Exp. 11 logs (Adam-NV beta_1 shuffle control).

For every requested seed it checks, in addition to the Exp. 7 checks (CSV with --epochs rows;
metadata arguments: adam_nv, constant schedule, sigma 0, label noise 0.2, batch 128,
deterministic, real data, default lr, software stack), that
  * shuffle_config_seed<k>.json exists and records the requested shuffle mode and seed,
  * beta1_dist_<mode>_seed<k>.jsonl exists with one record per epoch.
Exit code 1 if any check fails.

Example:
    python3 check_exp11_logs.py --log-dir ../logs_exp11_step_A --mode step --seeds 0 1 2 3 4 5
"""

import argparse
import json
import os
import sys

from check_exp07_logs import check_job


def check_seed(log_dir, seed, epochs, mode):
    problems = check_job(log_dir, "adam_nv", seed, epochs, None)
    cfg_path = os.path.join(log_dir, f"shuffle_config_seed{seed}.json")
    dist_path = os.path.join(log_dir, f"beta1_dist_{mode}_seed{seed}.jsonl")
    if not os.path.isfile(cfg_path):
        problems.append("missing shuffle_config sidecar")
    else:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        if cfg.get("shuffle_mode") != mode:
            problems.append(f"sidecar mode {cfg.get('shuffle_mode')!r}, expected {mode!r}")
        if cfg.get("seed") != seed:
            problems.append(f"sidecar seed {cfg.get('seed')!r}, expected {seed}")
    if not os.path.isfile(dist_path):
        problems.append("missing beta1 distribution log")
    else:
        with open(dist_path) as fh:
            n = sum(1 for line in fh if line.strip())
        if n != epochs:
            problems.append(f"beta1 distribution log has {n} records, expected {epochs}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mode", required=True, choices=["none", "step", "fixed"])
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    failed = 0
    print(f"{'job':24s} result")
    for seed in args.seeds:
        job = f"adam_nv_{args.mode}:{seed}"
        problems = check_seed(args.log_dir, seed, args.epochs, args.mode)
        print(f"{job:24s} {'OK' if not problems else 'FAIL: ' + '; '.join(problems)}")
        failed += bool(problems)
    print(f"\n{len(args.seeds) - failed}/{len(args.seeds)} jobs verified")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

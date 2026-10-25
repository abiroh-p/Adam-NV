"""Run part_b_resnet_v8.main() with AdamNV replaced by AdamNVShuffle (Exp. 11).

part_b_resnet_v8.py and adam_nv.py are NOT modified (their recorded hashes stay valid). This
wrapper

  1. removes its own options from the command line,
  2. replaces the AdamNV name inside part_b_resnet_v8 by a subclass that applies the requested
     beta_1 shuffle and logs the element-level beta_1 distribution,
  3. writes shuffle_config_seed<k>.json next to the run's logs, and
  4. calls part_b_resnet_v8.main() with the remaining arguments unchanged.

Use it ONLY with --optimizer adam_nv (the file names are those of a normal Adam-NV run, so give
every shuffle mode its own --log-dir).

Own options:
  --shuffle-mode {none,step,fixed}   required
  --dist-log-every N                 steps between beta_1 distribution records (default 391,
                                     one per epoch for batch size 128 on 50,000 images)

Example:
    python3 part_b_resnet_v8_shuffle.py --shuffle-mode step --optimizer adam_nv --schedule constant \\
        --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 --epochs 60 \\
        --batch-size 128 --data-dir ../data --log-dir ../logs_exp11_step --deterministic
"""

import argparse
import json
import os
import sys
import time


def _split_args():
    own_parser = argparse.ArgumentParser(add_help=False)
    own_parser.add_argument("--shuffle-mode", choices=["none", "step", "fixed"], required=True)
    own_parser.add_argument("--dist-log-every", type=int, default=391)
    own, rest = own_parser.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--seed", type=int, required=True)
    peek.add_argument("--log-dir", default=None)
    peek.add_argument("--optimizer", default=None)
    pk, _ = peek.parse_known_args(rest)
    if pk.optimizer != "adam_nv":
        sys.exit("part_b_resnet_v8_shuffle.py must be used with --optimizer adam_nv")
    if pk.log_dir is None:
        sys.exit("--log-dir is required")
    return own, rest, pk


def main():
    own, rest, pk = _split_args()
    os.makedirs(pk.log_dir, exist_ok=True)
    dist_path = os.path.join(pk.log_dir, f"beta1_dist_{own.shuffle_mode}_seed{pk.seed}.jsonl")
    if os.path.exists(dist_path):
        os.remove(dist_path)  # an interrupted earlier attempt must not leave partial records

    import part_b_resnet_v8 as v8
    from adam_nv_shuffle import AdamNVShuffle

    shuffle_seed = 1_000_003 + pk.seed

    class _Patched(AdamNVShuffle):
        def __init__(self, params, **kwargs):
            super().__init__(params, shuffle=own.shuffle_mode, shuffle_seed=shuffle_seed,
                             dist_log_path=dist_path, dist_log_every=own.dist_log_every, **kwargs)

    v8.AdamNV = _Patched

    config = {"shuffle_mode": own.shuffle_mode, "shuffle_seed": shuffle_seed, "seed": pk.seed,
              "dist_log_every": own.dist_log_every, "dist_log_file": os.path.basename(dist_path),
              "argv": rest, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(pk.log_dir, f"shuffle_config_seed{pk.seed}.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    sys.argv = [sys.argv[0]] + rest
    v8.main()


if __name__ == "__main__":
    main()

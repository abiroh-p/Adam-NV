"""Paired analysis for Exp. 6 (constant-beta1 Adam controls) against Exp. 5 logs.

Reads <opt><tag>_seed<k>.csv from one or more log directories (no recursion into other
names), compares FINAL-epoch val_acc per seed, and prints per-optimizer mean/std plus paired
differences versus a reference optimizer.

Example (after Exp. 6, with Exp. 5 logs unzipped next to it):
    python3 analyze_exp06.py --log-dirs logs_exp05 logs_exp06 --tag _lr3e-3 \
        --opts adam adam_nv adam_b083 adam_b080 --ref adam_nv

Sanity check on Exp. 5 data alone (no Exp. 6 needed):
    python3 analyze_exp06.py --log-dirs logs_exp05 --tag _lr3e-3 --opts adam adam_nv --ref adam
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats


def load_final(log_dirs, opt, tag, epochs):
    out = {}
    pat = re.compile(rf"^{re.escape(opt)}{re.escape(tag)}_seed(\d+)\.csv$")
    for d in log_dirs:
        for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
            m = pat.match(os.path.basename(f))
            if not m:
                continue
            df = pd.read_csv(f)
            if len(df) < epochs:
                print(f"[skip] {f}: {len(df)}/{epochs} epochs")
                continue
            out[int(m.group(1))] = float(df["val_acc"].iloc[epochs - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--tag", default="_lr3e-3")
    ap.add_argument("--opts", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="reference optimizer for paired differences")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    data = {o: load_final(args.log_dirs, o, args.tag, args.epochs) for o in args.opts}
    print("\nFinal-epoch val_acc (percent):")
    for o, d in data.items():
        v = np.array([d[k] for k in sorted(d)]) * 100
        if len(v):
            print(f"  {o:12s} n={len(v)} mean={v.mean():.3f} std={v.std(ddof=1) if len(v) > 1 else float('nan'):.3f}")
        else:
            print(f"  {o:12s} no complete runs found")

    ref = data[args.ref]
    print(f"\nPaired differences vs {args.ref} (percent points, same seeds):")
    for o, d in data.items():
        if o == args.ref:
            continue
        seeds = sorted(set(d) & set(ref))
        if len(seeds) < 2:
            print(f"  {o:12s} fewer than 2 shared seeds")
            continue
        diff = np.array([d[s] - ref[s] for s in seeds]) * 100
        t = stats.ttest_1samp(diff, 0.0)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=se)
        print(f"  {o:12s} n={len(seeds)} mean diff={diff.mean():+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"p={t.pvalue:.3f}  seeds higher: {(diff > 0).sum()}/{len(diff)}")
    print("\nNote: with n=4 these intervals are wide; read them as effect-size estimates, not proof.")


if __name__ == "__main__":
    main()

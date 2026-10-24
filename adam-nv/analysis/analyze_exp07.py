"""Paired analysis for Exp. 7 (label noise 0.2, constant-beta1 control, seeds 0-7).

Arms: adam, adam_nv, adam_nv_fixed (= Adam with constant beta1).
Primary metric : mean over the last 5 epochs of train_corrupt_noisy_label_acc
                 (fraction of corrupted training examples fitted to their WRONG label),
                 in percentage points = the memorization measure.
Secondary      : last-5 val_acc, last-5 train_clean_subset_acc, peak val_acc.
Statistics     : paired by seed; mean difference, t-based 95% CI, paired t p, exact sign-flip
                 p, Holm adjustment over the three contrasts of each metric.

Pre-registered decision rule (primary metric; M = adam_nv - adam, F = adam_nv_fixed - adam,
D = adam_nv - adam_nv_fixed):
  * effect reproduced   : the 95% CI of M excludes 0 and M > 0
  * EXPLAINED BY LOWER BETA1 : F >= 0.5 * M and the 95% CI of D includes 0
  * MOSTLY EXPLAINED, SMALL RESIDUAL : F >= 0.5 * M but the 95% CI of D is above 0
                         (adaptation adds at most D on top of the lower-beta1 effect)
  * ADAPTATION BEYOND BETA1  : the 95% CI of D excludes 0 with D > 0 and F < 0.25 * M
  * otherwise INCONCLUSIVE (extend seeds).

Example:
    python3 analyze_exp07.py --orig-dir Label_Diagnostic \
        --new-dirs logs_exp07_A logs_exp07_B logs_exp07_C --canary-dir logs_exp07_canary
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

NAME = "{opt}_scheduleconstant_sigma0.0_ln0.2_seed{seed}.csv"
ARMS = ["adam", "adam_nv", "adam_nv_fixed"]
LABEL = {"adam": "Adam (beta1=0.9)", "adam_nv": "Adam-NV", "adam_nv_fixed": "Adam constant beta1"}
METRICS = {
    "mem5": ("Memorization: corrupted examples fitted to wrong label, last-5 (pp)",
             lambda d: d["train_corrupt_noisy_label_acc"].tail(5).mean() * 100),
    "val5": ("Validation accuracy, last-5 (pp)", lambda d: d["val_acc"].tail(5).mean() * 100),
    "clean5": ("Clean-subset train accuracy, last-5 (pp)",
               lambda d: d["train_clean_subset_acc"].tail(5).mean() * 100),
    "valpeak": ("Peak validation accuracy (pp)", lambda d: d["val_acc"].max() * 100),
}


def find_runs(orig_dir, new_dirs, seeds):
    paths = {}
    for d in [orig_dir] + list(new_dirs):
        if not d:
            continue
        for opt in ARMS:
            for s in seeds:
                p = os.path.join(d, NAME.format(opt=opt, seed=s))
                if os.path.isfile(p) and (opt, s) not in paths:
                    paths[(opt, s)] = p
    return paths


def flip_p(d):
    obs = abs(d.mean())
    hits = total = 0
    for sg in itertools.product([1, -1], repeat=len(d)):
        total += 1
        hits += abs((np.array(sg) * d).mean()) >= obs - 1e-12
    return hits / total


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def contrast(table, a, b):
    seeds = sorted(set(table[a]) & set(table[b]))
    x = np.array([table[a][s] - table[b][s] for s in seeds])
    n = len(x)
    if n < 2:
        return None
    se = x.std(ddof=1) / np.sqrt(n)
    if se == 0:  # identical runs (e.g. a control equal to the reference): no spread
        lo = hi = x.mean()
        p = 1.0 if x.mean() == 0 else 0.0
    else:
        lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=se)
        p = stats.ttest_1samp(x, 0.0).pvalue
    return dict(a=a, b=b, n=n, mean=x.mean(), lo=lo, hi=hi,
                p=p, pflip=flip_p(x), up=int((x > 0).sum()))


def canary_report(orig_dir, canary_dir):
    print("\nCanary (re-run of seed 0 with current code vs the original logs):")
    for opt in ["adam", "adam_nv"]:
        po = os.path.join(orig_dir, NAME.format(opt=opt, seed=0))
        pc = os.path.join(canary_dir, NAME.format(opt=opt, seed=0))
        if not (os.path.isfile(po) and os.path.isfile(pc)):
            print(f"  {opt}: missing file(s)")
            continue
        a, b = pd.read_csv(po), pd.read_csv(pc)
        cols = [c for c in a.columns if c != "epoch_time_sec" and c in b.columns]
        if len(a) != len(b):
            print(f"  {opt}: different length {len(a)} vs {len(b)}")
            continue
        diff = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))
        print(f"  {opt}: max abs difference over {len(cols)} columns = {diff:.3e} "
              f"({'IDENTICAL' if diff == 0 else 'NOT identical'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True, help="existing Label_Diagnostic logs")
    ap.add_argument("--new-dirs", nargs="+", default=[])
    ap.add_argument("--canary-dir", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--fixed-beta1", type=float, default=0.81)
    args = ap.parse_args()

    paths = find_runs(args.orig_dir, args.new_dirs, args.seeds)
    data = {m: {a: {} for a in ARMS} for m in METRICS}
    beta1 = {a: [] for a in ARMS}
    for (opt, s), p in paths.items():
        df = pd.read_csv(p)
        if len(df) != 60:
            print(f"[skip] {p}: {len(df)}/60 epochs")
            continue
        for m, (_, fn) in METRICS.items():
            data[m][opt][s] = fn(df)
        if opt == "adam_nv":
            beta1[opt].append(df["beta1_mean"].mean())

    print("Seeds available per arm:", {a: sorted(data["mem5"][a]) for a in ARMS})
    if beta1["adam_nv"]:
        b = float(np.mean(beta1["adam_nv"]))
        print(f"Adam-NV time-averaged beta1 = {b:.3f}; control beta1 = {args.fixed_beta1:.2f}"
              + ("" if abs(b - args.fixed_beta1) <= 0.02 else "  <-- MISMATCH > 0.02"))

    pairs = [("adam_nv", "adam"), ("adam_nv_fixed", "adam"), ("adam_nv", "adam_nv_fixed")]
    results = {}
    for m, (title, _) in METRICS.items():
        print(f"\n=== {title} ===")
        for a in ARMS:
            v = np.array(list(data[m][a].values()))
            if len(v):
                print(f"  {LABEL[a]:22s} n={len(v)}  mean={v.mean():.2f}  sd={v.std(ddof=1) if len(v) > 1 else float('nan'):.2f}")
        rs = [contrast(data[m], a, b) for a, b in pairs]
        valid = [r for r in rs if r]
        if len(valid) == len(pairs):
            adj = holm([r["p"] for r in valid])
        else:
            adj = [float("nan")] * len(valid)
        for r, ap_ in zip(valid, adj):
            print(f"  {LABEL[r['a']]} - {LABEL[r['b']]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                  f"p={r['p']:.4f}  p_Holm={ap_:.4f}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']} seeds")
        results[m] = {(r["a"], r["b"]): r for r in valid}

    r = results.get("mem5", {})
    M = r.get(("adam_nv", "adam"))
    F = r.get(("adam_nv_fixed", "adam"))
    D = r.get(("adam_nv", "adam_nv_fixed"))
    print("\n=== Pre-registered decision (primary metric) ===")
    if not (M and F and D):
        print("  Not enough complete arms/seeds to apply the rule.")
    else:
        reproduced = M["lo"] > 0
        print(f"  Effect reproduced (Adam-NV - Adam CI above 0): {reproduced}  (M = {M['mean']:+.2f} pp)")
        if not reproduced:
            print("  OUTCOME: effect not reproduced; the control question is moot. Report that.")
        elif F["mean"] >= 0.5 * M["mean"] and D["lo"] <= 0 <= D["hi"]:
            print(f"  OUTCOME: EXPLAINED BY LOWER BETA1 (F = {F['mean']:+.2f} >= 0.5*M, D CI includes 0)")
        elif F["mean"] >= 0.5 * M["mean"] and D["lo"] > 0:
            print(f"  OUTCOME: MOSTLY EXPLAINED BY LOWER BETA1, SMALL RESIDUAL "
                  f"(F = {F['mean']:+.2f} >= 0.5*M; D = {D['mean']:+.2f}, CI above 0)")
        elif D["lo"] > 0 and F["mean"] < 0.25 * M["mean"]:
            print(f"  OUTCOME: ADAPTATION BEYOND BETA1 (D = {D['mean']:+.2f} CI above 0, F < 0.25*M)")
        else:
            print("  OUTCOME: INCONCLUSIVE; extend seeds before claiming either.")

    if args.canary_dir:
        canary_report(args.orig_dir, args.canary_dir)


if __name__ == "__main__":
    main()

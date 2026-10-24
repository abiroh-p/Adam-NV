"""Exp. 8 analysis: dose-response of constant-beta1 Adam against Adam-NV under 20% label noise.

Arms (all seeds 0-7, paired by seed):
  adam          beta1 = 0.9           (Label_Diagnostic seeds 0-3 + Exp. 7 seeds 4-7)
  adam_nv       Adam-NV               (Label_Diagnostic seeds 0-3 + Exp. 7 seeds 4-7)
  c081          constant beta1 = 0.81 (Exp. 7)
  c078          constant beta1 = 0.78 (Exp. 8, the late-training mean of Adam-NV's beta1)
  c070          constant beta1 = 0.70 (Exp. 8, a stronger-than-observed control)

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
M = adam_nv - adam; F_c = c - adam; D_c = adam_nv - c.   Holm over {F_078, F_070, D_078, D_070}.

Pre-registered rule (written before any Exp. 8 run):
  * lower constant beta1 EXPLAINS the effect : for some c in {0.78, 0.70}, F_c >= 0.5*M and the
                                               95% CI of D_c includes 0
  * ADAPTATION BEYOND ANY CONSTANT beta1     : for both c, F_c < 0.25*M and the 95% CI of D_c
                                               is above 0
  * otherwise PARTIAL / INCONCLUSIVE (report the dose-response and extend seeds).

Example:
    python3 analyze_exp08.py --orig-dir Label_Diagnostic \
        --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
        --c078-dirs logs_exp08_b078_A logs_exp08_b078_B \
        --c070-dirs logs_exp08_b070_A logs_exp08_b070_B logs_exp08_b070_C
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze_exp07 as e7  # same folder: reuses flip_p, holm, contrast, metric definitions

NAME = e7.NAME


def load_arm(opt, dirs, seeds):
    out = {}
    for d in dirs:
        for s in seeds:
            p = os.path.join(d, NAME.format(opt=opt, seed=s))
            if os.path.isfile(p) and s not in out:
                df = pd.read_csv(p)
                if len(df) == 60:
                    out[s] = e7.METRICS["mem5"][1](df)
                else:
                    print(f"[skip] {p}: {len(df)}/60 epochs")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--c078-dirs", nargs="+", default=[])
    ap.add_argument("--c070-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base_dirs = [args.orig_dir] + args.exp07_dirs
    table = {
        "adam": load_arm("adam", base_dirs, args.seeds),
        "adam_nv": load_arm("adam_nv", base_dirs, args.seeds),
        "c081": load_arm("adam_nv_fixed", args.exp07_dirs, args.seeds),
        "c078": load_arm("adam_nv_fixed", args.c078_dirs, args.seeds),
        "c070": load_arm("adam_nv_fixed", args.c070_dirs, args.seeds),
    }
    beta = {"adam": 0.90, "c081": 0.81, "c078": 0.78, "c070": 0.70}
    label = {"adam": "Adam (0.90)", "c081": "constant 0.81", "c078": "constant 0.78",
             "c070": "constant 0.70", "adam_nv": "Adam-NV"}

    print("Memorization (corrupted examples fitted to wrong label, last-5, pp):")
    for k in ["adam", "c081", "c078", "c070", "adam_nv"]:
        v = np.array([table[k][s] for s in sorted(table[k])])
        if len(v) > 1:
            print(f"  {label[k]:15s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
        else:
            print(f"  {label[k]:15s} n={len(v)} (not available)")

    pairs = [("c078", "adam"), ("c070", "adam"), ("adam_nv", "c078"), ("adam_nv", "c070")]
    res = {}
    for a, b in pairs:
        r = e7.contrast(table, a, b) if min(len(table[a]), len(table[b])) > 1 else None
        res[(a, b)] = r
    valid = [r for r in res.values() if r]
    adj = dict(zip([id(r) for r in valid], e7.holm([r["p"] for r in valid]))) if len(valid) == 4 else {}

    M = e7.contrast(table, "adam_nv", "adam") if len(table["adam_nv"]) > 1 else None
    if M:
        print(f"\nReference effect M = Adam-NV - Adam: {M['mean']:+.2f} pp "
              f"[{M['lo']:+.2f}, {M['hi']:+.2f}] ({M['up']}/{M['n']} seeds)")
    print("\nPaired contrasts (Holm over the four):")
    for (a, b), r in res.items():
        if not r:
            print(f"  {label[a]} - {label[b]}: not enough paired seeds")
            continue
        print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  p_Holm={adj.get(id(r), float('nan')):.4f}  sign-flip p={r['pflip']:.4f}  "
              f"higher in {r['up']}/{r['n']}")

    xs, ys = [], []
    for k in ["adam", "c081", "c078", "c070"]:
        for v in table[k].values():
            xs.append(beta[k]); ys.append(v)
    if len(set(xs)) >= 3:
        lr = stats.linregress(xs, ys)
        print(f"\nDose-response over constant beta1 (all runs pooled): slope = {lr.slope:+.1f} pp per unit "
              f"beta1 (r = {lr.rvalue:+.2f}, p = {lr.pvalue:.3g}); a negative slope means lower beta1 "
              f"gives more memorization. Descriptive only.")

    print("\n=== Pre-registered decision ===")
    if not (M and all(res.values())):
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    F = {c: e7.contrast(table, c, "adam") for c in ["c078", "c070"]}
    D = {c: res[("adam_nv", c)] for c in ["c078", "c070"]}
    explains = [c for c in F if F[c]["mean"] >= 0.5 * M["mean"] and D[c]["lo"] <= 0 <= D[c]["hi"]]
    beyond = all(F[c]["mean"] < 0.25 * M["mean"] and D[c]["lo"] > 0 for c in F)
    if M["lo"] <= 0:
        print("  Adam-NV effect not reproduced in this pooled set; the rule does not apply.")
    elif explains:
        print(f"  OUTCOME: LOWER CONSTANT BETA1 EXPLAINS THE EFFECT (control(s): {', '.join(label[c] for c in explains)})")
    elif beyond:
        print("  OUTCOME: ADAPTATION BEYOND ANY CONSTANT BETA1 (both controls < 0.25*M, D CIs above 0)")
    else:
        print("  OUTCOME: PARTIAL / INCONCLUSIVE; read the dose-response and extend seeds.")


if __name__ == "__main__":
    main()

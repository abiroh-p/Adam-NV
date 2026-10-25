"""Exp. 9 analysis: Adam-NV with response_mode=smooth under 20% label noise (seeds 0-7).

Arms (paired by seed):
  adam         beta1 = 0.9                         (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  adam_nv      Adam-NV, response_mode=distrust     (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  nv_smooth    Adam-NV, response_mode=smooth       (Exp. 9, beta_max = 0.999)

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
M = adam_nv - adam; F = nv_smooth - adam; D = adam_nv - nv_smooth.
Holm over the two contrasts {F, D}.

Pre-registered rule (written before any Exp. 9 run):
  1. Effect reproduced      : the 95% CI of M is above 0 (otherwise the rule does not apply).
  2. DISTRUST DIRECTION DRIVES THE EFFECT : F < 0.25*M and the 95% CI of D is above 0
                              (letting beta1 rise removes most of the extra memorization).
  3. DIRECTION DOES NOT MATTER           : F >= 0.5*M and the 95% CI of D includes 0
                              (the cause is the noise signal or per-element heterogeneity).
  4. otherwise PARTIAL / INCONCLUSIVE.

Example:
    python3 analyze_exp09.py --orig-dir Label_Diagnostic \
        --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C --smooth-dirs logs_exp09_A logs_exp09_B
"""

import argparse
import os

import numpy as np
import pandas as pd

import analyze_exp07 as e7  # same folder

SMOOTH_NAME = "adam_nv_scheduleconstant_sigma0.0_responsesmooth_betamax{bm}_ln0.2_seed{seed}.csv"


def load(name_fn, dirs, seeds):
    out = {m: {} for m in e7.METRICS}
    b1 = []
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, name_fn(s))
            if os.path.isfile(p):
                df = pd.read_csv(p)
                if len(df) != 60:
                    print(f"[skip] {p}: {len(df)}/60 epochs")
                    break
                for m, (_, fn) in e7.METRICS.items():
                    out[m][s] = fn(df)
                if "beta1_mean" in df.columns:
                    b1.append((df["beta1_mean"].mean(), df["beta1_min"].min(), df["beta1_max"].max()))
                break
    return out, b1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--smooth-dirs", nargs="+", default=[])
    ap.add_argument("--beta-max", default="0.999")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base = [args.orig_dir] + args.exp07_dirs
    A, _ = load(lambda s: e7.NAME.format(opt="adam", seed=s), base, args.seeds)
    N, b1n = load(lambda s: e7.NAME.format(opt="adam_nv", seed=s), base, args.seeds)
    S, b1s = load(lambda s: SMOOTH_NAME.format(bm=args.beta_max, seed=s), args.smooth_dirs, args.seeds)
    label = {"adam": "Adam (0.90)", "adam_nv": "Adam-NV (distrust)", "smooth": "Adam-NV (smooth)"}

    for m, (title, _) in e7.METRICS.items():
        print(f"\n=== {title} ===")
        table = {"adam": A[m], "adam_nv": N[m], "smooth": S[m]}
        for k, t in table.items():
            v = np.array([t[s] for s in sorted(t)])
            if len(v) > 1:
                print(f"  {label[k]:20s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
            else:
                print(f"  {label[k]:20s} n={len(v)} (not available)")
        pairs = [("adam_nv", "adam"), ("smooth", "adam"), ("adam_nv", "smooth")]
        if min(len(t) for t in table.values()) > 1:
            rs = [e7.contrast(table, a, b) for a, b in pairs]
            fam = [rs[1], rs[2]]
            adj = dict(zip([id(r) for r in fam], e7.holm([r["p"] for r in fam])))
            for (a, b), r in zip(pairs, rs):
                ph = adj.get(id(r))
                ph_txt = f"  p_Holm={ph:.4f}" if ph is not None else "  (reference contrast)"
                print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                      f"p={r['p']:.4f}{ph_txt}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")

    if b1n:
        print(f"\nbeta1 (time-averaged mean, lowest element-level min, highest max): "
              f"distrust {np.mean([x[0] for x in b1n]):.3f}, min {min(x[1] for x in b1n):.3f}, max {max(x[2] for x in b1n):.3f}")
    if b1s:
        print(f"                                                                   "
              f"smooth   {np.mean([x[0] for x in b1s]):.3f}, min {min(x[1] for x in b1s):.3f}, max {max(x[2] for x in b1s):.3f}")

    print("\n=== Pre-registered decision (primary metric) ===")
    table = {"adam": A["mem5"], "adam_nv": N["mem5"], "smooth": S["mem5"]}
    if min(len(t) for t in table.values()) < 2:
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    M = e7.contrast(table, "adam_nv", "adam")
    F = e7.contrast(table, "smooth", "adam")
    D = e7.contrast(table, "adam_nv", "smooth")
    print(f"  M = {M['mean']:+.2f} pp, F = {F['mean']:+.2f} pp, D = {D['mean']:+.2f} pp "
          f"(CI [{D['lo']:+.2f}, {D['hi']:+.2f}])")
    if M["lo"] <= 0:
        print("  OUTCOME: Adam-NV effect not reproduced in this pooled set; the rule does not apply.")
    elif F["mean"] < 0.25 * M["mean"] and D["lo"] > 0:
        print("  OUTCOME: DISTRUST DIRECTION DRIVES THE EFFECT (F < 0.25*M, D CI above 0)")
    elif F["mean"] >= 0.5 * M["mean"] and D["lo"] <= 0 <= D["hi"]:
        print("  OUTCOME: DIRECTION DOES NOT MATTER (F >= 0.5*M, D CI includes 0)")
    else:
        print("  OUTCOME: PARTIAL / INCONCLUSIVE; extend seeds before claiming either.")


if __name__ == "__main__":
    main()

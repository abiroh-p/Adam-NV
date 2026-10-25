"""Exp. 11 analysis: beta_1 shuffle control for Adam-NV under 20% label noise (seeds 0-7).

Arms (paired by seed):
  adam      Adam, beta1 = 0.9                    (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  adam_nv   Adam-NV                              (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  step      Adam-NV, beta_1 permuted every step  (Exp. 11)
  fixed     Adam-NV, one fixed permutation       (Exp. 11)
  none      Adam-NV through the shuffle wrapper, no permutation (Exp. 11, seeds 0-1): a canary
            that must reproduce the existing Adam-NV logs exactly, and the source of the
            element-level beta_1 distribution.

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
M = adam_nv - adam; for S in {step, fixed}: F_S = S - adam, D_S = adam_nv - S.
Holm over {F_step, F_fixed, D_step, D_fixed, fixed - step}.

Pre-registered rule (written before any Exp. 11 run):
  0. If the 95% CI of M is not above 0, the rule does not apply.
  1. S RETAINS the effect : F_S >= 0.5*M and the 95% CI of D_S includes 0.
  2. S REMOVES the effect : F_S < 0.25*M and the 95% CI of D_S is above 0.
  3. otherwise S is PARTIAL.
Interpretation of the pair (step, fixed):
  both retain          -> the effect does not need the element-wise link between beta_1 and the
                          element's own noise score (it follows from the distribution or magnitude
                          of beta_1);
  both remove          -> the effect requires that element-wise link;
  fixed retains, step removes -> it needs each element's beta_1 to be temporally consistent;
  anything else        -> inconclusive; report all three contrasts.

Example:
    python3 analyze_exp11.py --orig-dir Label_Diagnostic --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
        --step-dirs logs_exp11_step_A logs_exp11_step_C --fixed-dirs logs_exp11_fixed_B logs_exp11_fixed_C \
        --none-dirs logs_exp11_none_C
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import analyze_exp07 as e7  # same folder: flip_p, holm, contrast, METRICS, NAME

NAME = e7.NAME


def load_arm(opt, dirs, seeds):
    out = {m: {} for m in e7.METRICS}
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, NAME.format(opt=opt, seed=s))
            if os.path.isfile(p):
                df = pd.read_csv(p)
                if len(df) != 60:
                    print(f"[skip] {p}: {len(df)}/60 epochs")
                    break
                for m, (_, fn) in e7.METRICS.items():
                    out[m][s] = fn(df)
                break
    return out


def canary(orig_dirs, none_dirs):
    print("\nCanary: shuffle wrapper with mode=none against the existing Adam-NV logs")
    for d in none_dirs:
        for f in sorted(glob.glob(os.path.join(d, "adam_nv_scheduleconstant_sigma0.0_ln0.2_seed?.csv"))):
            base = os.path.basename(f)
            ref = next((os.path.join(o, base) for o in orig_dirs if os.path.isfile(os.path.join(o, base))), None)
            if ref is None:
                print(f"  {base}: no reference log")
                continue
            a, b = pd.read_csv(ref), pd.read_csv(f)
            cols = [c for c in a.columns if c != "epoch_time_sec" and c in b.columns]
            if len(a) != len(b):
                print(f"  {base}: different length {len(a)} vs {len(b)}")
                continue
            diff = float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))
            print(f"  {base}: max abs difference over {len(cols)} columns = {diff:.3e} "
                  f"({'IDENTICAL' if diff == 0 else 'NOT identical'})")


def distribution_summary(dirs, label):
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "beta1_dist_*_seed?.jsonl"))]
    if not files:
        return
    recs = []
    for f in files:
        rows = [json.loads(x) for x in open(f) if x.strip()]
        recs.append(rows)
    n = min(len(r) for r in recs)
    print(f"\nElement-level beta_1 distribution of the controller ({label}; mean over {len(recs)} runs)")
    keys = ["mean", "min", "q01", "q05", "q25", "q50", "q75", "q95", "frac_below_0.5", "frac_below_0.7"]
    print("  epoch " + " ".join(f"{k:>14s}" for k in keys))
    for e in [1, 10, 20, 40, 60]:
        if e <= n:
            print(f"  {e:5d} " + " ".join(f"{np.mean([r[e - 1][k] for r in recs]):14.4f}" for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--step-dirs", nargs="+", default=[])
    ap.add_argument("--fixed-dirs", nargs="+", default=[])
    ap.add_argument("--none-dirs", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base = [args.orig_dir] + args.exp07_dirs
    A = load_arm("adam", base, args.seeds)
    N = load_arm("adam_nv", base, args.seeds)
    S = load_arm("adam_nv", args.step_dirs, args.seeds)
    F = load_arm("adam_nv", args.fixed_dirs, args.seeds)
    label = {"adam": "Adam (0.90)", "adam_nv": "Adam-NV", "step": "Adam-NV, shuffle/step",
             "fixed": "Adam-NV, shuffle/fixed"}

    for m, (title, _) in e7.METRICS.items():
        print(f"\n=== {title} ===")
        table = {"adam": A[m], "adam_nv": N[m], "step": S[m], "fixed": F[m]}
        for k, t in table.items():
            v = np.array([t[s] for s in sorted(t)])
            if len(v) > 1:
                print(f"  {label[k]:24s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
            else:
                print(f"  {label[k]:24s} n={len(v)} (not available)")
        pairs = [("adam_nv", "adam"), ("step", "adam"), ("fixed", "adam"),
                 ("adam_nv", "step"), ("adam_nv", "fixed"), ("fixed", "step")]
        if min(len(t) for t in table.values()) > 1:
            rs = [e7.contrast(table, a, b) for a, b in pairs]
            fam = rs[1:]
            adj = dict(zip([id(r) for r in fam], e7.holm([r["p"] for r in fam])))
            for (a, b), r in zip(pairs, rs):
                ph = adj.get(id(r))
                ph_txt = f"  p_Holm={ph:.4f}" if ph is not None else "  (reference contrast)"
                print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                      f"p={r['p']:.4f}{ph_txt}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")

    canary(base, args.none_dirs)
    distribution_summary(args.none_dirs if args.none_dirs else args.step_dirs,
                         "mode none" if args.none_dirs else "pre-shuffle values, mode step")

    print("\n=== Pre-registered decision (primary metric) ===")
    table = {"adam": A["mem5"], "adam_nv": N["mem5"], "step": S["mem5"], "fixed": F["mem5"]}
    if min(len(t) for t in table.values()) < 2:
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    M = e7.contrast(table, "adam_nv", "adam")
    print(f"  M = Adam-NV - Adam = {M['mean']:+.2f} pp [{M['lo']:+.2f}, {M['hi']:+.2f}]")
    if M["lo"] <= 0:
        print("  Adam-NV effect not reproduced in this pooled set; the rule does not apply.")
        return
    verdict = {}
    for mode in ("step", "fixed"):
        Fm = e7.contrast(table, mode, "adam")
        Dm = e7.contrast(table, "adam_nv", mode)
        if Fm["mean"] >= 0.5 * M["mean"] and Dm["lo"] <= 0 <= Dm["hi"]:
            verdict[mode] = "RETAINS"
        elif Fm["mean"] < 0.25 * M["mean"] and Dm["lo"] > 0:
            verdict[mode] = "REMOVES"
        else:
            verdict[mode] = "PARTIAL"
        print(f"  shuffle/{mode}: F = {Fm['mean']:+.2f} pp, D = {Dm['mean']:+.2f} pp "
              f"[{Dm['lo']:+.2f}, {Dm['hi']:+.2f}]  -> {verdict[mode]}")
    s, f = verdict["step"], verdict["fixed"]
    if s == "RETAINS" and f == "RETAINS":
        print("  OUTCOME: the effect does NOT need the element-wise link between beta_1 and the element's "
              "own noise score; it follows from the distribution or magnitude of beta_1.")
    elif s == "REMOVES" and f == "REMOVES":
        print("  OUTCOME: the effect REQUIRES the element-wise link between beta_1 and the element's own noise score.")
    elif f == "RETAINS" and s == "REMOVES":
        print("  OUTCOME: the effect needs each element's beta_1 to be temporally consistent, not the link to its own score.")
    else:
        print("  OUTCOME: INCONCLUSIVE; report all contrasts and extend seeds.")


if __name__ == "__main__":
    main()

"""Exp. 12 analysis: constant beta_1 above 0.9 versus the smooth response under 20% label noise.

Arms (paired by seed, seeds 0-7):
  adam      Adam, beta1 = 0.9                              (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  adam_nv   Adam-NV, distrust response                     (Label_Diagnostic 0-3 + Exp. 7 seeds 4-7)
  smooth    Adam-NV, smooth response (logged mean beta1 0.92) (Exp. 9)
  c092      Adam, constant beta1 = 0.92                    (Exp. 12; matches the smooth response's mean)
  c095      Adam, constant beta1 = 0.95                    (Exp. 12; a stronger constant)
  c070..c081 (optional) the constants below 0.9 from Exp. 7 / 8, used only in the dose-response fit.

Primary metric: last-5-epoch mean of train_corrupt_noisy_label_acc (memorization), in pp.
S = smooth - adam (Exp. 9: about -1.47 pp). For c in {c092, c095}: F_c = c - adam, D_c = smooth - c.
Holm over {F_092, F_095, D_092, D_095}.

Pre-registered rule (written before any Exp. 12 run):
  0. The reduction must be reproduced in the pooled set: the 95% CI of S is below 0.
  1. A CONSTANT beta1 ABOVE 0.9 REPRODUCES the reduction (for c): F_c <= 0.5*S (at least half of the
     reduction) and the 95% CI of D_c includes 0.
  2. ADAPTATION ADDS A REDUCTION (for c): F_c > 0.25*S (less than a quarter of the reduction) and the
     95% CI of D_c is below 0.
  3. otherwise PARTIAL.
Overall reading: both c reproduce -> the direction result is explained by higher average beta1;
both add -> the smooth response lowers memorization beyond a constant beta1 of the same size;
otherwise inconclusive.

Example:
    python3 analyze_exp12.py --orig-dir Label_Diagnostic --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
        --smooth-dirs logs_exp09_A logs_exp09_B --c092-dirs logs_exp12_b092_A logs_exp12_b092_B \
        --c095-dirs logs_exp12_b095_A logs_exp12_b095_B \
        --low-dirs-081 logs_exp07_A logs_exp07_B --low-dirs-078 logs_exp08_b078_A logs_exp08_b078_B \
        --low-dirs-070 logs_exp08_b070_A logs_exp08_b070_B logs_exp08_b070_C
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import analyze_exp07 as e7  # same folder

SMOOTH_NAME = "adam_nv_scheduleconstant_sigma0.0_responsesmooth_betamax0.999_ln0.2_seed{seed}.csv"


def load(name_fn, dirs, seeds):
    out = {}
    for s in seeds:
        for d in dirs:
            p = os.path.join(d, name_fn(s))
            if os.path.isfile(p):
                df = pd.read_csv(p)
                if len(df) != 60:
                    print(f"[skip] {p}: {len(df)}/60 epochs")
                    break
                out[s] = e7.METRICS["mem5"][1](df)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--exp07-dirs", nargs="+", required=True)
    ap.add_argument("--smooth-dirs", nargs="+", required=True)
    ap.add_argument("--c092-dirs", nargs="+", default=[])
    ap.add_argument("--c095-dirs", nargs="+", default=[])
    ap.add_argument("--low-dirs-081", nargs="+", default=[])
    ap.add_argument("--low-dirs-078", nargs="+", default=[])
    ap.add_argument("--low-dirs-070", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    args = ap.parse_args()

    base = [args.orig_dir] + args.exp07_dirs
    fixed = lambda s: e7.NAME.format(opt="adam_nv_fixed", seed=s)
    table = {
        "adam": load(lambda s: e7.NAME.format(opt="adam", seed=s), base, args.seeds),
        "adam_nv": load(lambda s: e7.NAME.format(opt="adam_nv", seed=s), base, args.seeds),
        "smooth": load(lambda s: SMOOTH_NAME.format(seed=s), args.smooth_dirs, args.seeds),
        "c092": load(fixed, args.c092_dirs, args.seeds),
        "c095": load(fixed, args.c095_dirs, args.seeds),
    }
    low = {"c081": (0.81, args.low_dirs_081), "c078": (0.78, args.low_dirs_078), "c070": (0.70, args.low_dirs_070)}
    label = {"adam": "Adam (0.90)", "adam_nv": "Adam-NV", "smooth": "Adam-NV (smooth)",
             "c092": "constant 0.92", "c095": "constant 0.95"}

    print("Memorization (corrupted examples fitted to wrong label, last-5, pp):")
    for k in ["adam", "c092", "c095", "smooth", "adam_nv"]:
        v = np.array([table[k][s] for s in sorted(table[k])])
        if len(v) > 1:
            print(f"  {label[k]:18s} n={len(v)} mean={v.mean():.2f} sd={v.std(ddof=1):.2f}")
        else:
            print(f"  {label[k]:18s} n={len(v)} (not available)")

    pairs = [("c092", "adam"), ("c095", "adam"), ("smooth", "c092"), ("smooth", "c095")]
    res = {}
    for a, b in pairs:
        res[(a, b)] = e7.contrast(table, a, b) if min(len(table[a]), len(table[b])) > 1 else None
    valid = [r for r in res.values() if r]
    adj = dict(zip([id(r) for r in valid], e7.holm([r["p"] for r in valid]))) if len(valid) == 4 else {}
    S = e7.contrast(table, "smooth", "adam") if min(len(table["smooth"]), len(table["adam"])) > 1 else None
    if S:
        print(f"\nReference: smooth - Adam = {S['mean']:+.2f} pp [{S['lo']:+.2f}, {S['hi']:+.2f}] "
              f"(lower in {S['n'] - S['up']}/{S['n']} seeds)")
    print("\nPaired contrasts (Holm over the four):")
    for (a, b), r in res.items():
        if not r:
            print(f"  {label[a]} - {label[b]}: not enough paired seeds")
            continue
        print(f"  {label[a]} - {label[b]}: {r['mean']:+.3f} pp  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  p_Holm={adj.get(id(r), float('nan')):.4f}  sign-flip p={r['pflip']:.4f}  "
              f"higher in {r['up']}/{r['n']}")

    xs, ys = [], []
    for k, b in [("adam", 0.90), ("c092", 0.92), ("c095", 0.95)]:
        for v in table[k].values():
            xs.append(b); ys.append(v)
    for k, (b, dirs) in low.items():
        if dirs:
            t = load(fixed, dirs, args.seeds)
            for v in t.values():
                xs.append(b); ys.append(v)
    if len(set(xs)) >= 3:
        fit = stats.linregress(xs, ys)
        print(f"\nDose-response over all constant beta1 values (0.70 to 0.95, pooled runs): slope = "
              f"{fit.slope:+.1f} pp per unit beta1 (r = {fit.rvalue:+.2f}, p = {fit.pvalue:.3g}); descriptive only.")

    print("\n=== Pre-registered decision ===")
    if not (S and all(res.values())):
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    if S["hi"] >= 0:
        print("  The smooth-response reduction is not reproduced in this pooled set; the rule does not apply.")
        return
    verdict = {}
    for c in ("c092", "c095"):
        F = e7.contrast(table, c, "adam")
        D = res[("smooth", c)]
        if F["mean"] <= 0.5 * S["mean"] and D["lo"] <= 0 <= D["hi"]:
            verdict[c] = "REPRODUCES"
        elif F["mean"] > 0.25 * S["mean"] and D["hi"] < 0:
            verdict[c] = "ADAPTATION ADDS"
        else:
            verdict[c] = "PARTIAL"
        print(f"  {label[c]}: F = {F['mean']:+.2f} pp, D = {D['mean']:+.2f} pp [{D['lo']:+.2f}, {D['hi']:+.2f}] -> {verdict[c]}")
    v = set(verdict.values())
    if v == {"REPRODUCES"}:
        print("  OUTCOME: a constant beta1 above 0.9 reproduces the reduction; the direction result is explained by "
              "higher average beta1.")
    elif v == {"ADAPTATION ADDS"}:
        print("  OUTCOME: the smooth response lowers memorization beyond a constant beta1 of the same size.")
    else:
        print("  OUTCOME: PARTIAL / INCONCLUSIVE; read the dose-response and extend seeds.")


if __name__ == "__main__":
    main()

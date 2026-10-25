"""Exp. 10 analysis: 10-seed re-analysis of Exp. 5 / Exp. 6 at lr = 3e-3 (clean CIFAR-10, ResNet-18).

Arms (paired by seed, seeds 0-9): adam, adam_nv, adam_b083, adam_b080.
Seeds 0-3 come from Exp. 5 (adam, adam_nv) and Exp. 6 (controls); seeds 4-9 from Exp. 10.
File names: <optimizer>_lr3e-3_seed<k>.csv  (the layout used by run_e1_protocol_v2.sh).

Primary metric : mean validation accuracy over the last 5 epochs, in percentage points.
Secondary      : final-epoch accuracy, epochs to 90%, validation-loss AUC (sum over 60 epochs).
Statistics     : paired by seed; mean difference, t-based 95% CI, paired t p, exact sign-flip p
                 (2^n sign patterns), Holm over the five contrasts
                 {NV-Adam, b083-Adam, b080-Adam, NV-b083, NV-b080}.

Pre-registered rule (written before any Exp. 10 run; M = Adam-NV - Adam on the primary metric):
  1. Gain confirmed          : the 95% CI of M is above 0 (otherwise the lr 3e-3 gain is not
                               confirmed on 10 seeds and the Exp. 5 result is reported as such).
  2. EXPLAINED BY LOWER BETA1: gain confirmed, both controls have a mean gain over Adam of at
                               least 0.5*M, and for both controls the 95% CI of Adam-NV minus
                               control includes 0 with half-width <= 0.15 pp.
  3. ADAPTATION ADDS A GAIN  : gain confirmed and the 95% CI of Adam-NV minus control is above 0
                               for both controls.
  4. otherwise INCONCLUSIVE.
Confirmatory check (added after review): seeds 0-3 generated the hypothesis, so p-values pooling them
are optimistic. The script also reports seeds 4-9 alone (confirmatory) and states whether the gain M
is confirmed there (95% CI above 0). Report both analyses; the paper's headline claim must hold in both.

Example:
    python3 analyze_exp10.py --log-dirs logs_exp05_adam logs_exp05_adamnv logs_exp06 \
        logs_exp10_C logs_exp10_A logs_exp10_B
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["adam", "adam_nv", "adam_b083", "adam_b080"]
LABEL = {"adam": "Adam", "adam_nv": "Adam-NV", "adam_b083": "Adam, const. beta1=0.83",
         "adam_b080": "Adam, const. beta1=0.80"}


def metrics(df):
    return dict(last5=df["val_acc"].tail(5).mean() * 100, final=df["val_acc"].iloc[-1] * 100,
                ep90=float((df["val_acc"] >= 0.9).idxmax() + 1) if (df["val_acc"] >= 0.9).any() else np.nan,
                vauc=float(df["val_loss"].sum()))


def load(log_dirs, seeds, epochs):
    out = {a: {} for a in ARMS}
    for a in ARMS:
        pat = re.compile(rf"^{re.escape(a)}_lr3e-3_seed(\d+)\.csv$")
        for d in log_dirs:
            for f in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True):
                m = pat.match(os.path.basename(f))
                if not m or int(m.group(1)) not in seeds or int(m.group(1)) in out[a]:
                    continue
                df = pd.read_csv(f)
                if len(df) != epochs:
                    print(f"[skip] {f}: {len(df)}/{epochs} epochs")
                    continue
                out[a][int(m.group(1))] = metrics(df)
    return out


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


def contrast(data, a, b, metric):
    seeds = sorted(set(data[a]) & set(data[b]))
    d = np.array([data[a][s][metric] - data[b][s][metric] for s in seeds])
    n = len(d)
    if n < 2:
        return None
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=se) if se > 0 else (d.mean(), d.mean())
    p = stats.ttest_1samp(d, 0.0).pvalue if se > 0 else (1.0 if d.mean() == 0 else 0.0)
    return dict(a=a, b=b, n=n, mean=d.mean(), lo=lo, hi=hi, p=p, pflip=flip_p(d), up=int((d > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    data = load(args.log_dirs, set(args.seeds), args.epochs)
    print("Seeds available:", {a: sorted(data[a]) for a in ARMS})
    print("\nlr 3e-3, mean +- SD over seeds:")
    for a in ARMS:
        v = {m: np.array([data[a][s][m] for s in sorted(data[a])]) for m in ["last5", "final", "ep90", "vauc"]}
        if len(v["last5"]) > 1:
            print(f"  {LABEL[a]:26s} n={len(v['last5'])}  last5 {v['last5'].mean():.2f} +- {v['last5'].std(ddof=1):.2f}"
                  f" | final {v['final'].mean():.2f} | ep90 {v['ep90'].mean():.1f} | val-loss AUC {v['vauc'].mean():.2f}")
        else:
            print(f"  {LABEL[a]:26s} not available")

    pairs = [("adam_nv", "adam"), ("adam_b083", "adam"), ("adam_b080", "adam"),
             ("adam_nv", "adam_b083"), ("adam_nv", "adam_b080")]
    for metric in ["last5", "final"]:
        res = [contrast(data, a, b, metric) for a, b in pairs]
        print(f"\nPaired contrasts, metric = {metric} (pp; Holm over the five):")
        if all(res):
            adj = holm([r["p"] for r in res])
        else:
            adj = [float("nan")] * len(res)
        for r, ap_, (a, b) in zip(res, adj, pairs):
            if r is None:
                print(f"  {LABEL[a]} - {LABEL[b]}: not enough paired seeds")
                continue
            print(f"  {LABEL[a]} - {LABEL[b]}: {r['mean']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.4f}  "
                  f"p_Holm={ap_:.4f}  sign-flip p={r['pflip']:.4f}  higher in {r['up']}/{r['n']}")
        if metric == "last5":
            primary = res

    print("\n=== Pre-registered decision (primary metric = last5) ===")
    if not all(primary):
        print("  Not enough complete arms/seeds to apply the rule.")
        return
    M, F83, F80, D83, D80 = primary
    nseeds = min(r["n"] for r in primary)
    print(f"  seeds per contrast: {nseeds}; M = {M['mean']:+.3f} pp [{M['lo']:+.3f}, {M['hi']:+.3f}]")
    if nseeds < 10:
        print("  NOTE: fewer than 10 paired seeds; the rule was written for 10 seeds, treat as provisional.")
    half = lambda r: (r["hi"] - r["lo"]) / 2
    if M["lo"] <= 0:
        print("  OUTCOME: GAIN NOT CONFIRMED on these seeds (CI of Adam-NV - Adam includes 0).")
    elif (F83["mean"] >= 0.5 * M["mean"] and F80["mean"] >= 0.5 * M["mean"]
          and D83["lo"] <= 0 <= D83["hi"] and D80["lo"] <= 0 <= D80["hi"]
          and half(D83) <= 0.15 and half(D80) <= 0.15):
        print("  OUTCOME: EXPLAINED BY LOWER BETA1 (both controls reproduce >= half the gain; Adam-NV is "
              "indistinguishable from each control with half-width <= 0.15 pp)")
    elif D83["lo"] > 0 and D80["lo"] > 0:
        print("  OUTCOME: ADAPTATION ADDS A GAIN over both controls")
    else:
        print("  OUTCOME: INCONCLUSIVE (see the half-widths above; more seeds or a tighter design needed)")
    confirmatory(data)


def confirmatory(data):
    sub = {a: {s: v for s, v in data[a].items() if s >= 4} for a in ARMS}
    print("\n=== Confirmatory analysis: seeds 4-9 only (primary metric = last5) ===")
    pairs = [("adam_nv", "adam"), ("adam_b083", "adam"), ("adam_b080", "adam"),
             ("adam_nv", "adam_b083"), ("adam_nv", "adam_b080")]
    res = [contrast(sub, a, b, "last5") for a, b in pairs]
    if not all(res):
        print("  Not enough seeds 4-9 yet.")
        return
    for r, (a, b) in zip(res, pairs):
        print(f"  {LABEL[a]} - {LABEL[b]}: {r['mean']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  higher in {r['up']}/{r['n']}")
    M = res[0]
    print(f"  Gain confirmed on seeds 4-9 alone (CI of M above 0): {M['lo'] > 0}  (n = {M['n']})")


if __name__ == "__main__":
    main()

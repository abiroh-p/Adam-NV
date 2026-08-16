"""
E1 · Convergence Benchmarks — analysis script.

Reads all per-(optimizer, seed) CSV logs produced by train_resnet.py /
run_e1_suite.py and produces:

    1. figures/e1_convergence.png
       A 2-subplot figure (loss curves, accuracy curves) — Tier-1 version
       of the design doc's 2x2 (CIFAR-10 + CIFAR-100); since Tier 1 is
       CIFAR-10 only, this collapses to 1x2.
       Each optimizer's mean curve is plotted with a shaded band showing
       +/- 1 std across seeds, and distinct line styles (not just colors)
       for accessibility, per the design doc's figure spec.

    2. e1_results_table.csv  (and a printed Markdown version)
       Rows = optimizers, columns = final val acc (mean +/- std),
       epochs-to-90%, loss curve AUC — matching the design doc's
       "What to Report" spec for E1.

Usage:
    python analyze_e1.py --log-dir ../logs --out-dir ..

Only includes runs matching the *_seed*.csv pattern (excludes the
_suite_status.csv tracking file and any *_smoke.csv dry-run logs by
default — pass --include-smoke to include them, e.g. for testing
this script itself before real data exists).
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Distinct line styles per optimizer (not just color) for accessibility,
# per design doc figure spec. Extend this dict if you add optimizers.
LINESTYLES = {
    "adam":      "--",
    "adamw":     "-.",
    "adabelief": ":",
    "adam_nv":   "-",
}
COLORS = {
    "adam":      "#7f7f7f",
    "adamw":     "#1f77b4",
    "adabelief": "#2ca02c",
    "adam_nv":   "#d62728",
}
DISPLAY_NAME = {
    "adam":      "Adam",
    "adamw":     "AdamW",
    "adabelief": "AdaBelief",
    "adam_nv":   "Adam-NV (ours)",
}


def parse_run_filename(path):
    """Extract (optimizer, seed) from a log filename like 'adam_nv_seed0.csv'
    or 'adam_nv_seed0_smoke.csv'. Returns None if the filename doesn't match
    the expected pattern (e.g. _suite_status.csv)."""
    base = os.path.basename(path).replace(".csv", "")
    m = re.match(r"^(.*)_seed(\d+)(_smoke)?$", base)
    if not m:
        return None
    optimizer, seed, is_smoke = m.group(1), int(m.group(2)), bool(m.group(3))
    return optimizer, seed, is_smoke


def load_runs(log_dir, include_smoke=False):
    """Returns dict: optimizer -> list of per-seed run dicts, each with
    'epoch', 'val_acc', 'val_loss', 'train_loss' as numpy arrays."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, "*.csv"))):
        parsed = parse_run_filename(path)
        if parsed is None:
            continue  # skip _suite_status.csv etc.
        optimizer, seed, is_smoke = parsed
        if is_smoke and not include_smoke:
            continue

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f"[analyze_e1] WARNING: {path} is empty, skipping")
            continue

        data = {
            "seed": seed,
            "epoch": np.array([int(r["epoch"]) for r in rows]),
            "train_loss": np.array([float(r["train_loss"]) for r in rows]),
            "val_loss": np.array([float(r["val_loss"]) for r in rows]),
            "val_acc": np.array([float(r["val_acc"]) for r in rows]),
        }
        runs[optimizer].append(data)

    return runs


def aggregate_across_seeds(run_list):
    """Given multiple seed runs for one optimizer (possibly different
    lengths if a run was interrupted), align to the shortest common epoch
    count and compute mean/std curves across seeds."""
    min_len = min(len(r["epoch"]) for r in run_list)
    if any(len(r["epoch"]) != min_len for r in run_list):
        print(f"[analyze_e1] NOTE: seeds have differing epoch counts "
              f"({[len(r['epoch']) for r in run_list]}); truncating all to "
              f"{min_len} epochs for fair averaging.")

    epochs = run_list[0]["epoch"][:min_len]
    val_acc_stack = np.stack([r["val_acc"][:min_len] for r in run_list])
    val_loss_stack = np.stack([r["val_loss"][:min_len] for r in run_list])
    train_loss_stack = np.stack([r["train_loss"][:min_len] for r in run_list])

    return {
        "epochs": epochs,
        "val_acc_mean": val_acc_stack.mean(axis=0),
        "val_acc_std": val_acc_stack.std(axis=0),
        "val_loss_mean": val_loss_stack.mean(axis=0),
        "val_loss_std": val_loss_stack.std(axis=0),
        "train_loss_mean": train_loss_stack.mean(axis=0),
        "n_seeds": len(run_list),
    }


def epochs_to_90(val_acc_mean, epochs):
    """First epoch at which mean val_acc reaches 0.90, or None."""
    idx = np.argmax(val_acc_mean >= 0.90)
    if val_acc_mean[idx] >= 0.90:
        return int(epochs[idx])
    return None


def loss_curve_auc(val_loss_mean, epochs):
    """Area under the validation loss curve via trapezoidal rule —
    a single number summarizing convergence quality (lower = better).
    Uses np.trapezoid (numpy >=2.0) with a fallback to the older
    np.trapz name for compatibility with older numpy installs."""
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(val_loss_mean, x=epochs))


def make_figure(aggregated, out_path, synthetic_warning=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_loss, ax_acc = axes

    for opt, agg in aggregated.items():
        style = LINESTYLES.get(opt, "-")
        color = COLORS.get(opt, None)
        label = DISPLAY_NAME.get(opt, opt)
        ep = agg["epochs"]

        ax_loss.plot(ep, agg["val_loss_mean"], linestyle=style, color=color,
                     label=label, linewidth=2)
        ax_loss.fill_between(ep, agg["val_loss_mean"] - agg["val_loss_std"],
                              agg["val_loss_mean"] + agg["val_loss_std"],
                              alpha=0.15, color=color)

        ax_acc.plot(ep, agg["val_acc_mean"], linestyle=style, color=color,
                    label=label, linewidth=2)
        ax_acc.fill_between(ep, agg["val_acc_mean"] - agg["val_acc_std"],
                             agg["val_acc_mean"] + agg["val_acc_std"],
                             alpha=0.15, color=color)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Validation Loss")
    ax_loss.set_title("Validation Loss — CIFAR-10 / ResNet-18")
    ax_loss.legend(frameon=False)
    ax_loss.grid(alpha=0.25)

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation Accuracy")
    ax_acc.set_title("Validation Accuracy — CIFAR-10 / ResNet-18")
    ax_acc.legend(frameon=False)
    ax_acc.grid(alpha=0.25)
    ax_acc.axhline(0.90, color="black", linestyle=":", linewidth=1, alpha=0.4)

    if synthetic_warning:
        fig.suptitle("*** SYNTHETIC / SMOKE-TEST DATA — NOT REAL RESULTS ***",
                      color="red", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze_e1] Figure saved to {out_path}")


def make_table(aggregated, out_csv_path):
    fieldnames = ["optimizer", "n_seeds", "final_val_acc_mean", "final_val_acc_std",
                  "epochs_to_90pct", "val_loss_auc"]
    rows = []
    for opt, agg in aggregated.items():
        rows.append({
            "optimizer": DISPLAY_NAME.get(opt, opt),
            "n_seeds": agg["n_seeds"],
            "final_val_acc_mean": round(float(agg["val_acc_mean"][-1]), 4),
            "final_val_acc_std": round(float(agg["val_acc_std"][-1]), 4),
            "epochs_to_90pct": epochs_to_90(agg["val_acc_mean"], agg["epochs"]) or "not reached",
            "val_loss_auc": round(loss_curve_auc(agg["val_loss_mean"], agg["epochs"]), 2),
        })

    # Sort with adam_nv first (the "ours" row convention used in E1's spec)
    rows.sort(key=lambda r: (r["optimizer"] != "Adam-NV (ours)", r["optimizer"]))

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[analyze_e1] Results table saved to {out_csv_path}")

    # Also print a Markdown version for quick copy into a report/viva slide
    print("\n[analyze_e1] Markdown table:\n")
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "|" + "|".join(["---"] * len(fieldnames)) + "|"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in fieldnames) + " |")
    print()


def main():
    parser = argparse.ArgumentParser(description="E1 analysis: figure + results table")
    parser.add_argument("--log-dir", type=str, default="../logs")
    parser.add_argument("--out-dir", type=str, default="..")
    parser.add_argument("--include-smoke", action="store_true",
                         help="Include *_smoke.csv logs (for testing this script "
                              "before real training data exists).")
    parser.add_argument("--expected-seeds", nargs="+", type=int,
                         default=[0, 1, 2, 3],
                         help="Seeds required for every optimizer. Default: 0 1 2 3.")
    parser.add_argument("--allow-incomplete", action="store_true",
                         help="Generate exploratory output despite missing seeds. "
                              "Do not use this output for paper claims.")
    args = parser.parse_args()

    runs = load_runs(args.log_dir, include_smoke=args.include_smoke)
    if not runs:
        print(f"[analyze_e1] No matching logs found in {args.log_dir}. "
              f"Run train_resnet.py / run_e1_suite.py first"
              + ("" if args.include_smoke else ", or pass --include-smoke to "
                 "test against smoke-test logs."))
        return

    print(f"[analyze_e1] Found runs for optimizers: {list(runs.keys())}")
    missing = {}
    for opt, run_list in runs.items():
        seeds = sorted(r["seed"] for r in run_list)
        print(f"  {opt}: seeds {seeds}")
        absent = sorted(set(args.expected_seeds) - set(seeds))
        if absent:
            missing[opt] = absent

    if missing and not args.allow_incomplete:
        details = "; ".join(
            f"{opt}: {seeds}" for opt, seeds in sorted(missing.items())
        )
        raise SystemExit(
            "[analyze_e1] Refusing to generate a paper table with missing seeds "
            f"({details}). Run the missing E1 jobs, or pass --allow-incomplete "
            "for explicitly exploratory output."
        )
    if missing:
        print("[analyze_e1] WARNING: incomplete seed sets; output is exploratory only")

    aggregated = {opt: aggregate_across_seeds(run_list)
                  for opt, run_list in runs.items()}

    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    fig_path = os.path.join(args.out_dir, "figures", "e1_convergence.png")
    table_path = os.path.join(args.out_dir, "e1_results_table.csv")

    make_figure(aggregated, fig_path, synthetic_warning=args.include_smoke)
    make_table(aggregated, table_path)


if __name__ == "__main__":
    main()

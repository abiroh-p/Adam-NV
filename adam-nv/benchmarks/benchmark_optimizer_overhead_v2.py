"""Optimizer overhead benchmark, version 2 (interleaved, medians, memory).

Improvements over benchmark_optimizer_overhead.py:
  * Rounds instead of blocks: in every round ALL optimizers are timed, in a freshly randomized
    order, so slow drift (GPU clocks, thermal state, allocator state) and first-run effects hit
    all optimizers alike instead of falling on whichever optimizer runs first.
  * Per-run medians (robust to outlier steps) as well as means, and an explicit record of the
    position of each optimizer within its round, so an order effect can be checked.
  * Overhead is reported as a PAIRED ratio against the reference optimizer within the same round,
    with a bootstrap 95% confidence interval over rounds.
  * Memory: peak allocated GPU memory during the timed steps, and the size of the optimizer state.
  * Three timed regions, each bracketed by CUDA synchronization: forward+backward, optimizer step,
    and the total training step (zero_grad, forward, backward, step). The total is measured
    directly, not as a sum of the other two.
  * Metadata JSON with library versions, GPU, arguments and SHA-256 hashes of the source files.

Example (GPU; my estimate is about 40 minutes for 20 rounds of the four default optimizers, not measured):
    python3 benchmark_optimizer_overhead_v2.py --rounds 20 --output results_v2/overhead

CPU smoke test:
    python3 benchmark_optimizer_overhead_v2.py --device cpu --rounds 2 --warmup-steps 1 \
        --measured-steps 3 --optimizers adam adam_nv adamw --output /tmp/overhead_smoke
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def state_bytes(optimizer):
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def run_once(name, round_idx, position, args, device, build_model, build_optimizer):
    torch.manual_seed(args.seed + round_idx)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + round_idx)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(num_classes=10).to(device)
    optimizer = build_optimizer(name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        optimizer.step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total, fb, step = [], [], []
    for _ in range(args.measured_steps):
        synchronize(device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        synchronize(device)
        t1 = time.perf_counter()
        optimizer.step()
        synchronize(device)
        t2 = time.perf_counter()
        total.append(1000.0 * (t2 - t0))
        fb.append(1000.0 * (t1 - t0))
        step.append(1000.0 * (t2 - t1))

    row = {
        "round": round_idx, "position": position, "optimizer": name,
        "median_total_ms": float(np.median(total)), "mean_total_ms": float(np.mean(total)),
        "median_fwd_bwd_ms": float(np.median(fb)), "median_step_ms": float(np.median(step)),
        "mean_step_ms": float(np.mean(step)),
        "optimizer_state_mib": state_bytes(optimizer) / 2**20,
        "param_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
        "peak_alloc_mib": (torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else float("nan"),
    }
    del model, optimizer, x, y
    gc.collect()
    return row


def bootstrap_ci(values, rng, n_boot=10000):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.median(values[idx], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optimizers", nargs="+", default=["adam", "adam_nv", "adamw", "adabelief"])
    ap.add_argument("--reference", default="adam")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--measured-steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--source-dir", type=Path, default=ROOT / "e2_noisy_gradient" / "src",
                    help="folder with part_b_resnet_v8.py, adam_nv.py, ... (default: repo layout)")
    ap.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results_v2" / "overhead",
                    help="output prefix; writes <prefix>_runs.csv, <prefix>_summary.csv, <prefix>_metadata.json")
    args = ap.parse_args()
    if args.reference not in args.optimizers:
        ap.error("--reference must be one of --optimizers")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    sys.path.insert(0, str(args.source_dir))
    from part_b_resnet_v8 import build_model, build_optimizer  # noqa: E402

    rng = np.random.default_rng(args.seed)
    rows = []
    for r in range(args.rounds):
        order = list(rng.permutation(args.optimizers))
        for pos, name in enumerate(order):
            print(f"[benchmark] round {r + 1}/{args.rounds} position {pos + 1}/{len(order)}: {name}", flush=True)
            rows.append(run_once(name, r, pos, args, device, build_model, build_optimizer))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs_path = Path(str(args.output) + "_runs.csv")
    with runs_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def by(name, key):
        return np.array([r[key] for r in sorted((x for x in rows if x["optimizer"] == name), key=lambda x: x["round"])])

    ref_total, ref_step = by(args.reference, "median_total_ms"), by(args.reference, "median_step_ms")
    ref_peak, ref_state = by(args.reference, "peak_alloc_mib"), by(args.reference, "optimizer_state_mib")
    summary = []
    for name in args.optimizers:
        tot, stp = by(name, "median_total_ms"), by(name, "median_step_ms")
        peak, state = by(name, "peak_alloc_mib"), by(name, "optimizer_state_mib")
        ratio_total, ratio_step = tot / ref_total, stp / ref_step
        lo_t, hi_t = bootstrap_ci(ratio_total, rng)
        lo_s, hi_s = bootstrap_ci(ratio_step, rng)
        summary.append({
            "optimizer": name, "rounds": len(tot),
            "median_total_ms": float(np.median(tot)), "median_step_ms": float(np.median(stp)),
            "median_fwd_bwd_ms": float(np.median(by(name, "median_fwd_bwd_ms"))),
            "total_ratio_vs_ref": float(np.median(ratio_total)), "total_ratio_ci95_low": lo_t, "total_ratio_ci95_high": hi_t,
            "step_ratio_vs_ref": float(np.median(ratio_step)), "step_ratio_ci95_low": lo_s, "step_ratio_ci95_high": hi_s,
            "optimizer_state_mib": float(np.median(state)),
            "state_extra_vs_ref_mib": float(np.median(state - ref_state)),
            "peak_alloc_mib": float(np.median(peak)),
            "peak_extra_vs_ref_mib": float(np.median(peak - ref_peak)),
        })
    summary_path = Path(str(args.output) + "_summary.csv")
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    # Order-effect check: mean of the reference optimizer's median total time by position in the round.
    order_check = {}
    for pos in range(len(args.optimizers)):
        vals = [r["median_total_ms"] for r in rows if r["optimizer"] == args.reference and r["position"] == pos]
        if vals:
            order_check[pos] = float(np.mean(vals))

    meta = {
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "hostname": socket.gethostname(),
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [args.source_dir / "adam_nv.py", args.source_dir / "part_b_resnet_v8.py",
                                  Path(__file__)] if p.exists()},
        "reference_total_ms_by_position_in_round": order_check,
    }
    with open(str(args.output) + "_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n[benchmark] summary (medians over rounds; ratios are paired within a round against '%s')" % args.reference)
    print(f"{'optimizer':12s} {'step ms':>9s} {'total ms':>9s} {'total ratio [95% CI]':>26s} {'state MiB':>10s} {'peak extra MiB':>15s}")
    for s in summary:
        print(f"{s['optimizer']:12s} {s['median_step_ms']:9.2f} {s['median_total_ms']:9.2f} "
              f"{s['total_ratio_vs_ref']:8.3f} [{s['total_ratio_ci95_low']:.3f}, {s['total_ratio_ci95_high']:.3f}] "
              f"{s['optimizer_state_mib']:10.1f} {s['peak_extra_vs_ref_mib']:15.1f}")
    print("[benchmark] reference total time by position in round (ms):", {k: round(v, 2) for k, v in order_check.items()})
    print(f"[benchmark] files: {runs_path}, {summary_path}, {str(args.output) + '_metadata.json'}")


if __name__ == "__main__":
    main()

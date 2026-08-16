"""Formal repeated benchmark for Adam-NV optimizer overhead.

Measures both end-to-end training-step time and optimizer.step() time after a
configurable warm-up. CUDA is synchronized around every timed region. Each
repeat rebuilds the model and reuses the same seed across optimizers, enabling
paired comparisons.
"""

import argparse
import csv
import math
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
from scipy import stats
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "e2_noisy_gradient" / "src"
sys.path.insert(0, str(SOURCE))
from part_b_resnet_v8 import build_model, build_optimizer  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_repeat(optimizer_name: str, repeat: int, args, device: torch.device) -> dict:
    seed = args.seed + repeat
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(num_classes=10).to(device)
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    synchronize(device)

    total_times = []
    optimizer_times = []
    for _ in range(args.measured_steps):
        synchronize(device)
        total_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        synchronize(device)
        optimizer_start = time.perf_counter()
        optimizer.step()
        synchronize(device)
        optimizer_end = time.perf_counter()
        total_times.append(optimizer_end - total_start)
        optimizer_times.append(optimizer_end - optimizer_start)

    return {
        "optimizer": optimizer_name,
        "repeat": repeat,
        "seed": seed,
        "measured_steps": args.measured_steps,
        "mean_total_step_ms": 1000.0 * float(np.mean(total_times)),
        "mean_optimizer_step_ms": 1000.0 * float(np.mean(optimizer_times)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) > 1:
        half = float(stats.t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    else:
        half = math.nan
    return mean, sd, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizers", nargs="+", default=["adam", "adam_nv"],
                        choices=["adam", "adam_nv", "adamw", "adabelief"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measured-steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks" / "results" / "optimizer_overhead_runs.csv",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    rows = []
    for repeat in range(args.repeats):
        for optimizer in args.optimizers:
            print(f"[benchmark] repeat {repeat + 1}/{args.repeats}: {optimizer}")
            rows.append(run_repeat(optimizer, repeat, args, device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary_rows = []
    for optimizer in args.optimizers:
        selected = [row for row in rows if row["optimizer"] == optimizer]
        for metric in ("mean_total_step_ms", "mean_optimizer_step_ms"):
            mean, sd, low, high = mean_ci([float(row[metric]) for row in selected])
            summary_rows.append({
                "optimizer": optimizer,
                "metric": metric,
                "n_repeats": len(selected),
                "mean_ms": mean,
                "std_ms": sd,
                "ci95_low_ms": low,
                "ci95_high_ms": high,
            })
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[benchmark] runs: {args.output}")
    print(f"[benchmark] summary: {summary_path}")


if __name__ == "__main__":
    main()

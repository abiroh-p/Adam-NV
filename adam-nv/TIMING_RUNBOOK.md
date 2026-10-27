# Timing and memory benchmark, version 2 (one run on one free GPU)

**Why:** the first benchmark ran the optimizers in the same fixed order in every repeat (one Adam repeat was an outlier, 98 ms against about 106 ms, which lowered Adam's mean), reported means of per-step times, did not measure memory, and its end-to-end overhead (21.9 ms) was smaller than its optimizer-step overhead (26.8 ms) without an explanation. Version 2 randomizes the optimizer order in every round, reports medians and paired ratios with bootstrap intervals, records the position of each run within its round (to check for an order effect), and records optimizer-state and peak GPU memory.

## What it measures
For each round, every optimizer (Adam, Adam-NV, AdamW, AdaBelief) is timed once, in a fresh random order, on a random batch of 128 CIFAR-sized images with a freshly built ResNet-18. Three regions, each bracketed by CUDA synchronization: forward + backward, optimizer step, and the total training step (measured directly). 50 warm-up and 200 measured steps per run, 20 rounds by default.

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): the script runs, randomizes the order, writes the three output files, and computes the paired ratios and the optimizer-state sizes. On CPU the state is 85.3 MiB for Adam and 170.5 MiB for Adam-NV (four per-element buffers `m`, `v`, `rho`, `P` against two for Adam, 11,173,962 parameters). NOT verified: any GPU timing, GPU peak memory, and AdaBelief (needs the `adabelief_pytorch` package, which your E1 environment already used). The runtime of about 40 minutes is my estimate.

## Steps (Kaggle, one GPU; any account)
```bash
!unzip -o timing_benchmark_v2.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/benchmarks
!python3 benchmark_optimizer_overhead_v2.py --rounds 20 --source-dir /kaggle/working/Adam-NV/e2_noisy_gradient/src --output /kaggle/working/Adam-NV/benchmarks/results_v2/overhead
!cd /kaggle/working/Adam-NV/benchmarks && zip -r timing_v2_results.zip results_v2
```
Run it alone on the GPU (not while another job runs on the same device), because other jobs distort the timings. If you have two GPUs, use one and leave the other idle. The script prints a summary table at the end; send me that printout and the zip.

## Send back
`timing_v2_results.zip` (runs, summary and metadata files) and the printed summary table.

## Do not
Change any argument, run other jobs on the same GPU during the benchmark, or edit the script.

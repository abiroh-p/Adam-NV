"""
Response-Lag Diagnostic — per-step β₁/ρ around a noise transition
======================================================================
Directly answers the question the epoch-level correlation analysis in
the Phase 1.5 report (§4.2) could NOT answer: how many STEPS after a
sigma_t transition does Adam-NV's beta_1 actually respond?

Why this is needed
--------------------
Every existing log (Phase 1.4, Phase 1.5, beta2-sweep) records
beta1_mean/rho_mean once per epoch — a snapshot after that epoch's
final step. That data cannot distinguish "beta_1 moved within the
first 20 steps of the new epoch" from "beta_1 took 3 epochs to move,"
because nothing was logged in between. The report's "~3-4 epoch lag"
figure was inferred from the SHAPE of epoch-level correlation curves,
not measured directly. That inference may well be right, but it has
not actually been checked against step-level dynamics, and there's a
real reason it might not be: sigma2 = (v - m^2) / (v + lam|m| + eps)
is a ratio of two EMAs, and m's own decay rate IS beta_1 — which is
itself adaptive. That's a feedback loop, not a simple fixed-time-
constant filter, and "the slower EMA (beta_2) dominates" is a
plausible guess, not a verified conclusion.

This script logs beta1_mean and rho_mean at EVERY STEP (not sampled)
during a configurable window of epochs, so the actual step count
between a sigma_t flip and beta_1's response can be read directly off
the data — settling the question empirically instead of arguing about
EMA math in isolation.

Design
------
Runs a normal 60-epoch adam_nv training run under a real schedule
(same model/data/optimizer as part_b_resnet_v8.py — imports directly
from it, no duplicated logic), producing the SAME epoch-level CSV as
before for comparability with existing Phase 1.5 results. The only
addition: during --fine-log-start to --fine-log-end (inclusive,
1-indexed epochs), it also writes a second, step-level CSV logging
sigma_t/beta1_mean/rho_mean after every single optimizer step.

Default fine-log window (epochs 37-42) is deliberately NOT the first
intermittent cycle — early epochs are still exiting warmup (the k=100
suppression term in sigma2_tilde) and moment estimates are young, so
transitions there aren't representative of steady-state behavior. This
window covers a clean noisy->clean->noisy transition pair well into
training, matching where the epoch-level lag signal was actually
measured in the report.

Usage
-----
    # Real run, default window (epochs 37-42), matches existing seed0
    # intermittent run for direct comparison:
    python diagnose_response_lag.py --schedule intermittent --sigma 2.0 \\
        --burst-epochs 5 --gap-epochs 5 --seed 0 --epochs 60 \\
        --fine-log-start 37 --fine-log-end 42

    # Same idea on the phased schedule -- ONE clean transition, easier
    # to read the lag off a plot with no cycle-length confound:
    python diagnose_response_lag.py --schedule phased --sigma 2.0 \\
        --seed 0 --epochs 60 --fine-log-start 18 --fine-log-end 26

    # Smoke test:
    python diagnose_response_lag.py --schedule intermittent --sigma 2.0 \\
        --seed 0 --epochs 8 --fine-log-start 1 --fine-log-end 8 \\
        --smoke-test --synthetic-data

Output
------
logs/part_b/{run_name}.csv                    -- normal epoch-level log
logs/part_b/{run_name}_finegrained.csv        -- step-level log, columns:
    epoch, step_in_epoch, global_step, sigma_t, beta1_mean, rho_mean

How to read the result
-----------------------
Find the global_step where sigma_t flips. Find the global_step where
beta1_mean crosses halfway between its pre-flip and post-flip
steady-state values. The difference in steps, divided by steps/epoch,
gives the ACTUAL response lag in epochs -- compare that number against
the "~3 epoch peak lag" claimed in the report. If it comes out much
smaller (consistent with rho's alpha=0.95 / ~20-step memory), the
report's beta_2 attribution is wrong and something else explains the
epoch-level correlation pattern (candidates: interaction with the
cosine LR schedule, which also changes every step; or an artifact of
the epoch-level snapshot timing itself). If it comes out at ~3 epochs
(~1000+ steps), that's consistent with beta_2=0.999 dominating as
claimed.
"""

import argparse
import csv
import os
import time

import torch

from part_b_resnet_v8 import (
    build_model, build_optimizer, get_dataloaders, evaluate, DEFAULT_LR,
)
from noise import inject_noise
from noise_schedule import get_sigma_schedule, describe_schedule

import torch.nn as nn


DIAGNOSTIC_PROTOCOL_VERSION = 2


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: per-step beta_1/rho response around a noise transition")
    parser.add_argument("--schedule", type=str, default="intermittent",
                        choices=["phased", "ramp_up", "ramp_updown", "intermittent"])
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--phase-start", type=float, default=1.0 / 3.0)
    parser.add_argument("--phase-end", type=float, default=2.0 / 3.0)
    parser.add_argument("--burst-epochs", type=int, default=5)
    parser.add_argument("--gap-epochs", type=int, default=5)
    parser.add_argument("--start-clean", action="store_true", default=True)
    parser.add_argument("--start-noisy", dest="start_clean", action="store_false")
    parser.add_argument("--beta2", type=float, default=None,
                        help="Optional override, same meaning as in v8. "
                             "Leave unset to test the default (0.999) "
                             "mechanism the report's claim is about.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--fine-log-start", type=int, required=True,
                        help="First epoch (1-indexed, inclusive) to log "
                             "every step instead of once per epoch.")
    parser.add_argument("--fine-log-end", type=int, required=True,
                        help="Last epoch (1-indexed, inclusive) to log "
                             "every step instead of once per epoch.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    args = parser.parse_args()

    if args.fine_log_start < 1 or args.fine_log_end > args.epochs or args.fine_log_start > args.fine_log_end:
        parser.error("--fine-log-start/--fine-log-end must satisfy "
                      "1 <= start <= end <= --epochs")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else DEFAULT_LR["adam_nv"]

    sigma_schedule = get_sigma_schedule(
        args.schedule, total_epochs=args.epochs, sigma_max=args.sigma,
        phase_start=args.phase_start, phase_end=args.phase_end,
        burst_epochs=args.burst_epochs, gap_epochs=args.gap_epochs,
        start_clean=args.start_clean)
    schedule_desc = describe_schedule(
        args.schedule, phase_start=args.phase_start, phase_end=args.phase_end,
        burst_epochs=args.burst_epochs, gap_epochs=args.gap_epochs,
        start_clean=args.start_clean)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    if device.type == "cuda":
        noise_gen = torch.Generator(device=device)
    else:
        noise_gen = torch.Generator()
    noise_gen.manual_seed(args.seed + 1)

    beta2_suffix = f"_beta2{args.beta2}" if args.beta2 is not None else ""
    run_name = (f"adam_nv_schedule{args.schedule}_sigma{args.sigma}{beta2_suffix}"
                f"_seed{args.seed}_lagdiag_v{DIAGNOSTIC_PROTOCOL_VERSION}"
                + ("_smoke" if args.smoke_test else ""))

    print(f"[Lag Diagnostic] {run_name} | epochs={args.epochs} lr={lr} device={device}")
    print(f"[Lag Diagnostic] Schedule: {schedule_desc}")
    print(f"[Lag Diagnostic] Fine-grained (per-step) logging: "
          f"epochs {args.fine_log_start}-{args.fine_log_end}")

    train_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir,
        smoke_test=args.smoke_test, synthetic=args.synthetic_data,
        loader_generator=loader_gen, label_noise=0.0, label_noise_seed=0)

    model = build_model().to(device)
    optimizer = build_optimizer("adam_nv", model.parameters(), lr, beta2=args.beta2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    epoch_log_path = os.path.join(args.log_dir, f"{run_name}.csv")
    fine_log_path = os.path.join(args.log_dir, f"{run_name}_finegrained.csv")

    epoch_fieldnames = ["protocol_version", "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                        "lr", "sigma_t", "epoch_time_sec",
                        "beta1_mean", "beta1_min", "beta1_max", "sigma2_mean", "rho_mean"]
    fine_fieldnames = ["protocol_version", "epoch", "step_in_epoch", "global_step", "lr", "sigma_t",
                       "beta1_mean", "rho_mean"]

    global_step = 0

    with open(epoch_log_path, "w", newline="") as ef, \
         open(fine_log_path, "w", newline="") as ff:

        epoch_writer = csv.DictWriter(ef, fieldnames=epoch_fieldnames)
        epoch_writer.writeheader()
        fine_writer = csv.DictWriter(ff, fieldnames=fine_fieldnames)
        fine_writer.writeheader()
        n_fine_rows = 0

        for epoch in range(1, args.epochs + 1):
            sigma_t = sigma_schedule[epoch - 1]
            fine_log_this_epoch = args.fine_log_start <= epoch <= args.fine_log_end

            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            step_in_epoch = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                inject_noise(model, sigma=sigma_t, generator=noise_gen)
                optimizer.step()

                step_in_epoch += 1
                global_step += 1

                if fine_log_this_epoch:
                    b_mean, _, _ = optimizer.get_beta1_stats()
                    rho_vals = [
                        optimizer.state[p]['rho'].mean().item()
                        for group in optimizer.param_groups
                        for p in group['params']
                        if optimizer.state[p]
                    ]
                    rho_mean = sum(rho_vals) / len(rho_vals) if rho_vals else 0.0
                    fine_writer.writerow({
                        "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
                        "epoch": epoch, "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "sigma_t": sigma_t,
                        "beta1_mean": b_mean, "rho_mean": rho_mean,
                    })
                    n_fine_rows += 1

                running_loss += loss.item() * x.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += x.size(0)

            # Keep the diagnostic learning-rate trajectory identical to the
            # main experiment: one scheduler update per completed epoch.
            scheduler.step()
            epoch_time = time.time() - t0
            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)

            b_mean, b_min, b_max = optimizer.get_beta1_stats()
            s_mean, _, _ = optimizer.get_sigma2_stats()
            rho_vals = [
                optimizer.state[p]['rho'].mean().item()
                for group in optimizer.param_groups
                for p in group['params']
                if optimizer.state[p]
            ]
            rho_mean = sum(rho_vals) / len(rho_vals) if rho_vals else 0.0

            epoch_writer.writerow({
                "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0], "sigma_t": sigma_t,
                "epoch_time_sec": epoch_time,
                "beta1_mean": b_mean, "beta1_min": b_min, "beta1_max": b_max,
                "sigma2_mean": s_mean, "rho_mean": rho_mean,
            })
            ef.flush()
            ff.flush()

            fine_note = f" | logging every step ({step_in_epoch} rows)" if fine_log_this_epoch else ""
            print(f"[{run_name}] epoch {epoch:3d}/{args.epochs} | sigma_t={sigma_t:.3f} | "
                  f"val_acc={val_acc:.4f} | β₁={b_mean:.3f} | {epoch_time:.1f}s{fine_note}")

    print(f"[Lag Diagnostic] DONE — epoch log → {epoch_log_path}")
    print(f"[Lag Diagnostic] DONE — fine-grained log ({n_fine_rows} rows) → {fine_log_path}")


if __name__ == "__main__":
    main()

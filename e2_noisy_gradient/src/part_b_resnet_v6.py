"""
E2 · Part B — ResNet-18 / CIFAR-10 Noisy Gradient Confirmation
===============================================================
Trains ResNet-18 on CIFAR-10 with injected Gaussian gradient noise
at σ ∈ {0.05, 0.5} and compares Adam-NV vs Adam.

The clean baseline (σ=0.0) is reused from E1 logs — no re-training.
IMPORTANT: verify E1's config (epochs, batch_size, lr, scheduler,
architecture, optimizer defaults) matches this script exactly before
trusting Δacc — any mismatch invalidates the comparison silently.

Scientific claim validated here (H2b)
--------------------------------------
Under noisy gradient conditions, Adam-NV degrades more gracefully
than Adam, because adaptive β₁ reduces momentum accumulation of
corrupted gradient signals.

Primary metric: Δacc = acc(σ=0) − acc(σ>0)
A smaller Δacc for Adam-NV than Adam directly supports H2b.

Secondary: β₁ trajectory per epoch (Adam-NV only)
Should show lower mean β₁ under noise than clean — H2a on real data.

Noise formula (confirmed correct by calibration):
    g_noisy = g + σ · ‖g‖_F · ε̂,   ε̂ = ε / ‖ε‖_F,   ε ~ N(0, I)

Reproducibility
----------------
Noise injection uses its OWN seeded generator (seed+1), separate from
the model-init / DataLoader-shuffle RNG stream (seed). This isn't
strictly required here — each (optimizer, sigma, seed) combo runs in
its own subprocess, and inject_noise draws the same number of random
tensors regardless of sigma's magnitude (only sigma==0.0 skips the
draw), so optimizer-vs-optimizer and sigma-vs-sigma comparisons stay
RNG-fair even without this. But it's cheap insurance: if inject_noise
is ever changed to draw a variable number of times (e.g. skipping
small-gradient tensors), an implicit shared stream would silently
reintroduce the batch-order confound found in Part A. An explicit
separate generator makes the independence structural, not incidental.

Usage
-----
    # Single run:
    python part_b_resnet.py --optimizer adam_nv --sigma 0.5 --seed 0

    # Smoke test (fast, CPU, synthetic data):
    python part_b_resnet.py --optimizer adam --sigma 0.5 --seed 0 \
        --epochs 2 --smoke-test --synthetic-data

    # Force deterministic cuDNN (slower, tighter noise floor on Δacc):
    python part_b_resnet.py --optimizer adam_nv --sigma 0.5 --seed 0 \
        --deterministic

Outputs
-------
logs/part_b/{optimizer}_sigma{sigma}_seed{seed}.csv
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from adam_nv import AdamNV
from noise import inject_noise

# ── Optimizer registry ────────────────────────────────────────────────

def build_optimizer(name: str, params, lr: float, fixed_beta1: float = None,
                    gamma: float = None):
    name = name.lower()
    if name == "adam_nv":
        # E7-C ablation: gamma controls adaptation STRENGTH (not whether
        # beta_1 is fixed — that's adam_nv_fixed above). gamma=0 here means
        # beta_1 stays at beta_base=0.9 (the class default), which is a
        # DIFFERENT comparison than adam_nv_fixed's beta_base=<tuned plateau>.
        # This tests "does removing adaptation from the DEFAULT constant
        # hurt", not "does removing adaptation from a WELL-TUNED constant
        # hurt" (that's what E7-A already answered).
        kwargs = {}
        if gamma is not None:
            kwargs['gamma'] = gamma
        return AdamNV(params, lr=lr, **kwargs)
    elif name == "adam_nv_fixed":
        # E7 ablation: "is this just a tuned Adam?"
        # gamma=0 collapses AdamNV's adaptive beta_1 to a CONSTANT
        # (beta_1 = beta_base every step), while keeping the exact
        # bias correction P_t = beta_base^t (see AdamNV docstring FIX 3).
        # Setting beta_min == beta_base is redundant given gamma=0 (scale
        # is always 1 regardless), but keeps the invariant beta_min<=beta_base
        # explicit and self-documenting.
        if fixed_beta1 is None:
            raise ValueError("adam_nv_fixed requires --fixed-beta1")
        return AdamNV(params, lr=lr, gamma=0.0,
                       beta_base=fixed_beta1, beta_min=fixed_beta1)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError(f"Part B only uses adam_nv, adam_nv_fixed, and adam. Got: {name}")

DEFAULT_LR = {"adam_nv": 1e-3, "adam_nv_fixed": 1e-3, "adam": 1e-3}

# ── Model (identical CIFAR-adapted ResNet-18 as E1) ───────────────────

def build_model(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 with CIFAR-adapted stem.
    Identical to E1 — critical for fair comparison: any architecture
    difference would confound the noise-level effect we're measuring.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ── Data ──────────────────────────────────────────────────────────────

def apply_label_noise(dataset, noise_rate: float, seed: int, num_classes: int = 10):
    """
    E11: symmetric label noise. For `noise_rate` fraction of TRAINING
    samples, replace the true label with a uniformly random DIFFERENT
    class label. Corruption is fixed once (at dataset construction),
    not re-drawn per epoch — this matches standard label-noise-robustness
    literature setup. Only ever applied to train_set, never test_set.

    Uses its own independent RNG (numpy RandomState seeded separately
    from the loader/noise generators) so which samples get corrupted
    doesn't interact with batch order or gradient-noise draws — same
    RNG-independence principle established for inject_noise/DataLoader
    earlier in this project.
    """
    if noise_rate <= 0.0:
        return dataset
    import numpy as np
    rng = np.random.RandomState(seed)
    if not hasattr(dataset, 'targets'):
        raise ValueError("apply_label_noise requires a dataset with a "
                         "'targets' attribute (e.g. torchvision CIFAR10).")
    targets = list(dataset.targets)
    n = len(targets)
    n_corrupt = int(round(noise_rate * n))
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    for i in corrupt_idx:
        orig = targets[i]
        choices = [c for c in range(num_classes) if c != orig]
        targets[i] = int(rng.choice(choices))
    dataset.targets = targets
    return dataset


def get_dataloaders(batch_size: int, data_dir: str,
                    smoke_test: bool = False, synthetic: bool = False,
                    loader_generator=None, label_noise: float = 0.0,
                    label_noise_seed: int = 0):
    if synthetic:
        n_tr = 256 if smoke_test else 2000
        n_te = 128 if smoke_test else 500
        tr = torch.utils.data.TensorDataset(
            torch.randn(n_tr, 3, 32, 32), torch.randint(0, 10, (n_tr,)))
        te = torch.utils.data.TensorDataset(
            torch.randn(n_te, 3, 32, 32), torch.randint(0, 10, (n_te,)))
        return (torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                            shuffle=True,
                                            generator=loader_generator),
                torch.utils.data.DataLoader(te, batch_size=256, shuffle=False))

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    if label_noise > 0.0:
        train_ds = apply_label_noise(train_ds, label_noise, label_noise_seed)

    if smoke_test:
        train_ds = torch.utils.data.Subset(train_ds, range(512))
        test_ds  = torch.utils.data.Subset(test_ds,  range(256))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available(), generator=loader_generator)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available())
    return train_loader, test_loader

# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += loss_fn(out, y).item() * x.size(0)
        correct    += (out.argmax(dim=1) == y).sum().item()
        total      += x.size(0)
    model.train()
    return total_loss / total, correct / total

# ── Main training loop ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E2 Part B — ResNet-18 noisy gradient experiment")
    parser.add_argument("--optimizer", required=True,
                        choices=["adam_nv", "adam_nv_fixed", "adam"])
    parser.add_argument("--fixed-beta1", type=float, default=None,
                        help="Constant beta_1 value for --optimizer adam_nv_fixed "
                             "(E7 ablation: 'is this just a tuned Adam?'). "
                             "Required if --optimizer is adam_nv_fixed. "
                             "Pick this as the empirical converged plateau "
                             "value of full adaptive Adam-NV under the same "
                             "sigma, not an arbitrary guess.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override AdamNV's gamma (adaptation strength) "
                             "for --optimizer adam_nv. Class default is 1.0. "
                             "E7-C ablation: gamma=0 means no adaptation from "
                             "the DEFAULT beta_base=0.9, not from a tuned "
                             "constant (see adam_nv_fixed for that comparison).")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="E11: fraction of TRAINING labels to randomly "
                             "corrupt (symmetric noise, fixed once at dataset "
                             "construction). 0.0 = clean (default). Never "
                             "applied to test/validation set. Not supported "
                             "with --synthetic-data.")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Noise level: 0.0 (clean), 0.05 (mild), 0.5 (strong)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic cuDNN algorithms. "
                             "Slower, but removes conv-kernel nondeterminism "
                             "as a noise source in Δacc comparisons.")
    parser.add_argument("--log-every-n-steps", type=int, default=0,
                        help="If >0, also log per-step diagnostics "
                             "(g_before/g_after norms, beta1_mean, sigma2_mean) "
                             "every N optimizer steps, to a separate CSV. "
                             "Use this to check (a) whether inject_noise is "
                             "actually perturbing gradients in this script, "
                             "and (b) whether beta1/sigma2 move on a "
                             "within-epoch timescale that per-epoch logging "
                             "would miss. 0 = off (original per-epoch-only "
                             "behavior).")
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # else: deliberately leave cuDNN flags untouched (PyTorch defaults:
    # benchmark=False, deterministic=False) to match train_resnet.py (E1),
    # which never sets these. Setting benchmark=True here would introduce
    # a cuDNN-algorithm-selection difference between the E1 clean baseline
    # and Part B's noisy runs — an extraneous confound in Δacc that has
    # nothing to do with the injected noise being studied.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]

    if args.optimizer == "adam_nv_fixed" and args.fixed_beta1 is None:
        parser.error("--optimizer adam_nv_fixed requires --fixed-beta1")

    gamma_suffix = f"_gamma{args.gamma}" if args.gamma is not None else ""
    ln_suffix = f"_ln{args.label_noise}" if args.label_noise > 0.0 else ""
    # E12: batch_size was NOT part of run_name until now, which meant every
    # --batch-size value (32, 128, 256) for a given optimizer/sigma/seed
    # wrote to the SAME filename, silently overwriting earlier runs. Only
    # add the suffix when batch_size != 128 (the default used throughout
    # E2/E7/E11) so none of those existing filenames/results are affected.
    bs_suffix = f"_bs{args.batch_size}" if args.batch_size != 128 else ""
    run_name = (f"{args.optimizer}_sigma{args.sigma}{gamma_suffix}{ln_suffix}{bs_suffix}_seed{args.seed}"
                + ("_smoke" if args.smoke_test else ""))

    print(f"[Part B] {run_name} | epochs={args.epochs} lr={lr} "
          f"device={device} smoke={args.smoke_test} "
          f"deterministic={args.deterministic}")

    if args.sigma == 0.0:
        print("[Part B] WARNING: sigma=0.0 — this is a clean run. "
              "Consider reusing E1 logs instead of re-training.")

    # ── Dedicated RNG streams ──────────────────────────────────────
    # loader_gen: controls DataLoader shuffle order. Seeded from
    # args.seed so it's reproducible per-seed but independent of the
    # noise stream below.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    # noise_gen: controls gradient noise draws inside inject_noise.
    # Kept separate from loader_gen so noise-injection RNG consumption
    # can never perturb batch order (see module docstring above).
    if device.type == "cuda":
        noise_gen = torch.Generator(device=device)
    else:
        noise_gen = torch.Generator()
    noise_gen.manual_seed(args.seed + 1)

    if args.label_noise > 0.0 and args.synthetic_data:
        parser.error("--label-noise is not supported with --synthetic-data "
                     "(synthetic labels are already random; corrupting "
                     "random labels is meaningless).")

    # ── Setup ────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir,
        smoke_test=args.smoke_test, synthetic=args.synthetic_data,
        loader_generator=loader_gen, label_noise=args.label_noise,
        label_noise_seed=args.seed + 2000)

    model     = build_model().to(device)
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr,
                                fixed_beta1=args.fixed_beta1, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{run_name}.csv")

    is_adam_nv = args.optimizer in ("adam_nv", "adam_nv_fixed")

    # Fieldnames: base + Adam-NV diagnostics if applicable
    fieldnames = ["epoch", "train_loss", "train_acc",
                  "val_loss", "val_acc", "lr", "epoch_time_sec"]
    if is_adam_nv:
        fieldnames += ["beta1_mean", "beta1_min", "beta1_max",
                       "sigma2_mean", "rho_mean"]

    # ── Optional within-epoch step-level diagnostics ──────────────────
    # Separate CSV so it doesn't disturb the existing per-epoch schema
    # that earlier analysis already depends on.
    step_log_path = None
    step_writer = None
    step_f = None
    if args.log_every_n_steps > 0:
        step_log_path = os.path.join(args.log_dir, f"{run_name}_steps.csv")
        step_fieldnames = ["global_step", "epoch", "step_in_epoch",
                            "g_before_norm", "g_after_norm", "delta_norm"]
        if is_adam_nv:
            step_fieldnames += ["beta1_mean", "sigma2_mean"]
        step_f = open(step_log_path, "w", newline="")
        step_writer = csv.DictWriter(step_f, fieldnames=step_fieldnames)
        step_writer.writeheader()
        print(f"[Part B] Step-level diagnostics every "
              f"{args.log_every_n_steps} steps → {step_log_path}")

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for step_in_epoch, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out  = model(x)
                loss = loss_fn(out, y)
                loss.backward()

                do_step_log = (args.log_every_n_steps > 0 and
                                global_step % args.log_every_n_steps == 0)

                if do_step_log:
                    g_before_norm = sum(
                        p.grad.norm().item()
                        for p in model.parameters() if p.grad is not None)

                # ── Noise injection ──────────────────────────────────
                # Called AFTER backward, BEFORE step.
                # sigma=0.0 is a no-op (explicit guard in inject_noise).
                # Matched RNG directions; magnitudes scale with each run's gradient norm.
                # Uses its own generator — see reproducibility note above.
                inject_noise(model, sigma=args.sigma, generator=noise_gen)

                if do_step_log:
                    g_after_norm = sum(
                        p.grad.norm().item()
                        for p in model.parameters() if p.grad is not None)

                optimizer.step()

                if do_step_log:
                    step_row = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "g_before_norm": g_before_norm,
                        "g_after_norm": g_after_norm,
                        "delta_norm": g_after_norm - g_before_norm,
                    }
                    if is_adam_nv:
                        b_mean, _, _ = optimizer.get_beta1_stats()
                        s_mean, _, _ = optimizer.get_sigma2_stats()
                        step_row["beta1_mean"] = b_mean
                        step_row["sigma2_mean"] = s_mean
                    step_writer.writerow(step_row)
                    step_f.flush()

                global_step += 1

                running_loss += loss.item() * x.size(0)
                correct      += (out.argmax(dim=1) == y).sum().item()
                total        += x.size(0)

            scheduler.step()
            epoch_time = time.time() - t0

            train_loss = running_loss / total
            train_acc  = correct / total
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)

            row = {
                "epoch":          epoch,
                "train_loss":     train_loss,
                "train_acc":      train_acc,
                "val_loss":       val_loss,
                "val_acc":        val_acc,
                "lr":             scheduler.get_last_lr()[0],
                "epoch_time_sec": epoch_time,
            }

            # Adam-NV diagnostics — logged every epoch for β₁ trajectory
            if is_adam_nv:
                b_mean, b_min, b_max = optimizer.get_beta1_stats()
                s_mean, _, _         = optimizer.get_sigma2_stats()

                # rho mean — read directly from state
                rho_vals = [
                    optimizer.state[p]['rho'].mean().item()
                    for group in optimizer.param_groups
                    for p in group['params']
                    if optimizer.state[p]
                ]
                rho_mean = sum(rho_vals) / len(rho_vals) if rho_vals else 0.0

                row.update({
                    "beta1_mean": b_mean, "beta1_min": b_min,
                    "beta1_max":  b_max,  "sigma2_mean": s_mean,
                    "rho_mean":   rho_mean,
                })

            writer.writerow(row)
            f.flush()

            print(f"[{run_name}] epoch {epoch:3d}/{args.epochs} | "
                  f"train={train_acc:.4f} val={val_acc:.4f} "
                  f"loss={val_loss:.4f} | {epoch_time:.1f}s"
                  + (f" | β₁={row.get('beta1_mean', 0):.3f}" if is_adam_nv else ""))

    if step_f is not None:
        step_f.close()

    print(f"[Part B] DONE — {run_name} | log → {log_path}")


if __name__ == "__main__":
    main()

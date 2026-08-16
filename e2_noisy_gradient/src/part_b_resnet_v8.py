"""
E13 · Part B (v8) — Non-Stationary Noise Schedule Experiment
===============================================================
Extends part_b_resnet_v7.py with time-varying gradient noise, so the
question Adam-NV was actually built to answer can finally be tested:
does adapting beta_1 in real time help when noise CHANGES during
training, as opposed to being fixed at one level for the whole run
(v6/v7's constant-sigma design, used throughout E2/E7/E11/E12 and the
Phase 1.4 baseline suite)?

Everything else — model, data, optimizer registry, label-noise
corruption, RNG-stream separation, CSV schema — is byte-identical to
v7. Only the per-epoch sigma now comes from a schedule (noise_schedule.py)
instead of a single fixed --sigma value, and the schedule's sigma(t)
is logged alongside the usual per-epoch metrics. Do not treat this as
a v7 replacement for existing constant-sigma conditions — use v7 for
those; use v8 only for --schedule runs, so existing v6/v7 logs are
never touched.

Scientific question this addresses
------------------------------------
The E2/E11 report (Phase 1.4 baseline-noise suite writeup) found
Adam-NV statistically indistinguishable from plain Adam on final
accuracy AND convergence speed, under both constant gradient noise
(sigma=2.0) and constant label noise (20%), despite Adam-NV's internal
beta1/rho diagnostics confirming the adaptive mechanism responds
correctly to noise type and severity. The likely explanation flagged
in that report: every condition tested held sigma fixed for the full
60 epochs, so a fixed-beta1 baseline never had to adapt to anything —
it could just settle into whatever beta1 already suits that one
constant level.

This script removes that confound directly by making sigma vary
WITHIN a single run (phased / ramped / intermittent — see
noise_schedule.py). The primary diagnostic is not just final accuracy,
but whether Adam-NV's logged beta1_mean per epoch actually TRACKS the
schedule's sigma(t) per epoch within a single run — real-time
adaptation, not just a different plateau across separate runs.

Label noise is NOT scheduled in this version — apply_label_noise
corrupts labels once at dataset construction (same as v7), so
--label-noise here still means constant label noise for the full run,
exactly as in v7. A non-stationary label-noise schedule (corruption
rate changing mid-training) would need a dataset wrapper that can
re-corrupt per epoch, which is a larger change deliberately left out
of scope here — flag as a follow-up (Phase 1.6) rather than bolting
it on for this suite.

Noise formula (identical to v6/v7, confirmed correct by calibration):
    g_noisy = g + sigma(t) · ‖g‖_F · ε̂,   ε̂ = ε / ‖ε‖_F,   ε ~ N(0, I)
The only change is that sigma is now a function of epoch instead of a
constant — inject_noise() itself is untouched.

Reproducibility
----------------
Same dedicated RNG streams as v7 (loader_gen from --seed, noise_gen
from --seed+1, label-noise RNG from --seed+2000). The schedule itself
is deterministic given (schedule, total_epochs, sigma_max, schedule
params) — no extra randomness introduced by scheduling.

Usage
-----
    # Phased: clean (epochs 1-20) -> noisy sigma=2.0 (21-40) -> clean (41-60)
    python part_b_resnet_v8.py --optimizer adam_nv --schedule phased \
        --sigma 2.0 --seed 0

    # Linear ramp 0 -> 2.0 over the full run
    python part_b_resnet_v8.py --optimizer adam --schedule ramp_up \
        --sigma 2.0 --seed 0

    # Triangular ramp 0 -> 2.0 -> 0
    python part_b_resnet_v8.py --optimizer adam_nv --schedule ramp_updown \
        --sigma 2.0 --seed 0

    # Intermittent bursts: 5 noisy epochs / 5 clean epochs, repeating
    python part_b_resnet_v8.py --optimizer adam_nv --schedule intermittent \
        --sigma 2.0 --burst-epochs 5 --gap-epochs 5 --seed 0

    # Sanity check — should reproduce v7 --sigma 2.0 behavior exactly:
    python part_b_resnet_v8.py --optimizer adam_nv --schedule constant \
        --sigma 2.0 --seed 0

    # Smoke test (fast, CPU, synthetic data):
    python part_b_resnet_v8.py --optimizer adam_nv --schedule phased \
        --sigma 2.0 --seed 0 --epochs 6 --smoke-test --synthetic-data

Outputs
-------
logs/part_b/{optimizer}_schedule{schedule}_sigma{sigma_max}_seed{seed}.csv
Per-epoch rows include everything v7 logs, PLUS a `sigma_t` column
giving the actual sigma applied that epoch (read this alongside
beta1_mean to check real-time tracking).

--beta2 (new, additive) — beta2-sweep experiment
------------------------------------------------
The Phase 1.5 report (intermittent schedule) found a small, consistent
accuracy deficit for adam_nv, explained by a lag: v's EMA (beta_2, class
default 0.999, ~2.5-epoch memory) feeds directly into the sigma2 noise
signal (sigma2 = (v - m^2) / (v + lam|m| + eps)), so the noise estimate
-- and therefore beta_1 -- responds several epochs slower than the
schedule itself changes. --beta2 lets you test whether shrinking that
memory (e.g. --beta2 0.99 or 0.95) shrinks the lag and recovers the
intermittent deficit.

IMPORTANT CONFOUND: v is not dedicated to noise-tracking -- it is the
same v used in Adam's own step-size denominator (v_hat in the parameter
update). Changing --beta2 therefore changes two things at once: (a) how
fast the noise estimate tracks a changing environment, AND (b) Adam-NV's
general step-size normalization, independent of noise-adaptivity. If
this sweep shows an effect, it does NOT yet prove the effect is due to
faster noise-tracking specifically -- it only justifies building a
decoupled v_noise (separate beta_2_noise, isolated from the update
denominator) to confirm it. Treat this sweep as a screening step, not
as confirmation of the lag hypothesis by itself.

    # beta2 sweep example (single run):
    python part_b_resnet_v8.py --optimizer adam_nv --schedule intermittent \\
        --sigma 2.0 --beta2 0.95 --seed 0
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
from noise_schedule import get_sigma_schedule, describe_schedule
from experiment_metadata import write_metadata

try:
    from adabelief_pytorch import AdaBelief
except ImportError:
    AdaBelief = None  # only required if --optimizer adabelief is used

# ── Optimizer registry (identical to v7) ───────────────────────────────

def build_optimizer(name: str, params, lr: float, fixed_beta1: float = None,
                    gamma: float = None, beta2: float = None, beta2_noise: float = None,
                    response_mode: str = "distrust", beta_max: float = 0.999):
    name = name.lower()
    if name == "adam_nv":
        kwargs = {"response_mode": response_mode, "beta_max": beta_max}
        if gamma is not None:
            kwargs['gamma'] = gamma
        if beta2 is not None:
            kwargs['beta_2'] = beta2
        if beta2_noise is not None:
            kwargs['beta_2_noise'] = beta2_noise
        return AdamNV(params, lr=lr, **kwargs)
    elif name == "adam_nv_fixed":
        if fixed_beta1 is None:
            raise ValueError("adam_nv_fixed requires --fixed-beta1")
        kwargs = dict(gamma=0.0, beta_base=fixed_beta1, beta_min=fixed_beta1)
        if beta2 is not None:
            kwargs['beta_2'] = beta2
        if beta2_noise is not None:
            kwargs['beta_2_noise'] = beta2_noise
        return AdamNV(params, lr=lr, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=1e-4)
    elif name == "adabelief":
        if AdaBelief is None:
            raise ImportError(
                "AdaBelief requires the adabelief-pytorch package. "
                "Install with: pip install adabelief-pytorch")
        return AdaBelief(params, lr=lr, rectify=False, weight_decouple=False,
                         print_change_log=False)
    else:
        raise ValueError(
            "Part B (v8) uses adam_nv, adam_nv_fixed, adam, adamw, "
            f"and adabelief. Got: {name}")

DEFAULT_LR = {"adam_nv": 1e-3, "adam_nv_fixed": 1e-3, "adam": 1e-3,
              "adamw": 1e-3, "adabelief": 1e-3}

# ── Model (identical CIFAR-adapted ResNet-18 as E1/v7) ─────────────────

def build_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

# ── Data (identical to v7) ──────────────────────────────────────────────

def apply_label_noise(dataset, noise_rate: float, seed: int, num_classes: int = 10):
    """
    Symmetric label noise, fixed once at dataset construction. Identical
    to v7 — NOT scheduled in v8 (see module docstring). Only applied to
    train_set, never test_set.
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


class LabelDiagnosticDataset(torch.utils.data.Dataset):
    """Returns clean/noisy labels together for memorization diagnostics."""

    def __init__(self, clean_dataset, noisy_targets):
        self.clean_dataset = clean_dataset
        self.noisy_targets = list(noisy_targets)

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        image, clean_target = self.clean_dataset[index]
        noisy_target = self.noisy_targets[index]
        return image, clean_target, noisy_target, clean_target != noisy_target


def get_dataloaders(batch_size: int, data_dir: str,
                    smoke_test: bool = False, synthetic: bool = False,
                    loader_generator=None, label_noise: float = 0.0,
                    label_noise_seed: int = 0, dataset: str = "cifar10",
                    return_label_diagnostics: bool = False):
    if dataset not in ("cifar10", "cifar100"):
        raise ValueError(f"Unsupported dataset: {dataset}")
    num_classes = 10 if dataset == "cifar10" else 100
    if synthetic:
        n_tr = 256 if smoke_test else 2000
        n_te = 128 if smoke_test else 500
        tr = torch.utils.data.TensorDataset(
            torch.randn(n_tr, 3, 32, 32), torch.randint(0, num_classes, (n_tr,)))
        te = torch.utils.data.TensorDataset(
            torch.randn(n_te, 3, 32, 32), torch.randint(0, num_classes, (n_te,)))
        return (torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                            shuffle=True,
                                            generator=loader_generator),
                torch.utils.data.DataLoader(te, batch_size=256, shuffle=False))

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        dataset_class = torchvision.datasets.CIFAR10
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        dataset_class = torchvision.datasets.CIFAR100
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
    train_ds = dataset_class(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = dataset_class(
        root=data_dir, train=False, download=True, transform=test_tf)

    clean_targets = list(train_ds.targets)
    if label_noise > 0.0:
        train_ds = apply_label_noise(
            train_ds, label_noise, label_noise_seed, num_classes=num_classes)
    noisy_targets = list(train_ds.targets)

    if smoke_test:
        train_ds = torch.utils.data.Subset(train_ds, range(512))
        test_ds  = torch.utils.data.Subset(test_ds,  range(256))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available(), generator=loader_generator)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available())
    if not return_label_diagnostics:
        return train_loader, test_loader

    clean_eval_ds = dataset_class(
        root=data_dir, train=True, download=False, transform=test_tf)
    diagnostic_ds = LabelDiagnosticDataset(clean_eval_ds, noisy_targets)
    if smoke_test:
        diagnostic_ds = torch.utils.data.Subset(diagnostic_ds, range(512))
    diagnostic_loader = torch.utils.data.DataLoader(
        diagnostic_ds, batch_size=256, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available())
    return train_loader, test_loader, diagnostic_loader

# ── Evaluation (identical to v7) ────────────────────────────────────────

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


@torch.no_grad()
def evaluate_label_memorization(model: nn.Module, loader, device) -> dict:
    """Measure fitting to clean versus corrupted labels in one train-set pass."""
    model.eval()
    counts = {"clean": 0, "corrupt": 0}
    correct = {"clean": 0, "corrupt_clean": 0, "corrupt_noisy": 0}
    for x, clean_y, noisy_y, is_corrupt in loader:
        x = x.to(device)
        clean_y = clean_y.to(device)
        noisy_y = noisy_y.to(device)
        is_corrupt = is_corrupt.to(device).bool()
        prediction = model(x).argmax(dim=1)
        clean_mask = ~is_corrupt
        counts["clean"] += int(clean_mask.sum().item())
        counts["corrupt"] += int(is_corrupt.sum().item())
        correct["clean"] += int((prediction[clean_mask] == clean_y[clean_mask]).sum().item())
        correct["corrupt_clean"] += int(
            (prediction[is_corrupt] == clean_y[is_corrupt]).sum().item())
        correct["corrupt_noisy"] += int(
            (prediction[is_corrupt] == noisy_y[is_corrupt]).sum().item())
    model.train()
    return {
        "train_clean_subset_acc": correct["clean"] / counts["clean"] if counts["clean"] else None,
        "train_corrupt_clean_label_acc": (
            correct["corrupt_clean"] / counts["corrupt"] if counts["corrupt"] else None),
        "train_corrupt_noisy_label_acc": (
            correct["corrupt_noisy"] / counts["corrupt"] if counts["corrupt"] else None),
    }

# ── Main training loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E13 Part B (v8) — non-stationary noise schedule experiment")
    parser.add_argument("--optimizer", required=True,
                        choices=["adam_nv", "adam_nv_fixed", "adam",
                                "adamw", "adabelief"])
    parser.add_argument("--fixed-beta1", type=float, default=None,
                        help="Constant beta_1 for --optimizer adam_nv_fixed. "
                             "Required if --optimizer is adam_nv_fixed.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override AdamNV's gamma (adaptation strength) "
                             "for --optimizer adam_nv. Class default is 1.0.")
    parser.add_argument("--response-mode", choices=["distrust", "smooth"],
                        default="distrust",
                        help="Direction ablation: original 'distrust' lowers beta_1 "
                             "as the heuristic rises; 'smooth' raises beta_1 from "
                             "0.9 toward --beta-max. Default: distrust.")
    parser.add_argument("--beta-max", type=float, default=0.999,
                        help="Upper beta_1 bound for --response-mode smooth.")
    parser.add_argument("--beta2", type=float, default=None,
                        help="Override AdamNV's beta_2 (second-moment decay) "
                             "for --optimizer adam_nv/adam_nv_fixed. Class "
                             "default is 0.999 (matches all prior Adam-NV "
                             "runs). Lower values shrink v's noise-tracking "
                             "memory but ALSO change Adam's step-size "
                             "normalization -- see module docstring, "
                             "'beta2-sweep experiment', for the confound "
                             "this introduces. Ignored for other optimizers.")
    parser.add_argument("--beta2-noise", type=float, default=None,
                        help="AdamNV's beta_2_noise (FIX 6): gives the noise "
                             "estimator its OWN second-moment buffer, isolated "
                             "from the parameter update's v/beta_2. Default "
                             "None means v_noise=v (original coupled behavior, "
                             "exactly reproduces every prior Adam-NV result). "
                             "Use this instead of --beta2 to change noise-"
                             "tracking speed WITHOUT the step-size-"
                             "normalization confound the beta2-sweep found. "
                             "Ignored for other optimizers.")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of TRAINING labels to randomly corrupt, "
                             "fixed once at dataset construction (NOT "
                             "scheduled — see module docstring). 0.0 = clean.")
    parser.add_argument("--label-diagnostic-interval", type=int, default=1,
                        help="For label-noise runs, evaluate clean and corrupted "
                             "training subsets every N epochs. Default: 1.")

    # ── Schedule arguments (new in v8) ──────────────────────────────
    parser.add_argument("--schedule", type=str, default="constant",
                        choices=["constant", "phased", "ramp_up",
                                "ramp_updown", "intermittent"],
                        help="Per-epoch gradient-noise schedule. 'constant' "
                             "reproduces v7's fixed-sigma behavior exactly "
                             "(sanity check). Default: constant.")
    parser.add_argument("--sigma", type=float, required=True,
                        help="Peak/target noise level (sigma_max). Same "
                             "scale as v7's --sigma: 0.05 mild, 0.5 strong, "
                             "2.0 the level used throughout E2/E7/E11/E12 "
                             "and the Phase 1.4 baseline suite.")
    parser.add_argument("--phase-start", type=float, default=1.0 / 3.0,
                        help="'phased' only: fraction of training (0-1) "
                             "where the noisy phase begins. Default 1/3.")
    parser.add_argument("--phase-end", type=float, default=2.0 / 3.0,
                        help="'phased' only: fraction of training (0-1) "
                             "where the noisy phase ends. Default 2/3.")
    parser.add_argument("--burst-epochs", type=int, default=5,
                        help="'intermittent' only: consecutive noisy "
                             "epochs per cycle. Default 5.")
    parser.add_argument("--gap-epochs", type=int, default=5,
                        help="'intermittent' only: consecutive clean "
                             "epochs per cycle. Default 5.")
    parser.add_argument("--start-clean", action="store_true", default=True,
                        help="'intermittent' only: start with a clean "
                             "block before the first noisy burst. Default.")
    parser.add_argument("--start-noisy", dest="start_clean",
                        action="store_false",
                        help="'intermittent' only: start with a noisy "
                             "burst instead of a clean block.")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/part_b")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"],
                        default="cifar10",
                        help="Dataset for the run. Default: cifar10.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic cuDNN algorithms.")
    args = parser.parse_args()

    if args.label_diagnostic_interval < 1:
        parser.error("--label-diagnostic-interval must be at least 1")

    # ── Reproducibility (identical to v7) ────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]

    if args.optimizer == "adam_nv_fixed" and args.fixed_beta1 is None:
        parser.error("--optimizer adam_nv_fixed requires --fixed-beta1")

    if args.label_noise > 0.0 and args.synthetic_data:
        parser.error("--label-noise is not supported with --synthetic-data.")

    if args.beta2 is not None and args.optimizer not in ("adam_nv", "adam_nv_fixed"):
        print(f"[Part B v8] NOTE: --beta2 is ignored for --optimizer "
              f"{args.optimizer} (only used by adam_nv/adam_nv_fixed).")

    if args.beta2_noise is not None and args.optimizer not in ("adam_nv", "adam_nv_fixed"):
        print(f"[Part B v8] NOTE: --beta2-noise is ignored for --optimizer "
              f"{args.optimizer} (only used by adam_nv/adam_nv_fixed).")

    if args.schedule != "intermittent" and (
            args.burst_epochs != 5 or args.gap_epochs != 5):
        print("[Part B v8] NOTE: --burst-epochs/--gap-epochs are ignored "
              f"for --schedule {args.schedule} (only used by 'intermittent').")
    if args.schedule != "phased" and (
            args.phase_start != 1.0 / 3.0 or args.phase_end != 2.0 / 3.0):
        print("[Part B v8] NOTE: --phase-start/--phase-end are ignored "
              f"for --schedule {args.schedule} (only used by 'phased').")

    # ── Precompute the sigma(t) schedule once (see noise_schedule.py) ─
    sigma_schedule = get_sigma_schedule(
        args.schedule, total_epochs=args.epochs, sigma_max=args.sigma,
        phase_start=args.phase_start, phase_end=args.phase_end,
        burst_epochs=args.burst_epochs, gap_epochs=args.gap_epochs,
        start_clean=args.start_clean)
    schedule_desc = describe_schedule(
        args.schedule, phase_start=args.phase_start, phase_end=args.phase_end,
        burst_epochs=args.burst_epochs, gap_epochs=args.gap_epochs,
        start_clean=args.start_clean)

    # ── Dedicated RNG streams (identical to v7) ──────────────────────
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    if device.type == "cuda":
        noise_gen = torch.Generator(device=device)
    else:
        noise_gen = torch.Generator()
    noise_gen.manual_seed(args.seed + 1)

    # ── run_name / filename ───────────────────────────────────────────
    gamma_suffix = f"_gamma{args.gamma}" if args.gamma is not None else ""
    beta2_suffix = f"_beta2{args.beta2}" if args.beta2 is not None else ""
    beta2n_suffix = f"_beta2noise{args.beta2_noise}" if args.beta2_noise is not None else ""
    response_suffix = ("" if args.response_mode == "distrust" else
                       f"_response{args.response_mode}_betamax{args.beta_max}")
    ln_suffix = f"_ln{args.label_noise}" if args.label_noise > 0.0 else ""
    bs_suffix = f"_bs{args.batch_size}" if args.batch_size != 128 else ""
    dataset_suffix = "" if args.dataset == "cifar10" else f"_{args.dataset}"
    run_name = (f"{args.optimizer}_schedule{args.schedule}_sigma{args.sigma}"
                f"{gamma_suffix}{beta2_suffix}{beta2n_suffix}{response_suffix}{ln_suffix}{bs_suffix}"
                f"{dataset_suffix}_seed{args.seed}"
                + ("_smoke" if args.smoke_test else ""))

    print(f"[Part B v8] {run_name} | epochs={args.epochs} lr={lr} "
          f"device={device} smoke={args.smoke_test} "
          f"deterministic={args.deterministic} dataset={args.dataset}")
    print(f"[Part B v8] Schedule: {schedule_desc}")
    print(f"[Part B v8] Script version: part_b_resnet_v8")

    # ── Setup ────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        args.batch_size, args.data_dir,
        smoke_test=args.smoke_test, synthetic=args.synthetic_data,
        loader_generator=loader_gen, label_noise=args.label_noise,
        label_noise_seed=args.seed + 2000, dataset=args.dataset,
        return_label_diagnostics=args.label_noise > 0.0)
    if args.label_noise > 0.0:
        train_loader, test_loader, label_diagnostic_loader = loaders
    else:
        train_loader, test_loader = loaders
        label_diagnostic_loader = None

    num_classes = 10 if args.dataset == "cifar10" else 100
    model     = build_model(num_classes=num_classes).to(device)
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr,
                                fixed_beta1=args.fixed_beta1, gamma=args.gamma,
                                beta2=args.beta2, beta2_noise=args.beta2_noise,
                                response_mode=args.response_mode, beta_max=args.beta_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    loss_fn   = nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{run_name}.csv")
    metadata_path = write_metadata(log_path, vars(args), "part_b_resnet_v8")
    print(f"[Part B v8] Metadata: {metadata_path}")

    is_adam_nv = args.optimizer in ("adam_nv", "adam_nv_fixed")

    # Fieldnames: v7 schema + sigma_t (the schedule column, new in v8)
    fieldnames = ["epoch", "train_loss", "train_acc",
                  "val_loss", "val_acc", "lr", "sigma_t", "epoch_time_sec"]
    if is_adam_nv:
        fieldnames += ["beta1_mean", "beta1_min", "beta1_max",
                       "sigma2_mean", "rho_mean"]
    if label_diagnostic_loader is not None:
        fieldnames += ["train_clean_subset_acc", "train_corrupt_clean_label_acc",
                       "train_corrupt_noisy_label_acc"]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            sigma_t = sigma_schedule[epoch - 1]

            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out  = model(x)
                loss = loss_fn(out, y)
                loss.backward()

                # ── Noise injection at this epoch's scheduled sigma ──
                # Same inject_noise() as v7 — only the sigma value now
                # comes from the schedule instead of a constant.
                inject_noise(model, sigma=sigma_t, generator=noise_gen)

                optimizer.step()

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
                "sigma_t":        sigma_t,
                "epoch_time_sec": epoch_time,
            }

            if label_diagnostic_loader is not None and (
                    epoch % args.label_diagnostic_interval == 0 or epoch == args.epochs):
                row.update(evaluate_label_memorization(
                    model, label_diagnostic_loader, device))

            if is_adam_nv:
                b_mean, b_min, b_max = optimizer.get_beta1_stats()
                s_mean, _, _         = optimizer.get_sigma2_stats()
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
                  f"sigma_t={sigma_t:.3f} | "
                  f"train={train_acc:.4f} val={val_acc:.4f} "
                  f"loss={val_loss:.4f} | {epoch_time:.1f}s"
                  + (f" | β₁={row.get('beta1_mean', 0):.3f}" if is_adam_nv else ""))

    print(f"[Part B v8] DONE — {run_name} | log → {log_path}")


if __name__ == "__main__":
    main()

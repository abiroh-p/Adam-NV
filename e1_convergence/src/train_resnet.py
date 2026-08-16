"""
E1 · Convergence Benchmarks — single training run.

Trains ResNet-18 on CIFAR-10 with one optimizer / one seed, logging
per-epoch metrics (and, for Adam-NV, beta_1/sigma^2/P diagnostics) to a
CSV file under logs/. Designed to be called repeatedly by run_e1_suite.py
for every (optimizer, seed) combination.

Usage:
    python train_resnet.py --optimizer adam_nv --seed 0 --epochs 60
    python train_resnet.py --optimizer adam     --seed 0 --epochs 60
    python train_resnet.py --optimizer adamw    --seed 0 --epochs 60
    python train_resnet.py --optimizer adabelief --seed 0 --epochs 60

Quick smoke test (tiny, CPU-friendly, ~1 minute):
    python train_resnet.py --optimizer adam_nv --seed 0 --epochs 1 \
        --smoke-test --batch-size 32
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
from experiment_metadata import write_metadata

# ──────────────────────────────────────────────────────────────────────────
# Optimizer registry — single source of truth for E1's three Tier-1 baselines
# plus Adam-NV itself. AdaBelief needs the `adabelief-pytorch` package; we
# import it lazily so the script still runs without it for adam/adamw/adam_nv.
# ──────────────────────────────────────────────────────────────────────────

def build_optimizer(name, params, lr):
    name = name.lower()
    if name == "adam_nv":
        return AdamNV(params, lr=lr)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=1e-4)
    elif name == "adabelief":
        try:
            from adabelief_pytorch import AdaBelief
        except ImportError as e:
            raise ImportError(
                "AdaBelief requires `pip install adabelief-pytorch --break-system-packages`. "
                "Install it before running --optimizer adabelief."
            ) from e
        return AdaBelief(params, lr=lr, eps=1e-8, betas=(0.9, 0.999),
                          weight_decouple=False, rectify=False, print_change_log=False)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# Per-optimizer default LR. Tier 1 uses ONE literature-default LR per
# optimizer (no sweep) — see Experimental Design Doc, Section 3, E1.
DEFAULT_LR = {
    "adam_nv":   1e-3,
    "adam":      1e-3,
    "adamw":     1e-3,
    "adabelief": 1e-3,
}


def build_model(num_classes=10):
    """Standard ResNet-18 adapted for CIFAR-sized (32x32) inputs.

    The torchvision ResNet-18 is built for 224x224 ImageNet inputs (7x7
    stride-2 stem conv + maxpool), which over-downsamples 32x32 CIFAR
    images. We replace the stem with the standard CIFAR ResNet-18 variant:
    3x3 stride-1 conv, no initial maxpool. This is the conventional setup
    used in CIFAR optimizer-benchmark papers.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_dataloaders(batch_size, data_dir, smoke_test=False, synthetic=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if synthetic:
        # Fake CIFAR-10-shaped random data — used ONLY to validate that the
        # training loop runs end-to-end without bugs (shapes, loss, backward,
        # optimizer step, scheduler, logging). This is NOT a substitute for
        # real training and produces meaningless accuracy numbers. Use this
        # when CIFAR-10 cannot be downloaded (e.g. restricted sandbox network).
        n_train = 256 if smoke_test else 2000
        n_test = 128 if smoke_test else 500
        train_x = torch.randn(n_train, 3, 32, 32)
        train_y = torch.randint(0, 10, (n_train,))
        test_x = torch.randn(n_test, 3, 32, 32)
        test_y = torch.randint(0, 10, (n_test,))
        train_set = torch.utils.data.TensorDataset(train_x, train_y)
        test_set = torch.utils.data.TensorDataset(test_x, test_y)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        return train_loader, test_loader

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

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                              download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                             download=True, transform=test_tf)

    if smoke_test:
        # Tiny subset so a full "epoch" finishes in seconds on CPU — this
        # validates the training loop logic, NOT real convergence behavior.
        train_set = torch.utils.data.Subset(train_set, range(256))
        test_set = torch.utils.data.Subset(test_set, range(128))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available())
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    model.train()
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="E1 Convergence Benchmark — single run")
    parser.add_argument("--optimizer", required=True,
                         choices=["adam_nv", "adam", "adamw", "adabelief"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None,
                         help="Override default LR for this optimizer")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="../logs")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny dataset subset + verbose per-step prints, "
                              "for validating the training loop, not real results.")
    parser.add_argument("--synthetic-data", action="store_true",
                         help="Use random fake-CIFAR tensors instead of downloading "
                              "real CIFAR-10. ONLY for validating the training loop "
                              "runs without bugs — produces meaningless accuracy. "
                              "Use on networks that block the CIFAR-10 mirror.")
    parser.add_argument("--deterministic", action="store_true",
                         help="Force deterministic cuDNN algorithms.")
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]

    print(f"[E1] optimizer={args.optimizer} seed={args.seed} epochs={args.epochs} "
          f"lr={lr} device={device} smoke_test={args.smoke_test}")
    if args.synthetic_data:
        print("[E1] *** SYNTHETIC DATA MODE — results are meaningless, "
              "this run ONLY validates the training loop runs without errors ***")

    train_loader, test_loader = get_dataloaders(
        args.batch_size, args.data_dir, smoke_test=args.smoke_test,
        synthetic=args.synthetic_data)

    model = build_model(num_classes=10).to(device)
    optimizer = build_optimizer(args.optimizer, model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.log_dir, exist_ok=True)
    run_name = f"{args.optimizer}_seed{args.seed}" + ("_smoke" if args.smoke_test else "")
    log_path = os.path.join(args.log_dir, f"{run_name}.csv")
    metadata_path = write_metadata(log_path, vars(args), "e1_train_resnet")
    print(f"[E1] Metadata: {metadata_path}")

    is_adam_nv = args.optimizer == "adam_nv"
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                  "lr", "epoch_time_sec"]
    if is_adam_nv:
        fieldnames += ["beta1_mean", "beta1_min", "beta1_max",
                        "sigma2_mean", "sigma2_min", "sigma2_max",
                        "P_mean", "P_min", "P_max"]

    reached_90 = None  # epoch at which val_acc first reaches 90% (CIFAR-10)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += x.size(0)

                if args.smoke_test and step == 0:
                    print(f"  epoch {epoch} step {step}: loss={loss.item():.4f}")

            scheduler.step()
            epoch_time = time.time() - t0

            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = evaluate(model, test_loader, loss_fn, device)

            if reached_90 is None and val_acc >= 0.90:
                reached_90 = epoch

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_sec": epoch_time,
            }
            if is_adam_nv:
                b_mean, b_min, b_max = optimizer.get_beta1_stats()
                s_mean, s_min, s_max = optimizer.get_sigma2_stats()
                p_mean, p_min, p_max = optimizer.get_P_stats()
                row.update({
                    "beta1_mean": b_mean, "beta1_min": b_min, "beta1_max": b_max,
                    "sigma2_mean": s_mean, "sigma2_min": s_min, "sigma2_max": s_max,
                    "P_mean": p_mean, "P_min": p_min, "P_max": p_max,
                })

            writer.writerow(row)
            f.flush()

            print(f"[{run_name}] epoch {epoch:3d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"{epoch_time:.1f}s/epoch")

    print(f"[E1] DONE — {run_name}. Log saved to {log_path}")
    if reached_90:
        print(f"[E1] Reached 90% val acc at epoch {reached_90}")
    else:
        print("[E1] Did not reach 90% val acc within training budget")


if __name__ == "__main__":
    main()

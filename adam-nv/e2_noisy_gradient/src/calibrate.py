"""
E2 · Calibration Run
=====================
Runs ~100 mini-batches on CIFAR-10/ResNet-18 and measures what
σ=0.05 and σ=0.5 actually do to the gradients before we commit
to 12 GPU-hours of full experiments.

What we're checking
-------------------
By construction, SNR = 1/σ (constant). What we actually want to
verify is:

1. The absolute noise magnitude σ·‖g‖_F is non-trivial — i.e.,
   at σ=0.05, the noise is not so small it's invisible to the
   optimizer, and at σ=0.5, it's not so large that it completely
   dominates the gradient from step 1.

2. The real gradient norms ‖g‖_F are in a sensible range for
   ResNet-18 on CIFAR-10 (not near-zero from the start, not
   exploding). This confirms our model and data pipeline are sane.

3. The noise injection function produces the expected perturbation
   magnitude at each σ value, confirming the implementation matches
   the formula g + σ·‖g‖_F·ε, ε~N(0,I).

Outputs
-------
- Console table: per-sigma stats (mean grad norm, mean noise norm,
  noise/grad ratio, effective SNR)
- logs/calibration/calibration_report.csv: same stats, saved
- logs/calibration/calibration_figure.png: distribution of per-layer
  noise magnitudes at each sigma

Usage
-----
    # On Colab/Azure (real CIFAR-10):
    python calibrate.py --data-dir ../data

    # In sandbox (synthetic data, for testing this script):
    python calibrate.py --synthetic-data
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise import inject_noise, compute_snr

# ── Model (same CIFAR-adapted ResNet-18 as E1) ────────────────────────

def build_model():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_loader(data_dir, synthetic=False, batch_size=128):
    if synthetic:
        x = torch.randn(1280, 3, 32, 32)
        y = torch.randint(0, 10, (1280,))
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616)),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                       download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                        shuffle=True, num_workers=2,
                                        pin_memory=torch.cuda.is_available())


def run_calibration(model, loader, sigma, n_batches, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    records = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()

        # Compute RMS grad norm BEFORE injection — same formula for all sigma
        # RMS norm = Frobenius norm / sqrt(n_elements), averaged across tensors
        grad_rms_list = []
        for p in model.parameters():
            if p.grad is not None:
                grad_rms_list.append(
                    p.grad.norm().item() / (p.numel() ** 0.5)
                )
        mean_grad_rms = sum(grad_rms_list) / len(grad_rms_list) if grad_rms_list else 0.0

        if sigma > 0.0:
            # Save clean gradients, inject noise, measure perturbation
            pre_noise = {p: p.grad.clone()
                         for p in model.parameters() if p.grad is not None}
            inject_noise(model, sigma=sigma)

            noise_rms_list = []
            for p in model.parameters():
                if p.grad is not None:
                    n = p.numel()
                    noise_rms_list.append(
                        (p.grad - pre_noise[p]).norm().item() / (n ** 0.5)
                    )
            mean_noise_rms = sum(noise_rms_list) / len(noise_rms_list) if noise_rms_list else 0.0
        else:
            mean_noise_rms = 0.0

        records.append({
            'batch': i,
            'mean_grad_norm': mean_grad_rms,
            'mean_noise_norm': mean_noise_rms,
            'noise_to_grad_ratio': (mean_noise_rms / mean_grad_rms
                                    if mean_grad_rms > 0 else 0.0),
            'loss': loss.item(),
        })

        optimizer.step()

    return records


def make_figure(results_by_sigma, out_path):
    """
    Plot distribution of per-batch noise/grad ratios for each sigma.
    Expected: ratios cluster tightly around sigma (by construction).
    Any wide spread indicates numerical issues.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = {0.05: '#1f77b4', 0.5: '#d62728'}

    # Left: noise/grad ratio over batches
    ax = axes[0]
    for sigma, records in results_by_sigma.items():
        if sigma == 0.0:
            continue
        batches = [r['batch'] for r in records]
        ratios = [r['noise_to_grad_ratio'] for r in records]
        ax.plot(batches, ratios, color=colors.get(sigma, 'grey'),
                label=f'σ={sigma}', alpha=0.8, linewidth=1.5)
        ax.axhline(sigma, color=colors.get(sigma, 'grey'),
                   linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel('Mini-batch')
    ax.set_ylabel('‖noise‖_F / ‖grad‖_F')
    ax.set_title('Noise/Grad Ratio per Batch\n(dashed = expected σ)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # Right: gradient norm over batches (sanity check on model health)
    ax2 = axes[1]
    for sigma, records in results_by_sigma.items():
        batches = [r['batch'] for r in records]
        gnorms = [r['mean_grad_norm'] for r in records]
        ax2.plot(batches, gnorms, color=colors.get(sigma, 'green'),
                 label=f'σ={sigma}', alpha=0.8, linewidth=1.5)
    ax2.set_xlabel('Mini-batch')
    ax2.set_ylabel('Mean ‖g‖_F per layer')
    ax2.set_title('Gradient Norm over Batches\n(sanity check)')
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[calibrate] Figure saved to {out_path}")


def print_summary(results_by_sigma):
    """Print a clean summary table to console."""
    print("\n" + "="*65)
    print(f"{'σ':>6} | {'Mean ‖g‖_F':>12} | {'Mean ‖noise‖_F':>14} | "
          f"{'Noise/Grad':>10} | {'SNR':>6}")
    print("-"*65)
    for sigma, records in sorted(results_by_sigma.items()):
        mean_gn = sum(r['mean_grad_norm'] for r in records) / len(records)
        mean_nn = sum(r['mean_noise_norm'] for r in records) / len(records)
        mean_ratio = sum(r['noise_to_grad_ratio'] for r in records) / len(records)
        snr = (1.0 / sigma) if sigma > 0 else float('inf')
        print(f"{sigma:>6.2f} | {mean_gn:>12.4f} | {mean_nn:>14.4f} | "
              f"{mean_ratio:>10.4f} | {snr:>6.1f}")
    print("="*65)
    print("Expected: Noise/Grad ratio ≈ σ (by construction)")
    print("Red flag: Noise/Grad >> σ means gradient norms are very small")
    print("Red flag: Mean ‖g‖_F < 1e-4 means model is stuck or exploding\n")


def main():
    parser = argparse.ArgumentParser(description="E2 Noise Calibration")
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Number of mini-batches to run per sigma")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigmas", nargs="+", type=float,
                        default=[0.0, 0.05, 0.5],
                        help="Sigma values to calibrate")
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--log-dir", type=str, default="../logs/calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-data", action="store_true",
                        help="Use random tensors instead of real CIFAR-10 "
                             "(for testing this script without GPU/data)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] device={device} n_batches={args.n_batches} "
          f"sigmas={args.sigmas} synthetic={args.synthetic_data}")

    os.makedirs(args.log_dir, exist_ok=True)

    loader = get_loader(args.data_dir, synthetic=args.synthetic_data,
                        batch_size=args.batch_size)

    results_by_sigma = {}
    for sigma in args.sigmas:
        print(f"[calibrate] Running σ={sigma} for {args.n_batches} batches...")
        torch.manual_seed(args.seed)
        model = build_model().to(device)
        t0 = time.time()
        records = run_calibration(model, loader, sigma,
                                  args.n_batches, device)
        elapsed = time.time() - t0
        results_by_sigma[sigma] = records
        print(f"[calibrate] σ={sigma} done in {elapsed:.1f}s")

    print_summary(results_by_sigma)

    # Save CSV report
    csv_path = os.path.join(args.log_dir, "calibration_report.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["sigma", "batch", "mean_grad_norm",
                      "mean_noise_norm", "noise_to_grad_ratio", "loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sigma, records in results_by_sigma.items():
            for r in records:
                writer.writerow({"sigma": sigma, **r})
    print(f"[calibrate] Report saved to {csv_path}")

    # Save figure
    fig_path = os.path.join(args.log_dir, "calibration_figure.png")
    make_figure(results_by_sigma, fig_path)

    # Decision guidance
    print("\n[calibrate] CALIBRATION DECISION GUIDE:")
    print("  If Noise/Grad ratio at σ=0.05 is in [0.03, 0.08]  → σ=0.05 is good")
    print("  If Noise/Grad ratio at σ=0.5  is in [0.3,  0.7]   → σ=0.5  is good")
    print("  If either ratio is >> expected: gradient norms too small → reduce sigma")
    print("  If ratio at σ=0.5 >> 1.0: noise dominates completely → reduce sigma")
    print("  Confirm before starting Part A or Part B experiments.")


if __name__ == "__main__":
    main()

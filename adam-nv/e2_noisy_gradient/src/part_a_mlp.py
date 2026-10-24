"""
E2 · Part A — MLP Synthetic Mechanism Proof
=============================================
Trains a small MLP on a synthetic regression task under three noise
levels (σ=0.0, 0.05, 0.5) and logs β₁ and σ² trajectories per step.

Purpose
-------
Isolate the adaptive β₁ mechanism from real-data confounders (batch
norm, data augmentation, class imbalance). On a clean synthetic task,
any β₁ movement is caused entirely by the injected gradient noise —
not by task difficulty or data statistics.

This is the mechanism proof. Part B (ResNet-18/CIFAR-10) is the
confirmation at real scale.

Scientific claim validated here (H2a)
--------------------------------------
Adam-NV's mean β₁ decreases monotonically as σ increases, and the
per-step β₁ trajectory is correlated with the σ² (noise proxy) signal,
confirming the SNR → β₁ pathway is active and causal.

Dataset
-------
Synthetic regression: 5000 samples, input dim=20, targets generated
as a linear function of inputs plus small output noise. Intentionally
easy so that injected gradient noise (not task difficulty) is the
dominant source of gradient variance.

Outputs
-------
logs/part_a/
    part_a_sigma{σ}.csv     per-step log (loss, beta1, sigma2, rho)
figures/
    e2_part_a.png           3-panel mechanism figure

Usage
-----
    python part_a_mlp.py                         # all three sigma values
    python part_a_mlp.py --sigmas 0.0 0.5        # subset
    python part_a_mlp.py --steps 500 --log-every 5  # quick test
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adam_nv import AdamNV
from noise import inject_noise

# ── Dataset ───────────────────────────────────────────────────────────

def make_synthetic_dataset(n=5000, input_dim=20, seed=0):
    """
    Generate a simple linear regression dataset.

        y = X @ w_true + 0.01 * noise,   w_true ~ N(0, 1)

    The task is intentionally easy so that gradient noise from sigma
    injection — not task difficulty — drives all β₁ movement.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, input_dim, generator=rng)
    w_true = torch.randn(input_dim, 1, generator=rng)
    y = X @ w_true + 0.01 * torch.randn(n, 1, generator=rng)
    return X, y


def get_loader(X, y, batch_size=64):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


# ── Model ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    3-layer MLP for regression.
    Small enough to train fast on CPU, large enough that per-layer
    β₁ variation is meaningful (not just 1-2 tensors).

    Architecture: Linear(20→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1)
    """
    def __init__(self, input_dim=20, hidden=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── Training loop ─────────────────────────────────────────────────────

def train(sigma, n_steps, batch_size, lr, log_every, log_dir, seed, device):
    """
    Train Adam-NV on the synthetic regression task with gradient noise σ.

    Logs per-step:
        - loss
        - β₁ (mean/min/max)
        - σ² (mean)
        - rho (mean)

    Returns:
        records: list of dictionaries for plotting/analysis.
    """

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------

    # Keeps model initialization identical across sigma runs.
    torch.manual_seed(seed)

    # Dataset
    X, y = make_synthetic_dataset(seed=seed)
    X, y = X.to(device), y.to(device)

    # ------------------------------------------------------------
    # Dedicated RNG for DataLoader
    # Ensures identical minibatch ordering for every sigma.
    # ------------------------------------------------------------
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)

    ds = torch.utils.data.TensorDataset(X, y)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_gen,
    )

    # ------------------------------------------------------------
    # Dedicated RNG for gradient noise
    # Keeps noise generation independent from batch shuffling.
    # ------------------------------------------------------------
    if device.type == "cuda":
        noise_gen = torch.Generator(device=device)
    else:
        noise_gen = torch.Generator()

    noise_gen.manual_seed(seed + 1)

    # ------------------------------------------------------------
    # Model / Optimizer
    # ------------------------------------------------------------
    model = MLP().to(device)
    optimizer = AdamNV(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"part_a_sigma{sigma}.csv")

    fieldnames = [
        "step",
        "loss",
        "beta1_mean",
        "beta1_min",
        "beta1_max",
        "sigma2_mean",
        "rho_mean",
    ]

    records = []
    step = 0

    print(f"[Part A] σ={sigma} | n_steps={n_steps} | device={device}")

    t0 = time.time()

    with open(log_path, "w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while step < n_steps:

            for x_batch, y_batch in loader:

                if step >= n_steps:
                    break

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                loss.backward()

                # --------------------------------------------------
                # Gradient norm BEFORE noise
                # --------------------------------------------------
                g_before = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )

                # --------------------------------------------------
                # Inject Gaussian gradient noise
                # --------------------------------------------------
                inject_noise(
                    model,
                    sigma=sigma,
                    generator=noise_gen,
                )

                # --------------------------------------------------
                # Gradient norm AFTER noise
                # --------------------------------------------------
                g_after = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )

                optimizer.step()
                step += 1

                # --------------------------------------------------
                # Debug output
                # --------------------------------------------------
                if step == 1 or step % log_every == 0:
                    print(
                        f"step={step:4d} "
                        f"sigma={sigma:.2f} "
                        f"g_before={g_before:.6f} "
                        f"g_after={g_after:.6f} "
                        f"delta={g_after-g_before:.6f}"
                    )

                # --------------------------------------------------
                # Log optimizer statistics
                # --------------------------------------------------
                if step == 1 or step % log_every == 0:

                    b_mean, b_min, b_max = optimizer.get_beta1_stats()
                    s_mean, _, _ = optimizer.get_sigma2_stats()

                    rho_vals = []

                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            st = optimizer.state[p]

                            if st:
                                rho_vals.append(st["rho"].mean().item())

                    rho_mean = (
                        sum(rho_vals) / len(rho_vals)
                        if len(rho_vals) > 0
                        else 0.0
                    )

                    row = {
                        "step": step,
                        "loss": loss.item(),
                        "beta1_mean": b_mean,
                        "beta1_min": b_min,
                        "beta1_max": b_max,
                        "sigma2_mean": s_mean,
                        "rho_mean": rho_mean,
                    }

                    records.append(row)

                    writer.writerow(row)
                    f.flush()

    elapsed = time.time() - t0

    final_b1 = records[-1]["beta1_mean"] if records else None

    print(
        f"[Part A] σ={sigma} done | "
        f"{elapsed:.1f}s | "
        f"final_loss={records[-1]['loss']:.4f} | "
        f"final_beta1_mean={final_b1:.4f}"
    )

    return records

def make_figure(all_records, sigmas, out_path, log_every):
    """
    3-panel figure:
      Panel 1: Loss curves for each σ (smoothed)
      Panel 2: β₁ mean trajectory for each σ — the mechanism
      Panel 3: σ² (noise proxy) trajectory — the input signal to β₁

    The key visual: Panel 2 and Panel 3 should move together under
    noisy conditions — β₁ drops when σ² rises. This is H2a.
    """
    COLORS = {0.0: '#7f7f7f', 0.05: '#1f77b4', 0.5: '#d62728'}
    STYLES = {0.0: ':', 0.05: '--', 0.5: '-'}
    LABELS = {0.0: 'σ=0.0 (clean)', 0.05: 'σ=0.05 (mild)', 0.5: 'σ=0.5 (strong)'}

    def smooth(vals, window=10):
        """Simple moving average for readability."""
        if len(vals) < window:
            return vals
        kernel = np.ones(window) / window
        return np.convolve(vals, kernel, mode='valid')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for sigma in sigmas:
        recs = all_records[sigma]
        steps = [r["step"] for r in recs]
        losses = [r["loss"] for r in recs]
        beta1s = [r["beta1_mean"] for r in recs]
        sigma2s = [r["sigma2_mean"] for r in recs]

        color = COLORS.get(sigma, 'green')
        style = STYLES.get(sigma, '-')
        label = LABELS.get(sigma, f'σ={sigma}')

        # Smooth loss for readability (raw loss is noisy for regression)
        s_loss = smooth(losses, window=15)
        s_steps_loss = steps[len(steps) - len(s_loss):]

        axes[0].plot(s_steps_loss, s_loss, color=color, linestyle=style,
                     label=label, linewidth=2)

        axes[1].plot(steps, beta1s, color=color, linestyle=style,
                     label=label, linewidth=2)

        axes[2].plot(steps, sigma2s, color=color, linestyle=style,
                     label=label, linewidth=2)

    # Panel 0 — Loss
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss\n(smoothed)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25)

    # Panel 1 — β₁ trajectory (the mechanism)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Mean β₁")
    axes[1].set_title("β₁ Trajectory\n(H2a: drops under noise)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].axhline(0.9, color='black', linestyle=':', alpha=0.3,
                    linewidth=1, label='β_base=0.9')
    axes[1].axhline(0.1, color='black', linestyle=':', alpha=0.3,
                    linewidth=1, label='β_min=0.1')
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25)

    # Panel 2 — σ² (noise proxy signal that drives β₁)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Mean σ² (noise proxy)")
    axes[2].set_title("σ² Signal\n(input to β₁ adaptation)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(alpha=0.25)

    fig.suptitle("E2 Part A — MLP Synthetic: β₁ Adaptation Mechanism",
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Part A] Figure saved to {out_path}")


# ── Summary table ─────────────────────────────────────────────────────

def print_summary(all_records, sigmas):
    """
    Print final stats per sigma — the key numbers for the design doc
    comparison: final β₁ mean and final loss under each noise level.
    """
    print("\n" + "="*65)
    print(f"{'σ':>6} | {'Final Loss':>10} | {'Final β₁ mean':>14} | "
          f"{'Final σ² mean':>14}")
    print("-"*65)
    for sigma in sigmas:
        recs = all_records[sigma]
        last = recs[-1]
        print(f"{sigma:>6.2f} | {last['loss']:>10.4f} | "
              f"{last['beta1_mean']:>14.4f} | "
              f"{last['sigma2_mean']:>14.4f}")
    print("="*65)
    print("\nH2a check: β₁ mean should decrease as σ increases.")
    print("If β₁ stays at ~0.9 for all σ: warm-up k may be too large,")
    print("or n_steps too small for the noise signal to build up.\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E2 Part A — MLP synthetic mechanism proof")
    parser.add_argument("--sigmas", nargs="+", type=float,
                        default=[0.0, 0.05, 0.5])
    parser.add_argument("--n-steps", type=int, default=2000,
                        help="Total optimizer steps per sigma run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log metrics every N steps")
    parser.add_argument("--log-dir", type=str, default="../logs/part_a")
    parser.add_argument("--fig-dir", type=str, default="../figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Part A is CPU-friendly — MLP is tiny
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Part A] device={device} | sigmas={args.sigmas} | "
          f"n_steps={args.n_steps}")

    os.makedirs(args.fig_dir, exist_ok=True)

    all_records = {}
    for sigma in args.sigmas:
        records = train(
            sigma=sigma,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            log_every=args.log_every,
            log_dir=args.log_dir,
            seed=args.seed,
            device=device,
        )
        all_records[sigma] = records

    print_summary(all_records, args.sigmas)

    fig_path = os.path.join(args.fig_dir, "e2_part_a.png")
    make_figure(all_records, args.sigmas, fig_path, args.log_every)

    print("\n[Part A] DONE. Check the figure for H2a:")
    print("  ✓ β₁ should be lower for σ=0.5 than σ=0.0")
    print("  ✓ σ² should be higher for σ=0.5 than σ=0.0")
    print("  ✓ β₁ and σ² should move in opposite directions")


if __name__ == "__main__":
    main()
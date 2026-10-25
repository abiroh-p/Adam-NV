# Experiment 10: 10-seed extension at lr = 3e-3 (clean CIFAR-10, ResNet-18)

**Question:** Exp. 5 found Adam-NV above Adam at lr 3e-3 (+0.20 pp, 4 seeds), and Exp. 6 found that plain Adam with a constant β₁ of 0.83 or 0.80 reproduces the gain. With four seeds the controls' own gains over Adam were not individually significant and the Adam-NV-minus-control intervals were about ±0.26 pp wide. With ten seeds the intervals should shrink to about ±0.15 pp. Does the gain survive, and is it still explained by lower β₁?

## Design (24 new runs, seeds 4–9; seeds 0–3 are reused)

| Arm | Optimizer name | Seeds 0–3 | Seeds 4–9 |
|---|---|---|---|
| Adam | `adam` | Exp. 5 logs | **NEW** |
| Adam-NV | `adam_nv` | Exp. 5 logs | **NEW** |
| Adam, constant β₁ = 0.83 | `adam_b083` | Exp. 6 logs | **NEW** |
| Adam, constant β₁ = 0.80 | `adam_b080` | Exp. 6 logs | **NEW** |

All runs: lr 3e-3, 60 epochs, batch 128, `--deterministic`, tag `_lr3e-3`, `train_resnet.py` as used in Exp. 6 and in the AdaBelief extension (hash 2ab3aa02; it contains the two control optimizers; the other optimizer code paths are unchanged from Exp. 5).

## Pre-registered rule (before any Exp. 10 run; primary metric = last-5-epoch mean validation accuracy, paired by seed, 10 seeds)

M = Adam-NV − Adam. Holm over {NV−Adam, b083−Adam, b080−Adam, NV−b083, NV−b080}.
1. If the 95% CI of M is not above 0, the lr 3e-3 gain is **not confirmed** on 10 seeds.
2. Gain confirmed, both controls have a mean gain over Adam of at least 0.5·M, and for both controls the CI of (Adam-NV − control) includes 0 with half-width ≤ 0.15 pp: **explained by lower β₁**.
3. Gain confirmed and the CI of (Adam-NV − control) is above 0 for both controls: **adaptation adds a gain**.
4. Anything else: **inconclusive**.
5. **Confirmatory check (added after an external review of the first paper draft):** seeds 0–3 generated the hypothesis, so pooled p-values are optimistic. The analysis also reports seeds 4–9 alone; the headline claim must hold in both the pooled (0–9) and the confirmatory (4–9) analysis, and the paper reports both.
Secondary metrics: final-epoch accuracy, epochs to 90%, validation-loss AUC.

## Verified, and not verified
Verified (CPU only; no PyTorch in my environment): shell syntax; the analysis script on your real Exp. 5/6 logs (it reproduces M = +0.197 pp, Adam-NV 93.13, controls 93.11 / 93.14) and on a synthetic 10-seed set (the 10-seed code path and the decision rule run).
NOT verified: any GPU run, and the outcome.

## Before launching: expected code hashes
Each run's metadata records SHA-256 hashes. Expect (first 8 characters) `train_resnet.py` 2ab3aa02, `adam_nv.py` bcc7824b, `gono.py` 326bcedc, as in the AdaBelief extension. If different, stop and send the output.

## Account plan (about 50 min per run; 3 per GPU, about 2.5 h per optimizer)

| When | Account | Call |
|---|---|---|
| now | C | `OPTS="adam adam_nv"` (about 5 h) |
| after Exp. 9 finishes | A | `OPTS="adam_b083"` (about 2.5 h) |
| after Exp. 9 finishes | B | `OPTS="adam_b080"` (about 2.5 h) |

## Steps (same notebook pattern as Exp. 5/6)

```bash
!unzip -o exp10_seed_extension.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e1_convergence/src
!python3 env_check.py                                        # must print PASS
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -5    # must end with OK
!python3 train_resnet.py --optimizer adam --seed 0 --epochs 1 --smoke-test --lr 3e-3 --run-tag _lr3e-3 --data-dir /kaggle/working/Adam-NV/data --log-dir ../smoke_gpu

# ACCOUNT C  (Save Version -> Save & Run All)
!OPTS="adam adam_nv" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp10_C bash run_exp10_seed_extension.sh
!cd /kaggle/working/Adam-NV && zip -r exp10_logs_C.zip logs_exp10_C

# ACCOUNT A (later)
!OPTS="adam_b083" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp10_A bash run_exp10_seed_extension.sh
!cd /kaggle/working/Adam-NV && zip -r exp10_logs_A.zip logs_exp10_A

# ACCOUNT B (later)
!OPTS="adam_b080" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp10_B bash run_exp10_seed_extension.sh
!cd /kaggle/working/Adam-NV && zip -r exp10_logs_B.zip logs_exp10_B
```
For accounts A and B use `--optimizer adam_b083` / `adam_b080` in the smoke test. If a session dies, run the same cell again; complete runs are skipped. If the 5-hour session is too long, run `OPTS="adam"` and `OPTS="adam_nv"` in two sessions with different log folders. The launcher runs `env_check.py` and `check_logs.py` itself; send their printouts.

## Analysis (put the Exp. 5 and Exp. 6 logs next to the new ones)

```bash
python3 analysis/analyze_exp10.py --log-dirs logs_exp05_adam logs_exp05_adamnv logs_exp06 \
    logs_exp10_C logs_exp10_A logs_exp10_B
```

## Send back
The `exp10_logs_*.zip` files, the unit-test and `env_check` outputs, the `check_logs.py` tables, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix optimizers or learning rates in a log folder, or edit any file.

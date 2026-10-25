# Experiment 9: direction of the β₁ adaptation under 20% label noise (CIFAR-10, ResNet-18)

**Question:** Exp. 7 and 8 showed that Adam-NV memorizes about 3.9 pp more corrupted labels than Adam, and that no constant β₁ between 0.9 and 0.7 reproduces it. Where in the controller does the effect come from? In `distrust` mode (the default used so far) β₁ can only fall below 0.9 where the noise estimate is high. In `smooth` mode β₁ moves in both directions, rising toward `beta_max` = 0.999 where the noise estimate is low. If memorization depends on the controller lowering β₁ where noise is high, `smooth` should reduce it.

## Design (8 new runs)

| Arm | Setting | Seeds | Source |
|---|---|---|---|
| Adam | β₁ = 0.9 | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV, distrust | default | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| **Adam-NV, smooth** | `--response-mode smooth --beta-max 0.999` | 0–7 | **NEW** |

All runs: 60 epochs, batch 128, lr 1e-3 (default), constant schedule, σ = 0, label noise 0.2, `--deterministic`, `part_b_resnet_v8.py` unchanged.

## Pre-registered rule (written before any Exp. 9 run; primary metric = last-5-epoch mean of `train_corrupt_noisy_label_acc`, paired by seed)

M = Adam-NV(distrust) − Adam (Exp. 7: +3.91 pp), F = Adam-NV(smooth) − Adam, D = Adam-NV(distrust) − Adam-NV(smooth). Holm over {F, D}.
1. If the 95% CI of M is not above 0 in the pooled set, the rule does not apply.
2. F < 0.25·M and the CI of D is above 0: **the distrust direction drives the effect.**
3. F ≥ 0.5·M and the CI of D includes 0: **the direction does not matter**; the cause lies in the noise signal (σ²) or in how β₁ varies across elements.
4. Anything else: **partial or inconclusive**; extend seeds.
Secondary metrics (validation accuracy, clean-subset accuracy, peak validation accuracy) and the β₁ statistics of both modes are printed by the analysis script and reported, not decisive.

## Limits of the interpretation (write these in the paper)
`smooth` also changes the average β₁ (it rises above 0.9, not just fails to fall), so it is not a pure "flip the sign" intervention. Exp. 8 shows that average β₁ between 0.9 and 0.7 does not move memorization, but β₁ above 0.9 is a different regime. A positive result (rule 2) says the lowering of β₁ under high noise estimates is *necessary within this controller*; it does not identify the quantity (σ² itself, element-wise heterogeneity) responsible.

## Verified, and not verified
Verified (CPU only; no PyTorch in my environment): shell syntax and a dry run of the launcher (it builds the exact `part_b_resnet_v8.py` command with `--response-mode smooth --beta-max 0.999`); the checker and analysis script on stand-in logs derived from Exp. 7 (this also found and fixed a crash on plain-Adam logs).
NOT verified: any GPU run, and the outcome itself.

## Before launching
Each launch writes `_code_hashes_*.txt` into its log folder. They must equal Exp. 7 and 8 (first 8 characters): 242e6a77, bcc7824b, 90e11e2c, eee61201, ab020aaf. If any differs, stop and send the output.

## Account plan (about 65 min per run; about 2.2 h per account)
| Account | Seeds |
|---|---|
| A | 0, 1 on GPU 0 and 2, 3 on GPU 1 |
| B | 4, 5 on GPU 0 and 6, 7 on GPU 1 |
| C | not needed |

## Steps (same setup as Exp. 7/8: your `src` folder already contains the old launcher; add the two new files)

```bash
!unzip -o exp09_smooth_response.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# ACCOUNT A
!SEEDS_GPU0="0 1" SEEDS_GPU1="2 3" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp09_A bash run_exp09_label_noise_smooth.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp09_A/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp09_logs_A.zip logs_exp09_A

# ACCOUNT B
!SEEDS_GPU0="4 5" SEEDS_GPU1="6 7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp09_B bash run_exp09_label_noise_smooth.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp09_B/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp09_logs_B.zip logs_exp09_B
```
If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each launch.

## Analysis (`analyze_exp07.py` and `analyze_exp09.py` in the same folder)

```bash
python3 analysis/analyze_exp09.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C --smooth-dirs logs_exp09_A logs_exp09_B
```

## Send back
Both `exp09_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument or edit any file.

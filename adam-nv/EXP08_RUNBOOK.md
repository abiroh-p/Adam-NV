# Experiment 8: constant-β₁ dose-response under 20% label noise (CIFAR-10, ResNet-18)

**Question:** in Exp. 7, Adam-NV memorized +3.91 pp more corrupted labels than Adam (8/8 seeds), while Adam with a constant β₁ = 0.81 did not (+0.64 pp, n.s.). But Adam-NV's β₁ is not constant: it drifts from 0.87 (epoch 1) to about 0.78 (epochs 50–60), with per-element minima near 0.62. Is the effect just late-training low momentum that a constant 0.81 missed, or does the per-element adaptation itself cause it?

## Design (16 new runs)

| Arm | Optimizer name | β₁ | Seeds | Source |
|---|---|---|---|---|
| Adam | `adam` | 0.90 | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV | `adam_nv` | adaptive | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Constant 0.81 | `adam_nv_fixed` | 0.81 | 0–7 | Exp. 7 |
| **Constant 0.78** | `adam_nv_fixed` | 0.78 | 0–7 | **NEW** (Adam-NV's late-training mean) |
| **Constant 0.70** | `adam_nv_fixed` | 0.70 | 0–7 | **NEW** (stronger than the observed mean, close to per-element lows) |

All runs: 60 epochs, batch 128, lr 1e-3 (default), constant schedule, σ = 0, label noise 0.2, `--deterministic`, `part_b_resnet_v8.py` unchanged. Nothing else is modified; this experiment reuses the Exp. 7 launcher through `FIXED_BETA1`.

## Pre-registered rule (before any Exp. 8 run; primary metric = last-5-epoch mean of `train_corrupt_noisy_label_acc`, paired by seed)

M = Adam-NV − Adam (Exp. 7: +3.91 pp), F_c = constant-c − Adam, D_c = Adam-NV − constant-c, for c ∈ {0.78, 0.70}. Holm over {F₀.₇₈, F₀.₇₀, D₀.₇₈, D₀.₇₀}.
1. For some c, F_c ≥ 0.5·M and the 95% CI of D_c includes 0: **a lower constant β₁ explains the effect.** Exp. 7's conclusion must then be reworded to "explained by late-training low momentum".
2. For both c, F_c < 0.25·M and the CI of D_c is above 0: **adaptation beyond any constant β₁ tested.**
3. Anything else: **partial or inconclusive**; report the dose-response and extend seeds.
The pooled dose-response slope over β₁ is descriptive only. Secondary metrics (validation accuracy, clean-subset accuracy) are reported by Exp. 7's script, not here.

## Verified, and not verified
Verified (CPU only, no PyTorch in my environment): the analysis script on stand-in data built from the Exp. 7 logs (it runs, handles missing arms, and applies the rule), and that `run_exp07_label_noise.sh` and `check_exp07_logs.py` accept any `FIXED_BETA1` (the checker compares the logged `fixed_beta1`).
NOT verified: any GPU run, and the dose-response outcome itself.

## Before launching: confirm the same code as Exp. 7
Each launch writes `_code_hashes_*.txt` into its log folder. They must equal Exp. 7's (SHA-256, first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf. If any differs, stop and send the output.

## Account plan (about 65 min per run; at most 3 rounds per account, about 3.3 h)

| Account | Invocation 1 | Invocation 2 |
|---|---|---|
| A | β₁ 0.78, seeds 0–3, `logs_exp08_b078_A` | β₁ 0.70, seeds 0–1, `logs_exp08_b070_A` |
| B | β₁ 0.78, seeds 4–7, `logs_exp08_b078_B` | β₁ 0.70, seeds 2–3, `logs_exp08_b070_B` |
| C | β₁ 0.70, seeds 4–7, `logs_exp08_b070_C` | none |

File names do not contain β₁, so every β₁ value uses its own log folder. Never reuse a folder for a different β₁.

## Steps (replace `<INPUT_ROOT>` and `<DATA_DIR>` as in Exp. 7)

```bash
!mkdir -p /kaggle/working/Adam-NV/e2_noisy_gradient
!cp -r <INPUT_ROOT>/Adam-NV/e2_noisy_gradient/src /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cp -r <INPUT_ROOT>/Adam-NV/tests /kaggle/working/Adam-NV/tests
!unzip -o exp07_label_noise_control.zip -d /kaggle/working/Adam-NV
!unzip -o exp08_dose_response.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# ACCOUNT A
!FIXED_BETA1=0.78 JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1" JOBS_GPU1="adam_nv_fixed:2 adam_nv_fixed:3" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp08_b078_A bash run_exp07_label_noise.sh
!FIXED_BETA1=0.70 JOBS_GPU0="adam_nv_fixed:0" JOBS_GPU1="adam_nv_fixed:1" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp08_b070_A bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp08_logs_A.zip logs_exp08_b078_A logs_exp08_b070_A

# ACCOUNT B
!FIXED_BETA1=0.78 JOBS_GPU0="adam_nv_fixed:4 adam_nv_fixed:5" JOBS_GPU1="adam_nv_fixed:6 adam_nv_fixed:7" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp08_b078_B bash run_exp07_label_noise.sh
!FIXED_BETA1=0.70 JOBS_GPU0="adam_nv_fixed:2" JOBS_GPU1="adam_nv_fixed:3" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp08_b070_B bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp08_logs_B.zip logs_exp08_b078_B logs_exp08_b070_B

# ACCOUNT C
!FIXED_BETA1=0.70 JOBS_GPU0="adam_nv_fixed:4 adam_nv_fixed:5" JOBS_GPU1="adam_nv_fixed:6 adam_nv_fixed:7" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp08_b070_C bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp08_logs_C.zip logs_exp08_b070_C
```
If a session dies, run the same cell again; complete runs are skipped. The checker prints at the end of each launch; its `--fixed-beta1` follows `FIXED_BETA1` automatically.

## Analysis (put `analyze_exp07.py` and `analyze_exp08.py` in the same folder)

```bash
python3 analysis/analyze_exp08.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
    --c078-dirs logs_exp08_b078_A logs_exp08_b078_B \
    --c070-dirs logs_exp08_b070_A logs_exp08_b070_B logs_exp08_b070_C
```

## Send back
The three `exp08_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix β₁ values in one log folder, or edit any file.

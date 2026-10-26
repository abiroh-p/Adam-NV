# Experiment 12: constant β₁ above 0.9 under 20% label noise (CIFAR-10, ResNet-18)

**Question:** Exp. 9 found that the smooth-response Adam-NV (logged mean β₁ 0.92, range 0.90–0.99) memorizes **1.47 pp fewer** corrupted labels than Adam, and 5.38 pp fewer than the distrust response. But it also raises the *average* β₁ above 0.9, and our constant controls only reached down to 0.70. Would plain Adam with a constant β₁ above 0.9 give the same reduction? If so, the direction result is explained by a higher average β₁; if not, the smooth controller does something a constant of the same size does not.

## Design (16 new runs)

| Arm | Optimizer name | β₁ | Seeds | Source |
|---|---|---|---|---|
| Adam | `adam` | 0.90 | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV, smooth | `adam_nv` with `--response-mode smooth` | adaptive, mean 0.92 | 0–7 | Exp. 9 |
| **Constant 0.92** | `adam_nv_fixed` | 0.92 | 0–7 | **NEW** (matches the smooth response's mean) |
| **Constant 0.95** | `adam_nv_fixed` | 0.95 | 0–7 | **NEW** (a stronger constant) |

All runs: 20% label noise, constant schedule, σ = 0, lr 1e-3 (default), 60 epochs, batch 128, `--deterministic`, `part_b_resnet_v8.py` unchanged. No new code: the Exp. 7 launcher takes any `FIXED_BETA1`.

## Pre-registered rule (before any Exp. 12 run; primary metric = last-5-epoch memorization, paired by seed)
S = smooth − Adam (Exp. 9: −1.47 pp). For each constant c ∈ {0.92, 0.95}: F_c = c − Adam, D_c = smooth − c. Holm over {F₀.₉₂, F₀.₉₅, D₀.₉₂, D₀.₉₅}.
0. The reduction S must be reproduced in the pooled set (95% CI below 0).
1. **A constant β₁ above 0.9 reproduces the reduction** (for c): F_c ≤ 0.5·S and the 95% CI of D_c includes 0.
2. **Adaptation adds a reduction** (for c): F_c > 0.25·S (less than a quarter of the reduction) and the 95% CI of D_c is below 0.
3. Otherwise **partial**.
Both constants reproduce → the direction result is explained by higher average β₁. Both "add" → the smooth response lowers memorization beyond a constant of the same size. Otherwise inconclusive. The script also prints a descriptive dose-response fit over all constants from 0.70 to 0.95.

## Verified, and not verified
Verified (CPU PyTorch 2.14.1): the fixed-β₁ optimizer with β₁ = 0.92 and 0.95 matches `torch.optim.Adam` to about 2e-9 (float64, 500 steps), and `part_b_resnet_v8.py` accepts these values (no range check). The analysis script ran on stand-in logs (existing constant-β₁ runs, **not** real Exp. 12 data, so its printed outcome in my test is meaningless), including the decision-rule branches.
NOT verified: any GPU run, and the outcome.

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf. If any differs, stop and send the output.

## Plan: four independent calls (about 65 min per run, 2 rounds per call, about 2.2 h each)
Run them on whichever accounts are free. File names do not contain β₁, so **each call needs its own log folder**.

| Call | β₁ | Seeds (GPU 0 / GPU 1) | Log folder |
|---|---|---|---|
| 1 | 0.92 | 0 1 / 2 3 | `logs_exp12_b092_1` |
| 2 | 0.92 | 4 5 / 6 7 | `logs_exp12_b092_2` |
| 3 | 0.95 | 0 1 / 2 3 | `logs_exp12_b095_1` |
| 4 | 0.95 | 4 5 / 6 7 | `logs_exp12_b095_2` |

## Steps (your `src` folder already holds the launcher; add only the analysis script and this runbook)

```bash
!unzip -o exp12_const_above_09.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# CALL 1 (change FIXED_BETA1, JOBS and LOG_DIR for the other calls)
!FIXED_BETA1=0.92 JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1" JOBS_GPU1="adam_nv_fixed:2 adam_nv_fixed:3" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp12_b092_1 bash run_exp07_label_noise.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp12_b092_1/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp12_logs_1.zip logs_exp12_b092_1
```
For call 2 use `JOBS_GPU0="adam_nv_fixed:4 adam_nv_fixed:5"`, `JOBS_GPU1="adam_nv_fixed:6 adam_nv_fixed:7"` and `logs_exp12_b092_2`. For calls 3 and 4 use `FIXED_BETA1=0.95` with the seed lists above and `logs_exp12_b095_1` / `logs_exp12_b095_2`. If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each call, and its `--fixed-beta1` follows `FIXED_BETA1` automatically.

## Analysis (`analyze_exp07.py` and `analyze_exp12.py` in the same folder)

```bash
python3 analysis/analyze_exp12.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
    --smooth-dirs logs_exp09_A logs_exp09_B \
    --c092-dirs logs_exp12_b092_1 logs_exp12_b092_2 --c095-dirs logs_exp12_b095_1 logs_exp12_b095_2 \
    --low-dirs-081 logs_exp07_A logs_exp07_B --low-dirs-078 logs_exp08_b078_A logs_exp08_b078_B \
    --low-dirs-070 logs_exp08_b070_A logs_exp08_b070_B logs_exp08_b070_C
```

## Send back
The `exp12_logs_*.zip` files, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, put two β₁ values in one log folder, or edit any file.

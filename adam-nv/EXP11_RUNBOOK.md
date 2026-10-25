# Experiment 11: β₁ shuffle control under 20% label noise (CIFAR-10, ResNet-18)

**Question:** Exp. 7 and 8 showed that Adam-NV memorizes about 3.9 pp more corrupted labels than Adam and that no constant β₁ between 0.9 and 0.7 reproduces it. Exp. 9 asks whether the direction of the adaptation matters. This experiment asks whether the effect needs the **element-wise link between β₁ and the element's own noise score**, or only the **distribution of β₁ values** that the controller produces.

## The intervention
After the controller computes its per-element β₁ tensor for a parameter, the tensor is permuted across that parameter's elements before it is used in the momentum update. The per-step distribution of β₁ is preserved; the link to each element's own gradients is not.
- `step`: a fresh permutation every step (the link is destroyed and each element's β₁ also fluctuates).
- `fixed`: one random permutation per parameter, drawn once (the β₁ trajectory of a source element is applied to a fixed, unrelated target element; temporal structure is preserved, the link to the target's own gradients is not).
- `none`: no permutation; the optimizer must reproduce the existing Adam-NV runs bit for bit (regression canary) and logs the element-level β₁ distribution.

The running product P_t uses the β₁ values actually applied, so the bias correction stays exact. `adam_nv.py`, `part_b_resnet_v8.py` and their recorded hashes are **not modified**: the new files are `adam_nv_shuffle.py` (a subclass) and `part_b_resnet_v8_shuffle.py` (a wrapper that swaps the optimizer class and then calls v8's `main()`).

## Design (18 new runs)

| Arm | Mode | Seeds | Source |
|---|---|---|---|
| Adam | β₁ = 0.9 | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Adam-NV | none | 0–7 | Label_Diagnostic (0–3) + Exp. 7 (4–7) |
| Shuffle / step | `step` | 0–7 | **NEW** |
| Shuffle / fixed | `fixed` | 0–7 | **NEW** |
| Canary | `none` | 0–1 | **NEW** (must equal the existing Adam-NV logs) |

All runs: 20% label noise, constant schedule, σ = 0, lr 1e-3 (default), 60 epochs, batch 128, `--deterministic`, `--optimizer adam_nv`.

## Pre-registered rule (before any Exp. 11 run; primary metric = last-5-epoch memorization, paired by seed)
M = Adam-NV − Adam; for each mode S, F_S = S − Adam and D_S = Adam-NV − S. Holm over {F_step, F_fixed, D_step, D_fixed, fixed − step}.
0. If the 95% CI of M is not above 0, the rule does not apply.
1. S **retains** the effect: F_S ≥ 0.5·M and the CI of D_S includes 0.
2. S **removes** the effect: F_S < 0.25·M and the CI of D_S is above 0.
3. Otherwise S is **partial**.
Interpretation: both retain → the effect does not need the element-wise link to the element's own noise score; both remove → it requires that link; `fixed` retains and `step` removes → it needs each element's β₁ to be temporally consistent; anything else → inconclusive.
The analysis also reports the element-level β₁ distribution (quantiles, extremes, fractions below 0.5 and 0.7) at every epoch.

## Verified, and not verified
Verified on a CPU PyTorch (2.14.1, float32/float64): 7 new unit tests all pass, including bit-identity of `none` with `AdamNV`, bit-identity of every mode when γ = 0, preservation of the β₁ multiset within each tensor, `fixed` reusing one permutation, reproducibility per seed, and the JSON-lines distribution log. The wrapper ran end to end on a real ResNet-18 (11,173,962 parameters) in synthetic smoke mode. The launcher was dry-run, and the checker and analysis were tested on synthetic logs with a known answer.
NOT verified: any GPU run, label-noise runs with real CIFAR-10 data, the outcome, and the time cost. Each step draws one random permutation per parameter tensor and the logging sorts 11 million values once per epoch, so runs will be slower than Adam-NV (my guess is 1.2 to 1.5 times; this is not measured). Plan for up to about 90 minutes per run.

## Before launching: hashes
Each launch writes `_code_hashes_*.txt`. Expect (first 8 characters): `part_b_resnet_v8.py` 242e6a77, `adam_nv.py` bcc7824b, `noise_schedule.py` 90e11e2c, `noise.py` eee61201, `experiment_metadata.py` ab020aaf; the two shuffle files are new. If any of the first five differs, stop and send the output.

## Account plan (about 85 min per run; 3 rounds per account, about 4.3 h)

| Account | Calls (separate log folders) |
|---|---|
| A | `MODE=step`, seeds 0 1 2 on GPU 0 and 3 4 5 on GPU 1 → `logs_exp11_step_A` |
| B | `MODE=fixed`, seeds 0 1 2 and 3 4 5 → `logs_exp11_fixed_B` |
| C | `MODE=step`, seeds 6 / 7 → `logs_exp11_step_C`; `MODE=fixed`, seeds 6 / 7 → `logs_exp11_fixed_C`; `MODE=none`, seeds 0 / 1 → `logs_exp11_none_C` |

## Steps (same setup as Exp. 7–9; your `src` folder already has the launcher pattern)

```bash
!unzip -o exp11_beta1_shuffle.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK

# ACCOUNT A
!MODE=step SEEDS_GPU0="0 1 2" SEEDS_GPU1="3 4 5" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_step_A bash run_exp11_beta1_shuffle.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_step_A/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp11_logs_A.zip logs_exp11_step_A

# ACCOUNT B
!MODE=fixed SEEDS_GPU0="0 1 2" SEEDS_GPU1="3 4 5" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_fixed_B bash run_exp11_beta1_shuffle.sh
!cat /kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_fixed_B/_code_hashes_*.txt
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp11_logs_B.zip logs_exp11_fixed_B

# ACCOUNT C
!MODE=step SEEDS_GPU0="6" SEEDS_GPU1="7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_step_C bash run_exp11_beta1_shuffle.sh
!MODE=fixed SEEDS_GPU0="6" SEEDS_GPU1="7" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_fixed_C bash run_exp11_beta1_shuffle.sh
!MODE=none SEEDS_GPU0="0" SEEDS_GPU1="1" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_none_C bash run_exp11_beta1_shuffle.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp11_logs_C.zip logs_exp11_step_C logs_exp11_fixed_C logs_exp11_none_C
```
If a session dies, run the same cell again; complete runs are skipped. The checker table prints at the end of each call.

## Analysis (`analyze_exp07.py` and `analyze_exp11.py` in the same folder)

```bash
python3 analysis/analyze_exp11.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --exp07-dirs logs_exp07_A logs_exp07_B logs_exp07_C \
    --step-dirs logs_exp11_step_A logs_exp11_step_C --fixed-dirs logs_exp11_fixed_B logs_exp11_fixed_C \
    --none-dirs logs_exp11_none_C
```

## Send back
All `exp11_logs_*.zip`, the unit-test output, the checker tables, the `_code_hashes` files, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix modes in one log folder, or edit any file.

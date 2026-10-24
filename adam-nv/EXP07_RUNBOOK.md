# Experiment 7: label noise (20%), constant-β₁ control + seed extension (CIFAR-10, ResNet-18)

**Question:** Adam-NV memorizes about 3.4 pp more corrupted labels than Adam in the existing label-diagnostic runs (seeds 0–3, all four positive, validation accuracy unchanged). Is that caused by the noise-aware β₁ controller, or only by running a lower β₁ than 0.9?

**Why 0.81:** in those runs Adam-NV's time-averaged β₁ was 0.808 (lr 1e-3, label noise 0.2). Exp. 6 used 0.83 because the clean lr 3e-3 runs averaged 0.829; the control must match the setting it is compared in.

## Design (18 new runs: 16 experimental + 2 canaries; Adam and Adam-NV seeds 0–3 are reused, not re-run)

| Arm | Optimizer name | Seeds | Source |
|---|---|---|---|
| Adam (β₁ = 0.9) | `adam` | 0–3 | existing `Label_Diagnostic` logs |
| | | 4–7 | NEW |
| Adam-NV | `adam_nv` | 0–3 | existing `Label_Diagnostic` logs |
| | | 4–7 | NEW |
| Adam, constant β₁ = 0.81 | `adam_nv_fixed` (γ = 0, β_base = β_min = 0.81) | 0–7 | NEW |
| Canary | `adam` and `adam_nv` seed 0 | 0 | NEW, re-run of existing logs |

All runs: 60 epochs, batch 128, lr 1e-3 (default), constant schedule, σ = 0, label noise 0.2, `--deterministic`, `part_b_resnet_v8.py` unchanged.

**Canary:** the older logs have no code hashes. The canary re-runs seed 0 of Adam and Adam-NV with the current code. Because runs are deterministic, the two CSVs must be IDENTICAL (except `epoch_time_sec`). If they are not, the 8-seed pooling is invalid and the seeds 0–3 must be re-run before any claim.

## Pre-registered rule (written before any new run; primary metric = last-5-epoch mean of `train_corrupt_noisy_label_acc`, paired by seed)

M = Adam-NV − Adam, F = constant-β₁ − Adam, D = Adam-NV − constant-β₁, all 8 seeds.
1. If the 95% CI of M is not above 0, the effect is not reproduced; report that and stop.
2. F ≥ 0.5·M and the CI of D includes 0: **explained by lower β₁**.
3. F ≥ 0.5·M and the CI of D is above 0: **mostly explained, small residual** (report D as the upper bound on the adaptation effect).
4. CI of D above 0 and F < 0.25·M: **adaptation beyond β₁**.
5. Anything else: **inconclusive**; extend seeds.
Holm adjustment over the three contrasts per metric. Secondary metrics (reported, not decisive): last-5 validation accuracy, last-5 clean-subset accuracy, peak validation accuracy. With 8 seeds the exact sign-flip p can go down to 0.0078.

## What is in this package (nothing existing is modified)

| File (repo-relative) | Status | Purpose |
|---|---|---|
| `e2_noisy_gradient/src/run_exp07_label_noise.sh` | NEW | per-GPU job launcher, environment check, code hashes, status files |
| `e2_noisy_gradient/src/check_exp07_logs.py` | NEW | completeness and provenance check of the new logs |
| `analysis/analyze_exp07.py` | NEW | paired analysis, decision rule, canary comparison |
| `tests/test_exp07_fixed_beta1.py` | NEW | the control configuration equals torch Adam with β₁ = 0.81 (tolerance about 1e-6) |

## What I verified, and what I could not

Verified (CPU, no PyTorch in my environment): shell syntax and a dry run of the launcher; the analysis and check scripts on synthetic logs derived from your `Label_Diagnostic` CSVs; and from those CSVs, the existing effect (Adam-NV − Adam = +3.40 pp memorization on the last-5 metric, +3.43 pp at the final epoch, all four seeds positive; validation −0.03 pp).

NOT verified: any GPU run, the PyTorch unit tests (including the new one and the older ones that RESEARCH_STATUS.md lists as never executed), and that your Kaggle stack is still Python 3.12.13 / torch 2.10.0+cu128 / torchvision 0.25.0+cu128 (the launcher aborts if not).

## Account plan (each run takes about 50 min; each account about 2.4 h)

| Account | Jobs |
|---|---|
| A | `adam_nv_fixed` seeds 0–3, `adam` seeds 4, 5 |
| B | `adam_nv_fixed` seeds 4–7, `adam` seeds 6, 7 |
| C | canaries (`adam:0`, `adam_nv:0`), then `adam_nv` seeds 4–7 |

## Steps (same notebook pattern; replace `<INPUT_ROOT>` and `<DATA_DIR>` with the paths you used for the label-diagnostic runs)

```bash
!mkdir -p /kaggle/working/Adam-NV/e2_noisy_gradient
!cp -r <INPUT_ROOT>/Adam-NV/e2_noisy_gradient/src /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cp -r <INPUT_ROOT>/Adam-NV/tests /kaggle/working/Adam-NV/tests
!unzip -o exp07_label_noise_control.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e2_noisy_gradient/src
!cd /kaggle/working/Adam-NV && python3 -m unittest discover -s tests -v 2>&1 | tail -8     # must end with OK
!python3 part_b_resnet_v8.py --optimizer adam_nv_fixed --fixed-beta1 0.81 --schedule constant --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 --epochs 1 --smoke-test --data-dir <DATA_DIR> --log-dir ../smoke_exp07

# ACCOUNT A  (Save Version -> Save & Run All)
!JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam:4" JOBS_GPU1="adam_nv_fixed:2 adam_nv_fixed:3 adam:5" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp07_A bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp07_logs_A.zip logs_exp07_A

# ACCOUNT B
!JOBS_GPU0="adam_nv_fixed:4 adam_nv_fixed:5 adam:6" JOBS_GPU1="adam_nv_fixed:6 adam_nv_fixed:7 adam:7" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp07_B bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp07_logs_B.zip logs_exp07_B

# ACCOUNT C  (canaries first, separate folder)
!JOBS_GPU0="adam:0" JOBS_GPU1="adam_nv:0" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp07_canary bash run_exp07_label_noise.sh
!JOBS_GPU0="adam_nv:4 adam_nv:5" JOBS_GPU1="adam_nv:6 adam_nv:7" DATA_DIR=<DATA_DIR> LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp07_C bash run_exp07_label_noise.sh
!cd /kaggle/working/Adam-NV/e2_noisy_gradient && zip -r exp07_logs_C.zip logs_exp07_canary logs_exp07_C
```
If a session dies, run the same cell again; complete runs are skipped.

## Analysis (after unzipping all three log zips next to the existing `logs/Label_Diagnostic`)

```bash
python3 analysis/analyze_exp07.py --orig-dir e2_noisy_gradient/logs/Label_Diagnostic \
    --new-dirs logs_exp07_A logs_exp07_B logs_exp07_C --canary-dir logs_exp07_canary
```

## Send back
The three `exp07_logs_*.zip`, the unit-test output, the smoke-test output, the checker table printed at the end of each launcher call, and any error verbatim. Do not rename or delete any log file.

## Do not
Change any argument, mix different β₁ values in one log folder, run other jobs in these folders, or edit any file.

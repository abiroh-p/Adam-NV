# Experiment 3 — bias-correction ablations (clean CIFAR-10, seeds 0–9)

**Question:** is Adam-NV's exact running-product bias correction empirically meaningful, and does it explain GONO's early-epoch lag?

**Variants (10 seeds each, 20 runs):**
- `gono_exact` = GONO with Adam-NV's exact correction. (The paper's GONO is `gono`, already run.)
- `adam_nv_basecorr` = Adam-NV with the standard base-beta1 normalizer.

Hypotheses and the analysis plan are in `PROTOCOL_V2.md` (section 2), fixed before the data exist.

**Time:** about 4.2 hours per variant on two GPUs. Run **one variant per session**; it is safer than one 8.4-hour session.

## What changed (all tested on CPU, not on your GPU)
- `adam_nv.py` (both copies, now byte-identical): new option `bias_correction`, default unchanged. The patched default is bit-identical to the exact file that ran in Experiments 1–2 (max difference 0.0, 6 trials x 700 steps, controller active).
- `gono.py`: new option `bias_correction` (`base` default = the paper).
- `train_resnet.py`: two new optimizer names. Existing optimizers are unchanged (0 differing cells versus the original script).
- Launcher: status and stdout files are now timestamped, so a rerun cannot overwrite earlier records.
- Tests: 28 pass (11 original, 17 new), including a test that the two `adam_nv.py` copies are identical.
- `analysis/analyze_e1_protocol_v2.py`: the paired / equivalence analysis, runnable by you. It reproduces the 10-seed numbers reported earlier.

## Steps (same notebook as before)
```bash
!unzip -o exp03_ablations.zip -d /kaggle/working/Adam-NV     # overwrites adam_nv.py, gono.py, train_resnet.py, launcher
%cd /kaggle/working/Adam-NV/e1_convergence/src
!python3 env_check.py                                        # must print PASS
!cd /kaggle/working/Adam-NV && python3 -m unittest tests.test_gono tests.test_adam_nv_variants -v   # must end with OK
!python3 train_resnet.py --optimizer gono_exact --seed 0 --epochs 1 --smoke-test --data-dir /kaggle/working/Adam-NV/data --log-dir ../smoke_gpu
!python3 train_resnet.py --optimizer adam_nv_basecorr --seed 0 --epochs 1 --smoke-test --data-dir /kaggle/working/Adam-NV/data --log-dir ../smoke_gpu

# SESSION 1 (use "Save Version -> Save & Run All"):
!OPTS="gono_exact" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp03 bash run_exp03_bias_ablation.sh

# SESSION 2 (repeat the unzip, env_check and smoke steps first):
!OPTS="adam_nv_basecorr" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp03 bash run_exp03_bias_ablation.sh

!cd /kaggle/working/Adam-NV && zip -r exp03_logs.zip logs_exp03
```
If a session dies, run the same cell again; complete runs are skipped.

## Send back
- `exp03_logs.zip` (one per session is fine),
- the unit-test output, the `env_check` output, and the `check_logs.py` table,
- any error, verbatim. Do not delete or rename any log file.

## Do not
Change any argument, run other optimizers in these folders, or edit any file.

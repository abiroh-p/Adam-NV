# Experiment 6 — constant-beta1 Adam controls (clean CIFAR-10, lr = 3e-3, seeds 0–3)

**Question:** is Adam-NV's +0.17 pt advantage over Adam at lr = 3e-3 (Exp. 5: 4/4 seeds, paired p = 0.031) caused by the *noise-aware* beta1, or simply by running a *lower* beta1 than Adam's 0.9?

**Why this control:** in `distrust` mode beta1 can only go down from beta_base = 0.9. In Exp. 5 its time-averaged value was about 0.83 (0.829 at lr 3e-3; 0.818 at lr 3e-4). Plain Adam with a constant lower beta1 is the cheapest way to separate the two explanations.

**Variants (4 seeds each, 8 runs, same seeds and lr as Exp. 5 so results pair by seed):**
- `adam_b083` = Adam, betas = (0.83, 0.999). The matched control.
- `adam_b080` = Adam, betas = (0.80, 0.999). Checks that the outcome is not specific to one value.

Reused from Exp. 5, not re-run: `adam_lr3e-3` and `adam_nv_lr3e-3` (seeds 0–3). Do not mix the code versions: `adam_nv.py` is unchanged (sha256 `bcc7824b…`), so those logs stay valid.

**Decision rule (fixed before the data exist), primary metric = final-epoch val_acc, paired by seed:**
1. `adam_b083` recovers most of the gain (mean(adam_b083 − adam) ≥ +0.10 pt, i.e. at least about 60% of Adam-NV's +0.17): the gain is explained by lower beta1. Do not claim noise-awareness helps at this lr. Next step: fix or replace the sigma² signal (see below) before more LR/batch-size experiments.
2. `adam_b083` stays at or below +0.05 pt over Adam while Adam-NV keeps +0.17: evidence for per-element adaptation. Extend to seeds 4–9 for Adam, Adam-NV and `adam_b083` before writing it up.
3. Anything in between: inconclusive at n = 4. Extend to seeds 4–9 (the thresholds are not tests; n = 4 cannot establish equivalence).
Both `adam_b083` and `adam_b080` are REPORTED; nothing is selected by test accuracy.

## What changed (nothing here was run on a GPU or with PyTorch installed in my environment)
- `train_resnet.py`: two new optimizer names, `adam_b083` and `adam_b080`. Existing optimizers are unchanged (the diff is only the new `elif` branches, two `DEFAULT_LR` entries and the argparse `choices`).
- `run_exp06_beta1_control.sh`: new launcher, same pattern as `run_exp05_lr_sensitivity.sh` (seeds 0 1 on GPU 0, seeds 2 3 on GPU 1, tag `_lr3e-3`).
- `tests/test_exp06_controls.py`: checks that Adam-NV with gamma = 0 and the default running-product correction matches torch Adam within about 1e-6. Not bit-exact: after about 153 steps P hits the 1e-7 clamp, so the docstring claim "exactly" is accurate only to that tolerance. If this test fails, report the max difference verbatim.
- `analysis/analyze_exp06.py`: paired analysis. Checked on the Exp. 5 logs; it reproduces Adam-NV − Adam = +0.173 pt, p = 0.031, 4/4 seeds.
- `adam_nv.py` and `gono.py` are untouched, so `tests.test_adam_nv_variants` (copy consistency) is unaffected.

## Steps (same notebook as before)
```bash
!unzip -o exp06_controls.zip -d /kaggle/working/Adam-NV     # overwrites train_resnet.py only; adds new files
%cd /kaggle/working/Adam-NV/e1_convergence/src
!python3 env_check.py                                        # must print PASS
!cd /kaggle/working/Adam-NV && python3 -m unittest tests.test_gono tests.test_adam_nv_variants tests.test_exp06_controls -v   # must end with OK
!python3 train_resnet.py --optimizer adam_b083 --seed 0 --epochs 1 --smoke-test --data-dir /kaggle/working/Adam-NV/data --log-dir ../smoke_gpu

# SESSION 1 (use "Save Version -> Save & Run All"):
!OPTS="adam_b083" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp06 bash run_exp06_beta1_control.sh

# SESSION 2 (repeat the unzip, env_check and smoke steps first):
!OPTS="adam_b080" DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/logs_exp06 bash run_exp06_beta1_control.sh

!cd /kaggle/working/Adam-NV && zip -r exp06_logs.zip logs_exp06
```
If a session dies, run the same cell again; complete runs are skipped.

## Analysis
```bash
python3 analysis/analyze_exp06.py --log-dirs logs_exp05 logs_exp06 --tag _lr3e-3 \
    --opts adam adam_nv adam_b083 adam_b080 --ref adam
python3 analysis/analyze_exp06.py --log-dirs logs_exp05 logs_exp06 --tag _lr3e-3 \
    --opts adam_nv adam_b083 --ref adam_nv
```
The second call gives the direct Adam-NV vs matched-control comparison.

## Send back
- `exp06_logs.zip` (one per session is fine),
- the unit-test output, the `env_check` output, and the `check_logs.py` table,
- any error, verbatim. Do not delete or rename any log file.

## Do not
Change any argument, run other optimizers in these folders, or edit any file.

## Known caveat for the write-up
The sigma² signal compares `v` (beta_2 = 0.999, memory about 1000 steps) with `m²` (memory about 10 steps), and neither is bias-corrected, so it also rises whenever gradients shrink during training. This experiment tests the beta1 explanation only; it does not show that sigma² measures noise. The `beta_2_noise` option already in `adam_nv.py` is the natural knob for a later experiment.

# Experiment 1 — GONO baseline, clean CIFAR-10, seeds 0–9

**Serves:** Protocol v2 primary comparison (Adam-NV vs GONO vs Adam, paired by seed).
**Runs:** 10 (GONO, seeds 0–9, 60 epochs, batch 128, lr 1e-3, `--deterministic`).
**Environment:** Kaggle, Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, same as the existing E1 logs.

## What is in this package

| File (repo-relative) | Status | Purpose |
|---|---|---|
| `e1_convergence/src/gono.py` | NEW | GONO optimizer, written from the paper's Algorithm 1 |
| `e1_convergence/src/train_resnet.py` | MODIFIED | adds `--optimizer gono` and `gono_global`; training loop untouched |
| `e1_convergence/src/run_e1_suite.py` | MODIFIED | adds `--status-file`; fixes "no status file when everything is skipped" |
| `e1_convergence/src/experiment_metadata.py` | MODIFIED | metadata now also stores SHA-256 of the source files |
| `e1_convergence/src/env_check.py` | NEW | refuses to start if the software stack differs from E1 |
| `e1_convergence/src/check_logs.py` | NEW | verifies completeness and provenance of the logs |
| `e1_convergence/src/run_e1_protocol_v2.sh` | NEW | two-GPU launcher |
| `tests/test_gono.py` | NEW | 9 tests |
| `REVIEW_modified_files.diff` | info | exact changes to the 3 modified files |

**Do not replace `e1_convergence/src/adam_nv.py`.** It is the file that produced the existing seeds 0–3.

## What I verified, and what I could not

Verified (CPU, PyTorch 2.14.1, not your 2.10.0):
- All 20 tests pass (11 existing + 9 new).
- GONO with λ=0 equals `torch.optim.Adam` (float64, tolerance 1e-11), for both cosine scopes.
- GONO matches an independent plain-formula implementation, and matches the authors' released file (byte-checked against your upload) up to two implementation details, which I isolated by patching a copy of their file: epsilon placement (dominant; 2.5e-6 parameter difference over 3,000 float64 steps) and a float32 rounding of β₁ₜ (~7e-8). With both matched the two agree to 9e-16, so the controller logic is identical. See the `gono.py` header.
- The patched `train_resnet.py` reproduces the original script's numbers exactly (0 differing cells) for `adam`, `adam_nv`, `adamw`, same seed.
- E1's older `adam_nv.py` and E2's current one give bit-identical updates at default settings (400 steps × 5 trials, controller active).

NOT verified (I have no GPU): CUDA execution, `--deterministic` interaction with the new ops, and your 2.10.0 stack. The three pre-flight steps below exist to catch this before the long run.

## Design decisions (state these in the paper)

1. **Cosine scope = per parameter tensor** (the scope in the authors' released code). The paper's Algorithm 1 reads as one cosine over the whole gradient; `gono_global` implements that and is reserved for an ablation.
2. **Independent re-implementation**, not copied code: the authors' repository has no license file.
3. **Epsilon convention = Adam's** (as in the paper's Algorithm 1), so GONO with λ=0 is exactly our Adam baseline. The authors' code uses a slightly different placement and is not exactly Adam at λ=0 despite its docstring. Report this as a limitation: for tiny-gradient elements the difference can reach about 1% of the parameter movement in my synthetic test.
4. Default GONO hyperparameters from the paper: λ=0.4, β₁ base 0.9, β₁ ∈ [0.5, 0.99], β₂=0.999, ε=1e-8. No tuning, same as the other baselines.
5. GONO's logged `beta1_*` are statistics over **tensors**, not elements. Never compare them numerically with Adam-NV's element-wise `beta1_*`.

## Steps (Kaggle, GPU T4 ×2)

```bash
# 0. Working copy with the repo layout (the data folder is copied with src)
!mkdir -p /kaggle/working/Adam-NV/e1_convergence /kaggle/working/Adam-NV/tests
!cp -r /kaggle/input/datasets/abiroh101/adammm/Adam-NV/e1_convergence/src /kaggle/working/Adam-NV/e1_convergence/src
# 1. Overlay this package (wherever the zip lives)
!unzip -o exp01_gono_clean.zip -d /kaggle/working/Adam-NV
%cd /kaggle/working/Adam-NV/e1_convergence/src

# 2. PRE-FLIGHT A: environment guard (must print PASS)
!python3 env_check.py

# 3. PRE-FLIGHT B: unit tests (must print OK)
!cd /kaggle/working/Adam-NV && python3 -m unittest tests.test_gono -v

# 4. PRE-FLIGHT C: one-epoch GPU smoke run (must finish without error)
!python3 train_resnet.py --optimizer gono --seed 0 --epochs 1 --smoke-test --log-dir ../smoke_gpu

# 5. The real run: use "Save Version -> Save & Run All" so it survives a browser disconnect
!bash run_e1_protocol_v2.sh

# 6. Package the logs
!cd /kaggle/working/Adam-NV/e1_convergence && zip -r exp01_logs.zip logs
```

If a session dies, run step 5 again: `--skip-existing` skips complete runs and restarts only the incomplete one (a partial run restarts from epoch 1).

## Expected time

About 48 minutes per run in your earlier single-process E1 runs, so about 4 hours per GPU for 5 seeds if two concurrent processes don't slow each other. They share the CPU for data augmentation, and I have not measured this on your setup. Please report the epoch time from the first epochs in `stdout_gono_gpu0.txt` and `stdout_gono_gpu1.txt`.

## Send back

1. `exp01_logs.zip` (CSVs, metadata JSONs, `_suite_status_*.csv`, `stdout_*.txt`).
2. The printed output of steps 2, 3, 4 and the final `check_logs.py` table.
3. Any error, verbatim.

## Do not

- Run `analyze_e1.py` on the new logs yet. I will send an updated analysis script with the paired and equivalence tests.
- Change `--epochs`, `--batch-size`, `--lr`, or any other argument. The existing E1 runs used these values.
- Start another experiment before this one passes `check_logs.py`.

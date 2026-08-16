# Adam-NV

Adam-NV is an experimental Adam variant whose per-parameter-element first
moment coefficient (`beta_1`) responds to a bounded gradient-noise heuristic.
The repository evaluates the mechanism on CIFAR models under constant and
non-stationary synthetic gradient noise.

## Current evidence

The completed experiments show that Adam-NV's controller responds to changing
gradient statistics. They do **not** establish a consistent accuracy,
convergence, or stability advantage over Adam. The appropriate current framing
is a mechanism-validation and diagnostic-redesign study.

Legacy step-level response claims are invalid: the original diagnostic stepped
the cosine scheduler once per minibatch instead of once per epoch. Corrected
diagnostics use protocol v2 and are not yet rerun on the required seeds.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for the machine before the
remaining requirements if the default wheel is not suitable.

## Authoritative code

- `e1_convergence/src/train_resnet.py` and `run_e1_suite.py`: E1 convergence
- `e2_noisy_gradient/src/part_b_resnet_v8.py`: all new E2/E13--E16 runs
- `e2_noisy_gradient/src/adam_nv.py`: current optimizer implementation
- `analysis/analyze_e13_e15.py`: paired E13--E15 statistical pipeline
- `e2_noisy_gradient/src/diagnose_response_lag.py`: protocol-v2 diagnostic

Files named `part_b_resnet_v1.py` through `part_b_resnet_v7.py` are historical
snapshots. See `e2_noisy_gradient/src/LEGACY_SCRIPTS.md`.

## Tests

```bash
python -m unittest discover -s tests -v
```

PyTorch-dependent numerical tests are skipped in analysis-only environments
without PyTorch. On the training machine, no optimizer test should be skipped.

## Required completion runs

Run commands from the indicated `src` directory.

### 1. Complete E1 baseline seeds

```bash
cd e1_convergence/src
python run_e1_suite.py --epochs 60 --optimizers adam adamw adabelief \
  --seeds 2 3 --skip-existing --deterministic
python analyze_e1.py --log-dir ../logs --out-dir ..
```

### 2. Replace the incomplete E14 run

The existing intermittent `beta_2=0.95`, seed-1 CSV stops at epoch 32.

```bash
cd e2_noisy_gradient/src
python part_b_resnet_v8.py --optimizer adam_nv --schedule intermittent \
  --sigma 2.0 --beta2 0.95 --seed 1 --epochs 60 --batch-size 128 \
  --log-dir ../logs/response_lag --deterministic
```

Then regenerate the publication-mode table (without `--allow-incomplete`):

```bash
cd ../..
python analysis/analyze_e13_e15.py
```

### 3. Rerun corrected multi-seed lag diagnostics

```bash
cd e2_noisy_gradient/src
python run_response_lag_v2_suite.py --epochs 60 --skip-existing --deterministic
```

Only `_lagdiag_v2` files containing `protocol_version=2` are valid. All older
files under `logs/diagnose` are explicitly invalidated.

### 4. Test controller direction

The original rule lowers `beta_1` when the heuristic increases. Because higher
momentum is the usual smoothing response to noise, the repository now includes
a matched reverse-direction ablation:

```bash
cd e2_noisy_gradient/src
python run_controller_direction_suite.py --epochs 60 --skip-existing --deterministic
```

### 5. Run CIFAR-100 generality testing

```bash
cd e2_noisy_gradient/src
python run_cifar100_generality_suite.py --epochs 60 --skip-existing --deterministic
```

This suite compares Adam and Adam-NV on clean CIFAR-100 and under intermittent
noise using four matched seeds.

### 6. Regenerate label-noise evidence

`part_b_resnet_v8.py` now logs training accuracy on uncorrupted examples and,
for corrupted examples, accuracy against both clean and noisy labels. These
curves directly test the claimed memorization transition.

```bash
cd e2_noisy_gradient/src
python part_b_resnet_v8.py --optimizer adam_nv --schedule constant --sigma 0 \
  --label-noise 0.2 --label-diagnostic-interval 1 --seed 0 --epochs 60 \
  --log-dir ../logs/label_noise_v2 --deterministic
```

Repeat for Adam and seeds 1--3 before making comparative claims.

### 7. Benchmark overhead formally

```bash
cd ../..
python benchmarks/benchmark_optimizer_overhead.py --device cuda \
  --warmup-steps 50 --measured-steps 200 --repeats 10
```

The benchmark synchronizes CUDA, records hardware/software metadata, separates
optimizer-step and end-to-end time, and reports means, standard deviations, and
95% confidence intervals.

## Statistical policy

- Primary accuracy metric: mean validation accuracy over the final five epochs.
- Match conditions by seed and use paired tests.
- Report mean paired differences and 95% confidence intervals.
- Treat the exact sign-flip result as a robustness check for small samples.
- Apply Holm correction within each declared experiment family.
- Do not infer that two conditions differ merely because one is significant and
  the other is not; use a direct contrast or interaction test.
- Phrase null findings as “no detectable advantage under the tested
  conditions,” not proof of equivalence or absence.

## Reproducibility notes

New training runs write a JSON metadata sidecar with arguments, command,
Python/PyTorch/torchvision versions, CUDA/GPU details, hostname, and RNG-stream
seeds. Gradient-noise runs with the same seed reuse matched Gaussian directions,
but the final perturbation tensors are not byte-identical: each optimizer's
direction is scaled by its own gradient norm.

See `RESEARCH_STATUS.md` for the issue-by-issue completion checklist.

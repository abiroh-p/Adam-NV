#!/usr/bin/env bash
# Experiment 5: learning-rate sensitivity on clean CIFAR-10 (Protocol v2).
# Learning rates 3e-4 and 3e-3 (the main protocol uses 1e-3), seeds 0-3, for the optimizers in OPTS.
# Per (optimizer, lr): seeds 0 1 on GPU 0 and 2 3 on GPU 1, about 1.7 h. All values are REPORTED;
# nothing is selected by test accuracy.
#   Account A:  OPTS="adam adam_nv" bash run_exp05_lr_sensitivity.sh   (4 combos, about 6.7 h)
#   Account B:  OPTS="gono"         bash run_exp05_lr_sensitivity.sh   (2 combos, about 3.3 h)
set -u
cd "$(dirname "$0")"
OPTS="${OPTS:-adam adam_nv gono}"
LRS="${LRS:-3e-4 3e-3}"
for OPT in $OPTS; do
  for LR in $LRS; do
    echo "[exp05] ===== $OPT at lr=$LR ====="
    OPT="$OPT" LR="$LR" TAG="_lr$LR" SEEDS_A="0 1" SEEDS_B="2 3" bash run_e1_protocol_v2.sh \
      || echo "[exp05] $OPT lr=$LR finished with problems (see check_logs output above)"
  done
done

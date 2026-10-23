#!/usr/bin/env bash
# Experiment 3: bias-correction ablations on clean CIFAR-10 (Protocol v2), seeds 0-9.
#   gono_exact        GONO with Adam-NV's exact running-product correction
#   adam_nv_basecorr  Adam-NV with the standard base-beta1 correction
# 20 runs, about 8.5 h on two GPUs (about 4.2 h per variant). Each variant is a separate
# launcher invocation, so a dead session loses at most the run in progress.
# To run one variant per session:  OPTS="gono_exact" bash run_exp03_bias_ablation.sh
set -u
cd "$(dirname "$0")"
OPTS="${OPTS:-gono_exact adam_nv_basecorr}"
for OPT in $OPTS; do
  echo "[exp03] ===== $OPT ====="
  OPT="$OPT" SEEDS_A="0 1 2 3 4" SEEDS_B="5 6 7 8 9" bash run_e1_protocol_v2.sh \
    || echo "[exp03] $OPT finished with problems (see check_logs output above)"
done

#!/usr/bin/env bash
# Experiment 6: constant-beta1 Adam controls at lr=3e-3 (Protocol v2), seeds 0-3.
#   adam_b083  Adam with constant beta1=0.83 (time-averaged beta1 of Adam-NV in Exp. 5)
#   adam_b080  Adam with constant beta1=0.80
# Per optimizer: seeds 0 1 on GPU 0 and 2 3 on GPU 1, about 1.7 h (about 3.4 h for both).
# Run adam_b083 first; it decides the story. One optimizer per session is safer:
#   OPTS="adam_b083" bash run_exp06_beta1_control.sh
set -u
cd "$(dirname "$0")"
OPTS="${OPTS:-adam_b083 adam_b080}"
LRS="${LRS:-3e-3}"
for OPT in $OPTS; do
  for LR in $LRS; do
    echo "[exp06] ===== $OPT at lr=$LR ====="
    OPT="$OPT" LR="$LR" TAG="_lr$LR" SEEDS_A="0 1" SEEDS_B="2 3" bash run_e1_protocol_v2.sh \
      || echo "[exp06] $OPT lr=$LR finished with problems (see check_logs output above)"
  done
done

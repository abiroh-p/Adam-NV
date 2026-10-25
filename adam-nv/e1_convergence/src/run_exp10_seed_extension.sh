#!/usr/bin/env bash
# Experiment 10: seed extension (seeds 4-9) at lr = 3e-3 for Adam, Adam-NV and the two
# constant-beta1 controls, so that every arm of Exp. 5 / Exp. 6 has 10 seeds (0-9).
#
# One optimizer per call is the safest unit (6 runs, 3 per GPU, about 2.5 h):
#   OPTS="adam adam_nv"  -> about 5 h in one session
#   OPTS="adam_b083"     -> about 2.5 h
#   OPTS="adam_b080"     -> about 2.5 h
#
# Environment variables (all optional except DATA_DIR / LOG_DIR in practice):
#   OPTS      optimizers to run               (default "adam adam_nv")
#   SEEDS_A   seeds for GPU 0                 (default "4 5 6")
#   SEEDS_B   seeds for GPU 1                 (default "7 8 9")
#   DATA_DIR, LOG_DIR, EPOCHS are passed through to run_e1_protocol_v2.sh
#
# Complete logs are skipped, so the script is safe to re-run after an interrupted session.
# Use a separate LOG_DIR per account / per call.

set -u
cd "$(dirname "$0")"
OPTS="${OPTS:-adam adam_nv}"
SEEDS_A="${SEEDS_A:-4 5 6}"
SEEDS_B="${SEEDS_B:-7 8 9}"

for OPT in $OPTS; do
  echo "[exp10] ===== $OPT at lr=3e-3, seeds: $SEEDS_A | $SEEDS_B ====="
  OPT="$OPT" LR="3e-3" TAG="_lr3e-3" SEEDS_A="$SEEDS_A" SEEDS_B="$SEEDS_B" bash run_e1_protocol_v2.sh \
    || echo "[exp10] $OPT finished with problems (see check_logs output above)"
done

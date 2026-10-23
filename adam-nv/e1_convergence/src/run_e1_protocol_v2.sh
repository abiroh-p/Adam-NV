#!/usr/bin/env bash
# Protocol v2 launcher for E1-style runs (clean CIFAR-10, ResNet-18, 60 epochs).
#
# Experiment 1 (GONO baseline, seeds 0-9) with defaults:
#     bash run_e1_protocol_v2.sh
# Any other single optimizer / seed set (e.g. extra Adam-NV seeds):
#     OPT=adam_nv SEEDS_A="4 5 6" SEEDS_B="7 8 9" bash run_e1_protocol_v2.sh
#
# Environment variables (all optional):
#     DATA_DIR  dataset root            (default ./data)
#     LOG_DIR   where logs are written  (default ../logs)
#     EPOCHS    epochs per run          (default 60)
#     OPT       optimizer name          (default gono)
#     SEEDS_A   seeds for GPU 0         (default "0 1 2 3 4")
#     SEEDS_B   seeds for GPU 1         (default "5 6 7 8 9")
#     LR        learning-rate override  (default: optimizer default 1e-3)
#     TAG       file-name suffix, e.g. _lr3e-4 (use together with LR)
#
# With >= 2 GPUs the two seed sets run concurrently, one process per GPU.
# With 1 GPU everything runs sequentially. Existing complete logs are skipped
# (--skip-existing checks the row count), so the script is safe to re-run
# after an interrupted session.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-../logs}"
EPOCHS="${EPOCHS:-60}"
OPT="${OPT:-gono}"
SEEDS_A="${SEEDS_A:-0 1 2 3 4}"
SEEDS_B="${SEEDS_B:-5 6 7 8 9}"
LR="${LR:-}"
TAG="${TAG:-}"
LR_ARGS=()
[ -n "$LR" ] && LR_ARGS+=(--lr "$LR")
[ -n "$TAG" ] && LR_ARGS+=(--run-tag "$TAG")
CHECK_LR_ARGS=()
[ -n "$LR" ] && CHECK_LR_ARGS+=(--lr "$LR")
[ -n "$TAG" ] && CHECK_LR_ARGS+=(--run-tag "$TAG")

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)   # timestamp keeps status/stdout of every invocation

python3 env_check.py || { echo "Environment check failed; not starting runs."; exit 2; }

NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")

run() {  # usage: run <gpu-index> <tag> <seed> [<seed> ...]
  local gpu=$1 tag=$2
  shift 2
  CUDA_VISIBLE_DEVICES=$gpu python3 run_e1_suite.py \
      --optimizers "$OPT" --seeds "$@" \
      --epochs "$EPOCHS" --batch-size 128 \
      --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" \
      --skip-existing --deterministic "${LR_ARGS[@]}" \
      --status-file "_suite_status_${OPT}${TAG}_${tag}_${TS}.csv" \
      > "$LOG_DIR/stdout_${OPT}${TAG}_${tag}_${TS}.txt" 2>&1
}

if [ "$NGPU" -ge 2 ]; then
  echo "[protocol-v2] 2 GPUs: seeds ($SEEDS_A) on GPU 0, seeds ($SEEDS_B) on GPU 1"
  run 0 gpu0 $SEEDS_A &
  run 1 gpu1 $SEEDS_B &
  wait
else
  echo "[protocol-v2] 1 GPU: running seeds ($SEEDS_A $SEEDS_B) sequentially"
  run 0 all $SEEDS_A $SEEDS_B
fi

echo "[protocol-v2] runs finished; checking logs"
python3 check_logs.py --log-dir "$LOG_DIR" --optimizers "$OPT" \
    --seeds $SEEDS_A $SEEDS_B --epochs "$EPOCHS" "${CHECK_LR_ARGS[@]}"

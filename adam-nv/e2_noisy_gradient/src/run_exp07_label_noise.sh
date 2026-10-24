#!/usr/bin/env bash
# Experiment 7: label-noise (20%) constant-beta1 control and seed extension.
# Uses the EXISTING part_b_resnet_v8.py unchanged (no code edits).
#
# Jobs are "<optimizer>:<seed>" pairs; optimizer is one of adam, adam_nv, adam_nv_fixed.
# adam_nv_fixed runs Adam-NV with gamma=0 and beta1 fixed at $FIXED_BETA1 (plain Adam
# with a constant lower beta1; see tests/test_exp07_fixed_beta1.py for the equivalence).
#
# Example (one account, two GPUs):
#   JOBS_GPU0="adam_nv_fixed:0 adam_nv_fixed:1 adam:4" \
#   JOBS_GPU1="adam_nv_fixed:2 adam_nv_fixed:3 adam:5" \
#   DATA_DIR=/path/to/e2_noisy_gradient/data LOG_DIR=../logs_exp07 bash run_exp07_label_noise.sh
#
# Environment variables:
#   JOBS_GPU0 / JOBS_GPU1  space-separated jobs for GPU 0 / GPU 1 (run sequentially per GPU)
#   DATA_DIR      dataset root                  (default ../data)
#   LOG_DIR       where logs are written        (default ../logs_exp07)
#   EPOCHS        epochs per run                (default 60)
#   FIXED_BETA1   beta1 for adam_nv_fixed       (default 0.81)
#   DRY_RUN=1     print the commands only (skips the environment check)
#
# Complete logs (EPOCHS data rows) are skipped, so the script is safe to re-run after an
# interrupted session. File names do not contain the beta1 value, so use a separate
# LOG_DIR for every different FIXED_BETA1.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp07}"
EPOCHS="${EPOCHS:-60}"
FIXED_BETA1="${FIXED_BETA1:-0.81}"
JOBS_GPU0="${JOBS_GPU0:-}"
JOBS_GPU1="${JOBS_GPU1:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$JOBS_GPU0$JOBS_GPU1" ]; then
  echo "[exp07] No jobs given. Set JOBS_GPU0 and/or JOBS_GPU1 (see the header of this script)."
  exit 2
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp07] Environment check failed; not starting runs."; exit 2; }
import sys
import torch, torchvision
want = {"python": "3.12.13", "torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}
have = {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__}
ok = True
for k in want:
    flag = "OK " if have[k] == want[k] else "MISMATCH"
    ok &= have[k] == want[k]
    print(f"[env_check] {flag} {k}: have {have[k]}, want {want[k]}")
if not torch.cuda.is_available():
    print("[env_check] MISMATCH CUDA not available"); ok = False
else:
    print("[env_check] OK  GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("[env_check]", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  sha256sum part_b_resnet_v8.py adam_nv.py noise_schedule.py noise.py experiment_metadata.py \
    > "$LOG_DIR/_code_hashes_${TS}.txt"
fi

run_job() {  # usage: run_job <gpu> <optimizer>:<seed>
  local gpu=$1 opt=${2%%:*} seed=${2##*:}
  local extra=()
  if [ "$opt" = "adam_nv_fixed" ]; then extra=(--fixed-beta1 "$FIXED_BETA1"); fi
  local base="${opt}_scheduleconstant_sigma0.0_ln0.2_seed${seed}"
  local csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then
    echo "[exp07] skip $opt seed $seed (complete log exists)"
    return 0
  fi
  local cmd=(python3 part_b_resnet_v8.py --optimizer "$opt" "${extra[@]}"
             --schedule constant --sigma 0 --label-noise 0.2 --label-diagnostic-interval 1
             --seed "$seed" --epochs "$EPOCHS" --batch-size 128
             --data-dir "$DATA_DIR" --log-dir "$LOG_DIR" --deterministic)
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run][GPU $gpu] CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    return 0
  fi
  local start=$SECONDS status=success err=""
  CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" > "$LOG_DIR/stdout_${base}_${TS}.txt" 2>&1 \
    || { status=failed; err="exit code $?"; }
  echo "${opt},${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp07] GPU $gpu: $opt seed $seed -> $status ($((SECONDS - start)) s)"
}

run_list() {  # usage: run_list <gpu> <jobs...>
  local gpu=$1; shift
  echo "optimizer,seed,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  for job in "$@"; do run_job "$gpu" "$job"; done
}

if [ -n "$JOBS_GPU0" ]; then run_list 0 $JOBS_GPU0 & fi
if [ -n "$JOBS_GPU1" ]; then run_list 1 $JOBS_GPU1 & fi
wait

if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp07] runs finished; checking logs"
python3 check_exp07_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --fixed-beta1 "$FIXED_BETA1" \
    --jobs $JOBS_GPU0 $JOBS_GPU1

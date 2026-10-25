#!/usr/bin/env bash
# Experiment 9: Adam-NV with response_mode=smooth under 20% label noise (seeds 0-7).
# Uses the EXISTING part_b_resnet_v8.py unchanged. Same protocol as Exp. 7/8 except
# --response-mode smooth --beta-max 0.999 (beta1 may rise above 0.9 when the noise
# estimate is low, instead of only falling below it).
#
# Jobs are plain seeds:
#   SEEDS_GPU0="0 1 2 3" SEEDS_GPU1="4 5 6 7" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp09_A \
#   bash run_exp09_label_noise_smooth.sh
#
# Environment variables: SEEDS_GPU0, SEEDS_GPU1, DATA_DIR (default ../data),
#   LOG_DIR (default ../logs_exp09), EPOCHS (default 60), BETA_MAX (default 0.999),
#   DRY_RUN=1 (print commands only).
# Complete logs (EPOCHS data rows) are skipped, so re-running after an interrupted session is safe.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp09}"
EPOCHS="${EPOCHS:-60}"
BETA_MAX="${BETA_MAX:-0.999}"
SEEDS_GPU0="${SEEDS_GPU0:-}"
SEEDS_GPU1="${SEEDS_GPU1:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ -z "$SEEDS_GPU0$SEEDS_GPU1" ]; then
  echo "[exp09] No seeds given. Set SEEDS_GPU0 and/or SEEDS_GPU1 (see the header of this script)."
  exit 2
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp09] Environment check failed; not starting runs."; exit 2; }
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

run_job() {  # usage: run_job <gpu> <seed>
  local gpu=$1 opt=adam_nv seed=$2
  local extra=(--response-mode smooth --beta-max "$BETA_MAX")
  local base="${opt}_scheduleconstant_sigma0.0_responsesmooth_betamax${BETA_MAX}_ln0.2_seed${seed}"
  local csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then
    echo "[exp09] skip $opt seed $seed (complete log exists)"
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
  echo "${opt}_smooth,${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp09] GPU $gpu: $opt seed $seed -> $status ($((SECONDS - start)) s)"
}

run_list() {  # usage: run_list <gpu> <jobs...>
  local gpu=$1; shift
  echo "optimizer,seed,status,elapsed_sec,error" > "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  for job in "$@"; do run_job "$gpu" "$job"; done
}

if [ -n "$SEEDS_GPU0" ]; then run_list 0 $SEEDS_GPU0 & fi
if [ -n "$SEEDS_GPU1" ]; then run_list 1 $SEEDS_GPU1 & fi
wait

if [ "$DRY_RUN" = "1" ]; then exit 0; fi
echo "[exp09] runs finished; checking logs"
python3 check_exp09_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --beta-max "$BETA_MAX" \
    --seeds $SEEDS_GPU0 $SEEDS_GPU1

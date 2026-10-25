#!/usr/bin/env bash
# Experiment 11: beta_1 shuffle control for Adam-NV under 20% label noise (seeds 0-7).
# Runs part_b_resnet_v8_shuffle.py (a wrapper; part_b_resnet_v8.py and adam_nv.py are unchanged).
# One call = one shuffle mode, and every mode needs its own LOG_DIR (file names are those of a
# normal Adam-NV run).
#
#   MODE=step  SEEDS_GPU0="0 1 2" SEEDS_GPU1="3 4 5" \
#   DATA_DIR=/kaggle/working/Adam-NV/data LOG_DIR=/kaggle/working/Adam-NV/e2_noisy_gradient/logs_exp11_step_A \
#   bash run_exp11_beta1_shuffle.sh
#
# MODE is none | step | fixed. SEEDS_GPU0 / SEEDS_GPU1 are space-separated seed lists, run
# sequentially per GPU. EPOCHS defaults to 60, DRY_RUN=1 prints the commands only.
# Complete logs (EPOCHS data rows plus the sidecar files) are skipped on re-run.

set -u
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-../data}"
LOG_DIR="${LOG_DIR:-../logs_exp11}"
EPOCHS="${EPOCHS:-60}"
MODE="${MODE:-}"
SEEDS_GPU0="${SEEDS_GPU0:-}"
SEEDS_GPU1="${SEEDS_GPU1:-}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in none|step|fixed) ;; *) echo "[exp11] Set MODE to none, step or fixed."; exit 2;; esac
if [ -z "$SEEDS_GPU0$SEEDS_GPU1" ]; then
  echo "[exp11] No seeds given. Set SEEDS_GPU0 and/or SEEDS_GPU1 (see the header of this script)."
  exit 2
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$DRY_RUN" != "1" ]; then
  python3 - <<'PY' || { echo "[exp11] Environment check failed; not starting runs."; exit 2; }
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
  sha256sum part_b_resnet_v8.py part_b_resnet_v8_shuffle.py adam_nv.py adam_nv_shuffle.py noise_schedule.py noise.py experiment_metadata.py \
    > "$LOG_DIR/_code_hashes_${TS}.txt"
fi

run_job() {  # usage: run_job <gpu> <seed>
  local gpu=$1 opt=adam_nv seed=$2
  local base="${opt}_scheduleconstant_sigma0.0_ln0.2_seed${seed}"
  local csv="$LOG_DIR/${base}.csv"
  if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -ge $((EPOCHS + 1)) ]; then
    echo "[exp11] skip $opt seed $seed (complete log exists)"
    return 0
  fi
  local cmd=(python3 part_b_resnet_v8_shuffle.py --shuffle-mode "$MODE" --optimizer "$opt"
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
  echo "${opt}_${MODE},${seed},${status},$((SECONDS - start)),${err}" >> "$LOG_DIR/_status_gpu${gpu}_${TS}.csv"
  echo "[exp11] GPU $gpu: $opt seed $seed -> $status ($((SECONDS - start)) s)"
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
echo "[exp11] runs finished; checking logs"
python3 check_exp11_logs.py --log-dir "$LOG_DIR" --epochs "$EPOCHS" --mode "$MODE" \
    --seeds $SEEDS_GPU0 $SEEDS_GPU1

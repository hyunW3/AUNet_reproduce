#!/usr/bin/env bash
# Sequential queue for the SVT 100M ablation on ece-agpu18 GPUs 5,6 (user-approved lanes).
# Each arm: wait until both GPUs hold no compute process, train, mark .DONE (or .FAILED).
# Re-running skips .DONE arms; an interrupted arm resumes from its last checkpoint.
# Usage: nohup ./svt_queue.sh > logs/queue.log 2>&1 &
set -uo pipefail
S=/home/hwbae/AUNet_svt
GPUS="${GPUS:-5,6}"
ARMS=(${ARMS:-stride4p57 rg_llama3_V32k rg_gpt2 rg_qwen2})
cd "$S"

gpus_idle() {
  local busy uuid
  for i in ${GPUS//,/ }; do
    uuid=$(nvidia-smi --query-gpu=uuid --id="$i" --format=csv,noheader)
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    [ "$busy" -eq 0 ] || return 1
  done
}

for arm in "${ARMS[@]}"; do
  [ -f "runs/$arm/.DONE" ] && { echo "$(date '+%F %T') skip $arm (.DONE)"; continue; }
  until gpus_idle; do echo "$(date '+%F %T') GPUs $GPUS busy; waiting"; sleep 300; done
  echo "$(date '+%F %T') START $arm on GPUs $GPUS"
  if GPUS="$GPUS" PORT="${PORT:-29771}" ./svt_train.sh "$arm" >> "logs/$arm.log" 2>&1 \
     && grep -q '"global_step": 53500,' "runs/$arm/metrics.jsonl"; then
    date '+%F %T' > "runs/$arm/.DONE"; echo "$(date '+%F %T') DONE $arm"
  else
    date '+%F %T' > "runs/$arm/.FAILED"; echo "$(date '+%F %T') FAILED $arm (see logs/$arm.log)"
  fi
done
echo "$(date '+%F %T') queue finished"

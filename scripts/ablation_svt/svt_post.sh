#!/usr/bin/env bash
# After rg_snapshot_repro / rg_randtrie_mcr finish training: fmha held-out BPB, then downstream
# (via svt_downstream_queue.sh) on ece-agpu18 GPUs 3,4 (user-assigned). Only idle GPUs are used.
# Usage: setsid nohup ./svt_post.sh >> logs/post.log 2>&1 < /dev/null &
set -uo pipefail
S=/home/hwbae/AUNet_svt
PY=/home/hwbae/AUNet/lingua/.venv/bin/python
cd "$S/lingua"
idle() { local apps; apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || return 1
         case "$apps" in *GPU-b56b3ba3*) return 1 ;; *) return 0 ;; esac; }   # GPU 3
for arm in rg_snapshot_repro rg_randtrie_mcr; do
  until [ -f "$S/runs/$arm/.DONE" ] || [ -f "$S/runs/$arm/.FAILED" ]; do sleep 600; done
  [ -f "$S/runs/$arm/.FAILED" ] && { echo "$(date '+%F %T') skip $arm (training FAILED)"; continue; }
  grep -q "\"tag\": \"$arm\"" "$S/heldout/heldout_fmha.jsonl" 2>/dev/null && continue
  ck="$S/runs/$arm/checkpoints/0000053504"
  # consolidate first (downstream eval does it; run it now so heldout_fmha has a consolidated dir)
  until idle; do sleep 300; done
  [ -d "$ck/consolidated" ] || CUDA_VISIBLE_DEVICES=3 "$PY" -c "from lingua.checkpoint import consolidate_checkpoints; consolidate_checkpoints('$ck')"
  CUDA_VISIBLE_DEVICES=3 "$PY" ../heldout/heldout_fmha.py "$ck/consolidated" ../heldout/bpb_windows.jsonl "$arm" \
    ../heldout/heldout_fmha.jsonl > "$S/heldout/${arm}_fmha.log" 2>&1 \
    && echo "$(date '+%F %T') heldout $arm $(tail -1 "$S/heldout/${arm}_fmha.log")" \
    || echo "$(date '+%F %T') heldout FAILED $arm"
done
cd "$S" && ARMS="rg_snapshot_repro rg_randtrie_mcr" GPU_POOL="3 4" ./svt_downstream_queue.sh
echo "$(date '+%F %T') post finished"

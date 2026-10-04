#!/usr/bin/env bash
# Eval queue for the SVT 100M ablation, run with the SNAPSHOT code (has the stride strategy) so
# every arm, incl. the re-scored lb_rg_100M baseline, uses identical eval code. Per arm:
#   (1) downstream: 6-bench, 0-shot, FULL sets, same settings as
#       runs/poc/portable_aunetlaw/eval_law100M_full_ece11.sh -> eval/<arm>/results.json
#   (2) held-out BPB: scripts/probes/bpb_windows_local.py on the 320 held-out DCLM windows
#       (reports/bpb_windows.jsonl, 1,228,802 B) -> heldout/bpb_results.jsonl
# Safe to run on several hosts at once (shared NFS): per-arm mkdir locks. Uses only GPUs in
# GPU_POOL that hold no compute process at all.
# Usage: GPU_POOL="0 7" setsid nohup ./svt_downstream_queue.sh >> logs/downstream_queue_$(hostname).log 2>&1 < /dev/null &
set -uo pipefail
S=/home/hwbae/AUNet_svt; H=/home/hwbae/AUNet
PY=$H/lingua/.venv/bin/python
GPU_POOL="${GPU_POOL:-0 7}"
ARMS=(${ARMS:-stride4p57 rg_llama3_V32k rg_gpt2 rg_qwen2 lb_rg_100M rg_snapshot_repro})
TASKS="[hellaswag,arc_easy,arc_challenge,boolq,piqa,winogrande]"
ACFG=apps/aunet/configs/eval_full_5bench_b200.yaml
INC=$H/lingua/eval_tasks
OUT=$S/eval
HO=$S/heldout
mkdir -p "$OUT"
cd "$S/lingua"
export TMPDIR=/var/tmp TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ckpt_of() {
  case $1 in
    lb_rg_100M) echo "$H/runs/poc/portable_aunetlaw/lb_rg_100M/checkpoints/0000053504" ;;
    *)          echo "$S/runs/$1/checkpoints/0000053504" ;;
  esac
}
ready() {   # training finished?
  case $1 in lb_rg_100M) return 0 ;; *) [ -f "$S/runs/$1/.DONE" ] ;; esac
}
tok_of() {  # patch tokenizer (stride ignores it: its checkpoint has no bpe_tokenizer_path)
  case $1 in
    rg_llama3_V32k) echo "$S/tokenizer_extra/llama3_V32k.model" ;;
    rg_gpt2)        echo "$S/tokenizer_extra/gpt2/tokenizer.json" ;;
    rg_qwen2)       echo "$H/tokenizer/qwen2/tokenizer.json" ;;
    rg_randtrie_mcr) echo "$S/tokenizer_extra/randtrie_V44500_s0.model" ;;
    *)              echo "$H/tokenizer/llama3/tokenizer.model" ;;
  esac
}
extra_of() {
  [ "$1" = stride4p57 ] && { echo ""; return; }   # fixed stride: train-config regex, nothing to force
  echo "greedy_question_loglikelihood=true regex_bpe_tokenizer_path=$(tok_of "$1") force_bpe_online_mode=greedy"
}
ds_done() { [ -f "$OUT/$1/results.json" ]; }
ho_done() { [ -f "$HO/$1.done" ]; }
free_gpu() {   # NB: no `cmd | grep -q` here — under pipefail grep's early exit SIGPIPEs nvidia-smi
  local g uuid apps
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader) || return 1
  for g in $GPU_POOL; do
    uuid=$(nvidia-smi --query-gpu=uuid --id="$g" --format=csv,noheader) || return 1
    [ -n "$uuid" ] || return 1
    case "$apps" in *"$uuid"*) ;; *) echo "$g"; return 0 ;; esac
  done
  return 1
}
run_arm() {  # arm gpu
  local arm=$1 g=$2 ck; ck=$(ckpt_of "$arm")
  if ! ds_done "$arm"; then
    CUDA_VISIBLE_DEVICES=$g "$PY" -m torch.distributed.run --nproc-per-node 1 --master-port $((29900 + g)) \
      -m apps.aunet.eval config=$ACFG harness.tasks=$TASKS validation=null \
      ckpt_dir="$ck" dump_dir="$OUT/$arm" harness.include_path=$INC $(extra_of "$arm") \
      > "$OUT/$arm.log" 2>&1
    ds_done "$arm" || { echo "$(date '+%F %T') $(hostname) FAILED downstream $arm"; touch "$OUT/$arm.FAILED"; return; }
    echo "$(date '+%F %T') $(hostname) DONE downstream $arm"
  fi
  if ! ho_done "$arm"; then
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$S/lingua" AUNET_ROOT=$H AUNET_TOK="$(tok_of "$arm")" \
      "$PY" "$HO/probes/bpb_windows_local.py" --family aunet --ckpt "$ck/consolidated" --tag "$arm" \
      --windows "$HO/bpb_windows.jsonl" --batch_size 2 --tok_path "$(tok_of "$arm")" \
      --out "$HO/bpb_results.jsonl" > "$HO/$arm.log" 2>&1 \
      && touch "$HO/$arm.done" \
      && echo "$(date '+%F %T') $(hostname) DONE heldout $arm $(tail -1 "$HO/$arm.log")" \
      || { echo "$(date '+%F %T') $(hostname) FAILED heldout $arm"; touch "$OUT/$arm.FAILED"; }
  fi
}

while :; do
  pending=0
  for arm in "${ARMS[@]}"; do
    ds_done "$arm" && ho_done "$arm" && continue
    [ -f "$OUT/$arm.FAILED" ] && continue                    # needs a human look; delete to retry
    pending=1
    ready "$arm" || continue
    mkdir "$OUT/.lock_$arm" 2>/dev/null || continue          # another host/worker owns it
    until g=$(free_gpu); do sleep 300; done
    echo "$(date '+%F %T') $(hostname) START $arm on GPU $g"
    ( run_arm "$arm" "$g"; rmdir "$OUT/.lock_$arm" ) &
    sleep 180                                                 # let the job claim its GPU before the next pick
  done
  [ "$pending" -eq 0 ] && break
  sleep 300
done
wait
echo "$(date '+%F %T') $(hostname) queue finished"

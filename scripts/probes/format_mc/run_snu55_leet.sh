#!/usr/bin/env bash
# nla_leet (+ clean) for the two main-table rows not covered on ece-agpu18: Transformer (Llama 1.8B paper ckpt)
# and BLT at the calibrated threshold 1.6093749 (official bytelatent, bs 1). snu55 policy: GPUs 2,3 only, and a
# worker takes a job only when its GPU has no compute process (same rule as ext_ci_snu55/worker.sh).
#   run_snu55_leet.sh <out_dir> [gpus="2 3"] [check_busy=1]
# 2026-10-04: user cleared GPUs 0,1 for this ("GPU0,1 써서 해결해"); run as <out> "0 0 1 1" 0 (two workers per GPU).
set -u
GPUS=${2:-2 3} CHECK=${3:-1}
A=/mnt/ssd2/hyun2/AUNet; S=$(cd "$(dirname "$0")" && pwd); O=$1; mkdir -p $O/logs
ITEMS=$(cd "$S/../../../reports/format_robustness/items" && pwd)
W=$A/runs/ext_ci_snu55
Q=$O/jobs.txt; [ -f $Q ] || printf "llama\nblt hellaswag\nblt arc_easy\nblt arc_challenge\nblt piqa\nblt boolq\n" > $Q
say(){ echo "$(date '+%F %T') $*" >> $O/queue.log; }
busy(){ local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1==g{print $2}')
        nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
pop(){ ( flock 9; l=$(head -n1 $Q); [ -n "$l" ] && tail -n +2 $Q > $Q.tmp && mv $Q.tmp $Q; echo "$l" ) 9>$Q.lock; }
run(){ local g=$1 m=$2 t=${3:-}
  if [ "$m" = llama ]; then
    cat > $O/llama.yaml <<EOF
name: "format_mc_llama_leet"
ckpt_dir: $A/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated
dump_dir: $O/llama
harness:
  log_samples: false
  limit: 2000
  tasks:
    - despace_mc
validation: null
generator:
  max_tokens: 4096
  dtype: bf16
EOF
    NV=$A/lingua/.venv/lib/python3.12/site-packages/nvidia     # this box's torch needs the venv's nvidia/*/lib
    ( cd $A/lingua && export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}" && \
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A/lingua FORMAT_ENTRY=apps.main.eval FORMAT_ITEMS=$ITEMS \
        FORMAT_CACHE=$O/llama.cache.jsonl FORMAT_VARIANTS=clean,nla_leet \
        MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29800 + g * 10 + RANDOM % 10)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
        .venv/bin/python -u $S/run_format_lingua.py config=$O/llama.yaml )
  else
    ( cd /tmp && CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$W/extra_site:$W/blt_official EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite \
        AUNET_LINGUA=$A/lingua HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 BLT_SUPPRESS_ATTN_ERROR=1 \
        /mnt/ssd2/hyun2/venvs/vllm011/bin/python -u $S/run_format_ext.py --family blt_official --threshold 1.6093749403933089 \
        --tasks $t --variants clean nla_leet --blt_weights $W/blt_weights --items_dir $ITEMS \
        --cache $O/blt1609_$t.cache.jsonl --out $O/blt1609_$t.json )
  fi; }
worker(){ local g=$1
  while true; do
    [ "$CHECK" = 1 ] && while busy $g; do sleep 120; done
    l=$(pop); [ -z "$l" ] && break
    say "start [$l] GPU$g"
    run $g $l > "$O/logs/${l// /_}.log" 2>&1
    say "end [$l] GPU$g exit=$?"
    sleep 20
  done; say "worker GPU$g done"; }
for g in $GPUS; do worker $g & sleep 30; done; wait; say ALL_DONE

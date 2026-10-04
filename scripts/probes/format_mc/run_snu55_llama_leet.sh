#!/usr/bin/env bash
# Transformer (Llama 1.8B paper ckpt) clean + nla_leet on snu55; waits until GPU <gpu> is empty (< 2 GB; a 12 GB gate OOMed when the BLT queue refilled the GPU).
#   run_snu55_llama_leet.sh <out_dir> <gpu>
set -u
A=/mnt/ssd2/hyun2/AUNet; S=$(cd "$(dirname "$0")" && pwd); O=$1 g=$2
ITEMS=$(cd "$S/../../../reports/format_robustness/items" && pwd)
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -gt 2000 ]; do sleep 30; done
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
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
cd $A/lingua
echo "$(date '+%F %T') start [llama] GPU$g" >> $O/queue.log
CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A/lingua FORMAT_ENTRY=apps.main.eval FORMAT_ITEMS=$ITEMS \
  FORMAT_CACHE=$O/llama.cache.jsonl FORMAT_VARIANTS=clean,nla_leet \
  MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29850 + g)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  .venv/bin/python -u $S/run_format_lingua.py config=$O/llama.yaml > $O/logs/llama.log 2>&1
echo "$(date '+%F %T') end [llama] GPU$g exit=$?" >> $O/queue.log

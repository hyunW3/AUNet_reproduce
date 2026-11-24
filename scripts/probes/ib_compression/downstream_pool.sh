#!/bin/bash
# 풀링 실험 3-arm 의 HS / ARC-E / BoolQ. GPU 는 비어 있는 0,2,3 만 쓴다(1 번은 타인 작업).
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
VENV=$L/lingua/.venv
LOG=$L/runs/pool_100M/downstream.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
cd $L/lingua
NV=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$L/lingua AUNET_TOK=$L/tokenizer/llama3/tokenizer.model TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/tr_ds_$$ TORCHINDUCTOR_CACHE_DIR=/tmp/ti_ds_$$
for t in "$@"; do
  C=$L/runs/pool_100M/$t/0000016000/consolidated
  OUT=$L/runs/pool_100M/ds_$t
  [ -f "$OUT/results.json" ] && { say "생략 $t"; continue; }
  [ -d "$C" ] || { say "건너뜀 $t"; continue; }
  say "시작 $t"
  CUDA_VISIBLE_DEVICES=${GPUS:-0,2,3} $VENV/bin/python -m torch.distributed.run \
    --nproc-per-node $(awk -F, '{print NF}' <<< "${GPUS:-0,2,3}") --master-port ${PORT:-29621} \
    -m apps.aunet.eval config=apps/aunet/configs/eval_pool_3task.yaml \
    ckpt_dir="$C" dump_dir="$OUT" >> "$LOG" 2>&1
  say "종료 $t results=$([ -f "$OUT/results.json" ] && echo yes || echo NO)"
done
say "DOWNSTREAM_POOL_DONE"

#!/bin/bash
# IB 측정 전량 실행 (held-out DCLM 320 윈도). GPU 2 만 비어 있으므로 순차 실행.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
O=$L/runs/ib_compression
LOG=$O/run.log
export PYTHONPATH=$L/lingua AUNET_TOK=$L/tokenizer/llama3/tokenizer.model TORCHDYNAMO_DISABLE=1
NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
EPS=0.001,0.002,0.005,0.01,0.02
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

go(){ # tag arm ckpt
  local tag=$1 arm=$2 ck=$3
  [ -f "$O/$tag.npz" ] && { say "생략 $tag"; return 0; }
  say "시작 $tag"
  $L/lingua/.venv/bin/python $L/scripts/probes/ib_compression/ib_extract.py \
     --arm "$arm" --ckpt "$ck" --out "$O/$tag.npz" --eps "$EPS" >> "$LOG" 2>&1
  say "종료 $tag exit=$? $([ -f "$O/$tag.npz" ] && echo ok || echo FAIL)"
}

go transformer_paper  llama   $L/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated
go bpebyte_paper      bpebyte $L/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
go bpebyte_retrain    bpebyte $L/runs/bpebyte_br_greedy_root_1.3B_a100x4/checkpoints/0000180000/consolidated
say "IB_ALL_DONE"

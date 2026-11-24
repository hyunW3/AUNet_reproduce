#!/bin/bash
# 30%/50% llama 체크포인트는 consolidated/ 가 없어서 run_milestone_eval_local.sh 의 사전검사에
# 걸려 즉시 exit 1 났다. consolidate 는 dcp_to_torch_save 라 CPU 작업이고 GPU 를 쓰지 않으므로,
# BLT 실행과 무관하게 먼저 돌린 뒤 평가를 큐에 넣는다. (~21GB 씩 디스크를 더 쓴다)
set -u
L=/mnt/ssd2/hyun2/AUNet
LOG=$L/reports_afterAAAIsub/milestone_evals.log
say(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
cd $L/lingua
NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
for step in 17877 29794; do
  pad=$(printf "%010d" $step); d=$L/runs/llama_1.8B_a100x4/checkpoints/$pad
  if [ -d "$d/consolidated" ]; then say "consolidated 이미 있음: $step"; continue; fi
  say "consolidate 시작: llama $step"
  .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$d'))" >> $LOG 2>&1
  say "consolidate 끝: $step -> $([ -d $d/consolidated ] && echo OK || echo 실패)"
done
say "consolidate 단계 완료"

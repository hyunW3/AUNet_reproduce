#!/bin/bash
# 빠진 마일스톤 평가 3건.
#   llama 17877(30%) / 29794(50%) : consolidated 가 어제 뒤늦게 생성돼 큐가 이미 지나갔다.
#   bpebyte 90000(50%)            : 원격에서 샤드만 받아와 consolidated 가 없어 사전검사에 걸렸다.
# GPU 4장이 비었으므로 llama 둘은 병렬, bpebyte 는 consolidate(CPU) 후 실행한다.
set -u
L=/mnt/ssd2/hyun2/AUNet
LOG=$L/reports_afterAAAIsub/milestone_evals.log
say(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }
cd $L/lingua
NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"

# bpebyte 90000 consolidate (CPU) — llama 평가와 병행
( d=$L/runs/bpebyte_br_greedy_root_1.3B_a100x4/checkpoints/0000090000
  if [ ! -d "$d/consolidated" ]; then
    say "consolidate 시작: bpebyte 90000"
    .venv/bin/python -c "
from lingua.checkpoint import consolidate_checkpoints
print(consolidate_checkpoints('$d'))" >> $LOG 2>&1
    say "consolidate 끝: bpebyte 90000 -> $([ -d $d/consolidated ] && echo OK || echo 실패)"
  fi ) &
CONS=$!

run_one(){  # arm step gpus port
  local arm=$1 step=$2 gpus=$3 pad=$(printf "%010d" $2) run out
  case $arm in llama) run=llama_1.8B_a100x4;; bpebyte) run=bpebyte_br_greedy_root_1.3B_a100x4;; esac
  out=$L/runs/$run/stage_$pad
  [ -f "$out/results.json" ] && { say "SKIP $arm $step"; return; }
  say "START $arm $step (GPU$gpus)"
  MASTER_PORT=$4 bash $L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh $arm $step $gpus \
      > $L/runs/$run/stage_${pad}.log 2>&1
  say "END   $arm $step exit=$? results=$([ -f $out/results.json ] && echo yes || echo NO)"
}

run_one llama 17877 0,1 29571 &
run_one llama 29794 2,3 29572 &
wait $CONS
wait
run_one bpebyte 90000 0,1,2,3 29573
say "MISSING_EVALS_DONE"

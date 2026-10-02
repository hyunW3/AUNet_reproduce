#!/bin/bash
# 비어 있던 마일스톤 평가 4건. 사용자님의 BLT robustness 실행과 겹쳐서 돌린다.
#
# GPU 0/1/3 은 BLT 가 18.6GB 씩 잡아 5.9GB 만 남아 평가(약 11GB)가 안 들어간다.
# GPU2 만 21.5GB 여유가 있어 거기서 순차로 돌린다. 남의 작업을 밀어내지 않는 것이 우선이라
# 여유가 THRESH 아래로 내려가면 다음 건을 시작하지 않고 기다린다.
set -u
L=/mnt/ssd2/hyun2/AUNet
LOG=$L/reports_afterAAAIsub/milestone_evals.log
GPU=${GPU:-2}
THRESH=${THRESH:-13000}
say(){ echo "$(date '+%F %T') $*" | tee -a $LOG; }

free_mb(){ nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU |
           awk -F', ' '{print $2-$1}'; }

run_one(){   # arm step
  local arm=$1 step=$2 pad=$(printf "%010d" $2)
  local run out
  case $arm in llama) run=llama_1.8B_a100x4;; bpebyte) run=bpebyte_br_greedy_root_1.3B_a100x4;; esac
  out=$L/runs/$run/stage_$pad
  if [ -f "$out/results.json" ]; then say "SKIP $arm $step (이미 있음)"; return; fi
  # 체크포인트 준비 확인 (consolidated 는 eval.py 가 없으면 자동 생성한다)
  if [ ! -d "$L/runs/$run/checkpoints/$pad" ]; then say "SKIP $arm $step (체크포인트 없음)"; return; fi
  while [ "$(free_mb)" -lt "$THRESH" ]; do sleep 180; done
  say "START $arm $step (GPU$GPU, 여유 $(free_mb)MiB)"
  bash $L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh $arm $step $GPU > $L/runs/$run/stage_${pad}.log 2>&1
  say "END   $arm $step exit=$? results=$([ -f $out/results.json ] && echo yes || echo NO)"
}

say "마일스톤 평가 큐 시작 (GPU$GPU)"
run_one llama 17877      # 30%
run_one llama 29794      # 50%
run_one llama 59589      # 100% (최종)
# BPEByte 90,000 은 전송이 끝난 뒤에만
for i in $(seq 1 120); do
  [ -f "$L/runs/bpebyte_br_greedy_root_1.3B_a100x4/checkpoints/0000090000/params.json" ] && break
  sleep 60
done
run_one bpebyte 90000    # 50%
say "ALL_DONE"

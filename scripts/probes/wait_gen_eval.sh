#!/bin/bash
# 생성 태스크(hellaswag_gen / arc_easy_gen) 두 건이 끝날 때까지 기다렸다가 가져와 병합하고
# viewer 를 다시 만든다. 이 둘은 문서당 비용이 판별형의 100배 수준이라 오래 걸린다
# (앞서 한 잡에 묶었을 때 29시간 동안 안 끝났다). 30분 주기로만 확인한다.
set -u
L=/mnt/ssd2/hyun2/AUNet; RUN=bpebyte_br_greedy_root_1.3B_a100x4; PAD=0000180000
R=/home/hwbae/AUNet/runs/$RUN
DONE=""
while [ "$DONE" != "hsgen aegen" ]; do
  DONE=""
  for t in hsgen aegen; do
    timeout 60 ssh -o BatchMode=yes ece-agpu11 "[ -f $R/stage_${PAD}_$t/results.json ]" 2>/dev/null \
      && DONE="${DONE:+$DONE }$t"
  done
  # 둘 다 죽었는데 결과도 없으면 무한 대기 방지
  if [ "$DONE" != "hsgen aegen" ]; then
    alive=$(timeout 60 ssh -o BatchMode=yes ece-agpu11 'pgrep -fc "[e]val_final_bpebyte_[ha]" 2>/dev/null || echo 0')
    if [ "${alive:-0}" -eq 0 ] 2>/dev/null; then echo "생성 잡이 모두 종료됐는데 결과 없음 (완료: ${DONE:-없음})"; break; fi
    sleep 1800
  fi
done
for t in hsgen aegen; do
  timeout 300 rsync -a --exclude='*.log' "ece-agpu11:$R/stage_${PAD}_$t/" "$L/runs/$RUN/stage_${PAD}_$t/" 2>/dev/null
done
python3 $L/scripts/probes/merge_final_eval.py
cd $L && python3 scripts/probes/build_stage_viewer.py 2>&1 | tail -5

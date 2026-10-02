#!/bin/bash
# 풀링 실험 arm 들의 despace 축. 100M 모델이라 1.3B 체크포인트용 러너 대신
# CKPT_DIR/OUT_DIR 오버라이드로 임의 체크포인트를 먹인다.
# despace 만 보는 이유: 풀링을 바꾸면 가장 먼저 움직일 축이고(재학습 1.3B 에서 무너진 축),
# 100M 에서 downstream 은 우연 수준에 가까워 신호가 약하다.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
LOG=$L/runs/pool_100M/despace.log
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
for t in "$@"; do
  C=$L/runs/pool_100M/$t/0000016000/consolidated
  OUT=$L/runs/pool_100M/despace_$t
  [ -f "$OUT/results.json" ] && { say "생략 $t"; continue; }
  [ -d "$C" ] || { say "건너뜀 $t (체크포인트 없음)"; continue; }
  say "시작 $t"
  DESPACE_TASKS="hellaswag,arc_easy,arc_challenge,piqa,boolq" \
  CFG=apps/aunet/configs/eval_despace_only_bpebyte.yaml \
  CKPT_DIR="$C" OUT_DIR="$OUT" MASTER_PORT=${PORT:-29611} \
    bash "$L/lingua/scripts/eval/robustness/run_robustness_local.sh" bpebyte 16000 "${GPUS:-1,2}" >> "$LOG" 2>&1
  say "종료 $t results=$([ -f "$OUT/results.json" ] && echo yes || echo NO)"
done
say "DESPACE_POOL_DONE"

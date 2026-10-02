#!/bin/bash
# 75%(135,000) 를 GPU2,3 에서 3번째 동시 평가로 돌린다.
# 전송/consolidate 는 이미 끝나 있고, 다른 두 건이 GPU0,1 과 0-3 을 쓰지만
# 카드당 9.3GB 라 24.5GB 안에 겹쳐 들어간다. CPU 는 64코어 중 load 8 이라 여유.
set -u
L=/mnt/ssd2/hyun2/AUNet
RUN=bpebyte_br_greedy_root_1.3B_a100x4
LOG=$L/reports_afterAAAIsub/milestone_evals.log
out=$L/runs/$RUN/stage_0000135000
if [ -f "$out/results.json" ]; then echo "$(date '+%F %T') 생략 135000 (결과 있음)" >> "$LOG"; exit 0; fi
echo "$(date '+%F %T') 평가 시작 0000135000 (GPU2,3) — 3건 동시" >> "$LOG"
MASTER_PORT=29592 bash "$L/lingua/scripts/eval/milestone/run_milestone_eval_local.sh" bpebyte 135000 2,3 \
    > "$L/runs/$RUN/stage_0000135000.log" 2>&1
echo "$(date '+%F %T') 평가 종료 0000135000 exit=$? results=$([ -f "$out/results.json" ] && echo yes || echo NO)" >> "$LOG"

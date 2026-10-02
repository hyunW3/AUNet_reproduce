#!/bin/bash
# 재학습본 1.3B robustness — 논문 학습본(runs/robustness_paper1p3b{,_ext}) 과 짝을 맞춘 실행.
#
# 왜 두 번 도는가: §12.1 의 5태스크 표는 config 두 개의 합이다.
#   base : hellaswag, arc_easy            (eval_robustness_*_local.yaml)
#   ext  : arc_challenge, piqa, boolq     (eval_robustness_*_local_ext.yaml)
# despace/pbp_mc 는 코드가 태스크를 하드코딩하므로 실행기에 환경변수로 넘겨야 한다.
#
# GPU 는 논문 실행과 같은 0,1,2 세 장을 쓴다(당시 헤더: "gpus=0,1,2"). 점수에는 영향이 없지만
# 비교에서 변수를 하나라도 줄인다. 3 번 카드는 비워 둔다.
#
# 체크포인트: llama 는 로컬에 있다(59,589 = 재학습 종료 스텝). bpebyte 재학습본 180,000 은
# 아직 이 박스에 없다 — ece-agpu11 이 지금 응답하지 않아 전송을 못 했다. 인자로 arm 을 골라
# llama 만 먼저 돌린다.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
GPUS=${GPUS:-0,1,2}
Q=$L/runs/robustness_retrain1p3b/queue.log
mkdir -p "$(dirname "$Q")" "$L/runs/robustness_retrain1p3b_ext"
say(){ echo "$(date '+%F %T') $*" | tee -a "$Q"; }

run(){  # arm step cfg_kind
  local arm=$1 step=$2 kind=$3
  local suffix out cfg pbp
  case "$kind" in
    base) suffix="";     pbp="hellaswag,arc_easy" ;;
    ext)  suffix="_ext"; pbp="arc_challenge,piqa,boolq" ;;
  esac
  out=$L/runs/robustness_retrain1p3b$suffix/$arm
  if [ -f "$out/results.json" ]; then say "생략 $arm/$kind (결과 있음)"; return 0; fi
  case "$arm" in
    llama)   cfg=apps/main/configs/eval_robustness_llama_local ;;
    bpebyte) cfg=apps/aunet/configs/eval_robustness_bpebyte_local ;;
    aunet)   cfg=apps/aunet/configs/eval_robustness_aunet2_local ;;
  esac
  [ "$kind" = ext ] && cfg="${cfg}_ext"
  say "시작 $arm/$kind (GPU$GPUS)"
  DESPACE_TASKS="$pbp" PBP_MC_TASKS="$pbp" CFG="${cfg}.yaml" OUT_DIR="$out" \
    bash "$L/lingua/scripts/eval/robustness/run_robustness_local.sh" "$arm" "$step" "$GPUS" \
    > "$L/runs/robustness_retrain1p3b$suffix/$arm.log" 2>&1
  say "종료 $arm/$kind exit=$? results=$([ -f "$out/results.json" ] && echo yes || echo NO)"
}

for spec in "$@"; do
  set -- ${spec//:/ }   # arm:step
  run "$1" "$2" base
  run "$1" "$2" ext
done
say "RETRAIN_ROBUSTNESS_DONE"

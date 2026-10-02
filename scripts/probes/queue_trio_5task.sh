#!/bin/bash
# 논문 프로토콜(5개 태스크) 대 통일 프로토콜(2개 태스크) 비교 — matched 3개 모델.
#
# 바꾸는 것은 태스크 구성 하나뿐이다. limit 은 2000 으로 통일 재측정과 동일하게 두어,
# "논문이 태스크를 다르게 고른 것"의 효과만 분리해서 본다(논문 despace 는 500 이었지만
# 항목 수까지 같이 바꾸면 두 요인이 섞인다).
#
# 커버리지는 축마다 다르다 — 코드가 지원하는 만큼만 넓힌다:
# 네 축 x 5개 태스크 전부. PIQA·WinoGrande 교란을 구현해서 커버리지를 맞췄다(config 주석 참조).
#
# GPU 는 BLT/H-Net 이 순차로 반납하는 것을 기다렸다 잡는다(모델당 ~8GB, BLT 는 18.6GB 를 쥐고 있다).
set -u
L=/mnt/ssd2/hyun2/AUNet
M=$L/main/main/1.3B
OUT=$L/runs/robustness_paper5task
NEED_MB=${NEED_MB:-11000}
mkdir -p "$OUT"

export DESPACE_TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande"
export PBP_MC_TASKS="hellaswag,arc_easy,arc_challenge,piqa,winogrande"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  awk -F', ' -v need="$NEED_MB" -v busy="$1" '
    { if (($3-$2) > need && index(busy, "["$1"]")==0) { print $1; exit } }'
}

BUSY=""
declare -A PIDS
launch() {   # arm ckpt step cfg gpu port
  local arm=$1 ck=$2 step=$3 cfg=$4 gpu=$5 port=$6
  echo "START $arm gpu=$gpu $(date '+%F %T')"
  ( CKPT_DIR="$ck" OUT_DIR="$OUT/$arm" CFG="$cfg" MASTER_PORT=$port \
      bash $L/lingua/scripts/eval/robustness/run_robustness_local.sh "$arm" "$step" "$gpu" > "$OUT/$arm.log" 2>&1
    echo "END   $arm exit=$? $(date '+%F %T')" ) &
  PIDS[$arm]=$!
}

i=0
for spec in \
  "llama|$M/llama_1.8B_paper/checkpoints/0000060000/consolidated|60000|apps/main/configs/eval_robustness_llama_5task_local.yaml" \
  "aunet|$M/aunet2_1.3B/checkpoints/0000180000/consolidated|180000|apps/aunet/configs/eval_robustness_aunet2_5task_local.yaml" \
  "bpebyte|$M/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated|180000|apps/aunet/configs/eval_robustness_bpebyte_5task_local.yaml" ; do
  IFS='|' read -r arm ck step cfg <<< "$spec"
  if [ -f "$OUT/$arm/results.json" ]; then echo "SKIP $arm"; continue; fi
  g=""
  while [ -z "$g" ]; do g=$(free_gpu "$BUSY"); [ -z "$g" ] && sleep 300; done
  BUSY="$BUSY[$g]"
  launch "$arm" "$ck" "$step" "$cfg" "$g" $((29650+i))
  i=$((i+1))
  sleep 60
done
wait
echo "TRIO5_ALL_DONE $(date '+%F %T')"

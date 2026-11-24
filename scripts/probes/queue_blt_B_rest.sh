#!/bin/bash
# BLT B-arm(θ=1.46875)의 남은 두 축. gpu3_chain 에서 떼어낸 것을 GPU 가 진짜 빌 때까지 기다렸다 돌린다.
# BLT 는 ~18.6GB 를 쓰므로, 5-태스크 trio 평가(~8GB)와 같은 카드에 겹치면 OOM 이 난다.
set -u
A=/mnt/ssd2/hyun2/AUNet
OUT=$A/reports_afterAAAIsub/blt_rob2000
NEED_MB=19500
mkdir -p "$OUT"
free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  awk -F', ' -v need="$NEED_MB" '($3-$2) > need {print $1; exit}'
}
port=28890
for AX in despace typo; do
  T="B_${AX}_n2000"
  if [ -f "$OUT/$T.json" ] && grep -q '"_minutes"' "$OUT/$T.json" 2>/dev/null; then echo "SKIP $T"; continue; fi
  G=""; while [ -z "$G" ]; do G=$(free_gpu); [ -z "$G" ] && sleep 300; done
  echo "START $T gpu=$G $(date '+%F %T')"
  ( cd /mnt/ssd2/hyun2/lzspacebyte/blt && BLT_SUPPRESS_ATTN_ERROR=1 TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES=$G MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 WORLD_SIZE=1 \
      LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
      .venv/bin/python $A/scripts/probes/blt_robustness_native.py \
        --threshold 1.46875 --attn_impl xformers --batch_size 1 --limit 2000 \
        --axes "$AX" --out "$OUT/$T.json" > "$OUT/$T.log" 2>&1 )
  echo "END   $T exit=$? $(date '+%F %T')"
  port=$((port+1))
done
echo "BLT_B_REST_DONE $(date '+%F %T')"

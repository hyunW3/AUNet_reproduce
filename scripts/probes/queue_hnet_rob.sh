#!/bin/bash
# H-Net 1-stage / 2-stage XL robustness on the unified suite, n=2000.
#
# H-Net needs ~5.4 GB and the four BLT n=2000 jobs hold ~18.6 GB each, so this waits for a card
# to actually free instead of racing them into an OOM (which is how the first attempt died).
# Unlike BLT, H-Net's harness masks padded rows, so batching should be safe -- --check_batch
# measures that instead of assuming it.
set -u
HP=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
OUT=/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/hnet_rob
N=${N:-2000}
BS=${BS:-8}
NEED_MB=${NEED_MB:-9000}
mkdir -p "$OUT"

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
  awk -F', ' -v need="$NEED_MB" '($3-$2) > need {print $1; exit}'
}

wait_gpu() {
  local g=""
  while :; do
    g=$(free_gpu)
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 300
  done
}

for V in hnet_1stage_XL hnet_2stage_XL; do
  TAG="${V}_n${N}"
  [ -f "$OUT/$TAG.json" ] && grep -q '"_minutes"' "$OUT/$TAG.json" 2>/dev/null && { echo "SKIP $TAG"; continue; }
  G=$(wait_gpu)
  echo "START $TAG gpu=$G $(date '+%F %T')"
  ( cd /mnt/ssd2/hyun2/hnet && HNET_REPO=/mnt/ssd2/hyun2/hnet CUDA_VISIBLE_DEVICES=$G \
      TOKENIZERS_PARALLELISM=false \
      "$HP" /mnt/ssd2/hyun2/AUNet/scripts/probes/hnet_robustness.py \
        --variant "$V" --limit "$N" --batch_size "$BS" --check_batch \
        --out "$OUT/$TAG.json" > "$OUT/$TAG.log" 2>&1 )
  echo "END   $TAG exit=$? $(date '+%F %T')"
done
echo "HNET_ALL_DONE $(date '+%F %T')"

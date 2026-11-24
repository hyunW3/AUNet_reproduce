#!/bin/bash
# GPU3 전용 체인: H-Net 을 먼저 돌리고, 중단해 둔 BLT B-arm 두 축을 이어서 돌린다.
#
# 왜 이 순서인가: H-Net(배치 가능, 몇 시간)을 BLT 뒤에 두면 전체 완료가 13h+4h 로 늘어나지만,
# 앞에 두면 총 makespan 은 그대로이면서 H-Net 결과가 훨씬 일찍 나온다. BLT B_despace/B_typo 는
# 40분만 돌다 중단됐으므로 버리는 계산량이 작다.
set -u
HP=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
A=/mnt/ssd2/hyun2/AUNet
HOUT=$A/reports_afterAAAIsub/hnet_rob
BOUT=$A/reports_afterAAAIsub/blt_rob2000
N=2000
mkdir -p "$HOUT" "$BOUT"

for V in hnet_1stage_XL hnet_2stage_XL; do
  T="${V}_n${N}"
  if [ -f "$HOUT/$T.json" ] && grep -q '"_minutes"' "$HOUT/$T.json" 2>/dev/null; then echo "SKIP $T"; continue; fi
  echo "START $T $(date '+%F %T')"
  ( cd /mnt/ssd2/hyun2/hnet && HNET_REPO=/mnt/ssd2/hyun2/hnet CUDA_VISIBLE_DEVICES=3 \
      TOKENIZERS_PARALLELISM=false \
      "$HP" $A/scripts/probes/hnet_robustness.py --variant "$V" --limit $N --batch_size 8 \
        --check_batch --out "$HOUT/$T.json" > "$HOUT/$T.log" 2>&1 )
  echo "END   $T exit=$? $(date '+%F %T')"
done

port=28880
for AX in despace typo; do
  T="B_${AX}_n${N}"
  if [ -f "$BOUT/$T.json" ] && grep -q '"_minutes"' "$BOUT/$T.json" 2>/dev/null; then echo "SKIP $T"; continue; fi
  echo "START $T $(date '+%F %T')"
  ( cd /mnt/ssd2/hyun2/lzspacebyte/blt && BLT_SUPPRESS_ATTN_ERROR=1 TOKENIZERS_PARALLELISM=false \
      CUDA_VISIBLE_DEVICES=3 MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 WORLD_SIZE=1 \
      LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
      .venv/bin/python $A/scripts/probes/blt_robustness_native.py \
        --threshold 1.46875 --attn_impl xformers --batch_size 1 --limit $N \
        --axes "$AX" --out "$BOUT/$T.json" > "$BOUT/$T.log" 2>&1 )
  echo "END   $T exit=$? $(date '+%F %T')"
  port=$((port+1))
done
echo "GPU3_CHAIN_DONE $(date '+%F %T')"

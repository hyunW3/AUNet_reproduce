#!/bin/bash
# BLT-1B robustness at n=2000, both thresholds, packed across 4 GPUs.
#
# BLT must run at batch_size 1 (padding goes through the entropy patcher and moves the
# segmentation), so wall-clock scales linearly with n. At n=500 the axes cost, per arm:
#   noise 145 min · typo 83 min · despace 116 min · pbp 18 min
# so n=2000 is ~4x that: 2880 GPU-min for the two arms. Packing 8 jobs onto 4 cards with the
# two long noise axes on their own cards balances to ~13 h wall-clock.
#
#   GPU0: A-noise  + A-pbp        GPU2: A-despace + A-typo
#   GPU1: B-noise  + B-pbp        GPU3: B-despace + B-typo
set -u
B=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT=/mnt/ssd2/hyun2/AUNet/reports_afterAAAIsub/blt_rob2000
N=${N:-2000}
mkdir -p "$OUT"

# arm  -> theta
th() { case $1 in A) echo 1.335442066192627;; B) echo 1.46875;; esac; }

run_chain() {           # gpu  port_base  arm  axis1[,axis2...]
  local gpu=$1 port=$2 arm=$3; shift 3
  for ax in "$@"; do
    local tag="${arm}_${ax}_n${N}"
    if [ -f "$OUT/$tag.json" ] && grep -q '"_minutes"' "$OUT/$tag.json" 2>/dev/null; then
      echo "SKIP $tag"; continue
    fi
    echo "START $tag gpu=$gpu $(date '+%F %T')"
    ( cd "$B" && BLT_SUPPRESS_ATTN_ERROR=1 TOKENIZERS_PARALLELISM=false \
        CUDA_VISIBLE_DEVICES=$gpu \
        MASTER_ADDR=127.0.0.1 MASTER_PORT=$port RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
        .venv/bin/python /mnt/ssd2/hyun2/AUNet/scripts/probes/blt_robustness_native.py \
          --threshold "$(th $arm)" --attn_impl xformers --batch_size 1 --limit "$N" \
          --axes "$ax" --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1 )
    echo "END   $tag exit=$? $(date '+%F %T')"
    port=$((port+1))
  done
}

run_chain 0 28840 A noise pbp     &
run_chain 1 28850 B noise pbp     &
run_chain 2 28860 A despace typo  &
run_chain 3 28870 B despace typo  &
wait
echo "ALL_DONE $(date '+%F %T')"

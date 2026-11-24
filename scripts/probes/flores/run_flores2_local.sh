#!/bin/bash
# FLORES-200 X->en generation for the SECOND language set: fi hu sv tr ru vi.
# Chosen on morphological type to test the §6 BPB result at the generation level —
# agglutinative (fi/hu/tr) should show the largest BPEByte-over-AU-Net gap and isolating (vi) the
# smallest or a sign flip, mirroring ΔBPB (+0.404 hu / +0.330 fi ... −0.069 vi).
# Protocol identical to the first set: dev, n=200, 5-shot from devtest, greedy, 288-byte budget.
ROOT=/mnt/ssd2/hyun2/AUNet
LOGS=$ROOT/runs/_logs; mkdir -p "$LOGS"
cd $ROOT/lingua
NV=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | paste -sd: -)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # gpu tag app cfg port
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -m torch.distributed.run --nproc-per-node 1 \
      --master-port $5 -m $3 config=$4 > "$LOGS/flores2_$2.log" 2>&1
  echo "  $2 rc=$?"
}
( run 0 llama      apps.main.eval  apps/main/configs/eval_flores2_llama_local.yaml       29911 ) &
( run 2 aunet2     apps.aunet.eval apps/aunet/configs/eval_flores2_aunet2_local.yaml     29913 ) &
( run 3 bpebyte_rg apps.aunet.eval apps/aunet/configs/eval_flores2_bpebyte_rg_local.yaml 29914 ) &
wait
echo "=== FLORES set 2 done $(date -Is)"

#!/bin/bash
# WMT X->en for the matched trio, same protocol as the FLORES runs (5-shot, greedy,
# max_gen_toks=96, n=200). FLORES vs WMT then isolates DOMAIN: encyclopedic/wiki vs news.
# Existing eval_gen WMT numbers are limit=50 (±1.2 BLEU noise) and have no BPEByte arm.
ROOT=/mnt/ssd2/hyun2/AUNet
LOGS=$ROOT/runs/_logs
mkdir -p "$LOGS"
cd $ROOT/lingua
NV=.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | paste -sd: -)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # gpu arm app cfg
  CUDA_VISIBLE_DEVICES=$1 .venv/bin/python -m torch.distributed.run --nproc-per-node 1 \
      --master-port $5 -m $3 config=$4 > "$LOGS/wmt_$2.log" 2>&1
  echo "  $2 rc=$?"
}
( run 0 llama      apps.main.eval  apps/main/configs/eval_wmt_llama_local.yaml       29901 ) &
( run 2 aunet2     apps.aunet.eval apps/aunet/configs/eval_wmt_aunet2_local.yaml     29903 ) &
( run 3 bpebyte_rg apps.aunet.eval apps/aunet/configs/eval_wmt_bpebyte_rg_local.yaml 29904 ) &
wait
echo "=== WMT done $(date -Is)"

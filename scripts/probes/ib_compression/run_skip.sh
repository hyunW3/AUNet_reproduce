#!/bin/bash
# U-Net skip 가설 측정 (held-out DCLM 320 윈도). GPU 2.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
export PYTHONPATH=$L/lingua AUNET_TOK=$L/tokenizer/llama3/tokenizer.model TORCHDYNAMO_DISABLE=1
NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
$L/lingua/.venv/bin/python $L/scripts/probes/ib_compression/ib_skip.py \
    --out $L/runs/ib_compression/skip.json >> $L/runs/ib_compression/skip.log 2>&1
echo "SKIP_DONE exit=$?" >> $L/runs/ib_compression/skip.log

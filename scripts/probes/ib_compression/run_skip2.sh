#!/bin/bash
# skip / 풀링 / random-init 측정 전량 (320 윈도). eps 는 차원 보정(sqrt 규칙) 적용.
set -uo pipefail
L=/mnt/ssd2/hyun2/AUNet
export PYTHONPATH=$L/lingua AUNET_TOK=$L/tokenizer/llama3/tokenizer.model TORCHDYNAMO_DISABLE=1
NV=$L/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU:-2}
LOG=$L/runs/ib_compression/skip3.log
P=$L/lingua/.venv/bin/python
$P $L/scripts/probes/ib_compression/ib_skip.py \
    --out $L/runs/ib_compression/trained_calib.json >> $LOG 2>&1
echo "TRAINED_DONE exit=$?" >> $LOG
$P $L/scripts/probes/ib_compression/ib_skip.py --random \
    --out $L/runs/ib_compression/random_calib.json >> $LOG 2>&1
echo "RANDOM_DONE exit=$?" >> $LOG

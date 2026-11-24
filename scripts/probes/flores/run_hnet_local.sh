#!/bin/bash
# H-Net FLORES-200 X->en, dev split, n=200, 5-shot greedy, 288-byte budget.
#
# Runs on the LOCAL box: snu233 cannot build mamba-ssm (CUDA 13.0 toolkit vs a cu128 torch),
# while the conda `spacebyte` env here already has mamba-ssm 2.3.1 + causal-conv1d + flash-attn.
# Waits for any running lingua eval to release the GPUs first -- H-Net 2stage XL needs ~3.2 GB
# of weights and the box only has ~3.8 GB free while one is in flight.
#
#   bash run_hnet_local.sh            # wait, then run both variants
ROOT=/mnt/ssd2/hyun2/AUNet
PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
OUT=$ROOT/reports_afterAAAIsub/flores
mkdir -p "$OUT" "$ROOT/runs/_logs"

echo "waiting for GPUs (lingua eval in flight)... $(date -Is)"
while pgrep -f "apps.aunet.eval|apps.main.eval" > /dev/null; do sleep 60; done
sleep 30
echo "GPUs free at $(date -Is)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() {  # gpu model langs tag
  CUDA_VISIBLE_DEVICES=$1 nohup $PY $ROOT/scripts/probes/flores/hnet_flores.py \
      --model $2 --repo /mnt/ssd2/hyun2/hnet \
      --config_path /mnt/ssd2/hyun2/hnet/configs/$2.json \
      --split dev --tasks_dir $ROOT/lingua/eval_tasks --langs $3 \
      --limit 200 --batch_size 8 --out $OUT/$4.json \
      > $ROOT/runs/_logs/flores_$4.log 2>&1 &
}
run 0 hnet_2stage_XL de,fr,es hnet2s_dev_a
run 1 hnet_2stage_XL it,nl,ko hnet2s_dev_b
run 2 hnet_1stage_XL de,fr,es hnet1s_dev_a
run 3 hnet_1stage_XL it,nl,ko hnet1s_dev_b
wait
echo "H-Net FLORES done $(date -Is)"

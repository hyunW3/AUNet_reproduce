#!/bin/bash
# run a python script in the lingua venv with the std_bench environment (GPU via $GPU, default none)
VENV=/mnt/ssd2/hyun2/AUNet/lingua/.venv
NV=$VENV/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export PYTHONPATH=/mnt/ssd2/hyun2/AUNet/lingua AUNET_ROOT=/mnt/ssd2/hyun2/AUNet AUNET_LINGUA=/mnt/ssd2/hyun2/AUNet/lingua
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-}
cd /mnt/ssd2/hyun2/AUNet/lingua
exec $VENV/bin/python "$@"

#!/bin/bash
# run a python script in the BLT or H-Net environment of run_std_bench.sh: runext.sh blt|hnet script args...
FAM=$1; shift
A=/mnt/ssd2/hyun2/AUNet
export AUNET_ROOT=$A AUNET_LINGUA=$A/lingua EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU:-}
if [ "$FAM" = blt ]; then
  W=$A/runs/ext_ci_snu55; PY=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
  export PYTHONPATH=$W/extra_site:$W/blt_official
else
  PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
  export HNET_REPO=/mnt/ssd2/hyun2/hnet
fi
exec $PY "$@"

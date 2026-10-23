#!/bin/bash
# NUMCoT generation on snu188 (RTX A5000), tree /mnt/ssd2/hyunw3/AUNet, venv = conda env aunet_leak (torch 2.8 cu126,
# xformers 0.0.32) + system-site venv with mamba-ssm / causal-conv1d / flash-attn for H-Net.
# usage: bash run_snu188.sh <family> <gpu> [extra gen_items.py args, e.g. --limit 40 --out ...]
A=/mnt/ssd2/hyunw3/AUNet; D=$A/reports_afterNAACL/numcot; E=$A/ext; PY=$A/venv/bin/python; [ "$1" = hnet ] && PY=$A/venv_hnet/bin/python
fam=$1; gpu=$2; shift 2
mkdir -p $D/out
export AUNET_ROOT=$A HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 RUN_EXT_DIR=$A/scripts/probes/ext_ci EVAL_SUITE=$E/eval_suite \
  HNET_REPO=$E/hnet HF_HUB_CACHE=$E/hf_hub CUDA_VISIBLE_DEVICES=$gpu
case $fam in
  blt)  export PYTHONPATH=$E/blt_official:$E/eval_suite; unset BLT_SUPPRESS_ATTN_ERROR ;;
  hnet) export PYTHONPATH=$E/hnet:$E/eval_suite ;;
  *)    export PYTHONPATH=$A/lingua ;;
esac
cd /tmp && exec $PY $D/scripts/gen_items.py --family $fam --items $D/items.jsonl --out $D/out/$fam.jsonl "$@"

#!/usr/bin/env bash
# run_snu55.sh <gpu> <model: llama|aunet|bpebyte|blt|hnet> [files...]
# snu55 wrapper around run_node.sh: LC root = /mnt/ssd2/hyun2/bltweak_lc (symlinks to the AU-Net ckpts,
# runs/ext_ci_snu55 BLT weights + repo, eval_suite; run_longctx.py / ext_models.py copied from ece-agpu18).
# Interpreters: lingua venv (trio), vllm011 + extra_site (official BLT, xformers), spacebyte conda (H-Net).
# env passthrough: SHARD, LIMIT, RESDIR
R=/mnt/ssd2/hyun2/bltweak_lc
export TRIO_PY=/mnt/ssd2/hyun2/AUNet/lingua/.venv/bin/python
export BLT_PY=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
export HNET_PY=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python HNET_REPO=/mnt/ssd2/hyun2/hnet
[ "$2" = blt ] && export EXT_PYTHONPATH=/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/extra_site
GPU=$1 MODEL=$2
shift 2
exec bash "$R/bltweak/run_node.sh" "$R" "$GPU" "$MODEL" "$@"

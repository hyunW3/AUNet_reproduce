#!/usr/bin/env bash
# snu20 paths for run_boolq_both.sh (BLT and H-Net jobs; the matched trio stays on snu55/gpusvr0908, whose checkpoint
# params point at /mnt/ssd2 tokenizer paths). BLT weights md5 = runs/ext_ci_snu55/blt_weights (73cd5016 / bbc9ae7d).
#   run_boolq_both_snu20.sh <out_dir> [gpus] [filter]
R=/mnt/ssd/hyun/aunet_boolq
E=/mnt/ssd/hyun/aunet_lc/ext
export LW=$R/lingua ITEMS=$R/items X=$E BLTPATH=$E/blt_official BLTPY=$E/venv_blt/bin/python
export HNETPY=$E/venv/bin/python HNET_REPO=/mnt/ssd/hyun/visual/hnet EVAL_SUITE=/mnt/ssd/hyun/visual/eval_suite
export HF_HUB_CACHE=/mnt/ssd/hyun/visual/cache/hf_hub
exec bash $R/scripts/probes/boolq_both/run_boolq_both.sh "$1" "${2:-0 1 2 3}" "${3:-^(blt|hnet)}"

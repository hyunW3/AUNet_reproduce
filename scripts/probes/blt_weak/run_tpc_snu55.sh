#!/usr/bin/env bash
# run_tpc_snu55.sh <gpu> <echo.jsonl> <out.jsonl> [shard i/n]
# Context-side segmentation transplant (blt_transplant_ctx.py) with snu55's official BLT env
# (vllm011 python + runs/ext_ci_snu55/extra_site, xformers); LC root /mnt/ssd2/hyun2/bltweak_lc.
R=/mnt/ssd2/hyun2/bltweak_lc
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/extra_site \
  BLT_REPO=$R/ext/blt_official EVAL_SUITE=$R/ext/eval_suite LC_SCRIPTS=$R/scripts/longctx HF_HUB_OFFLINE=1 \
  /mnt/ssd2/hyun2/venvs/vllm011/bin/python "$HERE/blt_transplant_ctx.py" --ckpt $R/ext/blt_weights \
  --echo "$2" --out "$3" ${4:+--shard $4}

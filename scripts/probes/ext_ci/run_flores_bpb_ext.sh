#!/bin/bash
# Per-language held-out BPB for the external byte models (BLT-1B official, H-Net 1/2-stage XL)
# on the FLORES dev source windows. FLORES is parallel, so window k is the SAME content in every
# language: content is controlled and only the language varies.
#
# BLT runs per BLT_execution_guide.md: official bytelatent + xformers (all windows) + batch 1 +
# released threshold. Verified beforehand with check_blt_window.py (4.075 B/patch on DCLM).
# snu55 policy: GPUs 2 and 3 only.
ROOT=/mnt/ssd2/hyun2/AUNet
W=$ROOT/runs/ext_ci_snu55
OUT=$ROOT/reports_afterAAAIsub/flores/bpb_ext
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
# Two interpreters: BLT needs the vllm011 build (torch 2.8 + xformers 0.0.32); H-Net needs
# flash-attn + mamba-ssm, which only the conda `spacebyte` env has. `extra_site` is a vllm011-only
# supplement and breaks the conda env (pandas built for another python), so it is BLT-only.
PY_BLT=/mnt/ssd2/hyun2/venvs/vllm011/bin/python
PY_HNET=/home/hyunwoong/miniconda3/envs/spacebyte/bin/python
export AUNET_LINGUA=$ROOT/lingua EVAL_SUITE=/mnt/ssd2/hyun2/eval_suite HNET_REPO=/mnt/ssd2/hyun2/hnet
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
cd $ROOT
run() {  # gpu family tag lang [extra...]
  local gpu=$1 fam=$2 tag=$3 lg=$4; shift 4
  local py=$PY_HNET pp=""
  if [ "$fam" = "blt_official" ]; then py=$PY_BLT; pp=$W/extra_site:$W/blt_official; fi
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$pp BLT_SUPPRESS_ATTN_ERROR=1 $py $ROOT/scripts/probes/ext_ci/run_ext.py \
     --family $fam --axis bpb --windows $ROOT/data/flores_bpb/$lg.jsonl \
     --blt_weights $W/blt_weights --out "$OUT/${tag}_${lg}.json" "$@" \
     > "$LOGS/flbpbext_${tag}_${lg}.log" 2>&1
  echo "  ${tag}/${lg} rc=$?"
}
( for lg in de nl it fr es en; do run 2 blt_official blt_official $lg; done ) &
( for lg in de nl it fr es en; do run 3 hnet hnet1s $lg --hnet_model hnet_1stage_XL; done
  for lg in de nl it fr es en; do run 3 hnet hnet2s $lg --hnet_model hnet_2stage_XL; done ) &
wait
echo "=== external per-language BPB done $(date -Is)"

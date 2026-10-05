#!/usr/bin/env bash
# run_aux.sh <lc_root> <gpu> <job: pp_blt|pp_trio|latency|transplant>   (env SHARD=i/n for transplant)
#   pp_blt   BLT patch spans (echo/insert) + windows patch counts & determinism -> patches/b2_blt_1b*.jsonl
#   pp_trio  same for BPEByte / AU-Net, token counts for Llama                 -> patches/b2_<tag>*.jsonl
#   latency  bs-1 scoring latency per OOD domain, all four models sequentially -> latency/<tag>.jsonl
# env: TRIO_PY (default ~/AUNet/lingua/.venv/bin/python)
set -u
R=$1 GPU=$2 JOB=$3
B=$R/bltweak D=$R/bltweak/data E=$R/ext
TPY=${TRIO_PY:-$HOME/AUNet/lingua/.venv/bin/python}
mkdir -p $B/patches $B/latency
cd $R/lingua
blt(){ CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= BLT_REPO=$E/blt_official EVAL_SUITE=$E/eval_suite HF_HUB_OFFLINE=1 \
       LC_SCRIPTS=$R/scripts/longctx $E/venv_blt/bin/python "$@"; }
trio(){ CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$R AUNET_TOK=$R/tokenizer/llama3/tokenizer.model PYTHONPATH=$R/lingua \
       LC_SCRIPTS=$R/scripts/longctx $TPY "$@"; }
TRIO="bpebyte:aunet:byte_greedyroot aunet:aunet:aunet_static llama:subword:subword_llama"
case $JOB in
  pp_blt)
    blt $B/patch_probe.py --family blt --ckpt $E/blt_weights --tag blt_1b --windows $D/ood_windows.jsonl \
        --out $B/patches/b2w_blt_1b.jsonl
    blt $B/patch_probe.py --family blt --ckpt $E/blt_weights --tag blt_1b --data $D/echo.jsonl $D/insert.jsonl \
        --out $B/patches/b2_blt_1b.jsonl ;;
  pp_trio)
    for m in $TRIO; do IFS=: read ck fam tag <<< "$m"
      trio $B/patch_probe.py --family $fam --ckpt $R/ckpt/$ck --tag $tag --windows $D/ood_windows.jsonl \
          --out $B/patches/b2w_$tag.jsonl
      [ $fam = aunet ] && trio $B/patch_probe.py --family aunet --ckpt $R/ckpt/$ck --tag $tag \
          --data $D/echo.jsonl $D/insert.jsonl --out $B/patches/b2_$tag.jsonl
    done ;;
  pp3_blt)
    blt $B/patch_probe.py --family blt --ckpt $E/blt_weights --tag blt_1b --data $D/probe3.jsonl \
        --out $B/patches/p3_blt_1b.jsonl ;;
  pp3_trio)
    trio $B/patch_probe.py --family aunet --ckpt $R/ckpt/bpebyte --tag byte_greedyroot --data $D/probe3.jsonl \
        --out $B/patches/p3_byte_greedyroot.jsonl ;;
  transplant)   # E2; CUDA_LAUNCH_BLOCKING avoids the intermittent async illegal-memory crash; resumable
    for i in $(seq 1 20); do
      CUDA_LAUNCH_BLOCKING=1 blt $B/blt_transplant.py --ckpt $E/blt_weights --echo $D/echo.jsonl \
          --out $B/results/transplant_blt_1b${SHARD:+_s${SHARD/\//of}}.jsonl ${SHARD:+--shard $SHARD} \
          && break
    done ;;
  latency)
    blt $B/latency.py --family blt --ckpt $E/blt_weights --tag blt_1b --windows $D/ood_windows.jsonl \
        --out $B/latency/blt_1b.jsonl
    for m in $TRIO; do IFS=: read ck fam tag <<< "$m"
      trio $B/latency.py --family $fam --ckpt $R/ckpt/$ck --tag $tag --windows $D/ood_windows.jsonl \
          --out $B/latency/$tag.jsonl
    done ;;
esac
echo "AUX_DONE $JOB $(date +%F_%T)"

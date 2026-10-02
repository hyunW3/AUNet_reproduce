#!/usr/bin/env bash
# run_patch_probe.sh <lc_root> <gpu> -- patch spans for BLT (entropy model only), BPEByte, AU-Net
# over the probe files; writes <lc_root>/bltweak/patches/<tag>.jsonl
# env: PP_ONLY=blt|trio (default both), TRIO_PY
R=$1 GPU=$2
B=$R/bltweak D=$R/bltweak/data O=$R/bltweak/patches
mkdir -p "$O"
cd "$R/lingua"
F="$D/dist512.jsonl $D/mkniah.jsonl $D/sniah_nc.jsonl $D/count.jsonl $D/copy.jsonl $D/hashhop.jsonl"
[ "${PP_ONLY:-}" != trio ] && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH= BLT_REPO=$R/ext/blt_official $R/ext/venv_blt/bin/python \
  $B/patch_probe.py --family blt --ckpt $R/ext/blt_weights --tag blt_1b --data $F \
  --out $O/blt_1b.jsonl > $O/blt.log 2>&1 &
[ "${PP_ONLY:-}" != blt ] && for m in bpebyte:byte_greedyroot aunet:aunet_static; do
  CUDA_VISIBLE_DEVICES=$GPU AUNET_ROOT=$R AUNET_TOK=$R/tokenizer/llama3/tokenizer.model PYTHONPATH=$R/lingua \
    LC_SCRIPTS=$R/scripts/longctx ${TRIO_PY:-$HOME/AUNet/lingua/.venv/bin/python} $B/patch_probe.py \
    --family aunet --ckpt $R/ckpt/${m%%:*} --tag ${m##*:} --data $F --out $O/${m##*:}.jsonl > $O/${m%%:*}.log 2>&1 &
done
wait
echo "PP_DONE $(date +%F_%T)" >> $O/blt.log

#!/usr/bin/env bash
# format_mc launcher for ece-agpu18 (GPUs 6/7 per the node rule: 0,5,6,7 + the 4,7 grant; 4,5 left to
# the long-context suite). Resumable: every unique score is cached, so re-running a killed job resumes.
#
#   run_ece.sh <gpu> <model: aunet|bpebyte|hnet|blt> <out_dir> [limit=2000] [tasks=all five, comma list] [group A|B (blt)]
set -u
GPU=$1 MODEL=$2 OUT=$3 LIMIT=${4:-2000} TASKS=${5:-hellaswag,arc_easy,arc_challenge,piqa,boolq} GROUP=${6:-}
W=$HOME/AUNet_fmt; S=$W/scripts/format_mc; X=$HOME/AUNet_lc/ext
mkdir -p "$OUT"
TAG=$MODEL; [ "$MODEL" = blt ] && TAG=blt_${TASKS//,/_}${GROUP:+_$GROUP}
case $MODEL in
  aunet|bpebyte)
    cat > "$OUT/$MODEL.yaml" <<EOF
name: "format_mc_$MODEL"
ckpt_dir: $W/ckpt/$MODEL
dump_dir: $OUT/$MODEL
harness:
  log_samples: false
  limit: $LIMIT
  tasks:
    - despace_mc
validation: null
generator:
  max_tokens: 16384
  dtype: bf16
EOF
    cd $HOME/AUNet/lingua
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$HOME/AUNet/lingua AUNET_ROOT=$HOME/AUNet_lc \
      FORMAT_ITEMS=$W/items FORMAT_CACHE=$OUT/$MODEL.cache.jsonl FORMAT_TASKS=$TASKS \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29700 + GPU * 10 + RANDOM % 10)) RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
      .venv/bin/python -u $S/run_format_lingua.py config=$OUT/$MODEL.yaml ;;
  hnet)
    cd /tmp
    CUDA_VISIBLE_DEVICES=$GPU EVAL_SUITE=$X/eval_suite HNET_REPO=$W/hnet_repo AUNET_LINGUA=$HOME/AUNet/lingua \
      $X/venv/bin/python -u $S/run_format_ext.py --family hnet --tasks ${TASKS//,/ } --limit $LIMIT \
      --items_dir $W/items --cache $OUT/hnet.cache.jsonl --out $OUT/hnet.json ;;
  blt)
    cd /tmp
    CUDA_VISIBLE_DEVICES=$GPU EVAL_SUITE=$X/eval_suite PYTHONPATH=$X/blt_official:$X/eval_suite \
      $X/venv_blt/bin/python -u $S/run_format_ext.py --family blt_official --tasks ${TASKS//,/ } --limit $LIMIT ${GROUP:+--group $GROUP} \
      --blt_weights $X/blt_weights --items_dir $W/items --cache $OUT/$TAG.cache.jsonl --out $OUT/$TAG.json ;;
esac
echo "RUN_DONE $MODEL $TASKS exit=$? $(date +%F_%T)"

#!/bin/bash
# Per-language held-out BPB on the FLORES dev source sentences.
#
# FLORES is parallel, so window k holds the SAME content in every language: the comparison is
# controlled for content and only the language varies. This is the test the sentence-level
# compound analysis could not do (see flores/compound_hypothesis.md §4-5) -- a training-time
# mechanism predicts AU-Net2's BPB disadvantage vs BPEByte is largest on German and tracks the
# FLORES BLEU ordering. Being a likelihood metric it carries none of the generation noise.
#
# Waits for the BLT FLORES chain (pid 2693508) to release the GPUs first.
ROOT=/mnt/ssd2/hyun2/AUNet
OUT=$ROOT/reports_afterAAAIsub/flores/bpb
LOGS=$ROOT/runs/_logs
mkdir -p "$OUT" "$LOGS"
BLTPID=0
if [ "$BLTPID" != "0" ]; then
  echo "waiting for BLT chain pid $BLTPID ... $(date -Is)"
  while kill -0 "$BLTPID" 2>/dev/null; do sleep 120; done
fi
sleep 30
echo "starting BPB sweep $(date -Is)"
cd $ROOT
declare -A CKPT=(
  [aunet2]=$ROOT/runs/aunet2_1.3B/checkpoints/0000180000/consolidated
  [bpebyte]=$ROOT/runs/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated
  [llama]=$ROOT/runs/llama_1.8B_paper/checkpoints/0000060000/consolidated
)
declare -A FAM=([aunet2]=aunet [bpebyte]=aunet [llama]=subword)
NV=$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib | paste -sd: -)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT/lingua${PYTHONPATH:+:$PYTHONPATH}   # apps.aunet lives there
i=0
for m in aunet2 bpebyte llama; do
  for lg in de nl it fr es en; do
    gpu=${FLBPB_GPU:-0}; i=$((i+1))   # one GPU; the rest of the box runs other jobs
    CUDA_VISIBLE_DEVICES=$gpu $ROOT/lingua/.venv/bin/python \
      $ROOT/scripts/probes/bpb_windows_local.py --family ${FAM[$m]} --ckpt ${CKPT[$m]} \
      --tag "${m}_${lg}" --windows $ROOT/data/flores_bpb/$lg.jsonl --batch_size 4 \
      --out "$OUT/${m}_${lg}.json" > "$LOGS/flbpb_${m}_${lg}.log" 2>&1
    echo "  ${m}/${lg} rc=$? gpu=$gpu"
  done
done
echo "=== BPB sweep done $(date -Is)"

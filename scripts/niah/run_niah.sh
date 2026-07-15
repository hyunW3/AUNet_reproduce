#!/usr/bin/env bash
# S-NIAH-1 retrieval sweep over the three 1.3B families (iso-byte context lengths).
#   subword : llama_1.8B_paper       @60k
#   aunet   : aunet2_1.3B            @180k
#   byte    : bpebyte_br_greedy_root @180k
# Inference-only. Env: GPU, LENGTHS, DEPTHS, NPC (n per cell), DIGITS, BATCH, REQUIRE_IDLE
set -euo pipefail
L=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet
PY=$L/lingua/.venv/bin/python
PROBE=$L/scripts/niah/niah_probe.py
TASKLIST=${TASKLIST:-"1"}            # space-separated: any of 1 2 3

export TRITON_CACHE_DIR=$L/scripts/fflm/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$L/scripts/fflm/.cache/inductor
export TMPDIR=$L/scripts/fflm/.cache/tmp
export PYTHONPATH=$L/lingua
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$L/reports/niah"

LENGTHS=${LENGTHS:-"512 1024 2048 4096 6144"}   # haystack bytes (iso-byte)
DEPTHS=${DEPTHS:-"0.1 0.3 0.5 0.7 0.9"}
NPC=${NPC:-4}
DIGITS=${DIGITS:-7}
BATCH=${BATCH:-8}

pick_gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' '; }
if [[ -n "${GPU:-}" ]]; then G=$GPU; else G=$(pick_gpu); fi
if [[ "${REQUIRE_IDLE:-0}" == "1" ]]; then
  echo "REQUIRE_IDLE=1: waiting for a GPU under 20% util ..."
  while true; do U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | head -1)
    [[ "$U" -lt 20 ]] && { G=$(pick_gpu); break; }; sleep 60; done
fi
echo "S-NIAH tasks=[$TASKLIST] on GPU $G | lengths=[$LENGTHS] depths=[$DEPTHS] npc=$NPC digits=$DIGITS"

CK_SUB=$L/runs/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated
CK_AUN=$L/runs/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
CK_BYT=$L/runs/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated

for T in $TASKLIST; do
  OUT=$L/reports/niah/results_S$T.jsonl
  PEROUT=$L/reports/niah/per_S$T.jsonl
  : > "$OUT"; : > "$PEROUT"
  run_one(){ local fam=$1 ckpt=$2 tag=$3
    echo "=============== S-NIAH-$T :: $tag ($fam) ==============="
    CUDA_VISIBLE_DEVICES=$G $PY "$PROBE" --family "$fam" --ckpt "$ckpt" --tag "$tag" \
      --task "$T" --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --value_digits $DIGITS \
      --batch_size $BATCH --out "$OUT" --per_out "$PEROUT" 2>&1 | grep -vE "FutureWarning|pynvml import"; }
  run_one subword "$CK_SUB" "subword_llama"
  run_one aunet   "$CK_AUN" "aunet_static"
  run_one aunet   "$CK_BYT" "byte_greedyroot"
  echo "Rendering S-NIAH-$T report ..."
  $PY $L/scripts/niah/niah_report.py --results "$OUT" --per "$PEROUT" \
      --outdir $L/reports/niah --suffix "_S$T" --task "$T" || true
done
echo "Done. See $L/reports/niah/"

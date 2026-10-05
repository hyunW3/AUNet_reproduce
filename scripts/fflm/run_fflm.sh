#!/usr/bin/env bash
# FFLM in-context state-tracking probe over the three 1.3B checkpoint families.
#
#   subword : llama_1.8B_paper          (flat BPE transformer)      @60k
#   aunet   : aunet2_1.3B               (byte->word static pooling) @180k
#   byte    : bpebyte_br_greedy_root    (byte, greedy-root pooling) @180k
#
# Inference-only. Picks the GPU with the most free memory (does not wait for
# training to finish unless REQUIRE_IDLE=1). Writes reports/fflm/results.jsonl
# then renders the table + plot.
#
# Env overrides: GPU, NUM_INSTR, N, N_SPARSE, MAX_READS, SHOTS, BATCH, REQUIRE_IDLE
set -euo pipefail

L=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet
PY=$L/lingua/.venv/bin/python
PROBE=$L/scripts/fflm/fflm_probe.py
OUT=$L/reports/fflm/results.jsonl
PERREAD=$L/reports/fflm/per_read.jsonl

export TRITON_CACHE_DIR=$L/scripts/fflm/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$L/scripts/fflm/.cache/inductor
export TMPDIR=$L/scripts/fflm/.cache/tmp
export PYTHONPATH=$L/lingua
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR" "$(dirname "$OUT")"

NUM_INSTR=${NUM_INSTR:-64}      # T = 2*NUM_INSTR symbols
N=${N:-200}                     # seqs per dense/indist regime
N_SPARSE=${N_SPARSE:-2000}      # sparse has ~1.6 reads/seq -> need many
MAX_READS=${MAX_READS:-16}      # cap reads scored per sequence
SHOTS=${SHOTS:-0}               # few-shot demo sequences prepended
BATCH=${BATCH:-32}

# ---- GPU selection ---------------------------------------------------------
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' '
}
if [[ -n "${GPU:-}" ]]; then G=$GPU; else G=$(pick_gpu); fi
if [[ "${REQUIRE_IDLE:-0}" == "1" ]]; then
  echo "REQUIRE_IDLE=1: waiting for a GPU under 20% util ..."
  while true; do
    U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | head -1)
    [[ "$U" -lt 20 ]] && { G=$(pick_gpu); break; }
    sleep 60
  done
fi
echo "FFLM probe on GPU $G | NUM_INSTR=$NUM_INSTR N=$N N_SPARSE=$N_SPARSE MAX_READS=$MAX_READS SHOTS=$SHOTS"

# ---- checkpoints (family, consolidated dir, tag) ---------------------------
CK_SUB=$L/runs/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated
CK_AUN=$L/runs/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated
CK_BYT=$L/runs/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated

: > "$OUT"       # fresh results file
: > "$PERREAD"   # fresh per-read diagnostic file

run_one() {
  local fam=$1 ckpt=$2 tag=$3
  echo "=================================================================="
  echo ">>> $tag ($fam)"
  CUDA_VISIBLE_DEVICES=$G $PY "$PROBE" \
    --family "$fam" --ckpt "$ckpt" --tag "$tag" \
    --regimes dense indist sparse \
    --num_instr $NUM_INSTR --n $N --n_sparse $N_SPARSE \
    --max_reads $MAX_READS --shots $SHOTS --batch_size $BATCH \
    --out "$OUT" --per_read_out "$PERREAD" 2>&1 | grep -vE "FutureWarning|pynvml import"
}

run_one subword "$CK_SUB" "subword_llama"
run_one aunet   "$CK_AUN" "aunet_static"
run_one aunet   "$CK_BYT" "byte_greedyroot"

echo "=================================================================="
echo "Rendering report ..."
$PY $L/scripts/fflm/report_fflm.py --results "$OUT" --outdir $L/reports/fflm
$PY $L/scripts/fflm/analyze_fflm.py --per_read "$PERREAD" --outdir $L/reports/fflm || true
echo "Done. See $L/reports/fflm/"

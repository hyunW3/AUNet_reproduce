#!/usr/bin/env bash
# Full PBP-NIAH grid on the real 1.3B paper checkpoints, parallel across 4 GPUs.
# S-NIAH-1/2/3 (lengths x depths) + MK-NIAH (K), canonical-vs-space cut per cell.
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT"
export AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"
OUT="$ROOT/reports/niah/pbp_1p3b"; mkdir -p "$OUT"

LLAMA="$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AUNET="$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BPE="$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
LENGTHS="512 1024 2048 4096 6144"; DEPTHS="0.1 0.3 0.5 0.7 0.9"; NPC=4; BATCH=8
filt() { grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org"; }

# one model, all three S-NIAH tasks, own output file, pinned to one GPU
sniah() { local fam=$1 ck=$2 tag=$3 gpu=$4
  for T in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_probe.py --family "$fam" --ckpt "$ck" --tag "$tag" \
      --task $T --pbp --lengths $LENGTHS --depths $DEPTHS --n_per_cell $NPC --batch_size $BATCH \
      --out "$OUT/pbp_${tag}.jsonl" 2>&1 | filt
  done; }

echo "=== S-NIAH-1/2/3 PBP (3 models in parallel on GPU 0/1/2) ==="
sniah subword "$LLAMA" subword_llama   0 > "$OUT/llama.log"   2>&1 &
sniah aunet   "$AUNET" aunet_static    1 > "$OUT/aunet.log"   2>&1 &
sniah aunet   "$BPE"   byte_greedyroot 2 > "$OUT/bpebyte.log" 2>&1 &

echo "=== MK-NIAH PBP (all 3 models, GPU 3) ==="
CUDA_VISIBLE_DEVICES=3 \
CKPT_SUBWORD_LLAMA="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated" \
CKPT_AUNET_STATIC="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated" \
CKPT_BYTE_GREEDYROOT="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated" \
$PY scripts/probes/mk_niah.py --models subword_llama aunet_static byte_greedyroot \
  --n 40 --target_bytes 2048 --needle_counts 1 2 4 8 --pbp --out "$OUT/mkniah_pbp.jsonl" \
  > "$OUT/mk.log" 2>&1 &

wait
echo "===== ALL DONE -> $OUT ====="

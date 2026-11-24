#!/usr/bin/env bash
# Forward-latency (ms/forward) for the four 1.3B models on ONE dedicated GPU (clean,
# exclusive timing). byte models bsz4 x 8192 B (32 KB/step); Llama bsz4 x 2048 tok;
# BLT-1B bsz4 x 4096 B (its context). ms/KB is the fair cross-model axis.
# Hardware note: measured on the local GPU (RTX A5000), not the paper's A100-80GB.
set -u
GPU="${GPU:-2}"
AR=/mnt/ssd2/hyun2/AUNet
BLT=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT="$AR/reports/throughput_1p3b"; mkdir -p "$OUT"
JQ="$OUT/models.jsonl"; : > "$JQ"
filt(){ grep -vE "FutureWarning|pynvml|warnings.warn|ProcessGroupNCCL|NCCL|Setting|xFormers|Please reinstall|Memory-effic|PyTorch 2.5|Update checkpoint"; }

# --- AUNet venv (llama / aunet / bpebyte) ---
cd "$AR/lingua"
export AUNET_ROOT="$AR" AUNET_TOK="$AR/tokenizer/llama3/tokenizer.model" PYTHONPATH="$AR/lingua"
export LD_LIBRARY_PATH="$AR/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=28860 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$AR/lingua/.venv/bin/python"

echo "== Llama 1.3B =="
CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_llama_forward \
  config=apps/main/configs/llama_1.8B_b200.yaml bsz=4 seq_len=1024 vocab_size=128256 attn_impl=sdpa \
  json_out="$JQ" 2>&1 | filt | tail -6
echo "== AU-Net 1.3B =="
CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_model_forward \
  config=apps/aunet/configs/aunet2_1.3B_b200.yaml bsz=4 seq_len=4096 attn_impl=sdpa \
  json_out="$JQ" 2>&1 | filt | tail -6
echo "== BPEByte 1.3B =="
CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_model_forward \
  config=apps/aunet/configs/bpebyte_root_greedy_1.3B_b200.yaml bsz=4 seq_len=4096 attn_impl=sdpa \
  json_out="$JQ" 2>&1 | filt | tail -6

# --- BLT venv ---
cd "$BLT"
export MASTER_PORT=28861 BLT_SUPPRESS_ATTN_ERROR=1 PYTHONPATH=""
echo "== BLT-1B =="
CUDA_VISIBLE_DEVICES=$GPU "$BLT/.venv/bin/python" bench_blt_forward.py \
  --ckpt_dir "$BLT/hf-weights/blt_1b" --entropy_ckpt_dir "$BLT/hf-weights/entropy_model" \
  --bsz 4 --seq 4096 --out "$OUT/blt.json" 2>&1 | filt | tail -6

echo "THROUGHPUT_1P3B_DONE  (results: $JQ + $OUT/blt.json)"

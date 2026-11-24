#!/usr/bin/env bash
# QUEUED throughput: wait until the whole node is idle (all GPUs < 1 GB used), then run
# the four 1.3B forward benches on ONE dedicated GPU with NO contention -- the only way to
# get reliable ms/forward on this shared 24 GB A5000 box. bsz2 x 4096 (memory-safe) +
# expandable_segments. Reports ms/forward + ms/KB (RTX A5000, not the paper's A100).
set -u
AR=/mnt/ssd2/hyun2/AUNet
BLT=/mnt/ssd2/hyun2/lzspacebyte/blt
OUT="$AR/reports/throughput_1p3b"; mkdir -p "$OUT"
JQ="$OUT/models.jsonl"
GPU="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
filt(){ grep -vE "FutureWarning|pynvml|warnings.warn|ProcessGroupNCCL|NCCL|Setting|xFormers|Please reinstall|Memory-effic|PyTorch 2.5|Update checkpoint"; }

echo "[queue] waiting for an idle node (all 4 GPUs < 1 GB used)..."
while true; do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{c++} END{print c+0}')
  [ "$busy" -eq 0 ] && break
  sleep 60
done
echo "[queue] node idle -> running throughput on GPU $GPU (clean, exclusive)"
: > "$JQ"

cd "$AR/lingua"
export AUNET_ROOT="$AR" AUNET_TOK="$AR/tokenizer/llama3/tokenizer.model" PYTHONPATH="$AR/lingua"
export LD_LIBRARY_PATH="$AR/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=28870 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
PY="$AR/lingua/.venv/bin/python"

echo "== Llama 1.3B =="; CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_llama_forward \
  config=apps/main/configs/llama_1.8B_b200.yaml bsz=2 seq_len=1024 vocab_size=128256 attn_impl=sdpa json_out="$JQ" 2>&1 | filt | tail -4
echo "== AU-Net 1.3B =="; CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_model_forward \
  config=apps/aunet/configs/aunet2_1.3B_b200.yaml bsz=2 seq_len=4096 attn_impl=sdpa json_out="$JQ" 2>&1 | filt | tail -4
echo "== BPEByte 1.3B =="; CUDA_VISIBLE_DEVICES=$GPU $PY -m apps.aunet.bench.bench_model_forward \
  config=apps/aunet/configs/bpebyte_root_greedy_1.3B_b200.yaml bsz=2 seq_len=4096 attn_impl=sdpa json_out="$JQ" 2>&1 | filt | tail -4

cd "$BLT"; export MASTER_PORT=28871 BLT_SUPPRESS_ATTN_ERROR=1 PYTHONPATH=""
echo "== BLT-1B =="; CUDA_VISIBLE_DEVICES=$GPU "$BLT/.venv/bin/python" bench_blt_forward.py \
  --ckpt_dir "$BLT/hf-weights/blt_1b" --entropy_ckpt_dir "$BLT/hf-weights/entropy_model" \
  --bsz 2 --seq 4096 --out "$OUT/blt.json" 2>&1 | filt | tail -4

echo "THROUGHPUT_1P3B_DONE"

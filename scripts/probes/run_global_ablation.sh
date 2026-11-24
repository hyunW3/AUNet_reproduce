#!/usr/bin/env bash
# Global-trunk ablation on the 1.3B AU-Net / BPEByte checkpoints (see global_ablation.py).
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"; PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/global_ablation"; mkdir -p "$O"
export PYTHONPATH="$ROOT/lingua" TMPDIR=/var/tmp
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
AU="$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
one(){ CUDA_VISIBLE_DEVICES=$3 $PY scripts/probes/global_ablation.py --ckpt "$2" --tag "$1" \
        --batch_size 2 --debug_boundaries "${@:4}" > "$O/$1.log" 2>&1; }
one aunet_static "$AU" "${GPU_AU:-2}" "$@" &
one byte_greedyroot "$BP" "${GPU_BP:-1}" "$@" &
wait; echo "GLOBAL ABLATION DONE"

#!/usr/bin/env bash
# (3) compare cloze, digits 1..20   (4) arith/reverse/copy cloze, digits 2..8
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; O="$ROOT/reports/statetrack"
D20="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
DC="2 3 4 5 6 7 8"

model() { local tag=$1 var=$2 ck=$3 gpu=$4
  env "$var=$ck" CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/numprobe.py --models $tag \
    --subtasks compare --digits $D20 --n 100 --batch_size 16 \
    --out "$O/numprobe_cmp_$tag.jsonl" > "$O/numprobe_cmp_$tag.log" 2>&1
  env "$var=$ck" CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/numprobe.py --models $tag \
    --subtasks arith_cloze reverse_cloze copy_cloze --digits $DC --n 100 --batch_size 16 \
    --out "$O/numprobe_clz_$tag.jsonl" > "$O/numprobe_clz_$tag.log" 2>&1
}
model subword_llama   CKPT_SUBWORD_LLAMA   main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated 0 &
model aunet_static    CKPT_AUNET_STATIC    main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated 1 &
model byte_greedyroot CKPT_BYTE_GREEDYROOT main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated 2 &
wait
echo "NUMPROBE2 DONE"

#!/usr/bin/env bash
# (1) S-NIAH-3 with the colon on the ANSWER side (--answer_sep ": ")
# (4) cloze arith/copy extended to 20 digits
set -u
cd "$(dirname "$0")/../.."
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model" PYTHONPATH="$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
PY="$ROOT/lingua/.venv/bin/python"; NO="$ROOT/reports/niah"; SO="$ROOT/reports/statetrack"
LEN="512 1024 2048 4096 6144"; DEP="0.1 0.3 0.5 0.7 0.9"
D20="2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org"; }

work() { local fam=$1 ck=$2 tag=$3 var=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/niah/niah_probe.py --family "$fam" --ckpt "$ck" --tag "$tag" \
    --task 3 --answer_sep ": " --lengths $LEN --depths $DEP --n_per_cell 4 --batch_size 8 \
    --out "$NO/answersep_$tag.jsonl" 2>&1 | filt
  env "$var=$ck" CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/numprobe.py --models $tag \
    --subtasks arith_cloze copy_cloze --digits $D20 --n 100 --batch_size 16 \
    --out "$SO/numprobe_clz20_$tag.jsonl" 2>&1 | filt
}
work subword "$ROOT/main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"          subword_llama   CKPT_SUBWORD_LLAMA   0 > "$NO/extra_llama.log"   2>&1 &
work aunet   "$ROOT/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"                aunet_static    CKPT_AUNET_STATIC    1 > "$NO/extra_aunet.log"   2>&1 &
work aunet   "$ROOT/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated" byte_greedyroot CKPT_BYTE_GREEDYROOT 2 > "$NO/extra_bpebyte.log" 2>&1 &
wait
echo "EXTRA DONE"

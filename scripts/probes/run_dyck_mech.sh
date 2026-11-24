#!/usr/bin/env bash
# Dyck mechanism probe: stack vs recency-copying.
#   GPUS="0 1" ./run_dyck_mech.sh
# Three configs x three models = 9 jobs, dealt round-robin into one SERIAL lane
# per GPU, so only ever one 1.3B model is resident per GPU (these are shared
# cards with other jobs on them -- do not stack two models on one GPU).
# k3/k4 are the pure-ASCII-bracket regime where BPEByte rg's lead is largest;
# k4-letters is the control where that lead is known to vanish (REPORT.md).
set -u; cd /mnt/ssd2/hyun2/AUNet
ROOT="$PWD"
export AUNET_ROOT="$ROOT" AUNET_TOK="$ROOT/tokenizer/llama3/tokenizer.model"
export PYTHONPATH="$ROOT/scripts/probes:$ROOT/lingua"
export LD_LIBRARY_PATH="$ROOT/lingua/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
PY="$ROOT/lingua/.venv/bin/python"
O="$ROOT/reports/statetrack/dyck_mech"; mkdir -p "$O"

LL="main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated"
AU="main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"
BP="main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"

GPUS=${GPUS:-"0 1"}
BS=${BS:-16}
read -ra G <<< "$GPUS"
NG=${#G[@]}

filt(){ grep -vE "FutureWarning|import pynvml|ProcessGroupNCCL|destroy_process_group|pytorch.org|warnings.warn"; }

# job spec: "cfgname|tag|ckenv|ckpt|extra args"
JOBS=()
for c in "dyck3_brackets|--k 3 --length 40 --maxdepth 6 --n 150" \
         "dyck4_brackets|--k 4 --length 40 --maxdepth 6 --n 150" \
         "dyck4_letters|--k 4 --length 40 --maxdepth 6 --n 150 --letters"; do
  name=${c%%|*}; extra=${c#*|}
  JOBS+=("$name|subword_llama|CKPT_SUBWORD_LLAMA|$LL|$extra")
  JOBS+=("$name|aunet_static|CKPT_AUNET_STATIC|$AU|$extra")
  JOBS+=("$name|byte_greedyroot|CKPT_BYTE_GREEDYROOT|$BP|$extra")
done

run_lane(){                         # $1 = lane index
  local lane=$1 gpu=${G[$1]} i
  for ((i=lane; i<${#JOBS[@]}; i+=NG)); do
    IFS='|' read -r name tag ckenv ck extra <<< "${JOBS[$i]}"
    local out="$O/${name}__${tag}.jsonl" log="$O/${name}__${tag}.log"
    rm -f "$out"
    echo "[gpu$gpu] START $name / $tag" >&2
    CUDA_VISIBLE_DEVICES=$gpu env "$ckenv=$ck" \
      $PY scripts/probes/dyck_mech.py --models "$tag" --batch_size "$BS" \
        --out "$out" $extra > "$log" 2>&1
    echo "[gpu$gpu] DONE  $name / $tag exit=$? -> $(grep -cE '^  ' "$log" 2>/dev/null || echo 0) result line(s)" >&2
  done
}

for ((l=0; l<NG; l++)); do run_lane "$l" & done
wait
echo "DYCKMECH DONE"

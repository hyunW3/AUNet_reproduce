#!/usr/bin/env python3
"""Generate the 4-family multilingual ratio-40 (84B bytes) pilot configs + orchestrator on ece.
Byte models: 13376 steps, bs24 ga8 (memory-safe A100-80GB, preserves 6.29M bytes/opt-step).
Llama iso-byte: 17600 steps, native bs32 ga4 seq2048. Data = 4 ml sources equal weight."""
import os
ROOT = "/home/hwbae/AUNet"
CFGD = f"{ROOT}/lingua/apps/aunet/configs"
LCFGD = f"{ROOT}/lingua/apps/main/configs"
RUN = f"{ROOT}/runs/ml_pilot_r40"
TOK = f"{ROOT}/tokenizer/llama3/tokenizer.model"
SOURCES = "      mlpilot_en: 1.0\n      mlpilot_fi: 1.0\n      mlpilot_zh: 1.0\n      mlpilot_code: 1.0"

def byte_cfg(tag, regex_block):
    return f"""# ML ratio-40 pilot — {tag}
dump_dir: {RUN}/{tag}
name: "ml_r40_{tag}"
steps: 13376
grad_acc_steps: 8
probe_freq: null
seed: 777
optim:
    lr: 2.0e-3
    warmup: 100
    weight_decay: 0.1
    lr_min_ratio: 0.01
    clip: 0.2
distributed:
    fsdp_type: full_shard
    compile: true
    model_dtype: bf16
    matmul_allow_tf32: false
    selective_activation_checkpointing: false
    tp_size: 1
    compile_cache_size_limit: 16
model:
    dimensions: [512, 768]
    layers: [3, 10]
    head_dims: [64, 128]
    residuals: [True]
    sliding_windows: [512, 4096]
    max_seqlens: [-1, 3200]
    block:
        rope_theta: 500000.0
        multiple_of: 256
    lambda_level: 0.0
    pooling_type: simple_indexed_matmul
    patch_read_delay: 0
data:
    root_dir: {ROOT}/data
    sources:
{SOURCES}
    batch_size: 24
    prefetch_size: 256
    seq_len: 8192
    n_views: 2
    load_async: true
    add_bos: true
    add_eos: true
    tokenizer:
        name: bytes
    regex:
{regex_block}
profiling:
  run: false
checkpoint:
    dump:
        every: 6688
        keep: 1
    eval:
        every: 9999999
        keep: 1
logging:
  freq: 10
async_eval_gpus: null
eval_milestones: []
eval: null
"""

RG = f"""        strategy:
            bpe_br: 1@1
        bpe_tokenizer_path: {TOK}
        bpe_online: true
        bpe_online_mode: greedy
        bpe_online_placement: root
        bpe_context_prefix: 0
        bpe_online_committed_view: false"""

BT = f"""        strategy:
            bpe_br: 1@1
        bpe_tokenizer_path: {TOK}
        bpe_online: true
        bpe_online_mode: bt
        bpe_online_placement: before_root
        bpe_context_prefix: 0
        bpe_online_committed_view: false"""

WORD = """        strategy:
            word1: 1@1"""

LLAMA = f"""# ML ratio-40 pilot — llama subword (iso-byte 17600 steps)
dump_dir: {RUN}/llama
name: "ml_r40_llama"
steps: 17600
grad_acc_steps: 4
probe_freq: null
seed: 777
optim:
  lr: 3.0e-3
  warmup: 110
  weight_decay: 0.1
  lr_min_ratio: 0.01
  clip: 0.2
distributed:
  fsdp_type: full_shard
  compile: true
  model_dtype: bf16
  matmul_allow_tf32: false
  selective_activation_checkpointing: false
  tp_size: 1
model:
  dim: 768
  n_layers: 14
  n_heads: 12
data:
  root_dir: {ROOT}/data
  sources:
{SOURCES}
  batch_size: 32
  prefetch_size: 1024
  seq_len: 2048
  n_views: 2
  load_async: true
  add_bos: true
  add_eos: true
  tokenizer:
    name: tiktoken
    path: {TOK}
profiling:
  run: false
checkpoint:
  dump:
    every: 8800
    keep: 1
"""

os.makedirs(RUN, exist_ok=True)
open(f"{CFGD}/ml_r40_bpebyte_rg.yaml","w").write(byte_cfg("bpebyte_rg", RG))
open(f"{CFGD}/ml_r40_bpebyte_bt.yaml","w").write(byte_cfg("bpebyte_bt", BT))
open(f"{CFGD}/ml_r40_aunet_word.yaml","w").write(byte_cfg("aunet_word", WORD))
open(f"{LCFGD}/ml_r40_llama.yaml","w").write(LLAMA)

DRIVER = f"""#!/bin/bash
# 4-family multilingual ratio-40 pilot (100M). Sequential on GPUs 4-7, resume-safe.
set -u
ROOT={ROOT}; cd $ROOT/lingua; source .venv/bin/activate
export TMPDIR=/var/tmp CUDA_VISIBLE_DEVICES=4,5,6,7
RUN={RUN}; mkdir -p $RUN; L=$RUN/orch.log
log(){{ echo "[$(date '+%F %T')] $*" | tee -a $L; }}
run_one(){{ # tag module config port target_steps
  local TAG=$1 MOD=$2 CFG=$3 PORT=$4 TGT=$5
  if [ -f $RUN/_done_$TAG ]; then log "$TAG done, skip"; return 0; fi
  log "START $TAG"
  torchrun --nproc-per-node 4 --master-port $PORT -m $MOD config=$CFG >> $RUN/$TAG.log 2>&1
  local last=$(grep -oE '"global_step": [0-9]+' $RUN/$TAG/metrics.jsonl 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  last=${{last:-0}}
  if [ "$last" -ge "$((TGT-20))" ]; then touch $RUN/_done_$TAG; log "DONE $TAG (step $last)"; else log "FAIL $TAG (reached $last/$TGT) — STOP"; exit 1; fi
}}
log "=== ML ratio-40 4-family pilot START (en/fi/zh/code) ==="
run_one bpebyte_rg apps.aunet.train apps/aunet/configs/ml_r40_bpebyte_rg.yaml 30011 13376
run_one bpebyte_bt apps.aunet.train apps/aunet/configs/ml_r40_bpebyte_bt.yaml 30012 13376
run_one aunet_word apps.aunet.train apps/aunet/configs/ml_r40_aunet_word.yaml 30013 13376
run_one llama      apps.main.train  apps/main/configs/ml_r40_llama.yaml       30014 17600
log "=== ALL TRAINING DONE ==="
for t in bpebyte_rg bpebyte_bt aunet_word llama; do
  m=$RUN/$t/metrics.jsonl
  [ -f "$m" ] && log "$t final $(grep -oE '\\"loss/out\\": [0-9.]+' $m | tail -1)"
done
log "=== PILOT COMPLETE — next: per-language held-out BPB eval ==="
"""
open(f"{RUN}/orch_ml_r40.sh","w").write(DRIVER)
os.chmod(f"{RUN}/orch_ml_r40.sh", 0o755)
print("configs + driver written under", CFGD, "and", RUN)
print("DONE")

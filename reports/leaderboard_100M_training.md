# Law-recipe leaderboard — 100M training scripts + ETAs

7 models retrained at the **AU-Net scaling-law recipe**, matched **γ10.4** budget (100M byte =
53504 steps @ global batch 48 / 21.0 GB; llama = 11891 steps @ global batch 192). 2 GPUs per model,
on ece-agpu11 (all 8) + ece-agpu18 (0–3); 7th model queued for the last free pair.

## Common wrapper (run on ece via ssh)
```bash
cd /home/hwbae/AUNet/lingua && source .venv/bin/activate
CFG=/home/hwbae/AUNet/runs/poc/portable_aunetlaw
DATA=/home/hwbae/AUNet/data
TOK=/home/hwbae/AUNet/tokenizer/llama3/tokenizer.model
# each launched as: CUDA_VISIBLE_DEVICES=<g> TMPDIR=/var/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   TORCHINDUCTOR_CACHE_DIR=/var/tmp/<x>  nohup setsid <cmd> >/tmp/lb_<name>_100M.log 2>&1 & disown
```

## The 7 training commands
```bash
# 1. BPEByte root_greedy (rg)          — ece11 GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 29601 -m apps.aunet.train \
  config=$CFG/bpebyte_rg_100M.yaml dump_dir=$CFG/lb_rg_100M \
  data.root_dir=$DATA data.regex.bpe_tokenizer_path=$TOK data.batch_size=12 grad_acc_steps=2

# 2. Llama (subword)                   — ece11 GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 --master-port 29602 -m apps.main.train \
  config=$CFG/llama_100M.yaml dump_dir=$CFG/lb_llama_100M \
  data.root_dir=$DATA data.tokenizer.path=$TOK data.batch_size=12 grad_acc_steps=8

# 3. BPEByte hybrid leaf_mid (offline_leaf + uniform_mid) — ece11 GPU 4,5
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc-per-node 2 --master-port 29603 -m apps.aunet.train \
  config=$CFG/bpebyte_rg_hybrid_leaf_100M.yaml dump_dir=$CFG/lb_hybrid_100M \
  data.root_dir=$DATA data.regex.bpe_tokenizer_path=$TOK data.batch_size=12 grad_acc_steps=2

# 4. AU-Net (word)                     — ece11 GPU 6,7
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node 2 --master-port 29604 -m apps.aunet.train \
  config=$CFG/aunet_100M.yaml dump_dir=$CFG/lb_aunet_100M \
  data.root_dir=$DATA data.batch_size=12 grad_acc_steps=2

# 5. ByteFlow global_topk (K=3200)     — ece18 GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 --master-port 29605 -m apps.aunet.train \
  config=$CFG/byteflow_100M.yaml dump_dir=$CFG/lb_byteflow_100M \
  data.root_dir=$DATA data.batch_size=12 grad_acc_steps=2

# 6. Entropy/BLT (low, 5x)             — ece18 GPU 2,3   (launcher wires entropy model + baked θ=0.665)
NPROC=2 BATCH_PER_GPU=12 SCALE=100M ENTROPY=low MASTER_PORT=29606 \
  DUMP_DIR=$CFG/lb_blt_low_100M DATA_ROOT=$DATA bash $CFG/train_blt.sh

# 7. Entropy/BLT (pretrained facebook/blt-entropy) — queued (runs/lb100M_blt_pretrained_queue.sh)
NPROC=2 BATCH_PER_GPU=12 SCALE=100M ENTROPY=blt MASTER_PORT=29607 \
  DUMP_DIR=$CFG/lb_blt_pretrained_100M DATA_ROOT=$DATA bash $CFG/train_blt.sh   # θ auto-calibrated on first run
```

## Batching
All byte models: `bs12 × ga2 × 2 GPU = global batch 48`. Llama: `bs12 × ga8 × 2 = 192` (small
micro-batch avoids the 128k-vocab logit OOM). Commands 1–5 are direct `torchrun`; 6–7 use
`train_blt.sh` (entropy-model path + θ calibration).

## Estimated training time (measured pace, 2 GPUs/model)
| model | steps | pace (steps/s) | **100M ETA** |
|---|---|---|---|
| rg / hybrid / AU-Net (byte) | 53504 | ~1.78 | **~8.3 h** |
| ByteFlow | 53504 | ~1.89 | **~7.9 h** |
| Llama | 11891 | ~0.59 | **~5.6 h** |
| BLT low / pretrained (entropy in-loop) | 53504 | ~0.37 | **~40 h** ⚠ |

**100M leaderboard is gated by BLT (~40 h)** — the entropy model runs on-GPU every train step
(`entropy_gpu: true`), ~5× the plain byte pace. The other six finish in ~8 h.

## 300M (same pattern, `*_300M.yaml`, global batch 64 byte / 256 llama, γ10.4 = 120752/26481 steps)
Estimated (not yet running; ~3× params, ~2.1 s/step byte):
| model | 300M ETA (est.) |
|---|---|
| rg / hybrid / AU-Net / ByteFlow (byte) | **~2.5–3 days** |
| Llama | ~1–1.5 days |
| BLT low / pretrained | **~1.5–2 weeks** ⚠ (entropy in-loop, dominant cost) |

The BLT runs dominate both scales; if that's too slow, options: give BLT 4 GPUs, or precompute the
entropy boundaries offline (`apps.aunet.precompute_entropy_boundaries`) so the train loop skips the
in-loop entropy forward.

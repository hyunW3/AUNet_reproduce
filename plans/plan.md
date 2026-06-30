# Plan: Reproduce AU-Net 2 1.3B (HellaSwag 64.2 / ARC-E 64.4)

## Context

The goal is to reproduce Table 2 row 1 of [Videau et al. 2025 (arXiv:2506.14761)](https://arxiv.org/abs/2506.14761) — **AU-Net 2 1.3B** trained on the DCLM corpus with data-to-model ratio 10 (60B LLaMA tokens ≈ 273B bytes). Target downstream scores:

- HellaSwag: 64.2 ±1.0
- ARC-Easy: 64.4 ±1.9

Key findings from investigation:

1. **No pretrained checkpoint exists.** Meta did not release weights. Reproduction requires training from scratch.
2. **The right config is `apps/aunet/configs/2B_1level.yaml`**, despite its "2B" name. Its architecture matches AU-Net 2 in the paper:
   - `dimensions: [512, 2048]` — byte stage + word stage (paper §3.2: byte=2048/4=512, word=2048)
   - `layers: [3, 25]` — 2 stages, symmetric encoder/decoder around inner trunk
   - `lr: 1.65e-3` matches paper's LR scaling law for C=5e20 FLOPs (Table 2 lists 5e20 for AU-Net 2)
   - `steps: 180000` × global batch 192 × seq_len 8192 ≈ 283B bytes ≈ 62B tokens ✓ matches Table 2 row 1
3. **AU-Net is byte-level** (`tokenizer: name: bytes`) — no separate tokenizer model needed.
4. **Eval is via `lm-eval` harness** (already a dependency in `requirements.txt`).

Hardware: 4× NVIDIA B200 (180 GB each), 2.8 TB free on workspace lustre fs.

Estimated cost: ~3-7 days of continuous training + ~500 GB DCLM subset download.

## Approach

### Step 1 — Environment setup

Create an isolated env and install lingua's deps. The repo ships `setup/create_env.sh` but pins versions for Meta's internal cluster, so we'll do a manual install verified for B200 (Blackwell) support.

- Use conda or venv at `/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/.venv`
- Install: PyTorch ≥ 2.5 with CUDA 12.4+ (B200 needs Blackwell-capable build), plus everything in `lingua/requirements.txt`
- Build `terashuf` (used by data prep): cloned and `make`'d automatically by the prep script
- Critical packages: `torch`, `lm-eval`, `datatrove`, `wandb`, `huggingface_hub`, `xformers`, `omegaconf`

### Step 2 — Download DCLM Baseline 1.0 subset

Use the lingua-provided script with `nchunks` set to limit the download to ~500-600 GB (sufficient for 273B bytes of training).

```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
python setup/download_prepare_hf_data.py \
    dclm_baseline_1.0 \
    <memory_gb> \
    --data_dir /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/data \
    --seed 42 \
    --nchunks 32
```

The script: (a) `snapshot_download`s a subset of parquet files from `mlfoundations/dclm-baseline-1.0`, (b) converts parquet → jsonl via `datatrove`, (c) terashuffles into chunked jsonl. Final layout: `data/dclm_baseline_1.0/dclm_baseline_1.0.chunk.{0..nchunks-1}.jsonl`.

Pick `nchunks` so the resulting shuffled jsonl is ~500 GB. May need to tune by sampling 1-2 chunks first.

### Step 3 — Adapt config for 4× B200

Create `apps/aunet/configs/aunet2_1.3B_b200.yaml` by copying `2B_1level.yaml` with these changes:

| Field | Original | New | Reason |
|---|---|---|---|
| `dump_dir` | — | `/NHNHOME/.../AUNet/runs/aunet2_1.3B` | Required, currently absent |
| `data.root_dir` | `/path/to/data` | `/NHNHOME/.../AUNet/data` | Point to downloaded data |
| `data.batch_size` | 12 | **48** | Preserves global batch = 192 across 4 GPUs (vs original 16 GPUs × 12) — B200's 180 GB memory accommodates this |
| `async_eval_gpus` | 8 | **0** | We only have 4 GPUs total — no spare for async eval |
| `distributed.compile` | true | true | Keep — large speedup |
| `optim.lr` | 1.65e-3 | unchanged | Matches paper |
| `steps` | 180000 | unchanged | Hits 60B-token target with global batch 192 |

All other model hyperparameters (dimensions, layers, sliding_windows, etc.) are kept unchanged from `2B_1level.yaml` since they define the AU-Net 2 architecture.

### Step 4 — Training

Launch as a single-node 4-GPU job. Lingua supports `torchrun` directly (the `lingua.stool` path is SLURM-only).

```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
torchrun --nproc-per-node 4 -m apps.aunet.train \
    config=apps/aunet/configs/aunet2_1.3B_b200.yaml
```

Run in background; checkpoints dumped every 2000 steps (`checkpoint.dump.every: 2000`). Resumes automatically from the latest checkpoint if interrupted (the train.py code reads existing checkpoints in `dump_dir/checkpoints` at startup; lines 314-317).

Monitor for: (1) OOM at startup with `batch_size=48` — back off to 32 or 24 if needed and proportionally increase steps; (2) loss going to NaN; (3) throughput vs paper's 225k bytes/sec/GPU (B200 should be ≥ this).

### Step 5 — Evaluation

After training completes (or at intermediate checkpoints to track progress):

```bash
python -m apps.aunet.eval \
    config=apps/aunet/configs/aunet2_1.3B_b200.yaml \
    ckpt_dir=/NHNHOME/.../AUNet/runs/aunet2_1.3B/checkpoints/<step>
```

`eval.py:231` calls `consolidate_checkpoints(cfg.ckpt_dir)` to merge FSDP shards before running `lm-eval`. The harness tasks already configured in `2B_1level.yaml` include `hellaswag` and `arc_easy` — the two we care about.

To save eval time during the run, you can manually trigger eval only at the final step rather than every 50k steps (or set `eval.harness.tasks` to just the two benchmarks).

## Critical files

- `apps/aunet/configs/2B_1level.yaml` — the base config; copy & modify, do not edit in place
- `apps/aunet/train.py` — training entry point; reads `dump_dir`, instantiates FSDP model, runs loop
- `apps/aunet/eval.py` — evaluation entry point; consolidates checkpoint and calls lm-eval
- `apps/aunet/hierarchical.py` — model architecture (no changes needed)
- `apps/aunet/data/data.py` — data loader (no changes needed)
- `setup/download_prepare_hf_data.py` — DCLM downloader
- `lingua/requirements.txt` — Python deps

## Verification

After each step:

- **Env**: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should report `True NVIDIA B200`. `python -c "import lm_eval, datatrove, xformers"` should succeed.
- **Data**: `ls /NHNHOME/.../AUNet/data/dclm_baseline_1.0/*.jsonl | wc -l` should equal `nchunks`. Total size from `du -sh` should be ~500 GB. Spot check: `head -1 *.chunk.0.jsonl | jq .text | head -c 200` returns readable English text.
- **Training start**: First 100 steps should log decreasing loss; `nvidia-smi` should show ~150-170 GB usage per B200; throughput should print bytes/sec/GPU after warmup. If loss is flat or NaN, stop and inspect.
- **Mid-training sanity**: After ~10% of steps, run eval on a checkpoint — HellaSwag should be in the 45-55 range (vs ~64 at convergence).
- **Final eval**: Run on the last checkpoint. Compare:
  - HellaSwag: target 64.2 ± 1.0 (acceptable range: 63.2 - 65.2)
  - ARC-Easy: target 64.4 ± 1.9 (acceptable range: 62.5 - 66.3)

Numbers outside these ranges suggest a config divergence (likely candidates: wrong global batch, wrong total bytes seen, lr mismatch, dtype issue).

## Risks & open questions

- **DCLM subset coverage**: The paper trained on a stream of DCLM but didn't specify exact subset. Random subset with seed=42 (as in `download_prepare_hf_data.py`) should be statistically equivalent at 60B-token scale.
- **B200 vs H100 numerics**: Paper used H100. B200 Blackwell tensor cores may use slightly different precision paths; should not affect final downstream accuracy more than ±0.5.
- **n_views=2 semantics**: Worth verifying in `data/data.py` that this doesn't double-count training tokens. If it does, the 180k-step target should be halved to 90k to match Table 2 row 1.
- **Torch compile + B200**: `compile: true` may hit Inductor codegen issues on sm_100. Have a fallback to `compile: false` (~30% slower).

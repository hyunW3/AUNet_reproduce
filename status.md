# AU-Net 2 1.3B Reproduction — Status Tracker

**Plan**: `plan.md` (in this directory) — full design.
**Target**: Table 2 row 1 of arXiv 2506.14761 — HellaSwag 64.2 / ARC-E 64.4.
**Hardware**: 4× B200 (180 GB), 2.8 TB free workspace.
**Started**: 2026-05-22

Update this file after every step. If the session is interrupted, the next session reads this file first to know where to resume.

---

## Overall progress

- [x] Step 1 — Environment setup
- [x] Step 2 — DCLM data download (2 shards → 16 chunks + val, 397 GB shuffled jsonl, 58.9M docs)
- [x] Step 3 — Config adapted for 4× B200 (apps/aunet/configs/aunet2_1.3B_b200.yaml)
- [x] Step 4 — Training (180k steps; finished ~2026-05-31)
- [x] Step 5 — Evaluation (HellaSwag 62.6 / ARC-E 66.5 — see Step 5 section)
- [x] Phase 2a — Llama 1B dm=10 subword baseline (`runs/llama_1B_dm10`, ckpt 30000) — HS 63.6 / ARC-E 67.3
- [x] Phase 2b — BPEByte br_bt 1.3B (`runs/bpebyte_br_bt_1.3B`, ckpt 180000) — HS 64.8 / ARC-E 64.0
- [ ] **Phase 3 (CURRENT) — BPEByte hyperparameter SCREEN: 3 configs, sequential** ← see next section

**Current step**: Post-reproduction — 1.3B 4-model benchmark comparison DONE; eval-efficiency infra (seg-cache + parallel precompute) running; 760M matched-scale training queued. See "Current status & plan (2026-06-21)" below.
**Last updated**: 2026-06-21 ~01:00 KST

---

## Current status & plan (2026-06-21)

The AU-Net2 1.3B reproduction (Steps 1–5) is done. Work since has been a **4-model 1.3B
comparison** (Llama 1B / AU-Net2 / BPEByte online-bt / BPEByte root_greedy), a **BLT-style
robustness + character-awareness suite**, an **eval-efficiency overhaul**, and a queued **760M
matched-scale run**.

### Done
- **1.3B consolidated benchmark table** → `runs/consolidated_1.3B_table.md`. 0-shot full:
  HS/ARCe/BoolQ/PIQA/Wino + MMLU + ARC-C + MBPP/HEval + CUTE. Llama edges binary tasks;
  online-bt leads HS(64.3)/ARC-C(37.8); root_greedy leads ARC-e(65.9); **byte models beat BPE
  on CUTE** (AU-Net2 23.9 / root_greedy 20.6 vs Llama 18.4) — the character-level win.
  Byte MBPP/HEval = N/A (online byte decode times out on long code-gen; it's generation, not
  cache-fixable). online-bt CUTE 0.0 = degenerate decode.
- **1.3B v4 root_greedy** trained @180k (HS 63.0 / ARC-e 65.9, leak-free).
- **Trunk warm-start** ablation (100M + 500M): scale-dependent — hurts strong br_bt at 100M
  (+0.02) but helps it most at 500M (−0.17); byte-warm ties/beats subword at 500M. See
  `runs/tokwarm/RESULT.md`, `trunk_warmup_poc.md`.
- **HellaSwag-typo** (8 ops × char/word) on 4 models — gate **passed** (root_greedy 52.1 >
  Llama 50.1, = AU-Net2 52.1).
- **CUTE in BLT Table-3 format** → `runs/cute_table3_format.md` (HellaSwag-Noise / Phonology-G2P
  rows pending the cache-backed eval below).
- **Eval infra:** OMP/RAYON thread caps (killed 68-thread/proc oversubscription); disk-backed
  **segmentation cache** (`regex_cutting.py`, validated reproducible, 21–71×/seq); **parallel
  precompute** (`precompute_segcache.py`, 100% key-match w/ real eval); real **PhonologyBench**
  G2P + **CMUdict**(dropped); corpus-scale **PBP** items (en/code/zh, 6k).

### Running / queued
1. **All-prompts precompute** (`run_precompute.sh` via `pack_evals.sh`) — one seg-cache per byte
   model covering all 15 loglikelihood tasks. GPU-free, parallel.
2. **Cache-backed eval packing** (`pack_evals.sh`, GPU-packing scheduler) — noise (HellaSwag-Noise
   + PhonologyBench-G2P), pbp/pbp_mc (ΔBPC en/code/zh + ΔAcc), typo_ds (boolq/piqa/arc typo,
   re-run cache-backed) on the 3 byte models. Then run `run_760M_chain.sh`.
3. **760M matched-scale training** (`run_760M_chain.sh`) — Llama → AU-Net2 → root_greedy,
   Chinchilla-optimal iso-text (Llama 7800 steps / byte 44000 steps), smoke-guarded, 4× B200.
   Configs: `apps/{main,aunet}/configs/{llama_760M_b200,aunet2_760M_b200,bpebyte_root_greedy_760M_b200}.yaml`
   (679.6M shared core verified).

### Reporting (as each finishes)
noisy-downstream drop table · robustness rows (fills cute_table3_format.md TODOs) · PBP ΔBPC/ΔAcc ·
per-doc root_greedy-vs-online-bt diagnostic (`runs/rootgreedy_diag.md`) · 760M milestone evals.

### Pipeline scripts (backed up in /tmp/pipeline_backup/)
`pack_evals.sh` (precompute→pack), `run_precompute.sh`, `run_robustness.sh`, `run_pbp.sh`,
`run_760M_chain.sh`, `run_postcute.sh` (orchestrator, done). Note: these live in the `lingua`
submodule and are git-untracked — a `git clean` there wipes them (happened once; restore from backup).

---

## Phase 3 — BPEByte hyperparameter screen (3 configs)

**Driver**: `runs/bpebyte_screen/driver.sh` (PID 2159279) runs the 3 configs **sequentially**, each compute-matched at **94B bytes** (60k steps × global batch 192 × seq 8192, or equivalent). It `rm -rf`s each dump dir before starting, and sends a Telegram alert (`alert_knock`) when each run starts/finishes and when the whole screen completes. Started 2026-06-08 01:31 KST.

| # | Config | Diff vs base | Steps | Global batch | LR | Warmup | Status |
|---|---|---|---|---|---|---|---|
| 1 | `bpebyte_screen_base` | — (lr 1.65e-3, gb 192) | 60,000 | 192 (bs12×4GPU×ga4) | 1.65e-3 | 3333 | 🟢 RUNNING — 19,010/60,000 (31.7%) |
| 2 | `bpebyte_screen_lr2e3` | LR → 2e-3 | 60,000 | 192 | 2e-3 | 3333 | ⏳ queued (driver starts it after base) |
| 3 | `bpebyte_screen_bb384` | global batch → 384, LR 2e-3, half steps | 30,000 | 384 (bs12×4GPU×ga8) | 2e-3 | 1667 | ⏳ queued |

All three share: AU-Net 2 architecture (`dimensions [512,2048]`, `layers [3,25]`), `bpe_br` regex split strategy (Llama-3 tokenizer boundaries), `compile: true`, clip 0.2, seed 777, milestone evals (hellaswag + arc_easy) at 30/60/90/100% of steps.

### Live snapshot (2026-06-08 11:35 KST)
- base: step 19,010 / 60,000, loss 0.62-0.67, grad 0.03, lr 1.36e-3, **1.90 s/step**, 210k wps/GPU, mem 46%, ~950 W/GPU
- base 30% milestone eval (step 18,000): **HellaSwag 41.0, ARC-E 45.3** (acc_norm)
- Side job on same GPUs: eval_gen smoke on aunet2_1.3B ckpt 180000 (PID 2247421, `harness.limit=5`) — transient

### ETA (at 1.90 s/step, quiet host)
- base: 40,990 steps left ≈ 21.6h → done **~2026-06-09 09:10 KST**
- lr2e3: 60k steps ≈ 31.6h → done **~2026-06-10 ~16:45 KST**
- bb384: 30k steps × ~3.8s (ga8) ≈ 31.6h → done **~2026-06-12 ~00:30 KST**
- **Screen complete: ~2026-06-12 early morning KST**

### How to check status
```bash
# Which run is the driver on?
pgrep -af "apps.aunet.train" | grep -oE 'bpebyte_screen_[a-z0-9]+' | head -1
tail -3 /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/bpebyte_screen/driver_outer.log

# Progress of the active run
grep -oE 'step: [0-9]+' /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/bpebyte_screen/*.log | tail -1

# Milestone eval results so far (all runs)
cat /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/bpebyte_screen/*/metrics.eval.jsonl
```

### When the screen finishes
Compare the three `runs/bpebyte_screen/*/metrics.eval.jsonl` final entries (HellaSwag / ARC-E acc_norm at 100% milestone) + final train/validation loss. The winner's hyperparameters feed the next full-scale BPEByte run.

### Phase 1-2 final results (for comparison)
| Run | HellaSwag | ARC-Easy | Notes |
|---|---|---|---|
| AU-Net 2 1.3B byte (`aunet2_1.3B`) | **62.6** | **66.5** | paper target 64.2±1.0 / 64.4±1.9 — HS 0.6 below band, ARC-E 0.2 above |
| Llama 1B dm10 subword (`llama_1B_dm10`) | **63.6** | **67.3** | BPE baseline |
| BPEByte br_bt 1.3B (`bpebyte_br_bt_1.3B`) | **64.8** | **64.0** | byte-level with BPE-boundary splitting, 180k steps |

### Installed versions
- torch 2.7.0+cu128, xformers 0.0.30, triton 3.3.0
- lm-eval 0.4.12, datatrove 0.9.0, huggingface_hub 0.36.2
- B200 (sm_100) detected, CUDA available

### Training history (step 1 → 79433 → killed → ckpt 78000)
- Steps 1-13: loss 6.05 → 4.37 (fast initial drop)
- Steps 234-10k: loss 2.31 → 0.95 (still in warmup, lr ramping)
- Steps 10k-50k: loss 0.95 → 0.97 (warmup ended; plateau as lr stabilises at 1.65e-3)
- Steps 50k-79k: loss 0.97 → 0.65-0.75 (slow improvement, grad norm settled to 0.03)
- iter time: 0.79 s / micro-batch × 4 grad_acc ≈ 3.16 s / optimizer step (when host is quiet)
- throughput: 125k bytes/sec/GPU on B200 (paper: 225k on H100 — gap explained by `compile: false`)
- checkpoints: saved every 2000 steps; latest healthy = 0000078000 (saved 2026-05-25 ~12:34)

### Step 4 — crash #9 and pause (2026-05-25 → present)
At step ~79433 the shared host's load avg climbed past 700 — another job started hammering
the node. NCCL collective timeout (default 10 min) tripped and all 4 ranks aborted at
15:39 with `[F525 ProcessGroupNCCL.cpp:1554] Terminating the process … due to collective
timeout or exception`.

Bumped `init_process_group(timeout=timedelta(hours=1))` in `lingua/distributed.py` and
relaunched (PID 2521496). The relaunch immediately got stuck in init — all 4 workers
went into uninterruptible D-state on I/O for an hour without ever producing a log line
past the torchrun banner. Killed the parent; the D-state workers cannot be killed.

Pause: training has been parked at ckpt 78000 ever since. Hourly load checks show a
monotonic climb (no plateau yet):

  | 2026-05-25 16:25 | 1135 |
  | 2026-05-25 17:27 | 1135 |  (post-crash, killed our process)
  | 2026-05-25 18:29 | 1420 |
  | 2026-05-25 19:31 | 1680 |
  | 2026-05-25 20:32 | 1889 |
  | 2026-05-25 21:33 | 2175 |
  | 2026-05-25 22:34 | 2396 |
  | 2026-05-25 23:35 | 2582 |
  | 2026-05-26 00:36 | 2810 |
  | 2026-05-26 01:37 | 3046 |
  | 2026-05-26 02:38 | 3306 |
  | 2026-05-26 03:39 | 3564 |
  | 2026-05-26 04:40 | 3793 |
  | 2026-05-26 05:41 | 4011 |
  | 2026-05-26 06:42 | 4242 |
  | 2026-05-26 07:43 | 4426 |
  | 2026-05-26 08:44 | 4595 |
  | 2026-05-26 09:45 | 4795 |
  | 2026-05-26 10:46 | 4929 |
  | 2026-05-26 11:47 | 5169 |
  | 2026-05-26 12:48 | 5366 |
  | 2026-05-26 13:49 | 5604 |
  | 2026-05-26 14:50 | 5846 |
  | 2026-05-26 15:51 | 6081 |

Hourly Read of `/proc/loadavg` will trigger a restart when load drops below 200.
Training resumes from ckpt 78000 — ~1500 steps will be re-traversed.

### Git / push status
- 6 commits made in `lingua/` (the modified Lingua checkout):
  1. `eb5ee0c` gitignore additions
  2. `2c4e172` b200: FA3 disable + DTensor-safe clip_grad
  3. `8bac5c1` b200: ENABLE_INTRA_NODE_COMM=0 (SymmetricMemory fix)
  4. `a04e7ac` data: dclm_baseline_1.0_2shards download option
  5. `517e927` config: aunet2_1.3B_b200.yaml
  6. `37271de` docs: plan.md + status.md
- `origin` set to `git@github.com:hyunW3/AUNet_reproduce.git`,
  old facebook origin renamed to `upstream`.
- **`git push -u origin main` failed** with `Permission denied (publickey)`:
  need to register `~/.ssh/id_ed25519.pub` at https://github.com/settings/ssh/new
  (or switch to HTTPS + PAT).

---

## Step 1 — Environment setup

**Status**: ✅ COMPLETED 2026-05-22
**Owner action**: install dependencies

### How to check status
```bash
# Should print: True NVIDIA B200
/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/.venv/bin/python -c \
    "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Should succeed without ImportError
/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/.venv/bin/python -c \
    "import lm_eval, datatrove, xformers, omegaconf, wandb, huggingface_hub"
```

### How to resume / start
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# B200 needs CUDA 12.4+ build of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Notes / blockers
_(empty)_

---

## Step 2 — DCLM data download

**Status**: ✅ COMPLETED 2026-05-22 (~35 min total)
**Output**: 16 chunks + 1 val file (397 GB jsonl, 58.9M documents) under `/NHNHOME/.../AUNet/data/dclm_baseline_1.0_2shards_shuffled/`
**`nchunks`**: 16
**.zst raw files deleted** to free 135 GB
**Source name in config**: `dclm_baseline_1.0_2shards_shuffled`

### How to check status
```bash
# Should print 32 (or chosen nchunks)
ls /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/data/dclm_baseline_1.0/*.jsonl 2>/dev/null | wc -l

# Should report ~500 GB
du -sh /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/data/dclm_baseline_1.0/

# Sample first record
head -1 /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/data/dclm_baseline_1.0/*.chunk.0.jsonl | python -c "import json,sys; print(json.loads(sys.stdin.read())['text'][:200])"
```

### How to resume / start
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
source .venv/bin/activate
# <memory_gb> = ~64 (RAM available to terashuf), nchunks=32 → ~500 GB
python setup/download_prepare_hf_data.py \
    dclm_baseline_1.0 \
    64 \
    --data_dir /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/data \
    --seed 42 \
    --nchunks 32
```

The script supports `resume_download=True` in `snapshot_download`, so re-running picks up where it left off. The parquet→jsonl and shuffle steps will re-execute if interrupted (they're cheap relative to download).

### Notes / blockers
_(empty)_

---

## Step 3 — Config adaptation

**Status**: ✅ COMPLETED 2026-05-22
**Output file**: `/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/apps/aunet/configs/aunet2_1.3B_b200.yaml`

### How to check status
```bash
test -f /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/apps/aunet/configs/aunet2_1.3B_b200.yaml && echo present || echo missing

# Verify key fields
grep -E "batch_size|dump_dir|root_dir|async_eval_gpus|steps" /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua/apps/aunet/configs/aunet2_1.3B_b200.yaml
```

Expected:
- `batch_size: 48`
- `async_eval_gpus: 0`
- `dump_dir: /NHNHOME/.../AUNet/runs/aunet2_1.3B`
- `root_dir: /NHNHOME/.../AUNet/data`
- `steps: 180000`

### How to resume / start
Copy `2B_1level.yaml` and apply the diff documented in `plan.md` Step 3.

### Notes / blockers
_(empty)_

---

## Step 4 — Training

**Status**: ✅ COMPLETED — reached step 180,000 (~2026-05-31); final checkpoint `0000180000`
**Run directory**: `/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/`
**Log**: `train.log` (active)
**Expected wall time**: ~158 hours (~6.6 days). `curr_iter_time=0.80s` reported per micro-batch (×4 grad_acc + opt = 3.16s/step actually observed).
**Step 623 @ 33min**: loss 6.05 → 1.42, grad norm 28 → 1.6, wps 125k/GPU, mem 132/178 GB, ~855W
**Per-step time**: 3.16 s (measured from log timestamps)
**GPU util**: 94-100% across all 4 B200s
**Throughput**: 500k bytes/sec total ≈ 56% of paper's 225k/H100×16=3.6M; gap is mainly `compile=false`

### B200/torch-2.7 fixes (in order applied)
1. **FA3 disabled** (train.py:22, eval.py:17) — xformers 0.0.30 ships sm_90 kernels only.
2. **Custom DTensor-safe clip_grad** (train.py:455-475) — manual `to_local()` + `torch.distributed.all_reduce` (default group) for the L2 grad norm; bypasses DTensor's functional collectives.
3. **`ENABLE_INTRA_NODE_COMM=0`** in `lingua/distributed.py:88` — *the actual root cause*: Meta's cluster-specific SymmetricMemory backend raises `get_group_info: no group info associated with the group name` on this cluster.
4. **batch_size 12 + grad_acc 4** — preserves paper's global batch = 192 across 4 GPUs.
5. **compile=false** + **SAC=false** — kept off after dead-ends; even with INTRA_NODE_COMM=0 these may need their own fixes for B200.

### Crash log archive
- `train.log.crash1` — FA3 (xformers Hopper-only kernels)
- `train.log.crash2` — get_group_info in `clip_grad_norm_` under compile
- `train.log.crash3` — OOM at bs=24, compile=off
- `train.log.crash4` — SAC + inplace LongTensor mutation
- `train.log.crash5` — get_group_info in `full_tensor()` under compile=on
- `train.log.crash6` — get_group_info in `full_tensor()` under compile=off (DTensor path)
- `train.log.crash7` — get_group_info in `torch.distributed.all_reduce` (default group) — revealed it's not DTensor, it's the env

### How to check status
```bash
# Latest checkpoint step
ls -1 /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/checkpoints/ 2>/dev/null | sort -n | tail -1

# Is training process alive?
pgrep -af "apps.aunet.train" || echo "no training process running"

# Tail the log
tail -50 /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/train.log 2>/dev/null
```

### How to resume / start
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
source .venv/bin/activate
# Lingua auto-resumes from latest checkpoint in dump_dir/checkpoints/
nohup torchrun --nproc-per-node 4 -m apps.aunet.train \
    config=apps/aunet/configs/aunet2_1.3B_b200.yaml \
    > /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/train.log 2>&1 &
echo "PID: $!"
```

### Milestones (update as we hit them)
- [x] First step completes without OOM
- [x] Step 1k — loss decreasing, throughput logged
- [x] Step 18k (10%) — intermediate eval
- [x] Step 90k (50%) — intermediate eval
- [x] Step 180k (100%) — training complete (~2026-05-31)

### Notes / blockers
_(empty)_

---

## Step 5 — Evaluation

**Status**: ✅ COMPLETED — results in `runs/aunet2_1.3B/eval/results.json`
**Targets**: HellaSwag 64.2 ± 1.0 (63.2-65.2), ARC-Easy 64.4 ± 1.9 (62.5-66.3)

### How to check status
```bash
# Results file (lm-eval writes a JSON)
find /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B -name "results*.json" -mtime -7
```

### How to resume / start
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua
source .venv/bin/activate
LATEST=$(ls -1 /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/checkpoints/ | sort -n | tail -1)
python -m apps.aunet.eval \
    config=apps/aunet/configs/aunet2_1.3B_b200.yaml \
    ckpt_dir=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs/aunet2_1.3B/checkpoints/$LATEST
```

### Final results
- HellaSwag: **62.64** (target 64.2 ± 1.0 → band 63.2-65.2) — ❌ 0.6 below band
- ARC-Easy:  **66.50** (target 64.4 ± 1.9 → band 62.5-66.3) — ❌ 0.2 above band
- Verdict: close to paper but HS slightly under-reproduced; plausibly DCLM-subset + numerics variance. Phase 2/3 (BPEByte) now exceeds paper's HS (64.8).

### Notes / blockers
_(empty)_

---

## How to resume after abort

1. Read this file top-to-bottom.
2. The "Current step" line and the unchecked boxes tell you where work stopped.
3. For each incomplete step, run the "How to check status" block — if it reports success, mark the step done and move on.
4. For the first step that's actually incomplete, run its "How to resume / start" block.
5. Update this file's "Current step" and "Last updated" lines, and check off completed milestones.

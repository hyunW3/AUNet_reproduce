# Scaling ladder — checkpoints & run commands (runbook)

Maps every cell (scale × recipe × family) to its **exact checkpoint folder**, its **config**, and the
**train / eval command**. Repo root `$A = /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet`;
`$L = $A/lingua`; venv `$L/.venv`. ece host `ece-agpu11` (user hwbae, `/home/hwbae/AUNet`, **not
mounted on B200** — reach via ssh). Byte-model training needs an **exec-capable TMPDIR** (`/tmp` is
noexec) → `TMPDIR=$A/../tmp`.

## 1. Checkpoint map

`bpp101` = deprecated old 760M. `⟳` = resuming. `⏳` = gated/not yet trained. `purged` = weights
deleted (keep=1 after eval); `metrics.jsonl`+eval `results.json` remain.

| Scale | Recipe | Family | Checkpoint folder |
|--:|:--|:--|:--|
| **100M** | law | rg | `ece:…/portable_aunetlaw/lb_rg_100M/checkpoints/0000053504` |
| | law | AU-Net | `ece:…/lb_aunet_100M/checkpoints/0000053504` |
| | law | Llama | `ece:…/lb_llama_100M/checkpoints/0000011891` |
| | law | hybrid | `$A/runs/main/100M/hybrid_100M/checkpoints/0000053504` |
| | γ10 | rg | `$A/runs/small/cmp_g10/v4_root_greedy` — **purged** |
| | γ10 | AU-Net | `$A/runs/main/100M_adhoc/aunet_100M/checkpoints/0000003344` |
| | γ10 | Llama | `$A/runs/small/cmp_100M/llama_100M/checkpoints/0000002200` |
| | γ10 | hybrid | `$A/runs/main/100M/hybrid_100M_g10/checkpoints/0000003344` |
| **300M** | law | Llama | `ece:…/llamalaw_300M/checkpoints/0000017836` |
| | law | rg | `ece:…/rgllaw_300M/checkpoints/0000040128` ⟳→`0000080256` |
| | law | AU-Net | `ece:…/aunetllaw_300M/checkpoints/0000040128` ⟳→`0000080256` |
| | law | hybrid | `$A/runs/main/300M/hybrid_300M/checkpoints/0000120752` |
| | γ10 | rg | `$A/runs/small/cmp_g10/rg_300M` — **purged** |
| | γ10 | AU-Net | `$A/runs/small/cmp_g10/aunet_300M/checkpoints/0000009900` |
| | γ10 | Llama | `$A/runs/small/cmp_g10/llama_300M` — **purged** |
| | γ10 | hybrid | `$A/runs/main/300M/hybrid_300M_g10/checkpoints/0000009900` |
| **760M** | γ10 (210bpp) | rg | `$A/runs/main/760M/rg_760M/checkpoints/0000060600` |
| | γ10 | AU-Net | `$A/runs/main/760M/aunet_760M/checkpoints/0000060600` |
| | γ10 | Llama | `$A/runs/main/760M/llama_760M/checkpoints/0000026600` |
| | law | hybrid | `$A/runs/main/760M/hybrid_760M/checkpoints/0000193867` ⏳ gated |
| | (bpp101) | rg/AU-Net/Llama | `$A/runs/main/760M_bpp101/{bpebyte_root_greedy_760M,aunet2_760M,llama_760M}/checkpoints/00000{29200,29200,12900}` |
| **1.3B** | law | rg | `$A/runs/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000` |
| | law | AU-Net | `$A/runs/1.3B/aunet2_1.3B/checkpoints/0000180000` |
| | law | Llama | `$A/runs/1.3B/llama_1.8B_paper/checkpoints/0000060000` |
| | law | hybrid | `$A/runs/1.3B/hybrid_1p3B_leaf_B3/checkpoints/0000180000` |

Downstream `results.json` live next to each ckpt: byte native + hybrid in `evals_5bench/` (or
`greedyQ_full/`,`leafQ_full/`); law-100M in `$A/runs/main/100M/*/eval_law_full/`; law-300M on ece
`portable_aunetlaw/eval_law300M_full/`; 760M-γ10 in `evals_g10/`.

## 2. Configs (all under `$L/apps/aunet/configs/` unless noted)

| Cell | Config |
|:--|:--|
| 100M γ10 rg | `bpebyte_100M_v4_root_greedy.yaml` (steps→3344) |
| 100M γ10 hybrid | `hybrid_100M_g10.yaml` |
| 100M law (rg/AU-Net/Llama/hybrid) | `$A/runs/poc/portable_aunetlaw/{bpebyte_rg,aunet,llama,bpebyte_rg_hybrid_leaf}_100M.yaml` |
| 300M γ10 rg | `bpebyte_300M_v4_root_greedy.yaml` (steps→9900) |
| 300M γ10 hybrid | `hybrid_300M_g10.yaml` |
| 300M law | `$A/runs/poc/portable_aunetlaw/{bpebyte_rg,aunet,llama,bpebyte_rg_hybrid_leaf}_300M.yaml` |
| 760M γ10 rg/AU-Net/Llama | `cmp_g10` run-dir `config.yaml` (in each ckpt folder) |
| 760M law hybrid | `hybrid_760M_aunetlaw.yaml` |
| 1.3B | each run-dir `config.yaml` |
| Llama (subword) train | `$L/apps/main/configs/llama_{100M_overtrain,300M}.yaml` |

## 3. Train — how to run each scale setting

Byte families (rg / AU-Net / hybrid) use `apps.aunet.train`; Llama uses `apps.main.train`. `--nproc`
= GPU count; the config's `dp_replicate` must equal it (γ10 uses dp4; hybrid_100M_g10/300M_g10 dp4;
760M dp4). Auto-resumes from the latest checkpoint in the run's `checkpoints/`.

```bash
cd $L
export TMPDIR=$A/../tmp TORCHINDUCTOR_CACHE_DIR=$A/../tmp/ind_<name>
# byte (rg / aunet / hybrid) — 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --nproc-per-node 4 --master-port 29xxx \
  -m apps.aunet.train config=apps/aunet/configs/<CONFIG>.yaml
# Llama subword
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/torchrun --nproc-per-node 4 --master-port 29xxx \
  -m apps.main.train config=apps/main/configs/<LLAMA_CONFIG>.yaml
```

Per-scale byte budgets (global batch × steps × 8192): **100M** law 48×53504=21.0 GB / γ10 768×3344=21.0 GB ·
**300M** law-hyb 64×120752=63 GB, γ10 768×9900=62.3 GB · **760M** γ10 60600 steps=~150 GB, law-hyb 96×193867=152 GB ·
**1.3B** 192×180000=283 GB.

## 4. Eval — how to run each family (full-sample downstream)

Common: `INC=$L/eval_tasks`, `TOK=$A/tokenizer/llama3/tokenizer.model`,
`TASKS=[hellaswag,arc_easy,arc_challenge,boolq,piqa,winogrande]`, `CFG=apps/aunet/configs/eval_full_5bench_b200.yaml`.
Prepend `TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1` (eval).

```bash
# rg (online-greedy) & AU-Net (word) — NATIVE regime, no override
torchrun --nproc-per-node 1 -m apps.aunet.eval config=$CFG harness.tasks=$TASKS validation=null \
  ckpt_dir=<CK> dump_dir=<CK_run>/evals_5bench harness.include_path=$INC

# hybrid — greedyQ (native root-greedy question)
torchrun ... -m apps.aunet.eval config=$CFG harness.tasks=$TASKS validation=null \
  greedy_question_loglikelihood=true ckpt_dir=<CK> dump_dir=<run>/greedyQ_full harness.include_path=$INC
# hybrid — leafQ (offline-leaf question + causal online answer)   ← force flag is MANDATORY
torchrun ... -m apps.aunet.eval config=$CFG harness.tasks=$TASKS validation=null \
  offline_question_loglikelihood=true regex_bpe_tokenizer_path=$TOK force_bpe_online_mode=greedy \
  ckpt_dir=<CK> dump_dir=<run>/leafQ_full harness.include_path=$INC

# Llama (subword)   [B200: eval_cmp_300M_llama.yaml · ece: eval_downstream_main.yaml]
torchrun ... -m apps.main.eval config=apps/main/configs/eval_cmp_300M_llama.yaml \
  harness.tasks=$TASKS harness.include_path=$INC ckpt_dir=<CK> dump_dir=<run>/evals_5bench
```

## 5. Automation scripts (as run)

| Script | Purpose |
|:--|:--|
| `$A/runs/hybrid100g10_chain.sh` | wait for 300M pipeline → train γ10 100M hybrid → eval greedyQ+leafQ |
| `$A/runs/hybrid300g10_eval_after.sh` | after γ10 300M hybrid trains → eval both regimes |
| `$A/runs/hybrid760_law_gate.sh` | **gated**: fire after 02:00 when all 4 B200 GPUs idle → train hybrid 760M law (152 GB) → eval |
| ece `…/portable_aunetlaw/resume_chain_300M.sh` | resume rgllaw/aunetllaw 300M (40128→80256) → eval |
| ece `…/portable_aunetlaw/eval_law300M_gpu7.sh` | GPU7 eval of law-300M runs |
| ece `…/portable_aunetlaw/eval_law100M_full_ece11.sh` | law-100M full downstream (rg/aunet/llama/hybrid/…) |

## 6. Reproduce a single number (example)

```bash
# 1.3B hybrid leafQ downstream:
A=/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet; cd $A/lingua
export TMPDIR=$A/../tmp TORCH_COMPILE_DISABLE=1
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc-per-node 1 --master-port 29711 -m apps.aunet.eval \
  config=apps/aunet/configs/eval_full_5bench_b200.yaml \
  harness.tasks=[hellaswag,arc_easy,arc_challenge,boolq,piqa,winogrande] validation=null \
  offline_question_loglikelihood=true regex_bpe_tokenizer_path=$A/tokenizer/llama3/tokenizer.model \
  force_bpe_online_mode=greedy \
  ckpt_dir=$A/runs/1.3B/hybrid_1p3B_leaf_B3/checkpoints/0000180000 \
  dump_dir=$A/runs/1.3B/hybrid_1p3B_leaf_B3/leafQ_repro harness.include_path=$A/lingua/eval_tasks
```

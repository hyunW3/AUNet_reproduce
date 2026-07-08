# Native µP + Proxy-Model Hyperparameter Search

## Context

Find optimal hyperparameters the way *"Scaling with Collapse: Efficient and
Predictable Training of LLM Families"* (arXiv 2509.25087, Cerebras) does in
Appendix B.1: tune **maximal-update-parameterization (µP)** base HPs once on a small
proxy model, then transfer zero-shot to every larger width in the family. Underpins
the "collapse" scaling work in `reports/collapse_scaling.md`.

**Paper's method (to replicate):** a 39M proxy (width `d_proxy=256`, 24 layers,
head 64) trained on 800M tokens; **randomly sample ~350 configs** of four base HPs —
base LR `η̄`, base init std `σ_W,base`, input multiplier `α_input`, output multiplier
`α_output` — keep the top performer by final train loss. Tuned values (Table 5):
σ_W,base=8.67e-2, η̄=1.62e-2, α_input=9.17, α_output=1.095. Transfer rule: reported
LR is the *base* µP LR; actual per-model LR = `η̄·(d_proxy/d_model)`.

**Problem:** lingua has **no µP** — `build_optimizer` (`lingua/lingua/optim.py:149`)
makes one uniform-LR `AdamW(model.parameters())`; init is fixed `dim**-0.5`; there
are no input/output multipliers. Implement native µP first, then the proxy search.

**Decisions (locked with user):** native µP in lingua (not the `mup` pip lib);
target **all three families** — Llama (plain, single width) + AU-Net(word) & BPEByte
(hierarchical, per-level widths); **scaled-down** sweep (~120 configs/family, ~40M
proxy, ~300M tokens) on the 4 idle **ece-agpu18** A100-80GB GPUs.

**Outcome:** a Table-5-equivalent of tuned base HPs per family, validated by a
coordinate/LR-transfer check, ready to transfer to the 100M–1.3B models.

## µP scaling rules (Adam variant; `w = d/d0`, d0 = base/proxy width)

| Param class | Modules | Init std | Base-LR mult | Forward mult |
|---|---|---|---|---|
| Input embedding | `tok_embeddings` | `σ_base` (width-indep.) | ×1 | ×`α_input` |
| Hidden / matrix | `wq,wk,wv,wo,w1,w2,w3`, transitions, `indexed_linear` | `σ_base·(d0/d)^0.5` | ×`(d0/d)` | — |
| Output / unembed | `vocab` (aunet) / `output` (llama) | `σ_base·(d0/d)` | ×`(d0/d)` | ×`α_output` |
| Norm / bias / scalar | RMSNorm weights | unchanged | ×1 | — |

Reduces exactly to current behavior when `σ_base=d0**-0.5`, `w=1`, `α=1` (strict
generalization → non-µP runs unchanged). Keep RoPE and `1/sqrt(head_dim)` attention
as-is: head_dim is width-invariant across our scaling ladder, so µP's `1/d`
attention rule is unnecessary (documented assumption; revisit only if head_dim ever
scales with width).

## Files to modify

1. **`lingua/lingua/optim.py`** — add nested `MupArgs` on `OptimArgs`
   (`enable=False`, `base_width`, `base_widths: Optional[List[int]]` for hierarchy,
   `base_lr`, `base_init_std`, `alpha_input`, `alpha_output`, `lr_vector_ratio=1.0`).
   Rewrite `build_optimizer` (line 149): when `enable`, build **param groups** by
   iterating `named_parameters()` (stable + DTensor-safe post-FSDP2), classify each
   param, set group base LR = `base_lr·(d0/d)` for matrix/output, `base_lr·ratio`
   for embed/vector; norms in a `weight_decay=0` group. Keep the existing `LambdaLR`
   (line 162) — it scales all groups' base LRs uniformly. Non-µP path (single group)
   untouched. Assert every param lands in exactly one group.

2. **`lingua/lingua/transformer.py`** — width-scaled init. Add µP fields to
   `BaseTransformerArgs` (`mup_enable/base_width/base_init_std`). Thread `base_std`,
   `width_ratio=base_width/dim` through `BaseTransformer.init_weights` (line ~613) →
   `TransformerBlock.init_weights` → `Attention.reset_parameters` (line ~435) &
   `FeedForward.reset_parameters` (line ~499); matrix std = `base_std·width_ratio^0.5`.
   Compose with existing depth `factor` on `wo`/`w2` (width-independent → safe).

3. **`lingua/apps/main/transformer.py`** (Llama) — add µP fields to
   `LMTransformerArgs`; in `forward` (line ~105) apply `h = α_input·tok_embeddings`
   and `logits = α_output·output(norm(h))`; in `reset_parameters` (line ~121) set
   embedding std=`σ_base`, output std=`σ_base·(d0/d)` (guard `weight_tying`: tied →
   single shared param in embedding group, both multipliers still applied). Add a
   `mup_param_groups()` helper returning `(width, base_width, class)` per param.

4. **`lingua/apps/aunet/hierarchical.py`** (AU-Net + BPEByte, the hard case) —
   **per-level µP**: `dimensions` is a list, so `mup_base_widths` is a list aligned
   to it; level `k` uses `w_k = dimensions[k]/base_widths[k]` (each level its own
   base — correct when only some levels widen, e.g. `[512,768]→[512,2048]`). Set
   per-level `block.mup_*` when building `CausalTransformerArgs` (line ~527).
   `SimpleTransition.reset_parameters` (line ~435): scale `trans_down`/`trans_up`
   each by its own fan-in width. `forward` (line ~581/620): `α_input` at
   `tok_embeddings`, `α_output` at `vocab(vocab_norm(x))`. `reset_parameters`
   (line ~678): embed std=`σ_base`, `vocab` std=`σ_base·(base_widths[0]/dimensions[0])`.
   Add `mup_param_groups()` walking `encoders.{k}/decoders.{k}/trunk/transitions.{k}/
   tok_embeddings/vocab` — single source of the width map (unit-tested).

5. **`lingua/apps/aunet/train.py`** + **`lingua/apps/main/train.py`** — right after
   config load and before model construction, mirror `args.optim.mup.*` init/multiplier
   fields into `args.model.*` (canonical namespace stays `optim.mup.*` for the sweep
   CLI). Both already call `build_optimizer` after `parallelize_model`.

## Proxy configs (new; ~40M, d_proxy=256, ~300M tokens, 1 GPU each)

- `apps/main/configs/mup_proxy_llama.yaml` (from a 100M llama config): `dim=256,
  n_layers=24, n_heads=4` (head_dim 64), `seq_len=2048, batch_size=32, grad_acc=1,
  steps≈4600, warmup≈460, weight_decay=0, cosine, lr_min_ratio=0.1`,
  `optim.mup.enable=true, base_width=256`.
- `apps/aunet/configs/mup_proxy_aunet_word.yaml` (from `aunet_100M_word.yaml`):
  `dimensions=[256,384], layers=[2,6], head_dims=[64,64]`, keep `word1` strategy,
  `optim.mup.base_widths=[256,384]`; cut `steps` to ~300M bytes.
- `apps/aunet/configs/mup_proxy_bpebyte.yaml` (from `bpebyte_100M_v4_root_greedy.yaml`):
  same width/layer shrink, keep the `bpe_br`/root_greedy `regex` block + tokenizer
  path override to ece-local `/home/hwbae/AUNet/tokenizer/llama3/tokenizer.model`.
- All: `eval=null`, `eval_milestones=[]`, `checkpoint.dump.every`>steps (clean
  rerun), `logging.freq=10` so `metrics.jsonl` `loss/out` is dense.

## Sweep harness (new; patterned on `scripts/gen_ml_r40.py` + `runs/_scripts/scaling_rg_driver.sh`)

- `scripts/gen_mup_sweep.py`: for each family × ~120 trials sample log-uniform around
  the paper's anchors — `σ_W,base∈[3e-2,2.5e-1]`, `η̄∈[4e-3,6e-2]`, `α_input∈[2,20]`,
  `α_output∈[0.5,3]` (fixed per-trial seed). Emit per-trial YAML (inject
  `optim.mup.enable/base_lr/base_init_std/alpha_input/alpha_output`) + `manifest.jsonl`.
- `runs/_scripts/mup_sweep_driver.sh`: idle-GPU-gated (`gpus_busy` awk pattern),
  ≤4 concurrent `torchrun --nproc-per-node 1` jobs each pinned to one ece GPU via
  `CUDA_VISIBLE_DEVICES`, sentinel `_done_<trial>` for resume, port bump per slot.
- `runs/_scripts/mup_rank.py`: read last `loss/out` per trial's `metrics.jsonl`, join
  `manifest.jsonl`, sort per family, write `mup_tuned_hps.md` (best 4-tuple + top-5).
  Re-run top-5 with a 2nd seed, rank by mean (kills ranking noise).

## Coordinate / transfer validation (before trusting the sweep)

- **Coord check**: with µP on, instantiate proxy + 2×/4× widths (Llama 256/512/1024;
  AU-Net level-1 384/768/1536), run ~50 steps, log per-layer activation RMS (reuse
  `RMSNorm.log_stats` `transformer.py:312`, `check_model_value_range`
  `distributed.py:339`) — should be ~width-invariant.
- **LR-transfer check**: for 3 widths sweep `base_lr` over ~6 log-spaced values at a
  tiny budget; plot final `loss/out` vs `base_lr`. **argmin base_lr must align across
  widths** — the definitive µP test.
- **Sanity/regression**: `enable=true, base_width=d, base_init_std=d**-0.5, α=1` must
  reproduce non-µP numerics; diff a short non-µP `metrics.jsonl` vs pre-change baseline
  to prove `enable=False` is byte-identical.

## Risks

- FSDP2 + param groups: iterate `named_parameters()`, assert full coverage (don't drop
  params). torch.compile: keep multipliers as Python floats (graph-folded).
- Hierarchical width map wrong-but-running → `mup_param_groups()` is single source +
  unit test group LRs = `base_lr·(d0/d)`; transitions span two widths (scale each
  fan-in independently). Tied embeddings: guarded edge case (all current configs
  untied). Ranking noise on 40M/300M-token runs → 2nd-seed re-rank of top-5.

## Verification (end-to-end)

1. Unit: `mup_param_groups()` LRs equal `base_lr·(d0/d)` for sampled params; group
   coverage == total params.
2. Regression: non-µP short run identical to baseline `metrics.jsonl`.
3. Coord check + LR-transfer check pass (width-invariant activations & aligned LR
   optima) for each family.
4. Full sweep runs on ece (4 GPUs), `mup_rank.py` emits `mup_tuned_hps.md` with a
   tuned (σ_W,base, η̄, α_input, α_output) per family; compare Llama's numbers against
   the paper's Table 5 as a plausibility anchor.
5. Transfer smoke: apply tuned base HPs to a 100M config, confirm loss ≤ the existing
   hand-tuned 100M baseline.

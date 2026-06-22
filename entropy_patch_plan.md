# Entropy-model patching (BLT-style) — prep plan

**Goal.** Add a causal, leak-free patch-boundary scheme driven by a trained **entropy model** (a
small byte-level LM), à la BLT: place a boundary where the next-byte entropy is high (the model is
"surprised"). This is a *learned* boundary source — orthogonal to the BPE-trie modes (greedy / bt /
prefix_free / prefix_vocab) and to V5's k-gram distill. Runs **after the entropy model is trained**.

## Why it fits this project
- **Causal + leak-free by construction:** entropy H_t is computed from bytes ≤ t only (the entropy
  model is autoregressive), so a boundary at t never depends on a future byte. Same leak-free+causal
  property as root_greedy / prefix_free / prefix_vocab, but the signal is *learned content surprise*
  rather than vocab-trie structure.
- **Reuses existing plumbing:** V5 already plugs a learned causal predictor into the boundary
  interface. The entropy patcher is the same shape with a different decision function.

## Two-phase structure
### Phase 1 — train the entropy model (prerequisite)
A small byte-level LM that outputs a next-byte distribution per position.
- Arch (proposal): flat byte transformer, dim 512 / 6–8 layers / head_dim 64, seq_len 8192,
  byte vocab 256 (+specials). ~30–50M params (BLT uses ~100M; smaller is fine for 100M-scale
  boundary signal). Train on the same DCLM shards.
- Budget (proposal): ratio-5…ratio-10 (the boundaries just need a decent entropy estimate, not a
  converged LM). Reuse `apps.main.train` (byte tokenizer) or a stripped AU-Net level-1.
- Output: a checkpoint that, given a byte prefix, gives p(next byte) → H_t = −Σ p log p.

### Phase 2 — entropy patching (what we PREPARE now)
1. **Precompute** per-byte entropy over the training corpus once (BLT does this offline) → store a
   compact per-shard entropy array, mirroring how V5 stores its distill table. Avoids running the
   entropy model inside the hot dataloader.
2. **Boundary rule** (BLT's two options; implement both, choose by config):
   - **Global threshold:** boundary at t iff `H_t > θ_g`.
   - **Approx-monotonicity:** boundary at t iff `H_t − H_{t-1} > θ_r` (entropy jumps up).
   θ chosen to hit a target mean patch length (~4 bytes, matched to greedy/bt for a fair compare).
3. **Placement:** boundary marks the START of the new patch (high-entropy byte) → **root** placement
   (consistent with v4/v7), keeping it leak-free.

## Integration (mirror V5 — minimal new code)
- `apps/aunet/data/entropy_patch.py`: `EntropyBoundaryPredictor` (analog of
  `DistilledBoundaryPredictor`): load precomputed entropies (or the entropy-model ckpt), expose
  `mask01(byte_seq)` / `boundary_positions(byte_seq)` applying the threshold/monotonicity rule +
  a max-run cap (bounds patch length, like distill). `EntropyIncrementalParser` (analog of
  `DistilledIncrementalParser`) for generation — runs the entropy model autoregressively, naturally
  causal, same feed/committed_levels/snapshot/restore interface.
- `regex_cutting.py`: add `bpe_online_mode == "entropy"` branch in `online_byte_boundaries`
  (like the `distilled` branch); `RegexArgs.bpe_entropy_model_path` + `bpe_entropy_threshold` /
  `bpe_entropy_mode` ("global" | "monotonic"). Placement = root.
- Config `apps/aunet/configs/r20_v8_entropy.yaml` (copy r20_v4_root_greedy; mode=entropy + paths).

## Experiment
100M ratio-40 (13376 steps), same eval as the family. Compare HS/ARC-E/PIQA/ARC-C against:
v4_root_greedy **31.92**, v6_prefix_free **32.70**, v7_prefix_vocab *(pending ~04:30)*; baselines
aunet 32.37 / llama 35.60. Headline question: does a *learned* surprise-based boundary beat the
*structural* trie boundaries at matched scale, while staying leak-free+causal?

## Open choices (need user input)
1. **Entropy model size / budget** — ~30M ratio-5 (cheap, fast) vs ~100M BLT-like (closer to paper).
2. **Boundary rule** — global threshold, approx-monotonicity, or run both.
3. **When to train the entropy model** — on the GPUs freed by v7 (~04:30) / warm_v1 (~07:00), and
   in what order relative to the still-paused 300M.

## Status
- 300M committed **PAUSED** (requeue runner killed; will not auto-start at `_v7_done`).
- Running: v7 prefix_vocab (0,4,5,7, ~04:30), warm_v1 (1,2,3,6, ~07:00).
- This doc = prep only; no entropy code written / nothing launched yet — awaiting the choices above.

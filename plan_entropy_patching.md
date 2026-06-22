# Entropy patching in AU-Net: an entropy-based member of the BPEByte family

**TL;DR.** BLT (Pagnoni et al. 2024, *Byte Latent Transformer*, arXiv:2412.09871) replaces a
fixed-vocabulary tokenizer with **entropy patching**: a small byte-level LM scores every byte's
next-byte entropy `H(x_t)`, and a patch boundary opens wherever entropy spikes. This document plans
how to drop entropy patching into our AU-Net stack **exactly the way `root_greedy` was dropped in** —
as a new boundary scheme (`bpe_online_mode: entropy`) that emits the same per-byte `level_mask` the
model already consumes, leaving `hierarchical.py` untouched. Entropy patching is *incremental/causal*
by construction (BLT §2.4: `f_p(x_{<i}) = f_p(x)_{<i}`), which is the **same leak-free property** that
makes `root_greedy` honest — so it slots directly into our train==generation comparison. The doc ends
with a 100M / 300M / 760M / 1.3B configuration matrix to compare **AU-Net (word patches) vs BPEByte
(BPE-trie patches) vs Entropy (entropy patches)**, all sharing one architecture and budget, against the
Llama subword baseline.

Companions: `BPEByte_root_greedy_method.md` (the leak-free trie scheme this mirrors),
`paper_exp_plan260618.md`, `cmp-300M-scope` / `status_300M.md`. Code touch-points:
`lingua/apps/aunet/data/byte_trie.py`, `…/data/regex_cutting.py`, `…/hierarchical.py`.

---

## 1. Why this is a *segmentation-function* change, not an architecture change

In our stack the only thing that distinguishes AU-Net (word), BPEByte `root_greedy`, and BPEByte
`br_bt` is the function **`bytes → per-byte boundary mask`**. Everything downstream is shared
(`BPEByte_root_greedy_method.md` §1). The contract the model consumes is a single integer array:

- `level_mask[t] = L` ⟹ byte `t` is a boundary for every level `< L`; for a 2-level model it is
  effectively **binary** (0/1) — 1 marks the first byte of a patch
  (`regex_cutting.py:396-401`, `data.py:244`).
- `level_mask[0]` is forced to the max level (BOS is a boundary at every level), both in data
  (`regex_cutting.py:400`) and defensively in the model (`force_first`, `hierarchical.py:621-622`).
- The model's only segmentation entry point is
  `HierarchicalTransformer.forward(token_values, level_mask, …)` (`hierarchical.py:555-563`); inside,
  `get_pool_mask` reads only `level_mask > i` (`hierarchical.py:614-641`), down-pooling gathers the
  boundary bytes (`MaxSumMask` + `trans_down`, `hierarchical.py:252-348`) and up-pooling broadcasts
  patch states back via `repeat_idx = mask.cumsum(1)` (`hierarchical.py:388`).

**Consequence:** an entropy patcher only has to emit `level_mask` in this layout. No change to
`hierarchical.py`. This is precisely how `root_greedy` is wired — through a `bpe_online_mode` dispatched
in `ByteTrie.boundaries` (`byte_trie.py:169-181`) and surfaced by
`RegexPool.online_levels_mask` (`regex_cutting.py:362-401`). Entropy patching becomes a new value of
`bpe_online_mode`.

---

## 2. BLT entropy patching — key components

### 2.1 The entropy model (BLT §4.2)
A small **byte-level autoregressive LM** trained on the *same* data distribution as the main model.
BLT default: **100M params, 14 layers, hidden dim 512, sliding-window attention 512 bytes**, byte
vocab. It produces, at every position, the next-byte distribution `p_e(·|x_{<i})` over the byte
vocabulary `V` and the entropy

```
H(x_i) = − Σ_{v∈V} p_e(x_i = v | x_{<i}) · log p_e(x_i = v | x_{<i})        (BLT Eq. 1)
```

Ablation (BLT Fig. 8): performance improves with entropy-model size and context window but **saturates
beyond ~50M params with a 512-byte window** (window 512 ≫ 128 ≫ 64). When the receptive field is small,
the entropy model can even be baked into a lookup table (BLT §4.2).

### 2.2 The two boundary rules (BLT §2.3)
A patch boundary (start of a new patch, `1` in the {0,1} boundary map) opens when:

```
Global constraint:               H(x_t) > θ_g
Approx.-monotonic constraint:    H(x_t) − H(x_{t-1}) > θ_r
```

The monotonic rule fires at points that *break* the approximately-monotone entropy decay inside a
patch (entropy is high at a word/unit start, then falls). BLT prefers the **monotonic** rule at large
scale because it is robust to *entropy drift* (§2.4 below).

### 2.3 Patch-size control (BLT §4.3)
`patch_size` (avg bytes/patch) is **free** — set the threshold to hit a target on the pretraining mix.
BLT's compute-optimal entropy patches and the Llama-3 BPE tokenizer both sit at **≈4.5 bytes/patch**;
BLT also studies **6** and **8** bytes/patch (the "patches scale better than tokens" axis — larger
patches → fewer global-transformer steps → reallocate FLOPs to a bigger trunk). To keep context length
constant across patch sizes, BLT holds *bytes-per-batch* fixed and shrinks the patch-sequence budget.

### 2.4 Entropy drift + incremental patching (BLT §2.4, §4.4)
- **Drift:** on repetitive/structured content (e.g. MCQ) the entropy model gets over-confident, entropy
  falls, patches grow without bound. Fix BLT uses at scale: **reset the entropy context at newlines**
  and use the **monotonic** rule (less drift-sensitive). There is also an inference-time threshold
  shift (θ 0.6→0.1) to buy more steps on hard inputs.
- **Incremental / causal:** BLT requires `f_p(x_{<i}) = f_p(x)_{<i}` — the boundary at position `i`
  must not depend on bytes ≥ `i`. Entropy patching satisfies this because `H(x_i)` is computed from a
  *causal* LM on `x_{<i}`. **BPE does not** (same prefix segments differently given different
  continuations). **This is the identical property that makes our `root_greedy` leak-free**
  (`BPEByte_root_greedy_method.md` §2): entropy patching is a drop-in causal boundary source.

### 2.5 BLT's own architecture (for reference / optional Variant B)
BLT does *not* use AU-Net's gather-pooling. Its three modules (BLT §3): a light **local encoder**
(hash n-gram byte embeddings, n∈3..8, 500k hashes; Perceiver-style **cross-attention** pooling, patch
queries attend to their bytes), a heavy **latent global transformer** (block-causal over patches), and
a light **local decoder** (byte queries cross-attend patch states). Ablations: hash n-grams are vital
(Table 8), cross-attn helps most in the decoder (Table 7), and a 1-layer encoder + heavy decoder is
best (Table 9). **We do not need any of this for the primary plan** — AU-Net's gather/repeat pooling
already plays the encoder/decoder role. It is listed only as an optional stretch variant (§5.3).

### 2.6 BLT headline results (context for our expectations)
8B, BLT-1T (Table 1): avg Llama-3 60.0, BLT-Space 58.0, **BLT-Entropy 61.1** (wins 4/7). Robustness/
character tasks (Table 3): BLT ≫ Llama-3 (CUTE 54.1 vs 27.5; HellaSwag-noise +7.4). Entropy patching
matches BPE scaling at iso-FLOPs and exceeds it on byte-level/robustness axes — the same profile we see
for `root_greedy` (`model_results_1.3B.md`).

---

## 3. How entropy patching maps onto AU-Net (analysis)

| axis | AU-Net word | BPEByte `br_bt` | BPEByte `root_greedy` | **Entropy (this plan)** |
|---|---|---|---|---|
| boundary source | whitespace/word regex | BPE trie, longest-match+backtrack | BPE trie, greedy dead-end | small byte LM, `H(x_t)` spike |
| lookahead to cut? | no | **yes** (re-feed) | no | **no** (causal LM on `x_{<t}`) |
| boundary visible when predicting it? | no | **yes** (before_root) | no (root) | **no** (boundary = patch start) |
| incremental `f_p(x_{<i})=f_p(x)_{<i}`? | yes | **no** | **yes** | **yes** |
| net | leak-free | leaky (×2) | **leak-free + causal** | **leak-free + causal** |
| avg bytes/patch | ~data-dependent | ~4.5 | ~coarser/irregular | **tunable (target 4.5)** |

**Placement.** A high-entropy byte is the *first* byte of the new patch, so the boundary marks a patch
start — structurally identical to `root` placement (`off=0`, boundary index `e`,
`regex_cutting.py:285-290`). The same argument that makes `root_greedy` leak-free in the up-pooling
decoder applies verbatim: predicting byte `e` reads `repeat_idx[e-1]`, which **excludes** the boundary
at `e` (`hierarchical.py:388`; `BPEByte_root_greedy_method.md` §2b). So entropy patching is leak-free
on both the *mode* axis (causal entropy) and the *placement* axis (patch-start boundary).

**Down-pool semantics.** AU-Net's `down()` gathers the boundary byte's hidden state
(`hierarchical.py:335-346`). Under entropy patching that boundary byte is the *highest-uncertainty*
byte in its neighborhood — a sensible representative for the trunk to attend over. This is a cleaner
match to AU-Net's "pick the boundary byte" pooling than `root_greedy` (whose greedy dead-ends are
linguistically arbitrary).

**What entropy patching buys over `root_greedy`.** (a) Boundaries are *data-driven* and *vocabulary-
free* — no Llama-3 BPE vocab dependency, so it answers the open audit question "is BPE a prior or just
a prediction-unit choice?" (`bpebyte-rootgreedy-audit` point 1) by providing the **non-BPE causal
control** the PoC never ran. (b) `patch_size` is a continuous knob, enabling the "patches scale better"
study (ps 4.5/6/8) that the trie cannot do. (c) Boundaries land on information-density spikes rather
than vocab dead-ends.

**Costs / asymmetries to keep honest.**
- The entropy model is an *external preprocessing artifact* (like the Llama-3 vocab is for
  `root_greedy`): trained **once** on DCLM, boundaries precomputed and disk-cached
  (`REGEX_SEG_CACHE`, `regex_cutting.py`), **not** counted in main-model params.
- **Unlike** the trie (O(n), free), entropy patching needs an entropy-model forward pass per byte at
  *generation* time to decide boundaries online. Mitigation: small window (512) → cheap; BLT notes it
  can be a lookup table. Report this as a small fixed inference overhead.
- One-time entropy-model **training compute** must be disclosed in any FLOP-matched claim.

---

## 4. Calibration protocol (do this before training anything)

1. **Train the entropy model once.** A flat byte-LM (vocab 258), dim 512, 14 layers, sliding-window
   512, RoPE θ=500000, on the same `dclm_baseline_1.0_2shards_shuffled` shard the main models use.
   Target ~50–100M params (Fig-8 saturation). Reuse `apps/main` transformer with `vocab_size: 258`, or
   the byte encoder block from `apps/aunet`. Save checkpoint to `tokenizer/entropy_model/`.
2. **Threshold sweep for iso-patch-size.** On a held-out DCLM slice, sweep `θ_g` (global) and `θ_r`
   (monotonic) and record avg bytes/patch. Pick `θ` that hits **4.5 bytes/patch** = the `root_greedy`
   / Llama-3 operating point, so trunk patch-counts (and thus trunk FLOPs) match `root_greedy`
   per-step. Record the chosen `θ` in the config and in eval JSON (audit point 5 — embed segmenter).
3. **Drift guard.** Enable newline context reset + use the **monotonic** rule for the ≥760M runs
   (BLT §4.4); validate on an MCQ-shaped probe that avg patch size on MMLU-like text ≈ that on prose.
4. **Leak verification.** Run the existing `probe_root_causal.py` / `probe_streaming_leak.py` on the
   entropy boundaries to confirm 0% future-byte leakage (must reproduce the `root_greedy` result, see
   `causal-segmentation-leak-tradeoff`).

---

## 5. Integration design

### 5.1 Primary (Variant A) — entropy boundaries, AU-Net pooling unchanged
Minimal surface, mirrors `root_greedy`:

1. **New boundary producer.** Add `entropy_boundaries(byte_array, entropy_model, θ, rule)` returning
   patch-start indices, alongside the trie modes in `byte_trie.py:169-181`. It runs the cached entropy
   model forward, computes `H(x_t)`, applies the global or monotonic rule, optional newline reset.
2. **Wire into the pool.** Dispatch a new `bpe_online_mode: entropy` in
   `RegexPool.online_byte_boundaries` (`regex_cutting.py:265-290`) and surface the mask through
   `online_levels_mask` (`regex_cutting.py:362-401`). Placement is fixed to `root` (patch-start).
   Add `RegexArgs` fields: `entropy_model_path`, `entropy_rule∈{global,monotonic}`, `entropy_theta`,
   `entropy_newline_reset` (`regex_cutting.py:90-132`).
3. **No model change.** `hierarchical.py` consumes the resulting `level_mask` unchanged.
4. **Caching.** Boundaries are a pure function of the prefix + (frozen entropy model, θ, rule) → reuse
   `REGEX_SEG_CACHE` keyed on those.

### 5.2 Generation (streaming)
Provide a streaming parser with the existing `ByteTrieIncrementalParser` interface
(`byte_trie.py:184-294`: `.feed`, `.committed_levels`, `.snapshot/.restore`, `.pos_at_root`) backed by
the entropy model instead of the trie. Because the rule is causal and never backtracks (a boundary,
once opened, is permanent), the streaming frontier is monotone and **train==generate** — same guarantee
as `root_greedy`. Hook into `generate_bt.py:120-121,184-195`.

### 5.3 Optional (Variant B) — full BLT pooling (stretch / ablation only)
If we want to test whether BLT's *architecture* (not just its boundaries) matters: add hash n-gram byte
embeddings + cross-attention pooling in the encoder/decoder. This is a real change to
`hierarchical.py`/`index_matmul.py` and is **out of scope for the headline AU-Net-vs-BPEByte
comparison** — keep it as a single 100M/300M ablation to attribute any gap to "boundaries" vs "pooling".

---

## 6. Configuration plan: 100M / 300M / 760M / 1.3B

**Design rule (iso-everything-but-segmentation).** At each scale, the AU-Net trunk, byte enc/dec,
optimizer, data, seq_len, global batch, steps, and warmup are **identical** across `{word, root_greedy,
entropy_global, entropy_monotonic}`. Only `data.regex` changes. The Llama subword model is the matched
external baseline (core-matched, so larger total params — the established framing). Entropy threshold is
tuned per scale to **4.5 bytes/patch** (= `root_greedy` operating point) for the primary comparison.

### 6.1 Shared AU-Net trunk per scale (from current configs)
All 2-level, `head_dims [64,128]`, `residuals [True]`, `sliding_windows [512,4096]`,
`max_seqlens [-1,3200]`, `rope_theta 500000`, `pooling simple_indexed_matmul`, byte vocab 258,
`seq_len 8192`.

| Scale | `dimensions` | `layers` | AU-Net params | trunk heads | source config |
|---|---|---|---|---|---|
| 100M | `[512, 768]` | `[3, 10]` | 98,591,488 | 6 | `bpebyte_100M_v4_root_greedy.yaml` |
| 300M | `[512, 1280]` | `[3, 13]` | ≈296,659,712 | 10 | `aunet_300M.yaml` |
| 760M | `[512, 1536]` | `[3, 24]` | 714,426,880 | 12 | `aunet2_760M_b200.yaml` |
| 1.3B | `[512, 2048]` | `[3, 25]` | 1,324,203,008 | 16 | `aunet2_1.3B_b200.yaml` |

### 6.2 Llama subword baseline per scale (core-matched)
| Scale | dim / L / heads | params | source config | steps / lr |
|---|---|---|---|---|
| 100M | 768 / 14 / 12 | 296,113,920 | `llama_100M_cmp.yaml` | 2200* / 3.0e-3 |
| 300M | 1280 / 15 / 20 | ≈623M | `llama_300M.yaml` | 8800 / 3.0e-3 |
| 760M | 1536 / 24 / 12 | 1,073,554,944 | `llama_760M_b200.yaml` | 12900 / 5.6e-3 |
| 1.3B | 2048 / 25 / 16 ("1.8B") | 1,809,946,624 | `llama_1.8B_b200.yaml` | 60000 / 3.0e-3 |

\* see `runs/cmp_100M/ratio20_table.md`: canonical ratio-20 readout is at ~6688 (aunet) / 8800 (llama)
steps (~9.2B subword ≈ 42B byte tokens).

### 6.3 Training budget per scale (applies to every byte variant at that scale)
| Scale | batch_size | grad_acc | bytes/step | steps | lr | warmup | status |
|---|---|---|---|---|---|---|---|
| 100M | 96 | 2 | 6.29M | 1672 (ablation) / 6688 (ratio-20) | 2.0e-3 | 100 | trunk trained |
| 300M | 48 | 4 | 6.29M | 6688 | 1.9e-3 | 400 | **configs only** |
| 760M | 12 | 6 | 2.36M | 29200 | 1.65e-3 | 2920 | trunk trained |
| 1.3B | 12 | 4 | 2.36M | 180000 | 1.65e-3 | 10000 | trunk trained |

### 6.4 Entropy model (shared across all four scales — trained once)
| field | value | rationale |
|---|---|---|
| architecture | flat byte transformer, vocab 258 | BLT §4.2 |
| dim / layers | 512 / 14 | BLT default |
| params | ~50–100M | Fig-8 saturation |
| attention window | 512 (sliding) | Fig-8 (512 ≫ 128 ≫ 64) |
| rope θ | 500000 | match stack |
| train data | `dclm_baseline_1.0_2shards_shuffled` | same as main |
| output | per-byte `H(x_t)`, disk-cached boundaries | not counted in main params |

### 6.5 New `data.regex` block (entropy variant)
```yaml
data:
  tokenizer: { name: bytes }
  regex:
    strategy: { bpe_br: 1@1 }
    bpe_online: true
    bpe_online_mode: entropy           # NEW dispatch — §5.1
    bpe_online_placement: root         # patch-start boundary → leak-free (§3)
    bpe_context_prefix: 0
    bpe_online_committed_view: false
    entropy_model_path: tokenizer/entropy_model/consolidated.pth   # NEW
    entropy_rule: monotonic            # global @100M/300M ablation; monotonic @760M/1.3B (drift)
    entropy_theta: <tuned for 4.5 bytes/patch on DCLM>             # §4 step 2
    entropy_newline_reset: true        # @760M/1.3B (BLT §4.4)
model:
  patch_read_delay: 0
```

### 6.6 Run matrix
Per scale, the byte-model family (each = one `data.regex` swap on the shared trunk):

| variant | mode | placement | leak-free | role |
|---|---|---|---|---|
| `aunet_word` | regex `word1` | — | yes | existing AU-Net baseline |
| `bpebyte_root_greedy` | trie greedy | root | yes | existing BPE-trie causal |
| `bpebyte_br_bt` | trie bt | before_root | no | leaky upper-bound diagnostic |
| **`entropy_global`** | entropy `H>θ_g` | root | yes | **NEW** (100M/300M ablation) |
| **`entropy_monotonic`** | entropy `ΔH>θ_r` | root | yes | **NEW** (all scales, headline) |
| **`entropy_ps6` / `entropy_ps8`** | entropy, larger θ | root | yes | **NEW** patch-scaling study (760M/1.3B) |
| `llama` | subword (external) | — | n/a | reference baseline |

**Recommended rollout (cheap → expensive):**
1. **100M** — full grid {word, root_greedy, br_bt, entropy_global, entropy_monotonic} at the ablation
   budget (1672 steps) to sanity-check wiring + leak probe, then the ratio-20 budget for a real read.
   This extends the existing v1–v6 ablation (`100M_ablation.md`) with an **entropy** leg (config
   `bpebyte_100M_entropy.yaml`; not labelled "v7" — that index is the unrelated `prefix_vocab` trie).
2. **300M** — {word, root_greedy, entropy_monotonic, llama}; the 300M tier is config-only and is the
   cleanest matched-comparison rung (`cmp-300M-scope`). Add entropy as a 4th leg of `orch.sh`.
3. **760M** — {root_greedy, entropy_monotonic, entropy_ps6, llama}; introduce drift guard + patch-size
   scaling. Reuse `run_760M_chain.sh` scaffolding.
4. **1.3B** — {entropy_monotonic, entropy_ps8} added beside the existing `aunet2_1.3B`,
   `bpebyte_br_greedy_root_1.3B`, `llama_1.8B_paper` — the headline 4-way at the paper scale.

### 6.7 Evaluation (reuse existing harness)
- **Primary:** train/val **BPB** (iso-byte, single readout at the byte endpoint; convert Llama
  tokens→bytes at the measured 4.5483 B/tok) — report honestly, entropy patches will likely sit near
  `root_greedy` (0.86) above leaky `bt` and Llama (`BPEByte_root_greedy_method.md` §4).
- **Downstream 0-shot:** HellaSwag, ARC-E/C, PIQA, BoolQ (`model_results_1.3B.md` set).
- **Byte-level axes (where byte models win):** CUTE, HellaSwag-typo, PBP cut-invariance
  (`run_eval_pbp.sh`), noisy-downstream robustness.
- **Patch statistics:** avg/percentile bytes/patch on prose vs MCQ (drift check), boundary-leak probe.
- **Rigor (audit fixes):** ≥3 seeds + CIs on the 1.3B BPB endpoints; embed ckpt + segmenter (entropy
  model hash, θ, rule) in eval JSON (`bpebyte-rootgreedy-audit` points 3, 5).

---

## 7. Risks & open questions
- **Entropy-model fairness.** Disclose its one-time training FLOPs and the per-byte generation
  overhead; do not fold it into "byte model params." Mirror the disclosure we owe for `root_greedy`'s
  reliance on the Llama-3 vocab.
- **Drift on structured text** could distort MCQ patch sizes → distort eval FLOPs. Validate the newline
  reset + monotonic rule (§4 step 3) before trusting downstream numbers.
- **AU-Net gather-pool vs BLT cross-attn-pool.** Our pooling reads a single boundary byte; BLT pools a
  whole patch via cross-attention + n-grams. If entropy patches underperform `root_greedy`, run
  Variant B (§5.3) at 100M to attribute the gap to pooling vs boundaries — don't conflate them.
- **Threshold transfer across scale.** θ calibrated on the entropy model is data-, not main-model-,
  dependent, so one θ per target patch-size should transfer across all four scales; re-measure
  bytes/patch at each scale to confirm.
- **This finally runs the missing non-BPE causal control** the audit flagged: entropy vs root_greedy
  isolates "is the win from BPE structure or just from having *any* causal information-density
  boundary?" — a result worth its own paragraph.

## 8. Implementation status & next steps

**DONE (code; tests green). Refined after the 3-pass review in `review_blt.md`.**
- Core module `lingua/apps/aunet/data/entropy_patch.py`: pure boundary rules
  (`entropy_patch_starts` global/monotonic — the latter is BLT's verbatim adjacent-difference
  equation), `calibrate_theta` (candidate-value search, robust to the step-function plateau),
  `EntropyModel` (custom **fork-safe fp32** loader — no forced `.cuda()`; per-byte `H(x_t)` via a
  **bounded, newline-reset, overlap-chunked** window), `EntropyPatcher` (logs first-seq bytes/patch
  and warns if θ is uncalibrated), `EntropyIncrementalParser` (streaming, root, margin-0, **mirrors
  the batch window + newline reset exactly** so train==generate), and a **document-level** calibration
  CLI.
- **Parity is by construction:** both paths feed `[BOS]+window` with ≥`sliding_window` real left
  context and reset at `\n`; RoPE relative-invariance + `overlap ≥ sliding_window` make the batch
  (long, chunked) and streaming (short, bounded) forwards score each byte identically; fp32 removes
  bf16 near-threshold flips. (Closes review CRITICAL #2/#3.)
- Wiring `lingua/apps/aunet/data/regex_cutting.py`: `RegexArgs.{entropy_model_path, entropy_rule,
  entropy_theta, entropy_newline_reset}`; `bpe_online_mode: entropy` handled in `online_byte_boundaries`
  (training), `_online_levels_mask_bytes` (eval), and `make_incremental_bt_parser` (generation); needs
  **no** BPE tokenizer; asserts `bpe_context_prefix == 0`; seg-cache signature folds in the entropy
  params **and a size+mtime stamp of `consolidated.pth`** (no stale boundaries on retrain, review #7).
  `hierarchical.py` unchanged.
- `.gitignore` fix: `lingua/.gitignore`'s bare `data/` rule was silently ignoring the new module;
  added `!apps/aunet/data/**` negation (real `data/` artifacts still ignored). (Closes CRITICAL #1.)
- Configs: entropy model `apps/main/configs/entropy_byte_50M.yaml`; runs `bpebyte_100M_entropy.yaml`,
  `bpebyte_300M_entropy.yaml`, `bpebyte_entropy_760M_b200.yaml`, `bpebyte_entropy_1.3B_b200.yaml`
  (renamed off the `v7` label — `v7` is the unrelated `prefix_vocab` trie experiment; θ is a
  calibration placeholder, guarded by the first-batch bytes/patch warning).
- Tests `apps/aunet/test_entropy_patch.py`: rules, both-rule calibration, step-plateau, streaming==batch,
  **newline-reset**, snapshot/restore (all pass); a **gated** (`ENTROPY_CKPT`) real-model test asserts
  causality `H(s)[:k]==H(s[:k])` and streaming≡batch on the model's own entropies across a newline and
  a **>max_seqlen** sequence, on CPU (and CUDA if `ENTROPY_TEST_CUDA`). (Closes review #6/#8/#9.)

**REMAINING (runtime — needs GPU/data).**
1. Train the entropy model (`entropy_byte_50M.yaml`); consolidate → fill `entropy_model_path`.
2. Calibrate θ on **document-shaped** DCLM via the CLI (`--jsonl-field text`, matching the run's
   `--newline-reset`); write `entropy_theta` into the 4 configs (+ a global-rule θ for the ablation).
   Verify achieved bytes/patch on held-out DCLM **and an MCQ-shaped probe** before any FLOP claim.
3. Leak probe (`probe_root_causal.py` / `probe_streaming_leak.py`) on entropy boundaries → 0%.
4. **Execution model for the entropy LM (infra, review #4):** in-worker inference now defaults to a
   fork-safe **CPU fp32** forward (works, but is a dataloader bottleneck at 760M/1.3B scale). For the
   large runs, precompute boundaries once on GPU into `REGEX_SEG_CACHE` (set `ENTROPY_DEVICE=cuda` in a
   non-forked main-process pass) so workers only hit the cache. Build this precompute pass before the
   ≥760M runs; the 100M sanity leg can run CPU-in-worker as-is.
5. Run the 100M leg → verify wiring + BPB + train==generate; roll out 300M → 760M → 1.3B per §6.6;
   evaluate per §6.7 with seeds/CIs. Emit the entropy-model stamp + θ/rule + its one-time train-FLOPs
   and per-byte generation overhead into eval JSON (fairness accounting, same gap the audit flagged).
</content>
</invoke>

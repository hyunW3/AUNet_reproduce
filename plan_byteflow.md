# plan_byteflow.md — Coding-Rate Based Chunking in AU-Net

Integrating **ByteFlow Net**'s *coding-rate based chunking* (arXiv 2603.03583v1, Deng et al., Rice / Amazon Science) as a third online-segmentation method on the AU-Net backbone, alongside whitespace AU-Net and BPEByte (root_greedy / online byte-trie).

> **Source-verification caveat.** The arXiv PDF (`2603.03583v1`) did not parse cleanly. The *method, equations, and algorithm* below were cross-checked across two independent extractions (arXiv HTML + community paper-notes) and are consistent → high confidence. The *exact numeric hyper-parameters and benchmark tables* (ε² default, K default, the 600M/1.3B accuracy figures) are extractor-approximated → **must be re-read from the PDF §3–§5 before being cited in our paper.** They are flagged `[VERIFY]` throughout. No public code repo was found (Rice/Amazon, no GitHub link in the paper or HF page as of 2026-06).

---

## 0. TL;DR

ByteFlow keeps the same encoder→downsample→global-trunk→upsample→decoder hierarchy AU-Net already has. Its single new idea is **how boundaries are chosen**: instead of a *data-side* deterministic function of the raw bytes (whitespace, BPE-trie), boundaries are chosen *inside the model* from the **learned encoder hidden states** `h`, by promoting the positions whose **marginal coding rate** ΔRₜ is largest.

So the integration is a **boundary-method swap on a fixed backbone**, identical in spirit to how this repo already compares `word1` vs `bt` vs `greedy/root`. The one structural consequence is that the boundary producer **moves from the data loader into `forward()`** (it needs learned `h`), which is the only nontrivial code change.

Two design forks dominate the plan and are decided up front:

1. **Top-K (paper) vs causal threshold (ours).** Global Top-K over the whole sequence is **non-causal** (membership of position t depends on future scores) → it leaks future into the autoregressive byte LM, which this project has spent significant effort eliminating (see `[[causal-segmentation-leak-tradeoff]]`, `[[byte-generate-until-hang]]`). **This exact concern was raised in the paper's ICLR-2026 review and the authors' answer is "rolling top-k"** (see §2.2): causal at *inference* (Top-K over already-seen tokens only) but **still global at *training*** (teacher-forcing Top-K sees the full sequence, which the authors concede "leaks minimal information"). That train/inference asymmetry is the parity gap this repo refuses to tolerate. **Recommendation: a causal ΔRₜ threshold** as primary (leak-free, identical at train==eval==gen), with **rolling top-k** (causal *at both* train and inference) as the first-class faithful alternative, and **global top-k** kept only as a non-causal upper-bound ablation.

2. **L2 approximation.** ByteFlow's own Appendix gives `Rε(h₁:ₜ) ∝ ‖h₁:ₜ‖₂²`, which makes ΔRₜ an O(d) per-step quantity → trivially streamable at generation. We adopt the L2 form as primary; full log-det is an ablation.

---

## 1. ByteFlow Net — key components

### 1.1 Architecture (5 stages — already matches AU-Net)

```
x₁:ₜ ∈ Vᵀ
   │  Local Encoder         shallow, SWA window w_local, "Canon" causal conv1d (k=4)
   ▼
h₁:ₜ ∈ ℝ^{T×d_local}
   │  Downsampling          ← coding-rate chunking selects K positions  (THE NEW PART)
   ▼
z₁:ₖ ∈ ℝ^{K×d_global}
   │  Global Transformer    deep+wide, full causal attention over K ≪ T tokens
   ▼
g₁:ₖ ∈ ℝ^{K×d_global}
   │  Upsampling            multilinear reconstruction (B bins) + large residual from h
   ▼
s₁:ₜ = hₜ + s̃ₜ
   │  Decoder               symmetric to encoder
   ▼
p̂(xₜ₊₁ | x₁:ₜ) = softmax(... W_out)
```

Mapping to our code (`lingua/apps/aunet/hierarchical.py`):

| ByteFlow stage | AU-Net component | File:lines |
|---|---|---|
| Local encoder | `encoders[0]` (`CausalTransformer`) | `hierarchical.py:210, 521` |
| Downsampling | `SimpleTransition.down` | `hierarchical.py:327` |
| Global transformer | `trunk` | `hierarchical.py:543` |
| Upsampling | `SimpleTransition.up` | `hierarchical.py:350` |
| Decoder | `decoders[0]` | `hierarchical.py:521` |
| Boundary selection | **currently data-side** `level_mask` via `RegexPool` | `data/regex_cutting.py` |

The "Canon layer" (causal depthwise conv1d, kernel 4) and "multilinear B-bin upsampling" are ByteFlow's encoder/upsample flavors; AU-Net's `SimpleTransition` (indexed-matmul down, repeat-gather + positional up) is functionally equivalent. **We keep AU-Net's backbone unchanged** so the comparison isolates the boundary mechanism. (Canon conv and B-bin upsample are listed as optional Phase-3 ablations in §6.)

### 1.2 The coding-rate criterion (the actual contribution)

**Lossy coding rate** of a prefix of byte representations (eq. 11):

```
Rε(h₁:ₜ) = ½ · log det( I + (d_local / ε²) · h₁:ₜ h₁:ₜᵀ )
```

Intuition: the log-det of the (regularized) covariance measures **how many independent directions** the representation spans. A position that pushes the representation into a *new* direction carries new information.

**Marginal coding rate** = information gain of adding position t (eq. 12):

```
ΔRₜ = Rε(h₁:ₜ) − Rε(h₁:ₜ₋₁)
```

Large ΔRₜ ⇒ position t introduces substantial new information ⇒ a natural chunk boundary. ΔRₜ depends **only on the prefix** `h₁:ₜ` → **the score itself is causal.**

**L2 streaming approximation** (Appendix B, eq. 32):

```
Rε(h₁:ₜ) ∝ ‖h₁:ₜ‖₂²    ⇒    ΔRₜ ∝ ‖h₁:ₜ‖₂² − ‖h₁:ₜ₋₁‖₂²  (a cheap O(d) running quantity)
```

**Boundary selection (paper = Top-K, eq. ~12 cont.):**

```
S = {1}  (always keep BOS)
S ← S ∪ argtop-(K−1) over t of ΔRₜ          # global Top-K → exactly K chunks
S = sort(S) = {s₁ < s₂ < … < s_K}
z = [h_{s₁}, …, h_{s_K}] · W_proj  ∈ ℝ^{K×d_global}
```

Top-K is chosen *by the paper* to keep a **static computation graph** (exactly K tokens every example → no ragged tensors, no OOM). **No learnable boundary parameters and no auxiliary loss** — selection is a deterministic forward op; gradients flow only through the *gathered* `h_{sᵢ}`. Trained with plain next-byte cross-entropy (reported BPB).

### 1.3 Reported config & results `[VERIFY against PDF]`

- Scales: **600M / 50B tok** and **1.3B / 500B tok**; data **FineWeb-Edu-100B**; seq_len 8192; AdamW (β 0.9/0.95), lr ~1e-3 cosine, bf16, clip 1.0.
- ε² "model-dependent"; K default ~3200 `[VERIFY]`; d_local 512–2048; d_global 1536–2048; E (local) 6; G (global) 20–24; B (upsample bins) 16.
- Claimed: ByteFlow **beats LLaMA-BPE** (val BPB 0.86 vs 0.89 @600M; 0.76 vs 0.79 @1.3B) and byte baselines (MambaByte/LlamaByte/SpaceByte), "matches/exceeds AU-Net." Ablation: coding-rate BPB 0.86 < cosine-sim 0.92 < entropy 0.91 @600M `[ALL VERIFY]`.

---

## 2. The integration problem (why this isn't a one-line data mode)

AU-Net's defining design choice (per the Explore map and `[[causal-segmentation-leak-tradeoff]]`): **the model never computes boundaries.** It consumes a precomputed integer `level_mask` from the data pipeline; `get_pool_mask` (`hierarchical.py:614`) turns `level_mask > i` into the per-level keep mask. Whitespace, offline-BPE, and online byte-trie all just emit this mask in `data/regex_cutting.py`.

Coding-rate chunking **cannot** be produced in the data loader, because it is a function of the **learned encoder output** `h`, which only exists after `encoders[0]` runs. Therefore:

> **The boundary producer must move into `forward()`**, between `encoders[0]` and `transitions[0].down`. This is the core structural difference from BPEByte and the main implementation work.

Two correctness hazards, both of which the project already has machinery and strong opinions about:

- **Causality / future leak.** Global Top-K is non-causal: whether position t survives depends on the ΔR of positions > t. For an AR byte LM this leaks future tokens into the trunk input — precisely the failure mode catalogued in `[[causal-segmentation-leak-tradeoff]]` (the "83% leak" artifact) and verified-fixed for root_greedy. We **must not** ship Top-K as the headline method.
- **Train / eval / generation parity.** This repo insists the boundaries used at training equal those at loglikelihood-scoring equal those at generation (`[[byte-generate-until-hang]]`; `_online_levels_mask_bytes` mirrors `online_levels_mask`). Top-K is undefined at single-step generation (no future to rank). A **causal threshold** is identical in all three regimes.

### 2.2 What the paper's own reviewers concluded (OpenReview `GhJIa921j7`, ICLR 2026)

The leak concern in §2 is **not ours alone** — it is the central thread of the paper's review:

- **Reviewer veFt:** *"The method relies on top-k over sequence, which (1) leaks minimal information from the future, (2) it is unclear how to apply it in an autoregressive setting."* Follow-up: *"Was this done by applying a top-k over the whole sequence, or with a 'rolling top-k', where at each position the top-k is taken with respect to the tokens seen so far?"*
- **Authors (rebuttal):**
  - *Inference* — *"the global Top-K selects from already-generated input tokens to promote to the global transformer. Because future tokens do not yet exist, there is no possibility of leakage — this is strictly causal."* They name this **"rolling top-k."**
  - **Buffer mechanism:** *"we maintain a buffer of k positions and dynamic chunking works when exceeding the buffer … when we have short sequences, we just run the pure byte model … and when we have longer sequences, we compress them into high-level units."*
  - *Training* — *"During teacher forcing, Top-K observes coding rates across the full sequence to determine which positions warrant global processing."* ⇒ **global at train, rolling at inference.** All attention masks are strictly causal, but the *boundary-selection set* is not, at train time.

**Takeaways for us:**
1. The causality risk is real and acknowledged; the published method has a **train/inference boundary mismatch** + an admitted *"leaks minimal information"* at training. We can strictly improve on it by being causal at **both** train and inference.
2. **"Rolling top-k" = streaming Top-K over a bounded buffer of k seen positions** (evict lowest-ΔR when a higher one arrives once the buffer is full). It is causal and gives a fixed ≤k trunk length (static graph). This is a first-class mode for us, distinct from the global Top-K.
3. The buffer's "pure byte model below k, compress above k" behavior maps cleanly onto our `max_seqlens[1]` cap: ≤cap positions ⇒ no eviction; >cap ⇒ selection. Our existing ragged+cap machinery already implements the buffer's spirit.

### 2.3 Decision: causal-threshold coding-rate chunking (primary)

```
score_t = ΔRₜ  (L2 form: ‖h₁:ₜ‖² − ‖h₁:ₜ₋₁‖², or its mean-removed/normalized variant)
boundary_t = 1  iff  score_t ≥ τ          # causal, depends only on prefix
```

- **Leak-free by construction** (per-position causal decision; no future read). Reuses the existing ragged handling: emit `boundary_t` as a binary `level_mask`, let `get_pool_mask` cap to `max_seqlens[1]=3200` exactly as today.
- **τ calibration to a target compression ratio** r (so the trunk sees the same token budget as BPEByte ≈ 4.4 bytes/token, and AU-Net-word ≈ 5–6 bytes/token — matched FLOPs). Options, in preference order:
  1. **Running-quantile / EMA threshold**: maintain an EMA of the (1−1/r) quantile of `score_t`; commit when `score_t` exceeds it. Causal, no labels, self-calibrating to ratio r. *(primary)*
  2. **Fixed τ** swept once per scale to hit r on a val shard. Simplest; risk of ratio drift as `h` changes during training.
  3. **Per-sequence quantile** (offline-in-batch): leaks within-sequence, **eval/ablation only**.
- **Rolling top-k** (paper's causal variant, §2.2) = first-class **secondary** method: streaming Top-K over a buffer of k ≈ `seq_len / target_ratio` seen positions (see §2.4 — *not* the `max_seqlens[1]` cap), applied *identically at train and inference* (fixing the paper's train/inference mismatch). Gives an exact bounded trunk length (static graph) vs. the threshold's variable count. Run head-to-head with the threshold.
- **Global top-k** = paper-faithful **non-causal ablation** only (per-sequence, exactly K, full-sequence ranking at train), to measure the leak-vs-quality gap the project cares about — expected to be the same shape as the root_greedy story (within-noise small scale, possibly diverging at 1.3B). The gap (global − rolling) *is* the train-time leak the authors admitted.

### 2.4 Rolling top-k edge cases (eviction + sub-k degenerate regime)

Two corner cases make exact rolling top-k a diagnostic-only mode and motivate the eviction-free designs. Both are unique to the top-k family — the **threshold** primary has neither.

**(a) Eviction of a previously promoted position.** Under the literal definition (at each step, top-k over tokens seen so far), once the buffer holds k positions, a newly generated token with `ΔR_{t+1} > min(buffer)` evicts the current argmin:
```
buffer full (k positions); token t+1 arrives
if ΔR_{t+1} > min(buffer):  evict argmin(buffer); insert t+1   # a PAST boundary disappears
```
Consequences for an AR byte LM (the rollback/recompute failure mode of `[[byte-generate-until-hang]]`):
- **Global-transformer KV-cache invalidation** → worst-case O(n²) re-prefill when a past key vanishes.
- **Non-monotone segmentation → retroactive change**: dropping boundary `p` re-routes the byte→chunk assignment and upsampling residual around `p`, altering the next-byte distribution of *already-sampled* tokens → a within-generation parity break we can't un-sample.
- It is exactly why the authors used **global** top-k at train (exact per-prefix rolling is an O(T·k) streaming select, awkward in one teacher-forced forward) and only rolling at inference.

→ **Forbid eviction.** Modes: `rolling_topk: commit` (monotone — committed boundaries are permanent; flush-on-overflow via the existing `committed_patch_idx` / incremental-parser machinery; a causal *greedy approximation* of top-k) is the **default**; `rolling_topk: exact` (evicts) is **diagnostic-only**, sits in the non-deployable column next to `global_topk`. The threshold never evicts by construction.

> **Scope — these consequences are generation-time only.** Loglikelihood/MC scoring (HellaSwag, ARC, BPB, …) does a **single full-sequence teacher-forced forward** and never autoregressively decodes, so it **never rolls back**: no KV-cache invalidation, no O(n²) re-prefill, no retroactive resampling (nothing is sampled). Eviction therefore imposes **no cost** on likelihood eval — consistent with `[[byte-generate-until-hang]]` ("loglikelihood never rolls back"). Two things still hold for the likelihood number to be *valid*, both cheap: (i) the boundary rule must be **causal** in that single pass — global top-k would leak future into position t's segmentation and inflate the score (the before_root "83% leak" artifact, `[[causal-segmentation-leak-tradeoff]]`); (ii) the boundary **policy still selects which segmentation (hence which loglikelihood) is computed**, so for parity, likelihood eval must use the *same* policy generation will (commit-monotone or threshold). With commit/threshold used everywhere, eviction simply never occurs and there is nothing to handle in the likelihood path.

**(b) Context ≤ k ⇒ every position is a boundary.** By definition `top-k(ΔR₁…ΔRₜ)` with t ≤ k returns all t positions → no compression (ratio 1) → "pure byte model" for the first k bytes, then compression switches on at t > k (the same k-crossing as the eviction cliff, seen from the other side). Implications:
- **Short-input distribution shift / eval confound**: benchmark continuations shorter than k bytes run *entirely uncompressed* under any top-k mode, while AU-Net (whitespace) and BPEByte (trie) chunk at all lengths → not apples-to-apples on short MC items. The threshold compresses at any length and avoids this.
- The first-k-uncompressed prefix is benign for long-context *training* (seq_len 8192 ≫ k) but matters for *short eval items and the start of generation*.

**buffer_k sizing (do not conflate with the `max_seqlens` cap).** A *fixed* top-k forces exactly k chunks whenever t > k → ratio = t/k. To match the BPEByte token budget (ratio ≈ 4.5) at `seq_len 8192`, set **`buffer_k ≈ seq_len / target_ratio ≈ 1820`** — *not* the `max_seqlens[1]=3200` ceiling (which would force ratio ≈ 2.56, i.e. ~1.8× weaker compression and ~1.8× more trunk FLOPs than its BPEByte partner, breaking the matched-FLOPs comparison). `max_seqlens[1]` stays a safety ceiling; `buffer_k` is the operating point. (For the **threshold** modes there is no `buffer_k`: the natural boundary count ≈1820 sits below the 3200 cap, so the cap rarely binds and ratio self-sets to `target_ratio`.)

### 2.5 Differentiability

Selection is hard (threshold/argtopk) → non-differentiable, **same as AU-Net's existing hard gather**. No straight-through estimator and no ratio loss are required (ByteFlow uses none). Gradients reach the encoder through the gathered `h` and through the `up()` residual `sₜ = hₜ + s̃ₜ`. This is a deliberate divergence from H-Net (which *learns* a router with an aux ratio loss); keeping it deterministic preserves a clean 3-way comparison where **only the boundary rule differs.**

---

## 3. Implementation design

Backbone unchanged. Add a model-side boundary module + thread a config flag. Files (all under `lingua/apps/aunet/`):

### 3.1 New: `coding_rate.py` (boundary module)

```python
# lingua/apps/aunet/coding_rate.py
def coding_rate_boundaries(h, target_ratio, eps2, mode="l2_quantile",
                           ema_state=None, max_seqlen=None):
    """
    h: [B, T, d_local] causal encoder output (encoders[0]).
    returns level_mask: [B, T] int (1 at promoted positions, 0 else; pos 0 forced 1).
    Causal: score_t uses only h[:, :t+1].
    """
    # L2 form: cumulative squared-norm energy is a prefix quantity -> ΔR_t cheap & causal.
    cum = (h.float() ** 2).sum(-1).cumsum(1)          # [B,T] = ‖h₁:ₜ‖²  (∝ Rε, eq.32)
    dR  = cum - torch.nn.functional.pad(cum, (1,0))[:, :-1]   # ΔRₜ (eq.12, L2)
    # optional: dR = log1p((d_local/eps2) * per_step_outer)   # full log-det (eq.11) ablation
    tau = ema_quantile(dR, 1 - 1/target_ratio, ema_state)     # causal EMA quantile
    mask = (dR >= tau).int()
    mask[:, 0] = 1                                              # always-keep BOS (force_first)
    return mask  # capping to max_seqlen handled downstream by get_pool_mask
```

- `mode`: `l2_quantile` (primary), `l2_fixed_tau`, `logdet_quantile` (ablation), `topk_per_seq` (non-causal ablation).
- The full log-det (eq.11) is computed via the rank-1 prefix update of `det(I + α hhᵀ)` so it stays O(T·d), not O(T·d³).

### 3.2 `hierarchical.py` — compute mask inside forward

Today (`hierarchical.py:564–573`):
```python
masks, _, nb_toks = self.get_pool_mask(level_mask, [...], force_first=True)   # level_mask from DATA
x = self.tok_embeddings(token_values)
for encoder, trans, mask in zip(self.encoders, self.transitions, masks):
    x = encoder(x, attn_impl="fmha")
    residuals.append(x); x = trans.down(x, mask, ...)
```
Coding-rate path (only the first level is dynamic; deeper levels, if any, can stay data-side or recurse):
```python
x = self.tok_embeddings(token_values)
x0 = self.encoders[0](x, attn_impl="fmha")        # h₁:ₜ — needed for boundaries
if self.coding_rate.enabled:
    level_mask = coding_rate_boundaries(x0, self.coding_rate.target_ratio,
                                        self.coding_rate.eps2, self.coding_rate.mode,
                                        self._cr_ema, max_seqlen=self.max_seqlens[1])
masks, _, nb_toks = self.get_pool_mask(level_mask, [...], force_first=True)
# then proceed with x0 as the level-0 features feeding trans.down (avoid recompute)
```
- `committed_patch_idx` / `patch_read_delay` machinery (`hierarchical.py:376–397`) stays at default 0; the causal threshold means the committed view == the training view, so no rollback path is needed (unlike streaming byte-trie, `[[byte-generate-until-hang]]`).
- Generation: feed bytes one at a time, maintain the cumulative `‖h₁:ₜ‖²` and EMA τ as parser state → commit a boundary exactly when `score_t ≥ τ`. Mirrors `ByteTrieIncrementalParser` (`data/byte_trie.py:184`) but state is two scalars per stream, not a trie cursor.

### 3.3 Config schema

Add a `coding_rate:` block to `HierarchicalArgs` (`hierarchical.py:33`) — model-side, *not* under `data.regex` (deliberate: it's a model mechanism):
```yaml
model:
    coding_rate:
        enabled: true
        mode: l2_quantile        # l2_quantile | l2_fixed_tau | logdet_quantile | rolling_topk | global_topk
        target_ratio: 4.5        # bytes/chunk; matches BPEByte token budget for iso-FLOPs
        eps2: 0.5                # [VERIFY default from PDF]; only used by logdet modes
        ema_decay: 0.99          # quantile-tracker decay (l2_quantile)
        tau: null                # only for l2_fixed_tau
        rolling_evict: commit    # rolling_topk: commit (monotone, no eviction; DEFAULT) | exact (evicts; diagnostic-only, §2.4a)
        buffer_k: null           # rolling_topk buffer size; default ≈ seq_len/target_ratio (≈1820), NOT max_seqlens[1] (§2.4); train==infer
        topk: null               # global_topk only (non-causal ablation; => seq_len/target_ratio)
        eval_mode: rolling_topk   # likelihood scoring for top-k family (§3.4): rolling_topk (causal, deployable) | global_topk (leaky upper bound). Ignored by threshold/commit-rolling.
```
When `enabled: true`, the `data.regex` block is ignored for boundary production (data still emits a placeholder `level_mask`; `n_views` stays 2). Add a smoke assertion that exactly one of {`regex` online, `coding_rate`} is active.

### 3.4 Eval parity — TWO required likelihood-scoring modes for the top-k family

For threshold / commit-rolling, eval is automatic: boundaries are prefix-deterministic, so the single `coding_rate_boundaries` call in `forward` already gives a correct causal likelihood for every position in one pass (same as BPEByte root_greedy, no separate eval path needed).

**But the top-k family is not prefix-deterministic, so likelihood scoring must split into two explicit modes** (these are *scoring* modes, distinct from the *training* mode):

1. **`global_topk` (leaky, single pass).** Boundaries chosen by Top-K over the *entire* sequence — including the answer/continuation tokens. Scoring answer token t then uses a segmentation that depended on tokens > t ⇒ **future leak** ⇒ this is **not a valid likelihood**, only a diagnostic **upper bound**. One forward pass, cheap. Report it labeled as leaky.

2. **`rolling_topk` (causal, correct).** Each answer token t's likelihood is computed with Top-K taken over **previous tokens only** (the prefix `< t`, never future). This is the only way to *correctly* measure the likelihood under top-k chunking. Two sub-cases:
   - **commit-monotone** (`rolling_evict: commit`): the per-prefix top-k *is* prefix-consistent, so a single forward with the committed segmentation already yields the correct causal per-token likelihoods — one pass, no extra cost.
   - **exact** (`rolling_evict: exact`): not prefix-consistent (eviction, §2.4a), so the segmentation must be **recomputed per scored prefix** (the boundaries in `[1,t]` differ from the full-sequence boundaries). Correct but O(answer_len) re-segmentations per item — offline-affordable for eval, never for generation. Diagnostic only.

So: `global_topk` and `rolling_topk` are reported as a **pair** — the gap between them *is* the future-leak inflation, the headline causal-leak measurement (§6 ablation 1). The deployable number is always `rolling_topk` (commit). `apps/aunet/eval` gains a `coding_rate.eval_mode: global_topk | rolling_topk` switch; threshold/commit-rolling ignore it (single path).

---

## 4. Configuration plan — 100M / 300M / 760M / 1.3B

We extend the **existing trunk-matched ladder** (`status_300M.md`, `[[cmp-300M-scope]]`). The coding-rate variant clones the byte-model config family, keeps the entire shared block identical to AU-Net / BPEByte, and only **(a)** removes the `regex` online boundary, **(b)** adds the `coding_rate` block. **Everything else (dims, layers, steps, lr, warmup, budget) is held equal to its BPEByte/AU-Net partner at that scale** so the comparison is clean.

### 4.1 Shared block (identical across all 4 scales, all 3 models)
`head_dims: [64,128]`, `residuals: [True]`, `sliding_windows: [512,4096]`, `max_seqlens: [-1,3200]`, `rope_theta 500000`, `multiple_of 256`, `pooling_type simple_indexed_matmul`, `lambda_level 0.0`, `patch_read_delay 0`; data `seq_len 8192`, `tokenizer bytes`, `n_views 2`, `add_bos/eos true`, source `dclm_baseline_1.0_2shards_shuffled`.

### 4.2 Per-scale ladder (byte trunk = 2nd element of `dimensions`/`layers`)

| Scale | `dimensions` | `layers` | trunk heads | bs / ga / steps | lr / warmup | budget |
|---|---|---|---|---|---|---|
| **100M** | `[512, 768]`  | `[3, 10]` | 6  | 96 / 2 / 13376 | 2.0e-3 / 100 | ratio-10 ≈ 21B bytes |
| **300M** | `[512, 1280]` | `[3, 13]` | 10 | 48 / 4 / 6688  | 1.9e-3 / 400 | 42B bytes (iso-token) |
| **760M** | `[512, 1536]` | `[3, 24]` | 12 | 12 / 6 / 29200 | 1.65e-3 / 2920 | ~69B bytes (iso-text Chinchilla) |
| **1.3B** | `[512, 2048]` | `[3, 25]` | 16 | 12 / 4 / 180000 | 1.65e-3 / 10000 | global batch 192 |

These rows are **copied verbatim** from the BPEByte/AU-Net partners (`bpebyte_root_greedy_{760M,1.3B}_b200.yaml`, `aunet_300M.yaml`, `bpebyte_100M_v4_root_greedy.yaml` / ablation driver) so trunk-param counts and token budgets match by construction. Llama partners (`main/configs/llama_{100M_cmp,300M,760M_b200,1B}*.yaml`) are unchanged.

### 4.3 `target_ratio` calibration (the one new knob to set per scale)

Set `target_ratio` so the trunk sees the **same number of chunks** as the BPEByte partner → matched trunk FLOPs and a fair quality comparison.
- BPEByte root_greedy on DCLM ≈ **4.3–4.6 bytes/token** (measure on a val shard with the existing byte-trie; record the empirical mean). Set `target_ratio` to that value at each scale → default **4.5**.
- Sanity gate before each full run: 25-step smoke (same gate as `runs/cmp_300M/orch.sh`) that logs realized bytes/chunk; abort if it drifts >±10% from `target_ratio` (EMA mis-tracking).
- Also run a **ratio-matched-to-AU-Net-word** variant (~5.5) as a secondary point so we can attribute any delta to ratio vs. rule.

### 4.4 New config files to create

```
lingua/apps/aunet/configs/byteflow_100M.yaml          # clone bpebyte_100M_v4_root_greedy.yaml
lingua/apps/aunet/configs/byteflow_300M.yaml          # clone bpebyte_300M_v4_root_greedy.yaml
lingua/apps/aunet/configs/byteflow_760M_b200.yaml     # clone bpebyte_root_greedy_760M_b200.yaml
lingua/apps/aunet/configs/byteflow_1.3B_b200.yaml     # clone bpebyte_root_greedy_1.3B_b200.yaml
```
Diff vs. the clone source in each: delete `data.regex.bpe_*` online keys, add the §3.3 `model.coding_rate` block, point `dump_dir`/`name` at `runs/byteflow_<scale>`. Launch scripts: clone `train_bpebyte_root_greedy_{760M,1.3B}.sh` → `train_byteflow_*.sh` (new ports), and add a `byteflow_*` leg to `run_760M_chain.sh` and `runs/cmp_300M/orch.sh`.

---

## 5. Experiment matrix (what we're actually comparing)

Per scale, on the **identical backbone + identical budget**, three boundary mechanisms × Llama baseline:

| # | Model | Boundary source | Causal/leak-free | Where boundaries live |
|---|---|---|---|---|
| 1 | **AU-Net (word)** | whitespace regex | yes | data-side (`regex_cutting`) |
| 2 | **BPEByte root_greedy** | causal byte-trie | yes (verified) | data-side (online) |
| 3 | **ByteFlow (ours, threshold)** | coding-rate ΔRₜ causal threshold | **yes (by construction)** | **model-side (`forward`)** |
| 3a | ByteFlow rolling top-k | streaming Top-K over buffer of seen tokens (train==infer) | **yes** | model-side |
| 3b | ByteFlow global top-k | full-sequence Top-K at train (paper's train rule) | **no (leaks at train)** | model-side — *ablation only* |
| 4 | Llama (BPE) | static BPE tokenizer | n/a | tokenizer |

**Primary metrics** (reuse `evaluation_flow.md` harness): val **BPB** (+95% CI per `bpb_ci_1.3B.md`), HellaSwag, ARC-easy/challenge, PIQA, WinoGrande, BoolQ; plus realized **bytes/chunk** and trunk-token throughput. **Headline question:** does *learned, information-theoretic* segmentation (3) beat *fixed-rule* segmentation (1,2) at matched FLOPs, and does the gap scale (the §2 leak/scale story — within-noise at 100M, watch 1.3B)?

**Rollout order** (cheapest signal first): 100M (driver `run_ablation_100M.sh` leg) → 300M (`orch.sh` leg, geometric midpoint) → 760M (`run_760M_chain.sh` leg) → 1.3B (B200 queue, behind the current jobs in `[[next-100m-aunet-llama]]`).

---

## 6. Ablations (after the main ladder)

1. **Causal threshold vs rolling top-k vs global top-k (3 vs rolling vs 3b)** — the three-way axis from §2.2/§2.3. `global − rolling` quantifies the train-time leak the authors conceded; `rolling vs threshold` compares the two leak-free designs (fixed-k buffer vs variable-count cap). The central scientific payoff given this project's leak focus.
2. **L2 (eq.32) vs full log-det (eq.11)** — does the cheap approximation cost quality?
3. **target_ratio sweep** {4.0, 4.5, 5.5, 6.5} at 100M — quality vs compression curve; locate the iso-FLOPs sweet spot.
4. **ε² sensitivity** (log-det modes) at 100M — `[VERIFY paper default first]`.
5. **EMA-quantile vs fixed-τ** — calibration robustness as `h` drifts during training.
6. **ByteFlow encoder extras** (Canon causal-conv k=4 in `encoders[0]`; B-bin multilinear `up()`) — only if (3) is competitive, to test whether the paper's gains are the *rule* or the *backbone trimmings*.

---

## 7. Risks & open questions

- **`[VERIFY]` paper numerics** — pull ε², K, lr, and the §5 result tables from the actual PDF before any are quoted. The 600M/1.3B accuracy figures in §1.3 are extractor-approximated and may be wrong.
- **Ratio stability** — `h` changes during training; a fixed τ will drift the compression ratio. Mitigated by EMA-quantile (§4.3) + the 25-step smoke gate. Watch the bytes/chunk log throughout, not just at start.
- **`max_seqlens[1]=3200` cap interaction** — if the threshold under-fires early in training, >3200 boundaries get clipped by `get_pool_mask` and the tail of long sequences loses structure. Log clip rate; raise cap or tighten τ if clipping is frequent.
- **Throughput** — boundaries are now data-dependent per step, so trunk token count varies batch-to-batch (the cap bounds the worst case, so static graph still holds via padding to the cap). Confirm `compile: true` tolerates this; if not, pad-to-cap is the fallback (matches ByteFlow's static-graph motivation).
- **Generation parity** — validate the incremental `‖h₁:ₜ‖²`+τ parser reproduces training boundaries on held-out text *before* trusting generation BPB (the parity bug class from `[[byte-generate-until-hang]]`).
- **Fairness framing** — keep AU-Net/BPEByte backbones byte-for-byte identical so a win is attributable to the boundary rule, not Canon conv / B-bin upsample. Defer those to ablation 6.
- **Train/inference boundary parity** — the published ByteFlow uses *global* Top-K at train but *rolling* Top-K at inference (§2.2), an asymmetry the authors concede "leaks minimal information." Our threshold and rolling-top-k modes are deliberately identical at train/eval/gen; **do not** copy the paper's split. If we report a global-top-k number, label it clearly as the leaky train-time upper bound, not the deployable model.

---

## 8. Next actions

1. `[VERIFY]` Re-extract ByteFlow §3 (eqs 11–18, 32) + §4–5 hyperparams/results from the PDF; correct §1.3 here.
2. Implement `lingua/apps/aunet/coding_rate.py` (§3.1) + `forward` hook (§3.2) + `HierarchicalArgs.coding_rate` (§3.3).
3. Measure BPEByte bytes/chunk on a DCLM val shard → fix `target_ratio` per scale (§4.3).
4. Create the 4 `byteflow_*.yaml` configs (§4.4); add a 100M smoke (25 steps) to `run_ablation_100M.sh`.
5. Run 100M → 300M → 760M → 1.3B (§5), then ablations (§6).

Related memory: `[[aunet-project-state]]`, `[[cmp-300M-scope]]`, `[[causal-segmentation-leak-tradeoff]]`, `[[byte-generate-until-hang]]`, `[[bpebyte-rootgreedy-audit]]`, `[[next-100m-aunet-llama]]`.

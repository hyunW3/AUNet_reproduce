# 300M-scale comparison — scope

**Goal.** Re-run the leak-free/causal vs. baseline comparison one scale up (≈3× the 100M trunk)
to test the central scaling claim: *the leak-free+causal penalty is within-noise at 100M but
+4 at 1.3B — where does 300M land?* Four models:

| # | Model | Boundaries | Leak-free | Causal |
|---|-------|-----------|-----------|--------|
| 1 | **AU-Net** (word) | offline word-patch (`bpe_br` / bt before_root) | — (baseline) | — |
| 2 | **Llama** (subword) | BPE (full lookahead) | — (baseline) | — |
| 3 | **BPEByte greedy_root** | online byte-trie, `mode=greedy`, `placement=root` | ✅ | ✅ |
| 4 | **BPEByte committed_view** (v1) | online byte-trie, `mode=bt`, `before_root`, `committed_view=true` | ✅ | ✅ |

Models 1/3/4 share **one** byte-trunk architecture; they differ *only* in the `data.regex`
block. Model 2 (Llama) is a separate single-level transformer, trunk-matched to the byte trunk.

---

## 1. Architecture (calibrated, not guessed)

Measured 100M sizes (printed `Model size:` in train logs): AU-Net byte trunk **98,591,488**;
Llama (dim768/14L + 128k untied emb) **296,113,920** (trunk ≈99M). 1.3B endpoints:
AU-Net `dims [512,2048] layers [3,25]`, Llama `dim2048/25L/16h`. The 300M point is the
geometric interpolation, calibrated to the trunk formula `≈ 1.23 × (9.44M + L₂·12·D₂²)` for
AU-Net (the 1.23 factor reproduces 98.6M exactly) and `12·dim²·n_layers` for Llama (reproduces
99M exactly).

### Byte models (AU-Net / greedy_root / committed_view) — identical arch
```yaml
model:
    dimensions: [512, 1280]      # 100M: [512,768]  → 1.3B: [512,2048]
    layers: [3, 13]              # 100M: [3,10]     → 1.3B: [3,25]
    head_dims: [64, 128]         # 1280/128 = 10 heads at level 2
    residuals: [True]
    sliding_windows: [512, 4096]
    max_seqlens: [-1, 3200]
    block: {rope_theta: 500000.0, multiple_of: 256}
    lambda_level: 0.0
    pooling_type: simple_indexed_matmul
    patch_read_delay: 0
```
→ **≈296M** total (3.0× the 100M trunk). Verified with the project's own `estimated_param_count`:
the 100M arch reproduces to 98.3M vs 98.59M actual (the byte-encoder level counts ×2, plus a
pooling/transition overhead ≈ `7.1M × D₂/768` → +11.8M here). seq_len 8192 unchanged.

### Llama (subword) — trunk-matched to ≈300M
```yaml
model:
    dim: 1280                    # 100M: 768  → 1.8B: 2048
    n_layers: 15                 # 100M: 14   → 1.8B: 25
    n_heads: 20                  # head_dim 64, matching the 100M head_dim
```
→ trunk **≈305M** (matched to the byte trunk ≈296M within −2.7%, per the project's "match
non-embedding params" rule; estimator is exact for plain Llama — reproduces 99.1M at 100M); with
the 128k untied embedding+output (328M) total ≈**633M**. The embedding-heavy
asymmetry is by construction — same framing as "AU-Net 1.3B vs Transformer-BPE 1.8B".
seq_len 2048 unchanged.

The **only** per-model deltas for the three byte models are the existing `data.regex` flags
(copy verbatim from the 100M configs):

| Model | `bpe_online_mode` | `bpe_online_placement` | `bpe_online_committed_view` |
|-------|-------------------|------------------------|-----------------------------|
| AU-Net (word) | `bt` | `before_root` | `false` |
| greedy_root | `greedy` | `root` | `false` |
| committed_view (v1) | `bt` | `before_root` | `true` |

---

## 2. Token budget

Per-step byte count (byte models): `batch 96 × Ngpu × grad_acc × seq 8192`. At 100M this was
6.29M bytes/step (bs96/4gpu/ga2). 300M needs more memory → keep tokens/step fixed by
**halving microbatch and doubling grad_acc** (bs48/ga4 on 4 GPU, or bs96/ga2 keeps if memory
allows on B200-178GB). Llama: 1.048M subword tok/step (bs32/4gpu/ga4/seq2048).

100M reference points: **ratio-20 = 6688 steps = 42.1B bytes ≈ 9.2B subword tok**; ratio-40
= 13376 steps = 84.1B bytes.

Two valid budget axes diverge ~3× at this scale — pick deliberately:

| Axis | What it holds | 300M budget | Per-param | Cost |
|------|---------------|-------------|-----------|------|
| **A — iso-token (RECOMMENDED primary)** | absolute tokens = 100M ratio-20 | **42B bytes / 9.2B subword** (6688 / 8800 steps) | ~31 subword-tok/param (still over-trained) | 1× |
| B — iso-ratio | tokens/param = 100M ratio-20 (~93/param) | 129B bytes / 28B subword (~20.5k steps) | ~93 | ~3× |

**Recommendation: Option A.** It builds the standard iso-data scaling ladder
(100M / 300M / 1.3B at a fixed 42B-byte budget) so any change in the leak gap is read directly
off the curve, and it's affordable. 300M at 42B is still in the over-train regime where the
1.3B +4 gap appeared. **If the gap is ambiguous at 42B, extend to 84B (ratio-40-equiv)** —
mirrors the 100M ratio-20→40 doubling that moved v6_prefix_free +1.7 HS. Reserve Option B
only if the iso-ratio regime is specifically demanded by a reviewer.

---

## 3. GPU-hours

300M ≈ 3.0× the 100M FLOPs. Measured 100M AU-Net throughput on 4×B200 = 5.3s/step (6.29M
bytes/step) → 300M ≈ **16 s/step (4×B200)** / ≈17 s/step (8×A100, A100≈0.5× B200 but 2× count).

**Option A (42B-byte primary):**

| Model | Steps | s/step | Wall (4×B200) | GPU-hrs (B200) |
|-------|------:|-------:|--------------:|---------------:|
| AU-Net (word) | 6688 | 16 | ~30 h | ~119 |
| greedy_root | 6688 | 16 | ~30 h | ~119 |
| committed_view | 6688 | 16 | ~30 h | ~119 |
| Llama | 8800 | ~10 (seq2048) | ~24 h | ~98 |
| **Total** | | | **~114 h serial** | **~455 B200-GPU-hrs** |

Parallelize on the 8×A100 ece node (2 GPU/model → all 4 concurrent, ~2× slower each) →
finishes in **~2 days wall**. Option B is ~3× these numbers (~6 days serial). Eval (0-shot
HellaSwag / ARC-E / ARC-C / PIQA, reusing `eval.py`, incl. `committed_view_loglikelihood`
leak-free scoring for #3/#4) is ~1 GPU-hr per model.

---

## 4. Eval plan (reuse existing harness)

- **Baselines (1,2):** standard `loglikelihood` 0-shot, 4 benchmarks.
- **Leak-free (3,4):** the matching leak-free scorer is mandatory or the comparison is unfair —
  greedy_root scored with its native causal boundaries; committed_view scored with
  `committed_view_loglikelihood` (full-seq bt + streaming committed index, matches v1 training
  `up()`). All already implemented in `apps/aunet/eval.py`.
- Report the same table as 100M (HS / ARC-E / PIQA / ARC-C, acc_norm) plus the **leak-gap
  column** = baseline_view − committed_view for #4, to place 300M on the 100M(≈0) → 1.3B(+4)
  curve.

---

## 5. Next actions
1. Generate the 4 configs from the 100M files with the model-block + budget edits above
   (`bpebyte_300M_{v4_root_greedy,v1_committed}.yaml`, `aunet_300M.yaml`, `llama_300M.yaml`).
2. Confirm the 300M byte model fits bs96/ga2 on B200 (else bs48/ga4); confirm Llama bs32/ga4.
3. Launch on freed B200 (1.3B greedy_root finishing) or 2-GPU-each on ece 8×A100.
4. Eval at 42B; extend to 84B only if the leak gap is ambiguous.

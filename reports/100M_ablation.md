# AU-Net 100M — Boundary-Scheme Ablation (v1–v5)

**What this is.** Five 100M BPEByte models, identical architecture/recipe, differing only in the
**patch-boundary scheme**. All trained for **1672 steps** (batch 96, seq 8192, grad_acc 2) on the
4×B200, then evaluated on HellaSwag / ARC-Easy. Run dir: `runs/ablation_100M/`.

_Last updated: 2026-06-29. **The 1672-step ablation below (v1–v5) was undertrained and could not
rank the schemes.** The canonical, properly-powered comparison is the **ratio-40** section
immediately below (same 100M arch, 13,376 steps ≈ 84B bytes, 4 benchmarks), followed by the
**ratio-10/20/40 budget trajectory** and the finished **v8 entropy-patching** study._

---

# Ratio-40 (84B bytes) — CANONICAL boundary-scheme comparison

**What this is.** The same 100M BPEByte/AU-Net architecture, trained to **13,376 steps / ratio-40
(≈84B bytes)** — long enough that the boundary scheme produces a real quality signal (unlike the
1672-step run below). Subword **Llama** (17,600 steps, iso-byte) and the **AU-Net word-patch**
model are included as baselines. Eval: `acc_norm %` on HellaSwag (n=10,042) / ARC-Easy (n=2,376) /
PIQA / ARC-Challenge, full datasets. Verified from each run's `eval_r40_allbench/results.json`.

| Model | Scheme | HS | ARC-E | PIQA | ARC-C | **Avg** |
|---|---|---:|---:|---:|---:|---:|
| **llama** | subword baseline | 35.60 | 45.20 | 65.34 | 24.74 | **42.72** |
| **v6-root** | prefix_free, **root** (leak-free) | 32.34 | 38.01 | 60.88 | 25.09 | **39.08** |
| **v6** | prefix_free, before_root | 32.70 | 37.04 | 61.21 | 25.34 | **39.07** |
| **v4** | root_greedy (leak-free) | 31.92 | 37.25 | 61.81 | 24.06 | **38.76** |
| **v7** | prefix_vocab, root (leak-free) | 31.86 | 38.59 | 61.97 | 22.35 | **38.69** |
| **aunet** | AU-Net word-patch baseline | 32.37 | 37.71 | 61.70 | 22.87 | **38.66** |
| **v1** | committed-view (bt, delayed mask) | 29.83 | 34.60 | 59.90 | 24.49 | **37.21** |
| _warm_rg_ | v4 + subword-warm trunk | 31.59 | 37.92 | 60.66 | 23.21 | _38.35_ |
| _warm_v1_ | v1 + subword-warm trunk | 29.02 | 33.71 | 56.64 | 20.56 | _34.98_ |


| Entropy model | HS | ARC-E | PIQA | ARC-C | **Avg** |
|---|---:|---:|---:|---:|---:|
| **v4** | root_greedy (leak-free) | 31.92 | 37.25 | 61.81 | 24.06 | **38.76** |
| **llama** | subword baseline | 35.60 | 45.20 | 65.34 | 24.74 | **42.72** |
| **aunet** | AU-Net word-patch baseline | 32.37 | 37.71 | 61.70 | 22.87 | **38.66** |
| MID (byte_50M, 10×)              | 31.94 | 36.57 | 60.88 | 23.12 | **38.13** |
| HIGH (byte_50M, 20×)             | 31.99 | 35.35 | 61.64 | 23.63 | **38.15** |
| LOW (byte_50M, 5×)               | 31.74 | 35.27 | 58.92 | 23.38 | **37.33** |
| BLT (Meta, 100M)                 | 30.19 | 35.94 | 58.81 | 22.95 | **36.98** |

(sorted by average; warm-start rows in italics. Run dirs: `runs/cmp_100M_ratio20/<model>/`.)

**Findings.**
- **Leak-free byte == word baseline.** Every leak-free byte scheme (v6-root 39.08, v4 38.76, v7
  38.69) ties or beats the AU-Net **word** baseline (38.66) on average — no penalty for going to
  raw bytes with a causal patch boundary. Subword **llama leads all by ~3.6 avg** — the byte↔subword
  gap the 300M/1.3B runs are meant to test for narrowing.
- **The before_root dead-end leak is worth ~+0.36 HS, but nets ~0 overall.** v6 (before_root,
  leaky) HS 32.70 → v6-root (leak-fixed) HS 32.34 (−0.36): the residual leak (the ~12.7% of patches
  whose boundary peeks at the breaking byte) inflated HS specifically. But v6-root *gains* ARC-E
  (+0.97), so on **average it's a wash** (39.08 vs 39.07) — the leak inflated HS, not overall
  capability. v6-root is the **best byte scheme on average** and best byte on ARC-C (25.09).
- **committed-view (v1) is the weakest byte scheme** (37.21) — the delayed 3rd-view mask
  underperforms by ~1.5–2 avg vs the trie/greedy schemes.
- **Trunk warm-start does NOT help at ratio-40.** Subword-warm-started trunks are *worse* on both
  schemes (warm_rg −0.41 avg vs v4; warm_v1 −2.2 vs v1) — consistent with the undertraining-regime
  hypothesis (warm-start helps only when more undertrained; at 84B these are well-trained).

**Scheme legend (ratio-40):** v1 = `bt`/before_root + committed_view · v4 = `greedy`/root ·
v6 = `prefix_free`/before_root · v6-root = `prefix_free`/root · v7 = `prefix_vocab`/root.
All of v4/v6-root/v7 are leak-free + causal (no boundary depends on a future byte).

## Training budget — wall-clock & the ratio-20 → ratio-40 trajectory

**Ratio = the AU-Net paper's data-to-model ratio** (Videau et al. 2025, "From Bytes to Ideas",
arXiv:2506.14761), γ = N_data / F_FLOPs-per-input-unit — a compute-normalized data-to-model ratio,
**not** a Chinchilla multiple. The paper notes **ratio-10 ≈ 2× the Kaplan/Chinchilla-optimal ratio**
(so ratio-5 ≈ 1× Chinchilla, ratio-40 ≈ ~8×). For byte models γ_byte = k²·γ_token with k ≈ 4.56
(bytes per LLaMA-3 token), which is why the byte-ratio maps to these large byte counts. Mirrors the
paper's Table 2 (AU-Net 2 1.3B @ ratio-10 = 60B LLaMA tokens ≈ 273B bytes), scaled to 100M.
Byte models: ratio-5/10/20/40 = **1672 / 3344 / 6688 / 13376** steps (6.29M bytes/step → ratio-40
≈ 84B bytes). Subword **Llama** is iso-byte so it runs more steps: 2200 / 4400 / 8800 / **17600**.
_(NB: the separate "5×/10×/20× Chinchilla" labels elsewhere refer to the byte_50M **entropy models**
LOW/MID/HIGH — a different axis, not the AU-Net data-to-model ratio.)_

**Wall-clock on 4× A100** (per-step time depends on the scheme):

| Scheme | s/step | ratio-10 | ratio-20 | ratio-40 |
|---|---:|---:|---:|---:|
| standard byte (v4/v6/v7/v6-root/aunet) | ~4.1 | ~3.8 h | ~7.6 h | **~15 h** |
| committed_view (v1) — extra delayed-mask view | ~8.1 | ~7.5 h | ~15 h | **~30 h** |
| Llama (subword) | ~2.1 | ~2.6 h | ~5.1 h | **~10.3 h** |
| v8 entropy (precompute cache) | ~4.0 + precompute | — | — | ~15 h **+ ~14 h one-time precompute** per entropy model |

(v8's entropy boundaries are precomputed once per entropy model — a separate ~14 h pass, reusable —
then training runs at standard-byte speed. The on-the-fly fp32 path was ~22.6 s/step ≈ 84 h and was
abandoned for the cache.)

### Budget trajectory — ratio-10 / 20 / 40, per-benchmark + average

Core 6 models at each budget (acc_norm %). ratio-10 = step 3344 (llama 4400), retrained fresh;
ratio-20 = step 6688 (llama 8800); ratio-40 = 13376 (llama 17600). v6 here = `prefix_free`/before_root.
v7 has no ratio-20 (no surviving step-6688 checkpoint). Eval = `eval_r{10,20,40}_allbench/`.

| Model | Ratio | HS | ARC-E | PIQA | ARC-C | **Avg** |
|---|---|---:|---:|---:|---:|---:|
| **llama** | r10 | 31.67 | 42.55 | 63.38 | 23.04 | **40.16** |
|  | r20 | 33.80 | 43.86 | 63.66 | 24.83 | **41.53** |
|  | r40 | 35.60 | 45.20 | 65.34 | 24.74 | **42.72** |
| **v6** | r10 | 29.46 | 33.33 | 59.36 | 23.04 | **36.30** |
|  | r20 | 31.01 | 35.94 | 60.55 | 25.34 | **38.21** |
|  | r40 | 32.70 | 37.04 | 61.21 | 25.34 | **39.07** |
| **v4** | r10 | 28.69 | 33.46 | 56.75 | 22.35 | **35.31** |
|  | r20 | 30.38 | 35.61 | 59.63 | 23.46 | **37.27** |
|  | r40 | 31.92 | 37.25 | 61.81 | 24.06 | **38.76** |
| **v7** | r10 | 28.67 | 33.80 | 56.80 | 22.44 | **35.43** |
|  | r20 | — | — | — | — | _—_ |
|  | r40 | 31.86 | 38.59 | 61.97 | 22.35 | **38.69** |
| **aunet** | r10 | 28.80 | 33.80 | 58.49 | 22.61 | **35.92** |
|  | r20 | 31.19 | 36.20 | 61.10 | 22.61 | **37.77** |
|  | r40 | 32.37 | 37.71 | 61.70 | 22.87 | **38.66** |
| **v1** | r10 | 28.27 | 31.65 | 55.11 | 21.08 | **34.03** |
|  | r20 | 28.88 | 33.63 | 57.56 | 22.35 | **35.61** |
|  | r40 | 29.83 | 34.60 | 59.90 | 24.49 | **37.21** |

Each budget doubling buys ~**+1.0 to +1.5 avg**, monotonically, and the scheme ordering is **preserved
at every budget** (llama ≫ v6 > v4 ≈ v7 ≈ aunet > committed-view v1). HS and ARC-E carry most of the
budget signal; PIQA/ARC-C move less. (ratio-20-only legacy schemes: v3_offline 37.57, v5_distilled 37.34.)

## v8 — entropy patching (BLT-style), ratio-40

Boundaries from a small byte-LM's next-byte surprise (monotonic ΔH, target 4.5 B/patch, root
placement, leak-free) instead of a BPE trie. **Entropy-model budget study**: our byte_50M at
5×/10×/20× Chinchilla (LOW/MID/HIGH) + Meta's `facebook/blt-entropy` 100M (the intended ceiling).

| Entropy model | HS | ARC-E | PIQA | ARC-C | **Avg** |
|---|---:|---:|---:|---:|---:|
| MID (byte_50M, 10×) | 31.94 | 36.57 | 60.88 | 23.12 | **38.13** |
| HIGH (byte_50M, 20×) | 31.99 | 35.35 | 61.64 | 23.63 | **38.15** |
| LOW (byte_50M, 5×) | 31.74 | 35.27 | 58.92 | 23.38 | **37.33** |
| BLT (Meta, 100M) | 30.19 | 35.94 | 58.81 | 22.95 | **36.98** |

**Findings.**
- **Entropy-model budget plateaus at MID.** 5×→10× gains ~0.8 avg (LOW 37.33 → MID 38.13); 10×→20×
  adds essentially nothing (38.13 → 38.15). More entropy-model training stops helping early.
- **Meta's BLT 100M is the *worst*, not the ceiling** (36.98 < all our 50M runs). A bigger/better
  entropy model did **not** give better downstream AU-Net boundaries — the learned-surprise signal
  it produces doesn't transfer to a stronger downstream model here (despite matched 4.5 B/patch).
- **Entropy patching < trie patching** at 100M: every v8 (37–38 avg) lands **below** the structural
  trie/greedy schemes (v4 38.76 / v6 39.07 / v7 38.69). The learned boundary loses to the vocab-trie
  boundary for downstream quality at this scale.

---

_(Below: the original v1–v5 1672-step ablation — kept as a record of why it could not rank the
schemes. Superseded by the ratio-40 table above.)_

## TL;DR / correction

- The **full-dataset** eval (entire HellaSwag 10,042 + ARC-E 2,376) shows **all five variants are
  statistically tied and barely above the 25% 4-choice random floor** (HellaSwag ~27%, ARC-E ~30%).
- The **1672-step 100M models are too undertrained** for the boundary scheme to produce any quality
  signal — so this ablation, as trained, **cannot rank the schemes on accuracy.**
- The earlier **limit-150** numbers (HellaSwag 38–44%) were **noise** — those 150 examples were
  unrepresentatively easy. Do not use them.

## Variant configurations

Only the `data.regex` boundary settings differ:

| Variant | `bpe_online` | `bpe_online_mode` | `bpe_online_placement` | `committed_view` | npz |
|---|---|---|---|---|---|
| **v1_committed** | true | bt | before_root | **true** | — |
| **v2_online** | true | bt | before_root | false | — |
| **v3_offline** | **false** | bt | before_root | false | — |
| **v4_root_greedy** | true | **greedy** | **root** | false | — |
| **v5_distilled** | true | **distilled** | before_root | false | `tokenizer/boundary_v56/distilled_boundary.npz` |

## Eval setup

- Config: `apps/aunet/configs/eval_ablation.yaml` — `max_tokens: 16384`, `validation: null`,
  **nproc 1** (single GPU, B200).
- Tasks (**all likelihood-based — no text generation**):
  - `hellaswag`, `arc_easy` — multiple-choice **loglikelihood**: score each of the 4 endings, pick the
    most likely. `acc_norm` = length-normalized loglikelihood (the standard HellaSwag metric).
  - `hellaswag_gen_ll`, `arc_easy_gen_ll` — gen-framed **single-letter likelihood** (argmax over the
    answer letter A/B/C/D).
  - The true generation tasks (`hellaswag_gen`, `arc_easy_gen`, `generate_until` → exact_match) were
    **dropped** — the online-boundary variants (v2/v4) hang in autoregressive online-boundary decode
    (see Issues).
- 4-choice random floor = **25%** for both HellaSwag and ARC-Easy.

## Results — FULL DATASET (canonical)

HellaSwag n=10,042 · ARC-Easy n=2,376 · acc_norm % ± stderr:

| Variant | HellaSwag | ARC-Easy | HS gen_ll | ARCe gen_ll | avg acc_norm |
|---|---:|---:|---:|---:|---:|
| v1_committed   | 26.9 ±0.4 | 31.1 ±0.9 | 24.8 | 26.4 | 29.0 |
| v2_online      | 27.7 ±0.4 | 31.1 ±0.9 | 24.7 | 25.4 | **29.4** |
| v3_offline     | 27.7 ±0.4 | 29.8 ±0.9 | 25.0 | 26.7 | 28.7 |
| v4_root_greedy | 27.5 ±0.4 | 30.0 ±0.9 | 25.2 | 25.7 | 28.7 |
| v5_distilled   | 27.6 ±0.4 | 31.0 ±0.9 | 25.3 | 23.8 | 29.3 |

**Significance:** HellaSwag spread is 0.8 pt vs ±0.4 stderr (v1↔v2: z≈1.4, **not** significant);
ARC-E spread 1.3 pt vs ±0.9 (**not** significant); avg range 0.7 pt. **No variant is measurably
better than another.** All sit ~2 pt (HS) / ~5 pt (ARC-E) above the 25% random floor → the models
have barely learned the tasks at 1672 steps.

Results dir per variant: `<variant>/evals_full/results.json`.

## Results — limit 150 (SUPERSEDED, noise; kept as a caution)

acc_norm %, only 150 samples (±~8 pt 95% CI — too coarse to rank):

| Variant | HS acc_norm | ARC-E acc_norm | avg |
|---|---:|---:|---:|
| v1_committed | 38.0 | 38.7 | 38.3 |
| v2_online | 39.3 | 36.0 | 37.7 |
| v3_offline | 42.7 | 32.7 | 37.7 |
| v4_root_greedy | 40.0 | 32.7 | 36.3 |
| v5_distilled | 38.7 | 36.7 | 37.7 |

These are **~10–15 pt inflated vs the full dataset** (the 150 examples were easy) and reorder under
noise. **Lesson:** never rank ablation variants on a 150-sample eval. Dirs: `evals_deferred/`
(v1/v3 orig config) and `evals_uniform/` (uniform re-eval). The uniform re-eval confirmed v2/v4/v5
reproduce bit-identically and v1/v3 wobble 1–2 examples from `nproc 4→1` sharding — but it's all moot
given the full-dataset result.

## Verdict: scale v4 / v5 to 1B?

- **On accuracy: the data supports nothing** — all five are tied at near-random, so there is no
  measured-quality basis to scale any particular variant. Picking off these numbers = picking noise.
- **v4_root_greedy: do not scale.** No accuracy edge + worst operational profile (greedy decode that
  infinite-looped, commits invalid patches, CPU-heavy).
- **v5_distilled: the only one with a non-accuracy reason to scale** — a causal, cheap, **prefix-stable**
  k-gram boundary predictor where the batch-training mask and streaming-generation mask are identical
  by construction (zero commitment lag, zero rollback). That's a generation-time advantage to validate
  at scale *if* that property matters — not an accuracy claim.
- **Prerequisite before spending ~22 h/variant of 1B compute:** train the variants **longer** (full
  ablation step count, not 1672) so they clear random by a real margin and the eval can actually
  discriminate. The experiment is under-powered at the *training* stage, not the eval.
- Highest-value 1B comparison if pursued: **v3_offline (real-BPE) vs v5_distilled (its causal
  approximation)**, with the running 1.3B online (~v2) as the third point. v4 adds least.

## Issues found & fixed (boundary code; both root-caused to special tokens ≥256 in the byte stream)

1. **v5 distilled — `OverflowError: 256 out of bounds for uint8`** (`boundary_distill.py`,
   `DistilledBoundaryPredictor.mask01`/`feed`). The byte sequence carries BOS=256, which overflows a
   `uint8` array in numpy≥2. **Fix:** cast to `int64` (real bytes 0–255 hash identically via the
   FNV→uint64 path; ≥256 match no trained k-gram → prior). Training never hit it (training chunks are
   pure 0–255), so v5 trained fine but couldn't eval.
2. **v4 greedy — infinite loop** (`byte_trie.py`, `greedy_tokenize_boundaries`). A byte not in the
   trie root (BOS=256) gives `i==pos` → `longest_match = i-pos = 0` → `pos += 0` → loops forever,
   CPU pegged. **Fix:** `longest_match = max(1, i-pos)` (commit the unknown byte as a 1-byte patch,
   matching bt-mode's min-advance). `bt` mode never hit it (its default `longest_match=1`).

Committed to the `lingua` submodule `main` as **`2946db6`** (`boundary_distill.py` force-added — the
`apps/aunet/data` dir is gitignored). Not pushed; parent submodule pointer not yet bumped.

## Operational notes

- v5 was retrained from scratch (the original run OOM'd only because eval procs were co-resident);
  ~39 min on 4×B200.
- An attempt to offload v2/v4/v5 evals to **snu55** was abandoned: its driver (CUDA 12.2) can't run
  the cu126 envs; a hand-built cu121 env hit triton/nccl/numpy mismatches; and sshd rate-limited.
  Once the hanging `_gen` tasks were dropped, the evals were ~2-min loglikelihood runs that fit on the
  B200 alongside the 1.3B training (~91 GB free/GPU). **Lesson: no need for snu55.**

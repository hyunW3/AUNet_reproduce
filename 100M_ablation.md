# AU-Net 100M — Boundary-Scheme Ablation (v1–v5)

**What this is.** Five 100M BPEByte models, identical architecture/recipe, differing only in the
**patch-boundary scheme**. All trained for **1672 steps** (batch 96, seq 8192, grad_acc 2) on the
4×B200, then evaluated on HellaSwag / ARC-Easy. Run dir: `runs/ablation_100M/`.

_Last updated: 2026-06-11. **Canonical numbers = the full-dataset eval below.** The earlier
limit-150 numbers were small-sample noise and are kept only as a cautionary record._

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

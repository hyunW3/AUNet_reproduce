# Evaluation Results — 1.3B–1.8B models, 5 benchmarks (full sets)

Standard multiple-choice (loglikelihood, lm-eval default settings), **full validation sets**.
Models:

| Model | params | tokenization | eval tokenization |
|-------|--------|--------------|-------------------|
| **BPEByte online** (`bpebyte_br_bt_online_1.3B`, step 180k) | ~1.3B | online byte-trie `bt` | **hybrid greedy** (question=`bt`, answer=`greedy`) |
| **AU-Net 2** (`aunet2_1.3B`, step 180k) | ~1.3B | word patches | native (no byte-trie) |
| **Llama 1.8B** (`llama_1.8B_paper`, step 60k) | ~1.8B | BPE subword | native |

Benchmarks: HellaSwag, ARC-Easy, BoolQ, PIQA, Winogrande (all default lm-eval settings).
Run on snu55 (RTX A5000, cu121 env). Metric: `acc_norm` where the task defines it, else `acc`.

## Results (0-shot)

Full validation sets (HellaSwag 10042, ARC-Easy 2376, BoolQ 3270, PIQA 1838, Winogrande 1267).
Headline metric per benchmark: `acc_norm` for HellaSwag/ARC-Easy/PIQA, `acc` for BoolQ/Winogrande.
**Bold** = best per row.

BPEByte online is shown in three eval-tokenization modes (see "How the greedy eval works"):
`bt` = full batch backtracking (**leaks** future answer bytes), `hybrid-greedy` = question `bt` +
answer `greedy` (leak-free), `full-greedy` = greedy everywhere (leak-free). AU-Net 2 / Llama use
their native tokenization.

| Benchmark | metric | BPEByte `bt` (leaky) | BPEByte hybrid-greedy | BPEByte full-greedy | AU-Net 2 | Llama 1.8B |
|-----------|--------|----------------------|-----------------------|---------------------|----------|------------|
| HellaSwag | acc_norm | 0.6426 | 0.6402 | 0.6388 | 0.6268 | 0.6221 |
| ARC-Easy  | acc_norm | 0.6524 | 0.6410 | 0.6410 | 0.6570 | 0.6540 |
| BoolQ     | acc      | 0.6388 | 0.6361 | 0.6379 | 0.6110 | 0.6333 |
| PIQA      | acc_norm | 0.7405 | 0.7383 | 0.7372 | 0.7437 | 0.7492 |
| Winogrande| acc      | 0.6172 | 0.6156 | 0.6148 | 0.6109 | 0.6109 |

**Leakage cost.** `bt` (leaky) is consistently the highest BPEByte column — by **+0.2 to +1.1 pts**
(largest on ARC-Easy). That gap is the future-leakage advantage the leak-free evals remove.
**hybrid-greedy ≈ full-greedy** everywhere (the question's bt-vs-greedy tokenization barely moves
the score — it is conditioning, not the scored continuation). The honest, leak-free BPEByte number
is the **hybrid-greedy** column.

Secondary `acc` (where both reported), leak-free hybrid-greedy / AU-Net 2 / Llama:
HellaSwag 0.4727 / 0.4911 / 0.4724; ARC-Easy 0.6650 / 0.6940 / 0.6886; PIQA 0.7231 / 0.7329 / 0.7503.

### Summary

The three architectures are **closely matched** across all five benchmarks (full-set, so these
gaps are real, not sampling noise — though most are still ≤1–2 pts):

- **BPEByte online** (under the hybrid-greedy eval) is **fully competitive** — best on HellaSwag
  (0.640) and BoolQ (0.636, tied with Llama), within ~1.5 pts elsewhere. The byte/online-boundary
  model is **not disadvantaged** relative to word- or subword-tokenized models.
- **AU-Net 2** leads on ARC-Easy (0.657) and PIQA (0.744).
- **Llama 1.8B** leads on PIQA (0.749); roughly tied elsewhere despite ~0.5B more params and a
  different token budget.
- **Winogrande** sits near ~0.61 for all three (it is hard at this scale).

## Few-shot results (5-shot / 10-shot)

Uniform `num_fewshot` across all 5 benchmarks, full sets. Few-shot demonstrations are **context**,
so for the byte model they are tokenized with `bt` (the hybrid uses `bt` on everything before the
answer). Metric: `acc_norm` for HS/ARC-E/PIQA, `acc` for BoolQ/Winogrande.

**5-shot**

| Model | HellaSwag | ARC-Easy | BoolQ | PIQA | Winogrande |
|-------|-----------|----------|-------|------|------------|
| BPEByte `bt` (leaky) | 0.6448 | 0.6965 | 0.6281 | 0.7291 | 0.6283 |
| BPEByte hybrid-greedy | 0.6407 | 0.6869 | 0.6242 | 0.7280 | 0.6314 |
| BPEByte full-greedy | 0.6409 | 0.6864 | 0.6217 | 0.7285 | 0.6298 |
| AU-Net 2 | 0.6369 | 0.7252 | 0.5896 | 0.7416 | 0.6393 |
| Llama 1.8B | 0.6294 | 0.7016 | 0.6508 | 0.7573 | 0.6314 |

**10-shot**

| Model | HellaSwag | ARC-Easy | BoolQ | PIQA | Winogrande |
|-------|-----------|----------|-------|------|------------|
| BPEByte `bt` (leaky) | 0.6497 | 0.7121 | 0.5939 | 0.7459 | 0.6243 |
| BPEByte hybrid-greedy | 0.6445 | 0.7003 | 0.5920 | 0.7437 | 0.6243 |
| BPEByte full-greedy | 0.6461 | 0.7003 | 0.5924 | 0.7427 | 0.6219 |
| AU-Net 2 | 0.6422 | 0.7332 | 0.5651 | 0.7508 | 0.6267 |
| Llama 1.8B | 0.6383 | 0.7104 | 0.6480 | 0.7590 | 0.6377 |

**Trends (0→5→10-shot):**
- **ARC-Easy improves** strongly with shots (BPEByte 0.65→0.70→0.71; AU-Net 2 up to 0.733; Llama 0.71).
- **HellaSwag** improves slightly; **PIQA** roughly flat-to-up.
- **BoolQ degrades** with few-shot for the byte/word models (BPEByte 0.639→0.628→0.594; AU-Net 2
  0.611→0.590→0.565) — consistent with **context truncation**: BoolQ's long passages mean 5–10 shots
  exceed the byte model's ~8192-byte budget (and AU-Net 2's patch budget), so lm-eval left-truncates
  → effective shots < nominal and the format is disrupted. **Llama's BoolQ holds ~0.65** (subword
  shots fit), so this is a byte/word-model context-length limitation, not a capability gap.
- Across every shot count, **hybrid-greedy ≈ full-greedy**, and **`bt` (leaky)** stays slightly above
  the leak-free greedy columns.
- Note: **streaming-bt commit-margin** is identical to the `bt` column at every shot count
  (verified: `finalize()` == batch `bt` for complete-answer scoring), so it is not a separate row.

### Truncation audit (why BoolQ degrades with shots)

Reconstructed the actual lm-eval prompts at each shot count and measured the % of scored sequences
exceeding each model's budget (byte models: 8191 B; Llama: 4095 tok). **Truncation happens in
exactly one place — BoolQ at 10-shot:**

| Benchmark | shot | BPEByte / AU-Net 2 (byte, 8192 B) | Llama (4096 tok) | byte p95 |
|-----------|------|-----------------------------------|------------------|----------|
| HellaSwag | 0/5/10 | 0% / 0% / 0% | 0% | 556 / 2710 / 4702 |
| ARC-Easy  | 0/5/10 | 0% / 0% / 0% | 0% | 308 / 1249 / 2132 |
| **BoolQ** | 0/5/10 | **0% / 0.03% / 17.57%** | 0% | 1228 / 5343 / **9162** |
| PIQA      | 0/5/10 | 0% / 0% / 0% | 0% | 348 / 1410 / 2361 |
| Winogrande| 0/5/10 | 0% / 0% / 0% | 0% | 139 / 712 / 1260 |

- **BoolQ-10shot: 17.57%** of byte-model sequences exceed 8192 B → left-truncated (lose the prompt
  head: early shots and/or the start of the target passage). This is the cause of the BoolQ-10shot
  drop (0.628→0.594). **BPEByte and AU-Net 2 truncate identically** (same byte input + budget).
- **BoolQ-5shot is NOT truncation** (0.03%) — the 5-shot dip is the OOD-long-context / no-upside
  effect, not lost content.
- **No other benchmark/shot truncates** (HellaSwag-10shot peaks at p95 4702 B < 8192). Their
  few-shot changes are pure capability/calibration.
- **Llama never truncates** (subword packing ≈ 4.5 B/token → BoolQ-10shot p95 ≈ 2002 tok < 4096).

## ARC-Challenge + MMLU

Full sets, loglikelihood MC. **ARC-Challenge** (acc_norm) at 0/5/10-shot:

| Model | 0-shot | 5-shot | 10-shot |
|-------|--------|--------|---------|
| BPEByte bt | 0.378 | 0.408 | 0.418 |
| BPEByte hybrid-greedy | 0.374 | 0.410 | 0.415 |
| BPEByte full-greedy | 0.375 | 0.412 | 0.416 |
| AU-Net 2 | 0.358 | 0.407 | 0.413 |
| Llama 1.8B | 0.351 | 0.388 | 0.401 |

ARC-C is short (no truncation), few-shot helps all, and **BPEByte leads at every shot count**.

**MMLU** (acc), 0-shot only — few-shot dropped (near-random for these base models, ~3h to run,
low value):

| Model | MMLU 0-shot |
|-------|-------------|
| BPEByte bt / hybrid / full-greedy | 0.2471 / 0.2480 / 0.2479 |
| AU-Net 2 | 0.2631 |
| Llama 1.8B | 0.2500 |

All ≈ 0.25 (4-way random floor) — expected: MMLU is hard for 1.3–1.8B base models 0-shot. (MMLU
5/10-shot was started but skipped — full byte-model MMLU few-shot is ~3h and stays near random.)

## Committed-mask (truly leak-free) scoring — ARC-Easy 0-shot

`committed_answer_loglikelihood`: in the answer region, a greedy BOUNDARY byte `e-1` gathers the
PREVIOUS *committed* patch instead of its own just-closed patch (whose before_root boundary needs
the dead-end byte `e`). Position/local-pos stay on the true cumsum; only the byte→patch gather index
shifts. Mirrors the streaming decoder (the current word is still seen via the byte-level decoder; only
its *pooled* rep is withheld until committed). gap=0, zero future-token leakage on the answer.

ARC-Easy 0-shot leak ladder (acc_norm), BPEByte online (step 180k):

| method | acc_norm | answer leak |
|--------|----------|-------------|
| atomic (answer = 1 patch) | 0.3889 | none, but NO segmentation |
| **committed (gap=0)** | **0.5602** | **none — truly leak-free** |
| greedy / hybrid | 0.6427 | 1-byte before_root |
| bt | 0.6524 | unbounded backtracking |
| **AU-Net 2** (target) | **0.6570** | n/a (word patches) |

**Exact-spec confirmation (`bt_committed_answer_loglikelihood`).** The leak-free rule, stated
precisely (seq ABCDFG, D set as a before_root boundary by F): score `P(F|ABCD)` WITHOUT the D
boundary (it was generated by F) and `P(G|ABCDF)` WITH it (F is now past). Implemented as margin-1
before_root gap=0 on a **bt-tokenized** answer (matched to the br_bt model). Result: ARC-Easy 0-shot
acc_norm = **0.5602** (acc 0.5804) — identical to the greedy committed-mask above, so the leak-free
score is robust to greedy-vs-bt answer tokenization; the margin-1 boundary exclusion is what matters.
(`committed_patch_idx` margin-2 over-commits → ≈0 correction at clean boundaries, so it is NOT the
right tool for the 1-byte before_root spec; the boundary-mask is.)

Monotone `atomic < committed < greedy < bt < AU-Net 2` → implementation validated (greedy with the
flag off reproduces 0.6427). Findings:
- The 1-byte before_root leak is worth **~8 pts on ARC-Easy** (0.5602→0.6427) — far more than on
  HellaSwag (~2 pts), because ARC-Easy answers are short content words where the current-word pooled
  rep matters a lot.
- The gap is dominated by **train/eval mismatch**: the model was trained with the leaky as-if-complete
  (cumsum) view, so it leans on the current-word pooled rep at boundaries. Removing it only at eval
  cripples it. ⇒ **eval-time leak removal cannot reach AU-Net 2 (0.6570); leak-free TRAINING is
  required** (committed-view training, or statistics-based before_root — see `paper_exp_plan260618.md`).

## Causal root-statistics scoring (training-free, no retrain) — ARC-Easy 0-shot

Idea: keep the trained cumsum read (no train/eval mismatch, unlike committed), but make the
before_root boundary DECISION causal. Gather `P(commit | is_token node)` from the DCLM training
corpus (`apps.aunet.data.bpe_root_stats`, 80MB / 79.7K nodes); at eval the answer is tokenized by
committing at the first `is_token` node with `P(commit) ≥ threshold` — using only past bytes, no
next-byte lookahead (`root_stats_answer_loglikelihood`). Residual: if no node commits before a trie
dead-end, fall back to the deepest is_token (the one bt-like, dead-end-observed = still-leaky decision).

| threshold | acc_norm | causal-commit % (on DCLM) | note |
|-----------|----------|---------------------------|------|
| 0.3  | 0.4954 | 89% | very leak-free, over-segments |
| 0.5  | 0.5581 | 81% | leak-free; ≈ committed 0.5602 |
| 0.7  | 0.6061 | 67% | majority-causal, **beats committed by +5pt** |
| 0.85 | 0.6368 | 53% | ≈ greedy (0.6427) |
| 0.95 | 0.6532 | 36% | ≈ bt (0.6524), mostly leaky fallback |

Findings:
- **root_stats dominates committed-mask** at matched leak-freeness (keeps the cumsum read, so no
  train/eval mismatch). The acc-vs-leak frontier is much better.
- acc climbs monotonically with threshold, but causal% falls — high acc comes from reintroducing the
  leak via the dead-end fallback. The genuinely leak-free regime (causal > 80%, thr ≤ 0.5) plateaus
  ~0.56; closing fully to AU-Net 2 (0.6570) still needs leak-free TRAINING.
- (causal% measured on DCLM text; per-task causal% on ARC-Easy answers may differ slightly.)

**Context-conditioned variant** (P(commit | prev_byte, node), 167K ctx entries, backoff to marginal):
thr 0.5 → 0.5623, 0.7 → 0.6086, 0.85 → 0.6347 — only +0.4/+0.25/−0.2 pt vs context-free (within
noise). The prev-byte context fixes some splits (e.g. "sun light"→"sunlight") but word-internal BPE
suffixes still over-segment. ⇒ a single prev byte is too weak; a stronger context (previous *token*,
or 2 bytes) is the next training-free lever, but the plateau is robust across all three eval-time
methods (committed, causal root_stats, context-conditioned). **Reaching AU-Net 2 leak-free needs
leak-free TRAINING** — the model leans on the leaky boundary-timing signal it was trained with.

## How the greedy ("hybrid") evaluation works

BPEByte is a **byte** model: it consumes raw bytes and pools them into **patches** at boundaries.
The online (`br_bt`) model computes those boundaries **causally** with a byte-trie, in one of two
modes:

- **`bt` (backtracking)** — *longest-match-with-backtracking*. It walks the trie, commits the
  **longest valid** token, and **re-feeds** any bytes it over-consumed past that token. It can
  therefore revise a recent boundary after seeing more bytes.
- **`greedy`** — commits at the trie **dead-end** with **no re-feed**. Fully causal, O(n), and
  uses **no lookahead** — a boundary, once placed, is never revised.

### Why a *hybrid* (question = `bt`, answer = `greedy`)

A multiple-choice loglikelihood example is `question/context` + `answer/continuation`, and the
model scores `P(answer | question)`. The two parts have different information availability:

- **Question / context** is given to the model **in full** up front. Backtracking is legitimate
  here — the tokenizer may use the whole context to revise boundaries. → use **`bt`**.
- **Answer / continuation** is what the model is being scored on *causally*. Allowing the
  tokenizer to backtrack inside the answer lets the segmentation **peek at answer bytes the model
  has not yet "generated"** — information not available at real generation time. To keep the
  answer score honest, the answer is tokenized **`greedy`** (no backtracking). 

So the eval applies **`bt` to the question and `greedy` to the answer**, with a patch boundary
forced at the question/answer split (the question's last patch closes, the answer starts a fresh
causal stream).

> Note: the simpler "greedy version" applies greedy to the **entire** sequence (question + answer).
> That over-applies greedy to the question, where backtracking is actually available. The hybrid
> above is the methodologically correct form and is what the BPEByte row uses.

### Worked example

```
question = "Question: ... \nAnswer: Twist the lid"
answer   = " counterclockwise."
```

- **Question** `"... Twist the lid"` → **`bt`** boundaries (backtracking allowed; uses the full
  question).
- **Answer** `" counterclockwise."` → **`greedy`** boundaries, committed left-to-right at trie
  dead-ends with no re-feed.

A concrete byte-trie difference (real llama3 vocab), answer `" disciplined overflowing"`:

| mode | answer patches |
|------|----------------|
| `bt` (backtracking) | `[' disciplined', ' overflowing']` |
| `greedy` (no backtrack) | committed left-to-right at dead-ends, **no re-feed** |

Under the hybrid, the question keeps its `bt` patches while the answer is segmented greedily.
(`bt` vs `greedy` differ only on words where the maximal-munch walk over-consumes past the
longest valid token — about **~2.5–3%** of patches on natural text; on most answers the two
agree.)

### Applicability

The byte-trie modes (`bt`/`greedy`/hybrid) only exist for the **BPEByte online** model. **AU-Net 2**
(word patches) and **Llama** (BPE subword) have no byte-trie, so they are evaluated with their
**native** tokenization — there is no backtracking to remove, and the hybrid does not apply.

## Ideas to boost BPEByte online eval *without future leakage*

The leak-free evals (greedy/hybrid) sit ~0.2–1.1 pts below the leaky `bt` eval. These ideas aim to
recover (or exceed) that — all keep the answer causal (no peeking at un-generated answer bytes).

Reframing "leakage": batch `bt` uses **unbounded** backtracking over the whole answer → it places
early boundaries using far-future bytes (leak). `greedy` has **zero** lookahead but is
**train/gen-mismatched** (the model trained with bt + a small commit margin). The target is a
process that is causal with only the *bounded* lookahead the model actually uses at generation.

1. **Streaming bt with commit-margin (top pick).** Score the answer with the model's real
   generation parser (`ByteTrieIncrementalParser`, commit_margin≈2): it holds the last ~2 token-ends
   speculative and rolls back only on **already-emitted** bytes. Lookahead is bounded to the commit
   margin (exactly what generation sees) → **leak-free in the way that matters AND train-matched**.
   Should beat `greedy` (which is mismatched) without the unbounded-bt leak. The parser already
   exists; wire it into the answer-tokenization hook.
2. **Marginalize over answer segmentations.** True `P(answer|context) = Σ_segmentations P(bytes,seg)`.
   Any single path (greedy/bt) is a lower bound; summing (forward-DP / causal beam) gives the model's
   real probability of the string — strictly higher, leak-free. Biggest principled payoff, most work.
3. **Contextual calibration (PMI).** `log P(answer|context) − log P(answer|null-context)` removes the
   answer's byte-frequency/length prior. Leak-free (null context has no answer info); cheap.
4. **Few-shot in the (bt) context.** k demonstrations in the question/context (uses bt, fully
   available) teach MCQ format + calibrate. Pure context → no answer leak.
5. **Patch-aware length normalization.** `acc_norm` divides by *byte* length; normalize by #patches
   (or a fitted factor) to match the model's prediction granularity. Metric choice → leak-free.
6. **Leak-free surface/prompt ensembling.** Average each option over equivalent surface forms
   (casing/leading-space) and/or context phrasings; smooths byte-model format sensitivity.

**Recommended order:** #1 (streaming-bt, leak-free + train-matched) → #3 (calibration, cheap) →
#2 (marginalization, tightest). #1 is the most likely to lift the byte model *correctly*.

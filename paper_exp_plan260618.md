# Paper experiment plan — 2026-06-18

Consolidated plan. All running experiments stopped on 2026-06-18 (MMLU few-shot killed; snu55
GPUs idle). B200 v4 1.3B training left running (separate project job; unreachable at write time,
irrelevant to the goal below which uses the step-180k checkpoints already on snu55).

## Models under comparison (all step 180k unless noted)

| Model | params | train tokenization | eval tokenization |
|-------|--------|---------------------|-------------------|
| **BPEByte online** (`bpebyte_br_bt_online_1.3B`) | ~1.3B | online byte-trie `bt` | hybrid greedy (question=`bt`, answer=`greedy`); leak-free variants below |
| **AU-Net 2** (`aunet2_1.3B`) | ~1.3B | word patches | native (no byte-trie) |
| **Llama 1.8B** (`llama_1.8B_paper`, step 60k) | ~1.8B | BPE subword | native |

## Completed (results in `evaluation_results.md`)

- 0/5/10-shot on 5 benchmarks (HellaSwag, ARC-Easy, BoolQ, PIQA, Winogrande), full sets, all methods.
- ARC-Challenge 0/5/10-shot; MMLU 0-shot (5/10-shot skipped — near-random, ~3h, low value).
- Truncation audit (only BoolQ-10shot leaks: 17.57% byte truncation at 8192 B).
- **Leak ladder (HellaSwag):** atomic 0.6175 (0 lookahead) < greedy 0.6388 (1-byte before_root
  leak) < bt 0.6426 (unbounded). Confirms the user's 1-byte before_root lookahead insight.

## ★ ACTIVE GOAL (2026-06-18)

**Achieve ARC-Easy 0-shot comparable to AU-Net 2 (0.6570 acc_norm) with BPEByte online and NO
future-token leakage on the answer.**

Baseline ARC-Easy 0-shot `acc_norm`:

| method | acc_norm | leak on answer |
|--------|----------|----------------|
| BPEByte `bt` | 0.6524 | unbounded (leaky) |
| BPEByte greedy / hybrid | 0.6410 | 1-byte before_root |
| **AU-Net 2** | **0.6570** | n/a (target) |
| Llama 1.8B | 0.6540 | n/a |

Note: even leaky `bt` (0.6524) trails AU-Net 2 (0.6570) here — ARC-Easy is AU-Net 2's strongest
benchmark, so this is a genuine target, not just a leak-removal exercise.

### RESULT (2026-06-18): committed-mask scoring done

ARC-Easy 0-shot acc_norm: atomic 0.3889 < **committed (gap=0) 0.5602** < greedy 0.6427 < bt 0.6524
< AU-Net 2 **0.6570**. Implementation validated (greedy flag-off reproduces 0.6427; ladder monotone).
The 1-byte leak is worth ~8 pts here. **Eval-time leak removal lands at 0.5602, ~10 pts under the
AU-Net 2 target** — the gap is mostly train/eval mismatch (model trained leaky). ⇒ goal needs
leak-free TRAINING (Step 2). See `evaluation_results.md` for the full table.

### Step 1 (DONE): #2 committed-mask scoring (gap=0) — truly leak-free
- The 1-byte before_root leak: a greedy boundary at byte `e-1` is set only on seeing dead-end
  byte `e`, so boundary byte `e-1` reading its own just-closed patch needs `e` (future).
- **Mechanism (refined).** For scoring a *complete* sequence, `repeat_idx = mask.cumsum` only
  increments at boundaries, so NON-boundary bytes already read the previous *completed* patch
  (causal, no leak). The leak is exactly the boundary bytes reading their own patch. So gap=0 =
  **`repeat_idx[t] = cumsum[t] − mask[t]` for answer-region positions** (subtract 1 only at answer
  boundaries; question region untouched — it is given context). Equivalent at boundaries to the
  verified `online_committed_patch_idx` helper (greedy → commit_margin 1).
- Wiring: new eval flag `committed_answer_loglikelihood` → hybrid boundaries (question=`bt`,
  answer=`greedy`, as `_answer_greedy_tail`) + regex hook stashes answer-region positions;
  `prefill()` applies the −mask correction at encoder level 0 (where the greedy boundaries live).
- Validation: flag OFF (cumsum) must reproduce greedy 0.6410; flag ON gives the leak-free number.
  Expect leak-free ≤ greedy (model trained with the leaky as-if-complete view → eval-time gap=0 is
  a train/eval mismatch that can only lose). Run ARC-Easy 0-shot, full set (2376).

### ★ Step 2 (ACTIVE, 2026-06-18): statistics-based before_root at INFERENCE (no retraining)

User decision: approach A can be done at **inference time** by gathering statistics from the
training corpus — no retraining. Rationale: committed scoring (0.5602) loses because it withholds
the current-word pooled rep (train/eval mismatch). Instead, KEEP the trained cumsum read (current
word available, matched to training) but make the BOUNDARY DECISION causal so it needs no future
byte → leak-free without mismatch.

Plan:
1. **Gather stats:** run the byte-trie tokenizer over a sample of the TRAINING corpus. At every
   `is_leaf` node visited during the walk, record whether the final tokenization committed there
   (went to root / placed a boundary) vs extended past it. Aggregate per leaf-node id →
   `P(go_to_root | leaf)`.
2. **Causal eval tokenization:** in the answer region, walk the trie; at each `is_leaf` node decide
   the boundary from `P(go_to_root | leaf)` (e.g. threshold 0.5 / argmax) using ONLY the trie path
   so far + the precomputed table — NO lookahead at the next byte. before_root placement, but the
   commit decision is now causal.
3. Score with normal cumsum read (no committed shift) → leak-free, no train/eval mismatch.
4. Run ARC-Easy 0-shot; compare to greedy 0.6427 / committed 0.5602 / AU-Net 2 0.6570. Expect this
   to beat committed (keeps the current-word rep) and approach greedy/AU-Net 2.

**RESULT (2026-06-18): root_stats sweep done.** ARC-Easy 0-shot acc_norm by threshold: 0.3→0.4954,
0.5→0.5581, 0.7→0.6061, 0.85→0.6368, 0.95→0.6532. root_stats DOMINATES committed (0.5602) at matched
leak-freeness (thr 0.7 = 67% causal, 0.6061 vs committed 0.560) because it keeps the trained cumsum
read. But high acc needs high threshold = more leaky dead-end fallback; genuinely leak-free regime
(causal>80%) plateaus ~0.56. So eval-time methods improve the leak/acc frontier but still can't reach
AU-Net 2 (0.6570) fully leak-free. ⇒ Step 3 (leak-free training) for the last ~1pt of fully-causal gap.
Possible refinements before training: context-conditioned stats (cut over-segmentation), min-token-len
rule, or re-measure causal% on ARC-Easy answers.

**RESULT (2026-06-18): context-conditioned (prev-byte) done.** thr 0.5→0.5623, 0.7→0.6086,
0.85→0.6347 — only +0.4/+0.25/−0.2pt vs context-free (within noise). Prev-byte context is too weak
(word-internal BPE suffixes still over-segment). Plateau robust across committed / causal root_stats /
context-conditioned. Training-free path is near its ceiling; next levers: stronger context (previous
TOKEN or 2 bytes) OR leak-free training (Step 3).

### Step 2b (2026-06-18): exact-spec bt-committed scoring (`bt_committed_answer_loglikelihood`)
User spec (ABCDFG, D boundary set by F): `P(F|ABCD)` excludes the D boundary (generated by F),
`P(G|ABCDF)` includes it. = margin-1 before_root gap=0 applied to a **bt-tokenized** answer (matches
the br_bt model's training boundaries). Implemented: prefill subtracts a per-position committed
delta = answer boundary mask at each bt boundary (boundary byte reads previous patch; the next byte
keeps it). NOTE: `committed_patch_idx` (margin 2) over-commits (delta≈0 at clean boundaries), so the
margin-1 boundary-mask is the faithful implementation of the 1-byte before_root spec.
**RESULT (2026-06-18): bt_committed ARC-Easy 0-shot acc_norm = 0.5602** (acc 0.5804) — identical to
greedy committed (0.5602), confirming the leak-free score is robust to greedy-vs-bt answer
tokenization. This is the definitive leak-free hybrid number under the exact spec. ~10pt below
AU-Net 2 (0.6570) ⇒ reaching parity leak-free needs leak-free TRAINING (Step 3).

### ★ Step 3 (2026-06-18): committed-view TRAINING = bt_committed at train time

User idea: extend bt_committed to training so `P(F|ABCD)` is learned under BOTH the D-not-boundary
and D-is-boundary views, matching what bt backtracking sees at generation.

**Mechanism.** Use the streaming-bt committed index (`online_committed_patch_idx`, commit_margin=2)
as the training `repeat_idx` in `up()` (the existing `committed_patch_idx` branch) instead of the
as-if-complete cumsum. Per-position this gives:
- `P(F|ABCD)` trained with D NOT a boundary (open patch) — the state generation is in when it
  proposes F (D's boundary undecided until F reveals the dead-end).
- `P(G|ABCDF)` trained with D a boundary (ABCD committed) — the post-backtrack state.
⇒ train/gen gap = 0; the mismatch that drops committed-eval to 0.5602 disappears → should recover
toward AU-Net 2 (0.6570) while staying leak-free.

**Infra already exists:** `bpe_online_committed_view` flag, `online_committed_patch_idx` ("3rd data
view → repeat_idx, train/gen gap=0"), `up()` committed_patch_idx branch. Work = wire the committed
index into the train data pipeline + flip the flag + train.

**Dual-view extension (optional):** also add F's loss under the D-is-boundary view (sum both), so the
model can DRIVE backtracking: compare `P(F|D-not-boundary)` vs `P(F|D-boundary)` to decide commit
(model-scored boundaries instead of trie longest-match), leak-free.

**Scope before launch** (user: "decide after I scope it"): 300M on snu55's 4 A5000s first.
Need: token budget, steps, GPU-hours/ETA, data pipeline change for the committed index. B200 busy;
snu55 GPUs intermittently grabbed by root's vLLM (saw all 4 occupied 13:45–14:00).

- **★ Statistics-based before_root backtracking in training (user idea, 2026-06-18).** During
  before_root backtracking the boundary placement (before_root `e-1` vs root `e`) needs the future
  dead-end byte. Precompute from the training corpus, per trie leaf node (`is_leaf`), the empirical
  `P(go to root | reached this leaf)` — i.e. how often the longest match extends past this leaf vs
  stops here. Then at train time replace the lookahead decision with this *causal* statistic
  (expected/soft boundary, or threshold on P), so the boundary depends only on the trie path so far
  (past) + a precomputed table, NOT the next byte. Removes the leak at its source while keeping the
  segmentation signal → candidate to recover the AU-Net 2 gap leak-free. Then re-train (300M first,
  cheap) and eval ARC-Easy with the same committed scoring.
- **Committed-view training** (`bpe_online_committed_view=True`): train with the gap=0 repeat_idx so
  the model never sees the leak; eval committed → matched. Direct A/B vs the statistics idea.
- **Word-boundary answer patches** — segment the answer on whitespace (causal, deterministic, no
  trie lookahead); mirrors AU-Net 2's word-patch scheme that wins ARC-Easy.
- Length/char normalization beyond `acc_norm`; ensemble of leak-free segmentations; atomic
  (1-patch answer) as the 0-lookahead floor for the ladder.

## Backlog (deferred until goal closed)

- #10 MBPP + HumanEval @ 0/5/10-shot (code gen + execution; expect near-zero pass@1 for base models).
- #11 CUTE noisy eval (BLT arXiv 2412.09871 Table 3 / arXiv 2603.03583 Table 2).
- #12 300M training + eval plan (`status_300M.md`).

## ★ RESULT (2026-06-19): leak-free+causal HellaSwag 32 GOAL MET (100M)

Goal: get a leak-free+causal byte model to HellaSwag >= 32 at 100M (v1_committed was 28.9).
- Inference tricks INSUFFICIENT: matched committed eval 28.9; 4-model leak-free+causal ensemble
  (v1_committed/v4_root_greedy/v5_distilled/v6_prefix_free) capped at 31.1.
- Best single leak-free+causal model: v6_prefix_free (first-match, no maximal-munch lookahead,
  never revised; committed_view=false) = 31.0 at ratio-20.
- TRAINING-SCHEME lever (longer training): resumed v6_prefix_free ratio-20 -> ratio-40 (step 13376).
  **HellaSwag acc_norm = 0.3270 (32.70%) >= 32 -> GOAL MET.** (+1.7 per token-budget doubling.)
- Conclusion: the 100M ~31 plateau was a compute limit, not fundamental; a leak-free+causal byte
  model (prefix-free first-match) clears 32 with ratio-40 training. (1.3B greedy_root, also
  leak-free+causal via root placement, hits 44.1 at just 30% steps -> scales well past 32.)

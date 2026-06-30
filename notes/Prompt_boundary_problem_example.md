# Prompt Boundary Problem (PBP) — evaluation design + worked examples

Reference: Vieira et al., *Sampling from Your Language Model One Byte at a Time*,
[arXiv:2506.14123](https://arxiv.org/abs/2506.14123).

## What the PBP is

A BPE prompt that ends **mid-token** forces a tokenization boundary the model never
saw in training, distorting the next-token distribution. The folklore symptom is
"don't end your prompt with a space" — the trailing space normally gets glued into
the *next* token at generation time, so as a standalone prompt-final token it is
off-distribution. Key facts from the paper:

- The problem **persists in code and in languages like Chinese** even where the
  English trailing-space heuristic would hide it (BPE token boundaries rarely line up
  with word/syntactic boundaries there).
- The diagnostic is **byte-level evaluation**: naively predicting the next character by
  directly applying the tokenizer to an arbitrary string prefix gives poor performance.
- Mitigations are **token-healing** (back up one token + constrain the next token to
  start with the unmatched suffix) and **marginalization over tokenizations** — both
  inference-time patches that byte/char models do not need.

## Why this is a winnable benchmark for AU-Net

Our `evaluation_results.md` story is "byte models are competitive but not clearly
ahead." PBP is different: it is a place where **both byte models should win by
construction**, turning a tie into an architectural advantage.

- **BPEByte** and **AU-Net 2** consume raw bytes and pool causally. A mid-word prompt
  boundary just leaves the current patch open with all its prefix bytes visible to the
  byte-level stages — the next byte is always well-defined. No forced "complete token,"
  so **no PBP**.
- **Llama (BPE subword)** is the victim: an arbitrary character prefix gets greedily
  re-tokenized, and a mid-token cut shifts the conditioning distribution.

Predicted result: byte models at Δ ≈ 0, Llama degrading — a clean separation.

> Note: PBP is a **prompt-side** boundary effect, orthogonal to the existing
> `*_answer_loglikelihood` **answer-side** leakage controls in `apps/aunet/eval.py`.
> Do not conflate the two.

---

## Experiment A — Byte-level BPC invariance at arbitrary cut points (primary)

**The trick:** score the *same underlying string* but move the cut so the prompt ends
mid-token. A faithful model should assign the full string nearly the same total
log-prob regardless of where you split prompt/continuation. BPE breaks this; bytes
don't. (`|` marks schematic llama3-style BPE token boundaries.)

**Canonical trailing-space case.** Underlying string: `"...bought some milk"`

| Variant | prompt (conditioning) | continuation (scored) |
|--------|------------------------|------------------------|
| aligned | `...bought some` | `▁milk` |
| mid-token (trailing space) | `...bought some▁` | `milk` |

- **Llama:** `▁milk` is normally **one token**. In the mid-token variant the space is
  already consumed, so the model must emit `milk` (no leading space) as a token-final
  completion — off-distribution. `logP("milk" | "...some ") ≪ logP(" milk" | "...some")`.
- **Byte models:** identical byte stream either way → `logP` essentially equal.

**Mid-word case** (no whitespace to hide behind). Underlying string: `"unbelievable"`,
BPE = `un|bel|iev|able`

| Variant | prompt | continuation |
|--------|--------|--------------|
| aligned (token edge) | `unbel` (`un|bel`) | `ievable` |
| mid-token | `unbe` (`un|be`) | `lievable` |

`be` as a prompt-final token never co-occurs in training with the `bel|iev|able` path
→ distorted distribution for Llama; flat for byte models.

**Metric:** `ΔBPC = BPC(mid-token cut) − BPC(aligned cut)` over many (string, cut) pairs.

Predicted (illustrative):

| Model | ΔBPC | reading |
|-------|------|---------|
| AU-Net 2 | ~0.00 | invariant — the headline |
| BPEByte | ~0.00 | invariant |
| Llama naive | +0.3–0.8 | PBP penalty |
| Llama + token-healing | +0.05–0.2 | mitigated, not zero |

**Harness mapping:** one `loglikelihood` request per variant — `(prompt, continuation)`
exactly as `EvalHarnessLM.loglikelihood` already consumes. No new scoring code.

---

## Experiment B — Downstream MCQ with adversarial prompt boundary (corroboration)

Reuse ARC-Easy. Standard lm-eval scores `P(" Carbon dioxide" | context)` with a
**leading-space** continuation.

```
Question: Which gas do plants absorb from the atmosphere?
Answer:
```

| Variant | prompt ends with | continuation scored |
|--------|-------------------|----------------------|
| canonical | `Answer:` | `▁Carbon▁dioxide` |
| +trailing space | `Answer:▁` | `Carbon▁dioxide` |
| mid-word | `Answer:▁Carb` | `on▁dioxide` |

- **Llama:** `▁Carbon` is one token; after a standalone space token, `Carbon` is a
  different (rarer) token-final unit. The *relative* ranking across the four options
  shifts → some flip → accuracy drops.
- **Byte models:** the scored byte string is the same; per-option rankings unchanged →
  accuracy flat.

**Metric:** `Δacc = acc(canonical) − acc(perturbed)`.

Predicted (illustrative):

| Model | canonical | +trailing space | mid-word | Δ |
|-------|-----------|------------------|----------|---|
| AU-Net 2 | 0.657 | 0.656 | 0.655 | ~0.00 |
| BPEByte | 0.641 | 0.640 | 0.639 | ~0.00 |
| Llama 1.8B | 0.654 | 0.61–0.63 | 0.58–0.62 | −0.03 to −0.07 |

**Harness mapping:** clone `apps/aunet/eval_typo_ds.py` → `eval_pbp.py`; the
`process_docs` transform appends a trailing space / truncates the gold's first word into
the prompt instead of applying a typo. Same sentinel + `expand_pbp_tasks` +
`summarize_pbp` plumbing.

---

## Experiment C — Multilingual + code (where PBP survives outside English)

**Chinese** — no spaces, so the "don't end with a space" heuristic is meaningless, yet
PBP persists. String: `中国的首都是北京` ("The capital of China is Beijing"), token
`北京` = Beijing.

| Variant | prompt | continuation |
|--------|--------|--------------|
| aligned | `中国的首都是` | `北京` |
| mid-token | `中国的首都是北` | `京` |

`北` alone ≠ the prefix of token `北京`; Llama must emit `京` as a token-final completion
→ distorted. Byte model: fine. **The point:** in Chinese essentially *every* cut is a
potential mid-token cut, so the aligned/mid-token gap shows up at almost any prompt length.

**Code** — partial identifiers, the classic case.

| Variant | prompt | continuation |
|--------|--------|--------------|
| aligned | `for i in ` | `range(10):` |
| mid-token | `for i in ra` | `nge(10):` |

`range` is one token; `ra` forces `ra`-then-`nge`, off-distribution for Llama. Byte
models complete the partial identifier natively.

**Metric:** same `ΔBPC`, stratified by **English / Chinese / code**.

Predicted (illustrative):

| Model | EN ΔBPC | ZH ΔBPC | code ΔBPC |
|-------|---------|---------|-----------|
| AU-Net 2 / BPEByte | ~0.00 | ~0.00 | ~0.00 |
| Llama naive | +0.3 | **+0.6–1.0** | **+0.5–0.9** |

The widened ZH/code gap is the paper's core argument and the strongest separation figure.

**Harness mapping:** same loglikelihood path as A; reuse `eval_tasks/flores/` data for ZH
and a held-out code slice of the training mix for the code stratum.

---

## Recommended order

1. **A** on English first (zero new scoring code — a cut-point task emitting
   `(prefix, continuation)` pairs).
2. Extend the **same** task to **C**'s ZH/code strata.
3. **B** as downstream-task corroboration once A confirms the gap.

## Implementation status

- [x] **A — cut-point BPC** (`lingua/apps/aunet/eval_pbp.py`). Sentinel `pbp`; scored directly
  through `EvalHarnessLM.loglikelihood` via the 3-request word-NLL decomposition
  (`P(boundary+word) − P(boundary)` vs `P(word | prefix+boundary)`). Wired into both
  `apps/aunet/eval.py` and `apps/main/eval.py`. Byte models give `delta_bpc == 0` by
  construction (verified with a mock byte LM); the BPE slice at `len(encode(prompt))`
  misaligns on a mid-token cut, surfacing the gap for Llama with no scoring changes.
- [x] **C — ZH/code strata** ship inside `eval_pbp.py` as the `code_space` and `zh_char`
  strata (`PBP_ITEMS`). Same machinery; no extra wiring.
- [x] **B — MCQ boundary** (`lingua/apps/aunet/eval_pbp_mc.py`). Sentinel `pbp_mc`. Trailing-
  space variant only, metric `acc` (argmax logprob). Design note: the perturbation must be
  identical across an item's options so it is a constant per-item logprob offset → cannot flip
  a byte model's argmax (`delta_acc == 0`, verified); a "commit each option's first char"
  variant and `acc_norm` were dropped as measurement artifacts (see module docstring).

### How to run

One-shot per checkpoint (autodetects byte vs Llama, runs `pbp`+`pbp_mc`):
`bash lingua/run_eval_pbp.sh <ckpt_folder> [items_jsonl] [ngpu]` — set `PBP_GEN=1` to
generate corpus-scale items first. Run it on the byte ckpt AND the Llama ckpt, then
summarize the comparison: `python runs/pbp_table.py <byte_run_dir> <llama_run_dir>`
(prints ΔBPC per stratum + the MCQ Δacc; byte rows ≈0, Llama rows positive).
Manual invocation:

```bash
cd lingua
# Experiment A + C (byte model: AU-Net 2 or BPEByte)
python -m apps.aunet.eval config=apps/aunet/configs/eval_pbp_b200.yaml ckpt_dir=<CKPT> dump_dir=<OUT>
# Experiment A + C (Llama baseline)
python -m apps.main.eval  config=apps/main/configs/eval_pbp_llama_1.8B_b200.yaml ckpt_dir=<CKPT> dump_dir=<OUT>
# Experiment B: set harness.tasks to [pbp_mc] (or add it alongside pbp) in the config.
```

Results land in `<OUT>/results.json` under keys `pbp_en_space` / `pbp_code_space` /
`pbp_zh_char` / `pbp_overall` (each with `bpc_aligned`, `bpc_misaligned`, `delta_bpc`, `n`)
and `pbp_mc_canonical` / `pbp_mc_space` (`acc`, `delta_acc`).

### Corpus-scale cut-point generator (`lingua/apps/aunet/data/pbp_gen.py`)

The curated `PBP_ITEMS` are a controlled first signal. For paper-scale ΔBPC, generate items
from real text: the generator uses the **llama3 BPE tokenizer** (the victim model's vocab) and
`get_token_offsets` to find character positions that fall **strictly inside a token**, then emits
`{stratum, prefix, boundary, word}` where committing `boundary` lands the cut mid-token.
`commit_chars=1` reproduces the canonical cases automatically — English token `▁milk` → commit
the space (trailing-space PBP); Chinese token `北京` → commit `北`.

```bash
cd lingua
TOK=tokenizer/llama3/tokenizer.model
# build a multi-stratum item file (append per stratum)
python -m apps.aunet.data.pbp_gen --tokenizer $TOK --stratum en_space   \
    --data data/dclm_baseline_1.0_2shards_shuffled --max-items 2000 --out runs/pbp_items.jsonl
python -m apps.aunet.data.pbp_gen --tokenizer $TOK --stratum code_space \
    --data <code_corpus_dir>  --max-items 1000 --out runs/pbp_items.jsonl
python -m apps.aunet.data.pbp_gen --tokenizer $TOK --stratum zh_char    \
    --data <flores_zh_or_zh_jsonl> --max-items 1000 --out runs/pbp_items.jsonl
# then run the eval against the generated set (both model families):
python -m apps.aunet.eval config=apps/aunet/configs/eval_pbp_b200.yaml \
    ckpt_dir=<CKPT> dump_dir=<OUT> pbp_items_path=runs/pbp_items.jsonl
python -m apps.main.eval  config=apps/main/configs/eval_pbp_llama_1.8B_b200.yaml \
    ckpt_dir=<CKPT> dump_dir=<OUT> pbp_items_path=runs/pbp_items.jsonl
```

`pbp_items_path` (EvalArgs, both entry points) overrides the curated set; unset → curated.
The generator reconstructs the source exactly (`prefix+boundary+word == source[:len]`), guarantees
the cut is strictly inside a token, dedupes, caps per-doc to diversify, and is deterministic by
`--seed`. Knobs: `--commit-chars` (mid-token depth), `--window-chars` (word length),
`--min-prefix-chars`, `--per-doc`, `--max-bytes`. Validated on mock tokenizers (EN trailing-space
+ ZH token-split); the only external dependency is the real llama3 `tokenizer.model`.

**B (MCQ)** stays curated — auto-generating valid MCQ distractors is a separate problem; enlarge
`PBP_MC_ITEMS` (e.g. seed from ARC-Easy) for more power.

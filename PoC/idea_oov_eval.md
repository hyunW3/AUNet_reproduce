# Idea: Out-of-Vocabulary (OOV) eval — real-vs-nonce copy/spell

Status: design note (not yet implemented). Target harness: `apps/aunet/eval.py` /
`apps/main/eval.py`, same machinery as `eval_pbp.py` (sentinel + direct `loglikelihood`).

## What "OOV" actually means for these models

Neither model class has a *true* OOV in the classic fixed-vocabulary sense — a BPE
tokenizer always falls back to byte/char tokens, so every string is representable. The
"OOV problem" decomposes into **two independent axes**:

1. **Fragmentation / boundary structure** — a rare/novel word shatters into many
   sub-word/byte units (high "fertility"), eating context budget and splitting the word
   across morphology-blind boundaries.
2. **Character access inside a unit** — can the model see the actual characters of a word
   it was never trained on, or is the unit an opaque (often undertrained) embedding ID?

### Where our three models sit

| model | fragmentation / boundary structure | character access inside a unit |
|---|---|---|
| subword (Llama) | BPE (fragments) | ✗ opaque token IDs |
| **BPEByte** | **BPE (fragments — identical to subword)** | **✓ reads raw bytes** |
| AU-Net (regex split) | whitespace (coarser) | ✓ reads raw bytes |

**Key constraint that shapes this eval:** BPEByte derives its pooling boundaries from the
**BPE tokenizer**, not AU-Net's regex/whitespace split. So BPEByte's segmentation
**fertility is identical to the subword baseline's**. That kills "BPE fertility" as a
covariate that distinguishes BPEByte from subword — both fragment a rare word the same way.

Therefore the discriminating axis between BPEByte and subword is **axis 2 (character
access)**, not axis 1 (fragmentation). A real-vs-nonce **copy/spell** task probes exactly
axis 2: BPEByte should win over subword *despite identical fertility*, because it pools the
real bytes of a never-seen word while the subword model only has a bag of opaque token IDs.

## Why nonce words are the cleanest probe

Domain jargon is a natural OOV source (rare in pretraining → fragments), but it **confounds
tokenization fragility with knowledge** — a model may fail medical jargon because it doesn't
know the medicine, not because of tokenization. Nonce / pseudowords remove that confound:

- **zero memorization possible** (the string has never been seen by any model),
- **fragmentation held ~constant** (match nonce length/shape, ideally BPE fertility, to a
  real control word),
- so the only remaining difference is **novelty + character access** — precisely what we
  want to isolate.

## Design — real-vs-nonce matched pairs

Build matched pairs: a real word and a phonotactically-plausible pseudoword of the same
length/shape (ideally same BPE fertility, so fragmentation is held constant and only novelty
differs).

| real | nonce (matched) |
|---|---|
| balloon | ballimp |
| rabbit | tovrick |
| elephant | brovanth |
| garden | farnick |
| trumpet | drumnack |

### Task 1 — Spell (letter-separation), loglikelihood-scored (CUTE-style)

```
prompt:  Spell the following word with a hyphen between each letter.
         Word: ballimp
         Spelling:
gold:    " b-a-l-l-i-m-p"
distractors:  " b-a-l-i-m-p"    (dropped letter)
              " b-a-l-l-m-i-p"  (swapped pair)
              " b-a-l-l-i-n-p"  (substituted letter)
```
`output_type: multiple_choice`, argmax over {gold, distractors}, metric `acc` + `acc_bytes`.
Real "balloon" → gold `" b-a-l-l-o-o-n"` with parallel distractors.

### Task 2 — Reverse (char manipulation, CUTE-style)

```
prompt:  Write the following word backwards.
         Word: tovrick
         Backwards:
gold:    " kcirvot"
distractors: " kcrivot" / " kcirovt" / " kcirvto"   (adjacent-swap corruptions)
```
A model that can't see characters can't reverse a word it never memorized → subword collapses
on the nonce column; byte / BPEByte stay flat.

### Task 3 — Copy-through-context (the actual "copy" test, induction)

Strongest because it needs *only* verbatim reproduction, no spelling notion — the model must
carry the unseen string through the context bottleneck and emit it byte-exact:

```
prompt:  A "brovanth" is a small wooden tool used by sailors.
         Question: What is the name of the small wooden tool?
         Answer:
gold:    " brovanth"
distractors: " broventh" / " bravanth" / " brovanthe"
```
Real control swaps in an existing rare word (`a "trowel" is a small wooden tool…`).

## Metric — an OOV delta, not an absolute

For each task, the headline is the **real→nonce drop** per model:

```
oov_delta = acc(real items) − acc(nonce items)
```

- Robust model → `oov_delta ≈ 0`.
- **Prediction:** subword has a large positive `oov_delta` (fine on memorized real words,
  collapses on nonce); **both** byte models stay near zero.
- The **BPEByte-vs-subword gap is the headline that survives the fertility objection**:
  both fragment identically, yet only BPEByte holds up — proving the win is character access,
  not segmentation granularity.

Report `acc_bytes` alongside `acc` for byte-fairness, exactly as in the other suites
(`mmlu_text`, typo suite, PBP).

## Build notes

- **Nonce generation:** substitute letters in real words while preserving bigram phonotactics
  and length; **verify each nonce is not accidentally a real word** (dictionary check).
- **Matched fertility:** prefer nonce words whose BPE tokenization length matches the real
  control, so axis-1 fragmentation is held constant and the residual is axis-2 only.
- **Scoring path:** single short words → generative exact-match is also viable (our
  `generate_until` only times out on long code-gen). Run both MC-loglikelihood and
  exact-match and cross-check.
- **Integration:** mirror `eval_pbp.py` — an `oov` sentinel popped from `harness.tasks`,
  `run_oov` calling `wrap.loglikelihood` directly, strata = {spell, reverse, copy}, equal-
  weight (macro) overall like the PBP fix in `eval_pbp.py` / `eval_pbp_mc.py`.

## Out of scope / cautions

- Jargon downstream QA (confounds knowledge — only use jargon for copy/spell/BPB where
  knowledge isn't required).
- This eval probes **axis 2 (character access)** by design. To also probe **axis 1
  (fragmentation)** and separate pure-byte AU-Net from both BPE-based models, a separate
  BPB-vs-fertility corpus probe would be needed — but note fertility does NOT separate
  BPEByte from subword, so that probe answers a different question.

## Result

# 1.3B model results — full benchmark dump

Models: **Llama 1.8B** (BPE, `llama_1.8B_paper` @60k) · AU-Net2 (byte, word patches) · BPEByte root_greedy. All @ final ckpt, 0-shot, full dataset. `—` = not run, `·` = metric absent. _(BPEByte online-bt is a secondary variant; its results are in [Supplements — BPEByte online-bt](#supplements--bpebyte-online-bt) at the end.)_
Headline table: `consolidated_1.3B_table.md`. BLT-format: `cute_table3_format.md`.
**Scoring mode** is noted per section: _likelihood_ = option-loglikelihood argmax (acc/acc_norm) or BPC; _generation_ = generate_until + exact_match.

> **Note.** The `root_greedy` column throughout this doc is the **new-code** run (`bpebyte_br_greedy_root_1.3B` @180k, training commit `0fbc3ee` + new eval). Its full comparison against the old training code and the old eval/2nd-mask is the 2×2 below.

## 2×2: training-code × eval-fix (root_greedy 1.3B, new code)

All four cells are the **same `root_greedy` 1.3B model family** @180k, byte-normalized eval, read from the canonical full-dataset eval dirs. The 2×2 isolates two independent code changes:

- **Training code** (`new/*` vs `old/*`): NEW = non-sliding `max(8192,len)` streaming-parser window (commit `0fbc3ee`); OLD = original 4096-sliding window (`_oldseg_` run).
- **Eval fix / 2nd-position mask** (`*/new` vs `*/old`): NEW peels leading special tokens → BOS@0 mask=1, trie on real bytes; OLD (`OLD_EVAL_NO_BOS_PEEL=1`) feeds BOS to the trie → spurious 2nd-position boundary, BOS@0 unset.

Columns are **train/eval** code: `new/new`, `new/old`, `old/new`, `old/old`. `Δtrain = new/new − old/new` (eval held new); `Δevalfix = new/new − new/old` (train held new). All full datasets.

| benchmark (0-shot) | new/new | new/old | old/new | old/old | Δtrain | Δevalfix |
| --- | --- | --- | --- | --- | --- | --- |
| HellaSwag acc_norm | 62.47 | 62.77 | 62.82 | 63.05 | -0.35 | -0.30 |
| ARC-Easy acc_norm | 66.84 | 66.08 | 65.57 | 65.99 | +1.26 | +0.76 |
| ARC-Challenge acc_norm | 37.54 | 36.77 | 36.60 | 36.52 | +0.94 | +0.77 |
| BoolQ acc | 62.05 | 61.90 | 60.21 | 60.00 | +1.83 | +0.15 |
| PIQA acc_norm | 74.32 | 74.76 | 73.78 | 73.61 | +0.54 | -0.44 |
| WinoGrande acc | 61.09 | 61.56 | 59.75 | 59.98 | +1.34 | -0.47 |
| MMLU acc | 25.49 | 25.47 | 24.51 | 24.49 | +0.98 | +0.01 |
| MMLU-text acc_norm | 34.14 | 34.51 | 33.93 | 34.08 | +0.21 | -0.37 |

**5-bench mean** (HS-norm, ARC-E-norm, BoolQ, PIQA-norm, Wino), full sets:
| shot | new/new | new/old | old/new | old/old | Δtrain | Δevalfix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-shot | 65.35 | 65.41 | 64.43 | 64.53 | +0.93 | -0.06 |
| 3-shot | 66.66 | 66.92 | 66.87 | 66.71 | -0.22 | -0.26 |
| 5-shot | 67.03 | 66.82 | 67.88 | 67.83 | -0.85 | +0.20 |

**Robustness aggregates** (limit=2000 by design):
| metric | new/new | new/old | old/new | old/old | Δtrain | Δevalfix |
| --- | --- | --- | --- | --- | --- | --- |
| HellaSwag-typo avg (acc_norm) | 51.88 | 52.07 | 51.56 | 52.06 | +0.32 | -0.19 |
| HellaSwag-noise avg (acc_norm) | 42.34 | 42.56 | 41.88 | 42.16 | +0.46 | -0.22 |

**Mean |Δ| over the 8 headline 0-shot benchmarks:** training-code = **0.93** pts, eval-fix = **0.41** pts.

**Takeaway.** The eval-fix/2nd-position-mask has only a **small, direction-inconsistent effect for root_greedy** (mean |Δevalfix| ≈ 0.41 pt; per-benchmark shifts ±0.3–0.8, both signs, all within single-run stderr; no systematic winner — `new/new`≈`new/old` and `old/new`≈`old/old`). In root placement the BOS-peel rarely changes the realized segmentation, but the front-of-sequence mask difference does perturb a few patch boundaries, so it is not bit-identical. The more consistent difference is the **training-code window fix** (mean |Δtrain| ≈ 0.93 pt): new code is modestly better on 0-shot knowledge/reasoning (BoolQ +1.8, MMLU +1.0, ARC-C/E ~+1) and most few-shot, while old code edges 5-shot BoolQ/WinoGrande. Net 5-bench means stay within ~1 pt across all four cells. (All single runs, no seeds — treat sub-1-pt gaps as noise.)

## 2×2 train/eval breakdown by shot

Columns are **train/eval** code (all = `root_greedy` 1.3B @180k, byte-normalized, full datasets, canonical eval dirs): `new/new` = new-train/new-eval · `new/old` = new-train/OLD-eval · `old/new` = OLD-train/new-eval · `old/old` = OLD-train/OLD-eval. Comparing `new/new`↔`new/old` and `old/new`↔`old/old` isolates the **eval-fix/2nd-mask**; `new/new`↔`old/new` and `new/old`↔`old/old` isolates the **training-code window**. Metric per row in the label (acc_norm, or acc for BoolQ/WinoGrande/MMLU-letter). MMLU-letter only at 0-shot. Single runs, no seeds.

### 0-shot

**Averages**

| aggregate | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| 5-bench mean (HS, ARC-E, BoolQ, PIQA, Wino) | 65.35 | 65.41 | 64.43 | 64.53 |
| all-8 mean (5-bench + ARC-C+ MMLU-letter + MMLU-text) | 52.99 | 52.98 | 52.15 | 52.22 |

**Individual benchmarks**

| benchmark | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| HellaSwag (acc_norm) | 62.47 | 62.77 | 62.82 | 63.05 |
| ARC-Easy (acc_norm) | 66.84 | 66.08 | 65.57 | 65.99 |
| ARC-Challenge (acc_norm) | 37.54 | 36.77 | 36.60 | 36.52 |
| BoolQ (acc) | 62.05 | 61.90 | 60.21 | 60.00 |
| PIQA (acc_norm) | 74.32 | 74.76 | 73.78 | 73.61 |
| WinoGrande (acc) | 61.09 | 61.56 | 59.75 | 59.98 |
| MMLU-letter (acc) | 25.49 | 25.47 | 24.51 | 24.49 |
| MMLU-text (acc_norm) | 34.14 | 34.51 | 33.93 | 34.08 |

### 3-shot

**Averages**

| aggregate | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| 5-bench mean (HS, ARC-E, BoolQ, PIQA, Wino) | 66.66 | 66.92 | 66.87 | 66.71 |
| all-7 mean (5-bench + ARC-C + MMLU-text) | 58.51 | 58.67 | 58.43 | 58.30 |

**Individual benchmarks**

| benchmark | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| HellaSwag (acc_norm) | 63.51 | 63.54 | 63.16 | 63.15 |
| ARC-Easy (acc_norm) | 71.38 | 71.38 | 71.55 | 71.38 |
| ARC-Challenge (acc_norm) | 41.30 | 41.04 | 39.33 | 39.33 |
| BoolQ (acc) | 62.63 | 63.15 | 62.32 | 62.63 |
| PIQA (acc_norm) | 74.43 | 74.16 | 74.43 | 73.88 |
| WinoGrande (acc) | 61.33 | 62.35 | 62.90 | 62.51 |
| MMLU-text (acc_norm) | 35.00 | 35.06 | 35.31 | 35.19 |

### 5-shot

**Averages**

| aggregate | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| 5-bench mean (HS, ARC-E, BoolQ, PIQA, Wino) | 67.03 | 66.82 | 67.88 | 67.83 |
| all-7 mean (5-bench + ARC-C + MMLU-text) | 58.82 | 58.64 | 59.22 | 59.23 |

**Individual benchmarks**

| benchmark | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| HellaSwag (acc_norm) | 63.76 | 63.61 | 63.65 | 63.71 |
| ARC-Easy (acc_norm) | 71.76 | 71.42 | 72.14 | 71.84 |
| ARC-Challenge (acc_norm) | 41.89 | 41.72 | 40.70 | 40.96 |
| BoolQ (acc) | 62.81 | 62.78 | 65.41 | 65.60 |
| PIQA (acc_norm) | 74.21 | 74.10 | 74.32 | 74.05 |
| WinoGrande (acc) | 62.59 | 62.19 | 63.85 | 63.93 |
| MMLU-text (acc_norm) | 34.71 | 34.61 | 34.50 | 34.55 |

## Core reasoning (likelihood — option loglikelihood → acc/acc_norm)

| metric | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| HellaSwag acc | 47.2 | 49.1 | 48.3 |
| HellaSwag acc_norm | 62.2 | 62.6 | 63.0 |
| ARC-Easy acc | 69.1 | 69.1 | 69.0 |
| ARC-Easy acc_norm | 65.4 | 65.7 | 65.9 |
| ARC-Challenge acc | 32.1 | 34.6 | 35.6 |
| ARC-Challenge acc_norm | 35.3 | 36.5 | 36.6 |
| BoolQ acc | 63.5 | 61.1 | 60.2 |
| PIQA acc | 74.9 | 73.0 | 73.7 |
| PIQA acc_norm | 75.3 | 74.2 | 73.4 |
| WinoGrande acc | 61.6 | 61.5 | 59.3 |
| MMLU acc | 24.7 | 26.3 | 24.5 |

## Core reasoning — 3-shot (likelihood — option loglikelihood → acc/acc_norm)

3-shot, fair & truncation-free, full sets, fixed root_greedy code. `num_fewshot=3`;
ARC-Challenge added 2026-06-24 (acc/acc_norm); letter-MMLU not re-run at few-shot (see MMLU-text
0/3/5-shot below). Single run, no seeds.

| metric | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| HellaSwag acc | 46.8 | 48.9 | 48.1 |
| HellaSwag acc_norm | 62.7 | 63.1 | 63.1 |
| ARC-Easy acc | 70.2 | 71.0 | 70.9 |
| ARC-Easy acc_norm | 70.0 | 72.6 | 71.7 |
| ARC-Challenge acc | 35.7 | 36.1 | 35.5 |
| ARC-Challenge acc_norm | 38.2 | 37.9 | 39.4 |
| BoolQ acc | 63.5 | 59.3 | 62.4 |
| PIQA acc | 75.2 | 73.0 | 73.2 |
| PIQA acc_norm | 75.1 | 74.0 | 74.3 |
| WinoGrande acc | 62.1 | 60.9 | 63.1 |
| MMLU acc | — | — | — |

5-bench mean: **root_greedy 66.9 > Llama 66.7 > AU-Net2 66.0** — root_greedy already overtakes the
1.8B baseline by 3-shot. (BoolQ note: AU-Net2 stays depressed at 59.3, confirming the few-shot BoolQ
weakness seen at 5-shot is a real trend, not a one-off.)

## Core reasoning — 5-shot (likelihood — option loglikelihood → acc/acc_norm)

5-shot, fair & truncation-free (0% for all three, verified), full sets, fixed root_greedy code
(BOS/A.1/window). `num_fewshot=5`; ARC-Challenge added 2026-06-24 (acc/acc_norm); letter-MMLU not
re-run at few-shot (see MMLU-text 0/3/5-shot below). Single run, no seeds.

| metric | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| HellaSwag acc | 47.2 | 49.0 | 48.5 |
| HellaSwag acc_norm | 62.9 | 63.7 | 63.7 |
| ARC-Easy acc | 70.9 | 70.8 | 70.5 |
| ARC-Easy acc_norm | 70.3 | 72.4 | 71.9 |
| ARC-Challenge acc | 34.6 | 36.9 | 35.8 |
| ARC-Challenge acc_norm | 38.5 | 39.5 | 39.9 |
| BoolQ acc | 65.6 | 59.1 | 65.5 |
| PIQA acc | 74.2 | 74.2 | 73.7 |
| PIQA acc_norm | 75.6 | 74.1 | 74.4 |
| WinoGrande acc | 62.6 | 63.6 | 64.1 |
| MMLU acc | — | — | — |

5-bench acc_norm/acc mean (HellaSwag·ARC-Easy·BoolQ·PIQA·WinoGrande): **root_greedy 67.9 > Llama 67.4
> AU-Net2 66.6**. At 0-shot the order is reversed (Llama 65.6 > AU-Net2 65.0 > root_greedy 64.4), so
**root_greedy overtakes the 1.8B subword baseline at 5-shot** with the strongest few-shot scaling.
AU-Net2 is held back by a **depressed BoolQ (59.1)** — confirmed a real trend, not a one-off: AU-Net2
BoolQ *degrades* with shots (61.1 → 59.3 → 59.1 at 0/3/5-shot), unlike root_greedy (60.2 → 62.4 → 65.5).

## MMLU — HellaSwag-style (cloze: score the full answer TEXT, not the A/B/C/D letter)

New task `mmlu_text` (`lingua/eval_tasks/mmlu_text/`, group over all 57 subjects; driver
`run_eval_mmlu_text.sh`). Pure cloze prompt `"{{question}}\nAnswer:"` with **no options listed**;
each of the 4 full answer strings is scored as the continuation, argmax of length-normalized
loglikelihood — exactly the HellaSwag protocol. This is a deliberate departure from standard
letter-MMLU (which scores `P(letter)` and is kept as `mmlu`). `acc_bytes` ≈ `acc_norm` (answers
are near-pure ASCII). Full sets (57 subjects, size-weighted), single run, se ≈ 0.004.

acc_norm (headline, per-codepoint) across 0/3/5-shot:

| MMLU-text acc_norm | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| 0-shot | 33.70 | 33.66 | 33.84 |
| 3-shot | 35.27 | 35.07 | 35.24 |
| 5-shot | 35.16 | 34.90 | 34.43 |

acc_bytes (per-UTF-8-byte; ≈ acc_norm since answers are near-pure ASCII):

| MMLU-text acc_bytes | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| 0-shot | 33.71 | 33.68 | 33.87 |
| 3-shot | 35.28 | 35.09 | 35.30 |
| 5-shot | 35.19 | 34.89 | 34.49 |

Reference — stock letter-MMLU `acc` (scores `P(A/B/C/D)`), 5-shot: Llama **24.7**, AU-Net2 **26.3**,
root_greedy **24.5**.

**The format change is the headline.** Stock letter-MMLU pins all three models at the **chance
floor** (~24–26). Scoring the real answer text instead lifts every model to **~34–35** — a
**+9–11 pt** swing of knowledge that the "A/B/C/D" token format completely buries. At 1.3B these
models *do* hold MMLU content; they just can't express it through a single letter.

**Between models it is a statistical tie at every shot count** (all spreads ≤ ~1.3 se):
- 0-shot: all three within 0.2 pt (~33.7).
- 3-shot: root_greedy 35.24 ≈ Llama 35.27 ≈ AU-Net2 35.07 — byte ties the 1.8B subword baseline.
- 5-shot: Llama 35.16 ≥ AU-Net2 34.90 ≥ root_greedy 34.43 — a ≤0.7 pt edge to subword
  (root_greedy↔Llama ≈ 1.3 se, not significant).

**Few-shot saturates by 3-shot, then plateaus.** All three peak at 3-shot and dip slightly at
5-shot (Llama 35.27→35.16, AU-Net2 35.07→34.90, root_greedy 35.24→34.43). The byte model's 3→5
dip is the largest (−0.8 pt, ~1.4 se) but stays within noise; it is consistent with the byte
model gaining less from very long few-shot prompts (cf. the C.5 truncation/long-context note).
Net: the byte model is fully competitive on knowledge-MMLU once the letter bottleneck is removed,
and the letter-vs-text gap (~10 pt) dwarfs all between-model gaps (~1 pt).

### Examples

*Scoring: the prompt is fixed; each choice is appended and scored by total log-prob; the argmax choice is the prediction (acc). `acc_norm` divides by byte/char length.*

### HellaSwag

**Prompt (doc_to_text):**

```
Roof shingle removal: A man is sitting on a roof. He
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] is using wrap to wrap a pair of skis.
[1] is ripping level tiles off.
[2] is holding a rubik's cube.
[3] starts pulling up roofing on a roof.  ← gold
```

### ARC-Easy

**Prompt (doc_to_text):**

```
Question: Which statement best explains why photosynthesis is the foundation of most food webs?
Answer:
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] Sunlight is the source of energy for nearly all ecosystems.  ← gold
[1] Most ecosystems are found on land instead of in water.
[2] Carbon dioxide is more available than other gases.
[3] The producers in all ecosystems are plants.
```

### ARC-Challenge

**Prompt (doc_to_text):**

```
Question: An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?
Answer:
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] Planetary density will decrease.
[1] Planetary years will become longer.
[2] Planetary days will become shorter.  ← gold
[3] Planetary gravity will become stronger.
```

### BoolQ

**Prompt (doc_to_text):**

```
Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units fr …[truncated]
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] no  ← gold
[1] yes
```

### PIQA

**Prompt (doc_to_text):**

```
Question: How do I ready a guinea pig cage for it's new occupants?
Answer:
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.  ← gold
[1] Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.
```

### WinoGrande

_Non-standard lm-eval API: fill the blank `_` with each option → two candidate **contexts**, score the
**shared suffix** under each, argmax wins. (`doc_to_text` is repurposed to the gold index, hence a
naïve dump shows "1"; the real eval scores suffix-given-context and is correct — acc 61.6, n=1267.)_

**Sentence (blank `_`):**

```
Sarah was a much better surgeon than Maria so _ always got the easier cases.
options: "Sarah" / "Maria"   gold = Maria
```

**Two candidate contexts (each scores the shared suffix):**

```
[0] Sarah was a much better surgeon than Maria so Sarah
[1] Sarah was a much better surgeon than Maria so Maria   ← gold
```

**Shared suffix scored under each (higher loglikelihood wins):**

```
always got the easier cases.
```

### MMLU (abstract_algebra)

**Prompt (doc_to_text):**

```
Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
```

**Choices (model scores each as a continuation; argmax = prediction):**

```
[0] A
[1] B  ← gold
[2] C
[3] D
```

## CUTE — character manipulation (generation — generate_until → exact_match)

All 14 subtasks (exact_match %), grouped by CUTE category; **bold** category rows are the means.

| category | subtask | Llama 1.8B | AU-Net2 | root_greedy |
|---|---|---|---|---|
| **composition** | **_mean_** | **32.6** | **59.3** | **44.5** |
|  | spell | 18.6 | 88.4 | 22.1 |
|  | spell_inverse | 11.2 | 48.5 | 56.8 |
|  | contains_char | 53.4 | 50.8 | 49.5 |
|  | contains_word | 47.0 | 49.6 | 49.8 |
| **orthographic** | **_mean_** | **47.1** | **41.0** | **45.3** |
|  | orth | 47.3 | 37.0 | 38.1 |
|  | sem | 46.9 | 45.0 | 52.5 |
| **sequence** | **_mean_** | **4.1** | **1.9** | **2.4** |
|  | ins_char | 8.1 | 1.7 | 2.4 |
|  | ins_word | 14.5 | 6.4 | 7.5 |
|  | del_char | 1.1 | 0.0 | 0.0 |
|  | del_word | 1.2 | 1.5 | 0.5 |
|  | sub_char | 0.3 | 0.8 | 2.3 |
|  | sub_word | 6.3 | 3.6 | 5.1 |
|  | swap_char | 0.4 | 0.0 | 0.0 |
|  | swap_word | 0.8 | 1.4 | 1.4 |
| **CUTE avg** | **(all 14)** | **18.4** | **23.9** | **20.6** |

Notes: AU-Net2's composition lead is driven by **spell** (88.4 — emit the bytes of a word). root_greedy's
**spell_inverse** (56.8) leads. Llama 1.8B leads **orthographic** (orth 47.3) and most **sequence** ops
(ins/del/swap of chars/words) — sequence is near-floor for all three at 1.3B scale.

### Examples by category

*Self-contained few-shot prompts; the model must emit the manipulated string (generate_until → exact_match).*

**composition** — `spell` (emit a word's letters, space-separated):

```
Spell out the word, putting spaces between each letter, based on the following examples:
    1. Spell out the word " alphabet ". Answer: " a l p h a b e t "
    2. Spell out the word " hello ". Answer: " h e l l o "
    3. Spell out the word " zebra ". Answer: " z e b r a "
    4. Spell out the word " tongue ". Answer: " t o n g u e "
    Question: Spell out the word " the ".
→ Answer: t h e
```

**orthographic** — `orth` (pick the word closer in Levenshtein distance):

```
Select the word that is closer in Levenshtein distance to the given word based on the following examples:
    1. Closer in Levenshtein distance to "bold": "cold", "brave". Answer: "cold"
    2. Closer in Levenshtein distance to "computer": "completed", "laptop". Answer: "completed"
    3. Closer in Levenshtein distance to "happy": "glad", "apply". Answer: "apply"
    4. Closer in Levenshtein distance to "camp": "ramp", "tent". Answer: "ramp"
    Question: Closer in Levenshtein distance to " is ": " gis ", " are ".
→ Answer: gis
```

**sequence** — `swap_char` (swap two specified letters in a word):

```
Swap the positions of two specified letters in a given word, based on the following examples:
    1. Swap " l " and " b " in " alphabet ". Answer: " abphalet "
    2. Swap " h " and " e " in " hello ". Answer: " ehllo "
    3. Swap " z " and " a " in " zebra ". Answer: " aebrz "
    4. Swap " u " and " e " in " tongue ". Answer: " tongeu "
    Question: Swap " e " and " h " in " the ".
→ Answer: teh
```

### Examples

*Self-contained 0-shot prompts; the model must emit the manipulated string. 14 subtasks across composition/orthographic/sequence.*

### CUTE / spell

**Prompt:**

```
Spell out the word, putting spaces between each letter, based on the following examples:

            1. Spell out the word " alphabet ". Answer: " a l p h a b e t "
            2. Spell out the word " hello ". Answer: " h e l l o "
            3. Spell out the word " zebra ". Answer: " z e b r a "
            4. Spell out the word " tongue ". Answer: " t o n g u e "

            Question: Spell out the word " the ".
```

**Answer:**

```
t h e
```

### CUTE / swap_char

**Prompt:**

```
Swap the positions of two specified letters in a given word, based on the following examples:

            1. Swap " l " and " b " in " alphabet ". Answer: " abphalet "
            2. Swap " h " and " e " in " hello ". Answer: " ehllo "
            3. Swap " z " and " a " in " zebra ". Answer: " aebrz "
            4. Swap " u " and " e " in " tongue ". Answer: " tongeu "

            Question: Swap " e " and " h " in " the ".
```

**Answer:**

```
teh
```

### CUTE / contains_char

**Prompt:**

```
Answer whether the specified letter is in the given word, based on the following examples:

            1. Is there a " a " in " alphabet "? Answer: "Yes"
            2. Is there a " z " in " alphabet "? Answer: "No"
            3. Is there a " u " in " hello "? Answer: "No"
            4. Is there a " o " in " hello "? Answer: "Yes"

            Question: Is there a " l " in " the "?
```

**Answer:**

```
No
```

## HellaSwag-typo (likelihood — acc_norm)

Typos applied to the **prompt/context only** (answers kept clean), scored by acc_norm. **2 modes × 4
ops = 8 cases**, plus per-op / per-mode means and the overall avg. **bold** = per-mode / avg rows.

- **char mode** (harsh): every alphabetic char independently corrupted w.p. **0.15**, anywhere (no first/last protection).
- **word mode** (realistic): **0.30** of eligible words (alpha, len≥4) get **one interior edit**, first & last letters preserved ("Cmabrigde" effect).
- ops: **delete** (drop a char) · **swap** (transpose adjacent) · **key** (QWERTY-adjacent substitution, fat-finger) · **insert** (QWERTY-adjacent insertion).

_The per-op numbers (8 cases + char/word means + avg, clean baseline included) are auto-generated in
the **HellaSwag-typo per op** table inside the robustness AUTO block below, kept fresh by
`gen_report.py`. The methodology above and the example below are the static reference for that table._

Pattern: **char mode is much harsher** than word mode for all three (every char at risk vs one
protected interior edit); **`key` is the single hardest op**. Byte models (AU-Net2, root_greedy) are
**more typo-robust** — higher clean *and* higher typo-avg, and a smaller relative drop.

### Example: clean vs. keyboard-adjacent typo

**Clean prompt:**

```
Roof shingle removal: A man is sitting on a roof. He
```

**key_char typo (keyboard-adjacent substitutions):**

```
Roof shingle removal: A man is sktting ob z roof. He
```

## Robustness / boundary

*The tables in the block below are auto-generated by `gen_report.py` between the AUTO markers:
HellaSwag-Noise (avg, per-strategy in BLT Table 12 layout, **and** per-strategy × target);
HellaSwag-typo per op; Phonology-G2P; noisy-downstream (avg **and** per typo-op for each benchmark);
PBP cut-point ΔBPC; corpus-scale pbp_mc; and the online-bt supplement. Static examples & notes follow
after the markers and are preserved across regenerations.*

<!-- AUTO_BEGIN robustness/boundary (gen_report.py) -->

## Robustness & boundary (auto-updated 2026-06-22 15:50)

### HellaSwag-Noise (acc_norm avg over 15 variants) + Phonology-G2P (exact_match)

| Model | HS-Noise avg | Phonology-G2P |
|-------|-------------|---------------|
| Llama 1.8B | 37.6 | 0.45 |
| AU-Net2 | 41.5 | 0.00 |
| BPEByte root_greedy | 42.2 | 0.10 |

#### HellaSwag-Noise per strategy (acc_norm, BLT Table 12 layout)

| Model | Antspeak | Drop | Randomcase | Repeat | Uppercase | Avg |
|---|---|---|---|---|---|---|
| Llama 1.8B | 31.0 | 36.8 | 34.2 | 38.2 | 48.0 | 37.6 |
| AU-Net2 | 33.5 | 38.1 | 44.3 | 38.7 | 53.1 | 41.5 |
| BPEByte root_greedy | 32.7 | 38.9 | 44.2 | 41.0 | 54.1 | 42.2 |

#### HellaSwag-Noise per strategy × target (acc_norm)

| Model | Strategy | Prompt | Completion | Both |
|---|---|---|---|---|
| Llama 1.8B | Antspeak | 32.5 | 24.9 | 35.7 |
| Llama 1.8B | Drop | 47.3 | 31.6 | 31.4 |
| Llama 1.8B | Randomcase | 40.5 | 29.8 | 32.2 |
| Llama 1.8B | Repeat | 46.6 | 33.7 | 34.5 |
| Llama 1.8B | Uppercase | 51.5 | 44.2 | 48.4 |
| AU-Net2 | Antspeak | 33.6 | 31.8 | 35.0 |
| AU-Net2 | Drop | 49.2 | 32.9 | 32.3 |
| AU-Net2 | Randomcase | 47.3 | 40.2 | 45.4 |
| AU-Net2 | Repeat | 47.4 | 34.2 | 34.4 |
| AU-Net2 | Uppercase | 53.7 | 50.5 | 55.2 |
| BPEByte root_greedy | Antspeak | 32.4 | 31.1 | 34.7 |
| BPEByte root_greedy | Drop | 49.7 | 33.8 | 33.2 |
| BPEByte root_greedy | Randomcase | 48.8 | 40.1 | 43.8 |
| BPEByte root_greedy | Repeat | 49.3 | 36.3 | 37.4 |
| BPEByte root_greedy | Uppercase | 54.7 | 51.5 | 56.1 |

### HellaSwag-typo per op (acc_norm; question/stem corrupted, choices clean)

| Model | clean | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | char-mean | word-mean | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 1.8B | 55.8 | 46.2 | 48.3 | 44.0 | 46.8 | 53.4 | 53.3 | 52.5 | 53.3 | 46.3 | 53.1 | 49.7 |
| AU-Net2 | 57.8 | 47.3 | 52.8 | 46.3 | 50.0 | 55.0 | 55.2 | 54.4 | 55.4 | 49.1 | 55.0 | 52.1 |
| BPEByte root_greedy | 57.5 | 48.9 | 51.8 | 46.9 | 49.4 | 54.4 | 55.1 | 54.8 | 55.1 | 49.3 | 54.9 | 52.1 |

### Noisy downstream (clean → typo-avg; drop in pts; per-benchmark metric in header)

| Model | boolq (acc) | piqa (acc_norm) | arc_easy (acc_norm) | arc_challenge (acc_norm) |
|---|---|---|---|---|
| Llama 1.8B | 63.2→59.2 (-4.1) | 74.9→72.6 (-2.3) | 65.1→54.5 (-10.7) | 35.2→30.8 (-4.5) |
| AU-Net2 | 60.5→59.3 (-1.2) | 74.1→72.5 (-1.6) | 65.1→57.7 (-7.5) | 36.0→31.8 (-4.2) |
| BPEByte root_greedy | 59.1→59.8 (+0.8) | 73.2→72.3 (-0.9) | 66.0→59.4 (-6.6) | 36.5→32.6 (-3.9) |

#### Noisy downstream — boolq per typo-op (acc)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama 1.8B | 58.9 | 57.2 | 56.9 | 57.7 | 61.4 | 60.4 | 60.4 | 60.6 | 59.2 |
| AU-Net2 | 58.0 | 59.3 | 57.0 | 59.1 | 60.1 | 60.9 | 60.0 | 60.4 | 59.3 |
| BPEByte root_greedy | 59.6 | 59.8 | 59.4 | 60.2 | 59.7 | 60.1 | 60.3 | 59.8 | 59.8 |

#### Noisy downstream — piqa per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama 1.8B | 71.7 | 71.4 | 70.9 | 70.9 | 74.2 | 73.8 | 73.7 | 74.0 | 72.6 |
| AU-Net2 | 72.0 | 72.5 | 71.1 | 71.9 | 72.5 | 73.8 | 73.0 | 73.0 | 72.5 |
| BPEByte root_greedy | 71.3 | 72.3 | 70.5 | 72.0 | 72.6 | 73.6 | 72.9 | 73.6 | 72.3 |

#### Noisy downstream — arc_easy per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama 1.8B | 49.0 | 51.6 | 47.2 | 50.0 | 60.7 | 59.7 | 58.1 | 59.5 | 54.5 |
| AU-Net2 | 53.2 | 57.1 | 48.4 | 53.1 | 63.3 | 63.0 | 60.3 | 62.9 | 57.7 |
| BPEByte root_greedy | 54.0 | 59.3 | 50.9 | 55.8 | 64.2 | 64.5 | 61.6 | 64.5 | 59.4 |

#### Noisy downstream — arc_challenge per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama 1.8B | 28.0 | 30.3 | 27.6 | 29.5 | 33.1 | 33.2 | 31.7 | 32.8 | 30.8 |
| AU-Net2 | 30.1 | 32.6 | 28.2 | 31.0 | 33.3 | 34.4 | 31.6 | 32.9 | 31.8 |
| BPEByte root_greedy | 30.2 | 33.9 | 28.7 | 31.1 | 34.2 | 35.0 | 33.1 | 34.7 | 32.6 |

### PBP — cut-point ΔBPC (lower=more cut-invariant) & MCQ-boundary ΔAcc

| Model | ΔBPC overall | ΔBPC en | ΔBPC code | ΔBPC zh | pbp_mc ΔAcc avg |
|-------|-------------|---------|-----------|---------|-----------------|
| Llama 1.8B | +0.710 | +1.184 | +0.837 | +0.110 | -8.92 |
| AU-Net2 | +0.000 | +0.000 | -0.000 | +0.000 | -0.03 |
| BPEByte root_greedy | +0.000 | -0.000 | +0.000 | -0.000 | +0.16 |

#### pbp_mc per-task ΔAcc (corpus-scale, limit 2000)

| Model | arc_easy | arc_challenge | hellaswag | curated |
|---|---|---|---|---|
| Llama 1.8B | -22.55 | -2.56 | -1.65 | -60.00 |
| AU-Net2 | +0.00 | -0.09 | +0.00 | +0.00 |
| BPEByte root_greedy | +0.00 | +0.43 | +0.05 | +0.00 |

## Supplements — BPEByte online-bt

Secondary byte variant(s) kept out of the main comparison above. Empty cells (—) mean that suite was not run for the variant.

### HellaSwag-Noise (acc_norm avg over 15 variants) + Phonology-G2P (exact_match)

| Model | HS-Noise avg | Phonology-G2P |
|-------|-------------|---------------|
| BPEByte online-bt | 43.6 | 0.00 |

#### HellaSwag-Noise per strategy (acc_norm, BLT Table 12 layout)

| Model | Antspeak | Drop | Randomcase | Repeat | Uppercase | Avg |
|---|---|---|---|---|---|---|
| BPEByte online-bt | 34.4 | 40.7 | 45.2 | 42.6 | 55.1 | 43.6 |

#### HellaSwag-Noise per strategy × target (acc_norm)

| Model | Strategy | Prompt | Completion | Both |
|---|---|---|---|---|
| BPEByte online-bt | Antspeak | 34.8 | 31.6 | 36.8 |
| BPEByte online-bt | Drop | 51.4 | 36.5 | 34.0 |
| BPEByte online-bt | Randomcase | 49.8 | 42.3 | 43.5 |
| BPEByte online-bt | Repeat | 50.9 | 37.3 | 39.5 |
| BPEByte online-bt | Uppercase | 56.4 | 52.4 | 56.4 |

### HellaSwag-typo per op (acc_norm; question/stem corrupted, choices clean)

| Model | clean | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | char-mean | word-mean | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPEByte online-bt | 58.6 | 49.8 | 52.9 | 49.2 | 51.2 | 55.8 | 56.8 | 56.2 | 56.5 | 50.8 | 56.3 | 53.6 |

### Noisy downstream (clean → typo-avg; drop in pts; per-benchmark metric in header)

| Model | boolq (acc) | piqa (acc_norm) | arc_easy (acc_norm) | arc_challenge (acc_norm) |
|---|---|---|---|---|
| BPEByte online-bt | 64.2→61.5 (-2.8) | 74.4→71.9 (-2.5) | 64.9→58.0 (-6.9) | 37.4→33.3 (-4.1) |

#### Noisy downstream — boolq per typo-op (acc)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| BPEByte online-bt | 59.5 | 60.5 | 59.8 | 61.0 | 62.9 | 62.8 | 62.0 | 63.3 | 61.5 |

#### Noisy downstream — piqa per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| BPEByte online-bt | 71.2 | 72.3 | 69.0 | 70.9 | 72.6 | 73.0 | 72.9 | 73.3 | 71.9 |

#### Noisy downstream — arc_easy per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| BPEByte online-bt | 52.6 | 57.8 | 50.0 | 54.7 | 62.0 | 64.0 | 60.6 | 62.3 | 58.0 |

#### Noisy downstream — arc_challenge per typo-op (acc_norm)

| Model | delete_char | swap_char | key_char | insert_char | delete_word | swap_word | key_word | insert_word | avg |
|---|---|---|---|---|---|---|---|---|---|
| BPEByte online-bt | 30.5 | 34.0 | 29.9 | 31.8 | 34.0 | 35.8 | 35.6 | 34.6 | 33.3 |

### PBP — cut-point ΔBPC (lower=more cut-invariant) & MCQ-boundary ΔAcc

| Model | ΔBPC overall | ΔBPC en | ΔBPC code | ΔBPC zh | pbp_mc ΔAcc avg |
|-------|-------------|---------|-----------|---------|-----------------|
| BPEByte online-bt | +0.001 | +0.001 | +0.000 | +0.000 | +0.03 |

#### pbp_mc per-task ΔAcc (corpus-scale, limit 2000)

| Model | arc_easy | arc_challenge | hellaswag | curated |
|---|---|---|---|---|
| BPEByte online-bt | -0.10 | +0.09 | +0.10 | +0.00 |

### Eval job status

- Done: 16/16
- Remaining: none — all complete

<!-- AUTO_END -->

## Robustness & boundary — examples & notes

- [ ]  Phonology-G2P 는 파악이 필요함.

_Scoring legend: HellaSwag-Noise, noisy-downstream, and PBP ΔBPC are **likelihood** (acc_norm / BPC);
Phonology-G2P is **generation** (exact_match)._

#### Example: HellaSwag-Noise

**uppercase:**

```
ROOF SHINGLE REMOVAL: A MAN IS SITTING ON A ROOF. HE
```

**randomcase:**

```
RooF ShiNGlE REmOvaL: a MAN IS SITTIng On A rooF. He
```

#### Example: Phonology-G2P

**Example item (grapheme → phoneme / IPA):**

```
{
 "grapheme": "abb",
 "phoneme": "æ b",
 "outputs": "ˈæb"
}
```

*5-shot generate_until; whitespace-normalized exact_match on the IPA output.*

#### Examples: noisy downstream

#### boolq (perturbed field: `passage`)

**Clean prompt:**

```
Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy ba …[truncated]
```

**swap_char typo:**

```
Ethanol fuel -- All biomass goes throuhg at least some of htese stpes: it needs to be grown, oclletced, dried, fermented, distilled, and burned. All of these steps reuqire resoruces and na nifarstrutcure. The toatl maoutn of enregy niput into teh rpocess compaerd to the energy released by ubrnnig the resulting ethnaol fuel is known as the neergy ba …[truncated]
```

#### piqa (perturbed field: `goal`)

**Clean prompt:**

```
Question: How do I ready a guinea pig cage for it's new occupants?
Answer:
```

**swap_char typo:**

```
Question: How do I ready a guinea pig cage fro it's new occupatns?
Answre:
```

#### PBP boundary probe (see PBP table in the auto block above)

*Experiment A (cut-point ΔBPC): score the same text with an aligned vs mid-token boundary cut; byte≈0, BPE>0. Experiment B (pbp_mc ΔAcc): commit a trailing space into the MCQ prompt; byte=0, BPE flips answers.*

#### Example A: cut-point ΔBPC — what it measures

**Setup.** Take one string and split it into three parts — `prefix`, `boundary`, `word` — where the
natural text is `prefix + boundary + word`. The `word` is scored under **two cuts of the same string**,
and we compare the cost of those identical `word` bytes:

```
item:  prefix = "The capital of France is"   boundary = " "   word = "Paris"
natural text = "The capital of France is Paris"

aligned     : score "Paris" with the boundary NOT committed (natural cut)
              logP(word) = logP(" Paris" | prefix) − logP(" " | prefix)      # telescopes → exact
misaligned  : score "Paris" with the boundary COMMITTED into the prompt (mid-token cut)
              prompt = "The capital of France is "   (note the trailing space)  → predict "Paris"

BPC = −logP(word) / (len(word_bytes) · ln2)
ΔBPC = BPC(misaligned) − BPC(aligned)      # ≥ 0; larger = more boundary-sensitive
```

**Why the two cuts differ for BPE but not bytes.** A BPE tokenizer normally encodes `" Paris"` as a
**single** token (leading space included). Committing the trailing space into the prompt forces the
continuation to be tokenized as `"Paris"` **without** its usual leading space — a rarer token
sequence the model scores lower → BPC rises → **ΔBPC > 0**. This is the classic "never end your
prompt with a trailing space" failure. A **byte** model consumes the identical raw bytes either way
(the space is just byte 0x20 in the stream), so the two conditionals are byte-identical → **ΔBPC = 0
exactly**.

**Result (this run):** Llama 1.8B ΔBPC_en = **+1.18** (strong boundary sensitivity); AU-Net2 /
root_greedy ≈ **0.000** (cut-invariant by construction). Same effect in code
(`for i in |range(10):`, Llama +0.84) and Chinese (mid-`北|京`, +0.11). _(The corpus-scale strata in
`runs/pbp_items.jsonl` use real DCLM/code/FLORES text; the curated items above isolate the mechanism.)_

#### Example B: MCQ prompt-boundary shift

**canonical: (context, ' '+option):**

```
context = 'Question: Which statement best explains why photosynthesis is the foundation of most food webs?\nAnswer:'
option[0] cont = ' Sunlight is the source of energy for nearly all ecosystems.'
```

**space: (context+' ', option) — trailing space committed:**

```
context = 'Question: Which statement best explains why photosynthesis is the foundation of most food webs?\nAnswer: '
option[0] cont = 'Sunlight is the source of energy for nearly all ecosystems.'
```

For a byte model the space shifts every option's logprob by the same constant → argmax unchanged (ΔAcc=0). For BPE it re-tokenizes the tail unevenly → ranking can flip (ΔAcc<0).

_(BPEByte online-bt results live in the **Supplements — BPEByte online-bt** subsection of the auto
block above.)_

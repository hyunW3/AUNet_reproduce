## Result

# 1.3B model results — full benchmark dump

Models: **Llama 1.8B** (BPE, `llama_1.8B_paper` @60k) · AU-Net2 (byte, word patches) · BPEByte online-bt · BPEByte root_greedy. All @ final ckpt, 0-shot, full dataset. `—` = not run, `·` = metric absent.
Headline table: `consolidated_1.3B_table.md`. BLT-format: `cute_table3_format.md`.
**Scoring mode** is noted per section: _likelihood_ = option-loglikelihood argmax (acc/acc_norm) or BPC; _generation_ = generate_until + exact_match.

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

## Core reasoning — 5-shot (likelihood — option loglikelihood → acc/acc_norm)

5-shot, fair & truncation-free (0% for all three, verified), full sets, fixed root_greedy code
(BOS/A.1/window). `num_fewshot=5`; ARC-Challenge/MMLU not re-run at 5-shot (`—`). Single run, no seeds.

| metric | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| HellaSwag acc | 47.2 | 49.0 | 48.5 |
| HellaSwag acc_norm | 62.9 | 63.7 | 63.7 |
| ARC-Easy acc | 70.9 | 70.8 | 70.5 |
| ARC-Easy acc_norm | 70.3 | 72.4 | 71.9 |
| ARC-Challenge acc | — | — | — |
| ARC-Challenge acc_norm | — | — | — |
| BoolQ acc | 65.6 | 59.1 | 65.5 |
| PIQA acc | 74.2 | 74.2 | 73.7 |
| PIQA acc_norm | 75.6 | 74.1 | 74.4 |
| WinoGrande acc | 62.6 | 63.6 | 64.1 |
| MMLU acc | — | — | — |

5-bench acc_norm/acc mean (HellaSwag·ARC-Easy·BoolQ·PIQA·WinoGrande): **root_greedy 67.9 > Llama 67.4
> AU-Net2 66.6**. At 0-shot the order is reversed (Llama 65.6 > AU-Net2 65.0 > root_greedy 64.4), so
**root_greedy overtakes the 1.8B subword baseline at 5-shot** with the strongest few-shot scaling.
AU-Net2 is held back by a **BoolQ collapse to 59.1 (~chance)** — suspiciously low, needs a seed-repeat.

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

| case | Llama 1.8B | AU-Net2 | root_greedy |
|---|---|---|---|
| clean (`hellaswag`) | 55.8 | **57.8** | 57.5 |
| delete_char | 46.2 | 47.3 | **48.9** |
| swap_char | 48.3 | **52.8** | 51.8 |
| key_char | 44.0 | 46.3 | **46.9** |
| insert_char | 46.8 | **50.0** | 49.4 |
| delete_word | 53.4 | **55.0** | 54.4 |
| swap_word | 53.3 | **55.2** | 55.1 |
| key_word | 52.5 | 54.4 | **54.8** |
| insert_word | 53.3 | **55.4** | 55.1 |
| **char mean** | 46.3 | 49.1 | **49.3** |
| **word mean** | 53.1 | **55.0** | 54.9 |
| **typo_avg (8)** | 49.7 | **52.1** | **52.1** |

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

*Full robustness & boundary tables (HellaSwag-Noise, Phonology-G2P, noisy-downstream, PBP cut-point ΔBPC, corpus-scale pbp_mc) are auto-generated below and refreshed by `gen_report.py`.*

## Robustness & boundary (auto-updated 2026-06-21 03:27)

### HellaSwag-Noise (likelihood, acc_norm avg over 15 variants) + Phonology-G2P (generation, exact_match)

- [ ]  Phonology-G2P 는 파악이 필요함.

| Model | HS-Noise avg | Phonology-G2P |
| --- | --- | --- |
| Llama 1.8B | 37.6 | 0.45 |
| AU-Net2 | 41.5 | 0.00 |
| BPEByte root_greedy | 42.2 | 0.10 |

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

### Noisy downstream (likelihood; clean → typo-avg; drop in pts; per-benchmark metric in header)

| Model | boolq (acc) | piqa (acc_norm) | arc_easy (acc_norm) | arc_challenge (acc_norm) |
| --- | --- | --- | --- | --- |
| Llama 1.8B | 63.2→59.2 (-4.0) | 74.9→72.6 (-2.3) | 65.1→54.5 (-10.6) | 35.2→30.8 (-4.4) |
| AU-Net2 | 60.5→59.3 (-1.2) | 74.1→72.5 (-1.6) | 65.1→57.7 (-7.5) | 36.0→31.8 (-4.2) |
| BPEByte root_greedy | 59.1→59.8 (+0.8) | 73.2→72.3 (-0.9) | 66.0→59.4 (-6.6) | 36.5→32.6 (-3.9) |

#### Examples

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

### PBP (likelihood) — cut-point ΔBPC (lower=more cut-invariant) & MCQ-boundary ΔAcc

| Model | ΔBPC overall | ΔBPC en | ΔBPC code | ΔBPC zh | pbp_mc ΔAcc avg |
| --- | --- | --- | --- | --- | --- |
| Llama 1.8B | +0.710 | +1.184 | +0.837 | +0.110 | -8.92 |
| AU-Net2 | +0.000 | +0.000 | -0.000 | +0.000 | -0.03 |
| BPEByte online-bt | +0.001 | +0.001 | +0.000 | +0.000 | +0.03 |
| BPEByte root_greedy | +0.000 | -0.000 | +0.000 | -0.000 | +0.16 |

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
online-bt / root_greedy ≈ **0.000** (cut-invariant by construction). Same effect in code
(`for i in |range(10):`, Llama +0.84) and Chinese (mid-`北|京`, +0.11). _(The corpus-scale strata in
`runs/pbp_items.jsonl` use real DCLM/code/FLORES text; the curated items above isolate the mechanism.)_

### pbp_mc per-task ΔAcc (corpus-scale, limit 2000)

| Model | arc_easy | arc_challenge | hellaswag | curated |
| --- | --- | --- | --- | --- |
| Llama 1.8B | -22.55 | -2.56 | -1.65 | -60.00 |
| AU-Net2 | +0.00 | -0.09 | +0.00 | +0.00 |
| BPEByte root_greedy | +0.00 | +0.43 | +0.05 | +0.00 |

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

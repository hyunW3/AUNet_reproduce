### 5-Benchmark Mean Trajectory

**Metrics:** `acc_norm`; `acc` for BoolQ and WinoGrande

| Shot | Byte (`root_greedy`) | AU-Net 2 | Llama 1.8B |
| --- | --- | --- | --- |
| 0 | 65.4 | 65.0 | **65.6** |
| 3 | **66.7** | 66.0 | **66.7** |
| 5 | 67.0 | 66.6 | **67.4** |

**Setting:** full datasets, `num_fewshot=0/3/5`, truncation-free for all three models, fixed `root_greedy` implementation (`BOS/A.1/window`). `root_greedy` column = the new-code run (`bpebyte_br_greedy_root_1.3B` @180k, cell A of the 2×2). ARC-Challenge and MMLU were not re-evaluated. Results are from a single run without multiple seeds.

## Core Reasoning — 0-shot

**Evaluation:** option log-likelihood → `acc_norm`

**Exception:** BoolQ, WinoGrande, and MMLU use `acc`.

| Benchmark | Metric | Llama 1.8B | AU-Net 2 | root_greedy |
| --- | --- | --- | --- | --- |
| HellaSwag | acc_norm | 62.2 | 62.6 | 62.5 |
| ARC-Easy | acc_norm | 65.4 | 65.7 | 66.8 |
| ARC-Challenge | acc_norm | 35.3 | 36.5 | 37.5 |
| BoolQ | acc | 63.5 | 61.1 | 62.1 |
| PIQA | acc_norm | 75.3 | 74.2 | 74.3 |
| WinoGrande | acc | 61.6 | 61.5 | 61.1 |
| MMLU | acc | 33.84 | 33.66 | 34.14 |
| average |  | 56.734 | 56.466 | 56.920 |

## Core Reasoning — 3-shot

| Benchmark | Metric | Llama 1.8B | AU-Net 2 | `root_greedy` |
| --- | --- | --- | --- | --- |
| HellaSwag | acc_norm | 62.7 | 63.1 | 63.5 |
| ARC-Easy | acc_norm | 70.0 | 72.6 | 71.4 |
| ARC-Challenge | acc_norm | 38.2 | 37.9 | **41.3** |
| BoolQ | acc | 63.5 | 59.3 | 62.6 |
| PIQA | acc_norm | 75.1 | 74.0 | 74.4 |
| WinoGrande | acc | 62.1 | 60.9 | 61.3 |
| MMLU | acc | 35.24 | 35.07 | 35.00 |
|  |  | 61.440 | 60.828 | 61.367 |

## Core Reasoning — 5-shot

| Benchmark | Metric | Llama 1.8B | AU-Net 2 | root_greedy |
| --- | --- | --- | --- | --- |
| HellaSwag | acc_norm | 62.9 | 63.7 | 63.8 |
| ARC-Easy | acc_norm | 70.3 | 72.4 | 71.8 |
| ARC-Challenge | acc_norm | 38.5 | 39.5 | **41.9** |
| BoolQ | acc | 65.6 | 59.1 | 62.8 |
| PIQA | acc_norm | 75.6 | 74.1 | 74.2 |
| WinoGrande | acc | 62.6 | 63.6 | 62.6 |
| MMLU | acc | 34.43 | 34.90 | 34.71 |
| average |  | 61.905 | 61.300 | 61.652 |
## Old vs New code — 2×2 (train/eval) breakdown by shot

All four columns are the **same `root_greedy` 1.3B @180k** model family, byte-normalized, full datasets. Columns = **train/eval** code: `new/new` (new-train, new-eval), `new/old`, `old/new`, `old/old`. **Training code**: new = non-sliding `max(8192,len)` parser window; old = 4096-sliding. **Eval fix**: new = peel BOS before the byte-trie (BOS@0 boundary); old = `OLD_EVAL_NO_BOS_PEEL=1` (spurious 2nd-position boundary). `new/new`↔`new/old` (and `old/new`↔`old/old`) isolates the eval fix; `new/new`↔`old/new` isolates the training code. Single runs, no seeds — treat sub-1-pt gaps as noise.

### 0-shot

**Averages**

| aggregate | new/new | new/old | old/new | old/old |
| --- | --- | --- | --- | --- |
| 5-bench mean (HS, ARC-E, BoolQ, PIQA, Wino) | 65.35 | 65.41 | 64.43 | 64.53 |
| all-8 mean (5-bench + ARC-C + MMLU-letter + MMLU-text) | 52.99 | 52.98 | 52.15 | 52.22 |

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

- [260624] Arc challenge few-shot 실험
    
    ## ARC-Challenge
    
    | Setting | Llama 1.8B | AU-Net2 | root_greedy |
    | --- | --- | --- | --- |
    | 3-shot (acc_norm) | 38.2 | 37.9 | **41.3** |
    | 5-shot (acc_norm) | 38.5 | 39.5 | **41.9** |
    
    ### Raw acc (for reference)
    
    | Setting | Llama 1.8B | AU-Net2 | root_greedy |
    | --- | --- | --- | --- |
    | 3-shot (acc) | 35.7 | 36.1 | **36.8** |
    | 5-shot (acc) | 34.6 | 36.9 | **37.7** |
    
    ### Observations
    
    - On the reported metric (**acc_norm**), **root_greedy is best at both 3-shot and 5-shot**.
    - Root_greedy improves from **41.3 → 41.9** when moving from 3-shot to 5-shot.
    - AU-Net2 gains substantially from additional shots (**37.9 → 39.5**).
    - Llama improves only slightly (**38.2 → 38.5**).
    - On **raw acc** the new-code run now leads as well (**36.8 / 37.7**), edging AU-Net2 (36.1 / 36.9) — so root_greedy is strongest on both raw acc and length-normalized acc_norm (the earlier acc_norm-only advantage no longer holds).
    
    This continues the pattern seen elsewhere:
    
    - **root_greedy tends to excel in normalized likelihood ranking (acc_norm)**,
    - while its weaknesses are concentrated in **BoolQ, PIQA, and WinoGrande**, not in science-style multiple-choice reasoning tasks such as ARC-Easy or ARC-Challenge.
- [260623]MMMLU 도 rank choice 형태로 바꿔봄. - 위의 테이블 업데이트함.
    
    ## Full MMLU (`mmlu_text`) — Acc_Norm Across Shot Counts
    
    | Shot | Byte (`root_greedy`) | AU-Net 2 | Llama 1.8B |
    | --- | --- | --- | --- |
    | 0 | 34.14 | 33.66 | 33.70 |
    | 3 | 35.00 | 35.07 | 35.27 |
    | 5 | 34.71 | 34.90 | 35.16 |
    | Letter-MMLU (5-shot) | 24.5 | 26.3 | 24.7 |
    - 기존의 방식 : letter likelihoood P(A/B/C/D)
        - acc
        
        | Shot | Byte (`root_greedy`) | AU-Net 2 | Llama 1.8B |
        | --- | --- | --- | --- |
        | 0 | 25.5 | 26.3 | 24.5 |
        | 3 |  |  |  |
        | 5 | 24.5 | 26.3 | 24.7 |
        | Letter-MMLU (5-shot) |  |  |  |

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

*Non-standard lm-eval API: fill the blank `_` with each option → two candidate **contexts**, score the
**shared suffix** under each, argmax wins. (`doc_to_text` is repurposed to the gold index, hence a
naïve dump shows "1"; the real eval scores suffix-given-context and is correct — acc 61.6, n=1267.)*

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
| --- | --- | --- | --- | --- |
| **composition** | ***mean*** | **32.6** | **59.3** | **56.1** |
|  | spell | 18.6 | 88.4 | 79.3 |
|  | spell_inverse | 11.2 | 48.5 | 45.6 |
|  | contains_char | 53.4 | 50.8 | 49.4 |
|  | contains_word | 47.0 | 49.6 | 50.3 |
| **orthographic** | ***mean*** | **47.1** | **41.0** | **40.5** |
|  | orth | 47.3 | 37.0 | 38.8 |
|  | sem | 46.9 | 45.0 | 42.2 |
| **sequence** | ***mean*** | **4.1** | **1.9** | **2.1** |
|  | ins_char | 8.1 | 1.7 | 2.3 |
|  | ins_word | 14.5 | 6.4 | 7.0 |
|  | del_char | 1.1 | 0.0 | 0.2 |
|  | del_word | 1.2 | 1.5 | 1.4 |
|  | sub_char | 0.3 | 0.8 | 0.4 |
|  | sub_word | 6.3 | 3.6 | 3.6 |
|  | swap_char | 0.4 | 0.0 | 0.1 |
|  | swap_word | 0.8 | 1.4 | 1.9 |
| **CUTE avg** | **(all 14)** | **18.4** | **23.9** | **23.0** |

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

## Robustness / boundary

*Full robustness & boundary tables (HellaSwag-Noise, Phonology-G2P, noisy-downstream, PBP cut-point ΔBPB, corpus-scale pbp_mc) are auto-generated below and refreshed by `gen_report.py`*

## Robustness & boundary

> **`root_greedy` column refreshed to new-code (180k, `bpebyte_br_greedy_root_1.3B`).** These tables were previously old-code (`_oldseg_`); now matched to the rest of the doc. Largest change is in **CUTE** (generation): spell 22.1→79.3, composition mean 44.5→56.1, CUTE-avg 20.6→23.0 (but spell_inverse −11.2, sem −10.3). Likelihood robustness (HS-Noise, HS-typo) and PBP barely move (<0.5 pt / ~0 ΔBPB). Llama 1.8B / AU-Net2 columns unchanged.

### HellaSwag-Noise (likelihood, acc_norm avg over 15 variants) + Phonology-G2P (generation, exact_match)

_n=2000 (first 2000 docs, `harness.limit=2000`)._

- [ ]  Phonology-G2P 는 파악이 필요함.

| Model | HS-Noise avg | Phonology-G2P |
| --- | --- | --- |
| Llama 1.8B | 37.6 | 0.45 |
| AU-Net2 | 41.5 | 0.00 |
| BPEByte root_greedy | 42.3 | 0.10 |

### Example: HellaSwag-Noise

**uppercase:**

```
ROOF SHINGLE REMOVAL: A MAN IS SITTING ON A ROOF. HE
```

**randomcase:**

```
RooF ShiNGlE REmOvaL: a MAN IS SITTIng On A rooF. He
```

---

## Detail

## Real HellaSwag Noise Examples

Generated using the actual noise functions and deterministic seeding from `eval_noise.py`.

**Dataset:** Rowan/HellaSwag validation

**Example index:** 2

### Original prompt

> Canoeing: Two women in a child are shown in a canoe while a man pulls the canoe while standing in the water, with other individuals visible in the background. The child and a different man
> 

### Original gold completion

> sit in a canoe while the man paddles.
> 

## Five Noise Strategies Applied to the Prompt

### `antspeak`

Uppercases the text and inserts spaces between characters.

> C A N O E I N G :   T W O   W O M E N   I N   A   C H I L D   A R E   S H O W N   I N   A   C A N O E …
> 

### `drop`

Independently removes approximately 10% of characters.

> Canoeing: Two womn in a child are shownina cane while a manulls the canoe wilestanding in the water with other individuls viible in the background. The child nd a ifeen man
> 

### `randomcase`

Independently converts each character to uppercase or lowercase with probability 0.5.

> CANOeiNG: two WOmen iN a chILD aRe shOwn In A CAnoe WHilE a mAn pulLS THe CANOe WHile StAndinG in THE WAtER…
> 

### `repeat`

Selects approximately 20% of characters and repeats each selected character 2–4 times in total.

> Canooooeing:::: Two wwwwomen in a     ccchiild are showwn innnn a ccccanoeeee while a   maaan pulls     ttttheeee cannnoe …
> 

### `uppercase`

Converts all characters to uppercase.

> CANOEING: TWO WOMEN IN A CHILD ARE SHOWN IN A CANOE WHILE A MAN PULLS THE CANOE WHILE STANDING IN THE WATER…
> 

## Noise Applied to the Gold Completion

| Strategy | Noised completion |
| --- | --- |
| `antspeak` | `S I T I N A C A N O E W H I L E T H E M A N P A D D L E S .` |
| `drop` | `sitin a caoewhle t man paddles.` |
| `randomcase` | `sIt in a caNOe WHiLe ThE mAn pAdDLeS.` |
| `repeat` | `siiiit in aa canoee whileee thee man ppppaadddddles....` |
| `uppercase` | `SIT IN A CANOE WHILE THE MAN PADDLES.` |

### Example: Phonology-G2P

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

## HellaSwag-typo (likelihood — acc_norm - 이것도 우리가 만든거)

Typos applied to the **prompt/context only** (answers kept clean), scored by acc_norm. **2 modes × 4 ops = 8 cases**, plus per-op / per-mode means and the overall avg. **bold** = per-mode / avg rows. **All rows use the first 2000 HellaSwag docs (`harness.limit=2000`)** — so the clean baseline here (56.8) is below the full-set 0-shot HellaSwag (62.5); compare clean→typo within this table, not against the Core 0-shot table.

- **char mode** (harsh): every alphabetic char independently corrupted w.p. **0.15**, anywhere (no first/last protection).
- **word mode** (realistic): **0.30** of eligible words (alpha, len≥4) get **one interior edit**, first & last letters preserved ("Cmabrigde" effect).
- ops: **delete** (drop a char) · **swap** (transpose adjacent) · **key** (QWERTY-adjacent substitution, fat-finger) · **insert** (QWERTY-adjacent insertion).

| case | Llama 1.8B | AU-Net2 | root_greedy |
| --- | --- | --- | --- |
| clean (`hellaswag`) | 55.8 | **57.8** | 56.8 |
| delete_char | 46.2 | 47.3 | **48.3** |
| swap_char | 48.3 | **52.8** | 51.8 |
| key_char | 44.0 | 46.3 | **47.0** |
| insert_char | 46.8 | **50.0** | 48.5 |
| delete_word | 53.4 | **55.0** | 54.6 |
| swap_word | 53.3 | 55.2 | **55.4** |
| key_word | 52.5 | **54.4** | 53.8 |
| insert_word | 53.3 | 55.4 | **55.5** |
| **char mean** | 46.3 | **49.1** | 48.9 |
| **word mean** | 53.1 | **55.0** | 54.8 |
| **typo_avg (8)** | 49.7 (-6.1) | **52.1 (-5.7)** | 51.9 (-4.9) |

### Example: clean vs. keyboard-adjacent typo

**Clean prompt:**

```
Roof shingle removal: A man is sitting on a roof. He
```

**key_char typo (keyboard-adjacent substitutions):**

```
Roof shingle removal: A man is sktting ob z roof. He
```

### Other benchmarks

_clean → typo-avg, n=2000 (first 2000 docs, `harness.limit=2000`); clean here is below the full-set 0-shot numbers._

| Model | boolq (acc) | piqa (acc_norm) | arc_easy (acc_norm) | arc_challenge (acc_norm) |
| --- | --- | --- | --- | --- |
| Llama 1.8B | 63.2→59.2 (-4.0) | 74.9→72.6 (-2.3) | 65.1→54.5 (-10.6) | 35.2→30.8 (-4.4) |
| AU-Net2 | 60.5→59.3 (-1.2) | 74.1→72.5 (-1.6) | 65.1→57.7 (-7.5) | 36.0→31.8 (-4.2) |
| BPEByte root_greedy | 62.3→60.2 (-2.1) | 74.4→72.3 (-2.0) | 66.5→60.0 (-6.5) | 37.3→33.5 (-3.8) |

### Examples

### boolq (perturbed field: `passage`)

**Clean prompt:**

```
Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy ba …[truncated]
```

**swap_char typo:**

```
Ethanol fuel -- All biomass goes throuhg at least some of htese stpes: it needs to be grown, oclletced, dried, fermented, distilled, and burned. All of these steps reuqire resoruces and na nifarstrutcure. The toatl maoutn of enregy niput into teh rpocess compaerd to the energy released by ubrnnig the resulting ethnaol fuel is known as the neergy ba …[truncated]
```

### piqa (perturbed field: `goal`)

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

### PBP (likelihood) — cut-point ΔBPB (lower=more cut-invariant) & MCQ-boundary ΔAcc

# Prompt boundary problem

- Δ 가 클수록 patch boundary problem (prompt 뒤에 “ “있고 없고에 따라서 prediction / logits이 크게 변하는 문제)가 크다.
- Metric : BPB (Bits per byte)
    
    $\text{BPB} = \frac{-\log P(\text{word})}{\text{len(word bytes)} \cdot \ln 2}$ (log likelihood 를 len(word byte) ln2 로 나눈값 
    
    $\Delta\text{BPB} = \text{BPB}(\text{misaligned cut}) - \text{BPB}(\text{aligned cut})$
    
    - $\log P(w \mid p,b)$ 를 측정할때 두가지 상황 고려
        - $w$ : answer
        - $p$ : prompt / $b$ : boundary (space)
    - prompt 뒤에 space가 붙는경우 / 아닌경우 (= where prompt boundary problem occurs)
        - 예시)
            - “The capital of France is”
            - “The capital of France is ”
        - 두상황에서의 log P(Paris) 측정 (”” : prompt 영역, 그뒤 : answer 영역)
            - likelihood of aligned cut : space가 안붙은경우
                - “The capital of France is” Paris
            - likelihood of misaligned cut : space가 prompt뒤에 붙는 경우
                - “The capital of France is ”Paris
    

| Model | ΔBPB overall | ΔBPB en | ΔBPB code | ΔBPB zh | pbp_mc ΔAcc avg |
| --- | --- | --- | --- | --- | --- |
| Llama 1.8B | +0.710 | +1.184 | +0.837 | +0.110 | -8.92 |
| AU-Net2 | +0.000 | +0.000 | -0.000 | +0.000 | -0.03 |
| BPEByte online-bt | +0.001 | +0.001 | +0.000 | +0.000 | +0.03 |
| BPEByte root_greedy | +0.000 | +0.000 | -0.000 | -0.000 | -0.16 |

### Example A: cut-point ΔBPB — what it measures

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

BPB = −logP(word) / (len(word_bytes) · ln2)
ΔBPB = BPB(misaligned) − BPB(aligned)      # ≥ 0; larger = more boundary-sensitive
```

### pbp_mc (Patch boundary problem in Multiple choice) per-task ΔAcc (corpus-scale, limit 2000)

acc_norm 차이 

| Model | arc_easy | arc_challenge | hellaswag | curated |
| --- | --- | --- | --- | --- |
| Llama 1.8B | -22.55 | -2.56 | -1.65 | -60.00 |
| AU-Net2 | +0.00 | -0.09 | +0.00 | +0.00 |
| BPEByte root_greedy | -0.20 | -0.09 | -0.20 | +0.00 |

### Example B: MCQ prompt-boundary shift

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
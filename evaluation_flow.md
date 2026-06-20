# Evaluation flow: how ARC-Easy and HellaSwag are scored

How `lm-eval` harness evaluates our byte-level models through lingua's
`EvalHarnessLM` wrapper. Both benchmarks use the **same likelihood-comparison
mechanism**; only the prompt construction differs.

## The shared pipeline

Both tasks are `output_type: multiple_choice`. For every document, lm-eval
builds one **`loglikelihood` request per answer choice** — a
`(context, continuation)` string pair — and the model picks the choice whose
continuation it finds most likely. No generation/sampling happens; everything
is a teacher-forced forward pass.

```
doc ──> 4 requests (ctx, cont_i) ──> model: Σ log P(cont_i bytes | ctx) ──> argmax ──> acc / acc_norm
```

### Step 1 — request construction (lm-eval task YAML)

| | ARC-Easy (`arc_easy.yaml`) | HellaSwag (`hellaswag.yaml` + `utils.py`) |
|---|---|---|
| dataset | `allenai/ai2_arc` (test split, 2376 docs) | `Rowan/hellaswag` (validation split, 10042 docs) |
| context | `"Question: {question}\nAnswer:"` | `"{activity_label}: {ctx_a} {ctx_b.capitalize()}"` (preprocessed) |
| continuations | the 4 answer texts, each with a leading space | the 4 sentence endings, each with a leading space |
| gold | `answerKey` index | `label` index |
| preprocessing | none | strip WikiHow artifacts: `" [title]"` → `". "`, drop `[...]` brackets, collapse double spaces |

**ARC-Easy example** (question answering — short factual continuations):

```
context      = "Question: Which factor will most likely cause a person to develop a fever?\nAnswer:"
cont A       = " a leg muscle relaxing after exercise"
cont B       = " a bacterial population in the bloodstream"     <- gold
cont C       = " several viral particles on the skin"
cont D       = " carbohydrates being digested in the stomach"
```

**HellaSwag example** (commonsense sentence completion — long narrative continuations):

```
context      = "Removing ice from car: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles."
cont A       = " , the man adds wax to the windshield and cuts it."
cont B       = " , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled."
cont C       = " , the man puts on a christmas coat, knitted with netting."
cont D       = " then, the man continues removing the snow on his car."    <- gold
```

Note HellaSwag endings are full sentences (~10-30 words) while ARC answers are
short phrases — this is why length normalization (acc_norm) matters more for
HellaSwag, and why acc_norm is the headline number for both.

### Step 2 — likelihood computation (lingua side)

`EvalHarnessLM.loglikelihood` — `lingua/apps/aunet/eval.py:170-183`:

```python
inputs = [req.args[0] + req.args[1] for req in requests]   # context + continuation concatenated
self.generator.max_gen_len = 1                              # prefill only, no generation
_, lls, greedy = self.generator.generate(inputs)
for p, ll, gr in zip(prompts, lls, greedy):
    p_len = len(self.generator.tokenizer.encode(p, add_bos=False, add_eos=False))
    results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))
```

The per-token log-probs come from the prefill pass —
`lingua/apps/aunet/generate.py:443-448`:

```python
x = logit[:-1]                                          # logits at positions 0..L-2
y = torch.tensor(p[1:])                                 # true next tokens 1..L-1
loglikelihood.append(-F.cross_entropy(x, y, reduction="none"))
greedy.append(x.argmax(dim=-1) == y)
```

`-cross_entropy(x, y)` ≡ `log_softmax(logits)[y]`, so element `j` is the
standard causal-LM term **`ll[j] = log P(token[j+1] | tokens[0..j])`**.

For our byte models (`tokenizer: bytes`) a "token" is a UTF-8 byte over a
~256-way softmax. The full sequence is `[BOS] + prompt_bytes + cont_bytes`;
slicing `ll[p_len:]` keeps exactly the terms that predict continuation bytes:

```
ll_B = log P(" "|ctx) + log P("a"|ctx+" ") + log P(" "|ctx+" a") + ...
     = one term per byte of " a bacterial population in the bloodstream"  (43 terms)
```

For the Llama subword baseline the same code yields one term per BPE token
(~8 terms for the same answer) — the mechanism is identical, only the
tokenizer changes.

The `greedy` flag (`argmax == y` for every continuation token) is returned
but unused by these two tasks (it serves exact-match-style tasks).

### Step 3 — scoring (lm-eval `api/task.py:1494-1546`)

```python
completion_len = np.array([float(len(i)) for i in choices])   # CHARACTER length
acc      = 1.0 if argmax(lls) == gold else 0.0                # raw summed log-prob
acc_norm = 1.0 if argmax(lls / completion_len) == gold else 0.0   # per-char log-prob
```

- **acc** — pick the highest raw sum. Systematically biased against long
  choices: more terms, each ≤ 0.
- **acc_norm** — divide by the choice's character count first, i.e. compare
  average log-prob per character. This is the number we report and the number
  the AU-Net paper reports (HellaSwag 64.2, ARC-E 64.4 targets).

Per-character normalization is also what makes byte-level and subword models
comparable: raw sums live on different scales (43 byte-terms vs 8
subword-terms), but both are normalized by the same character count.

Worked example (illustrative numbers, ARC question above):

| choice | chars | Σ log P (raw) | Σ log P / chars |
|---|---|---|---|
| A | 37 | −19.8 | −0.535 |
| **B (gold)** | 43 | **−17.2** | **−0.400** |
| C | 36 | −18.9 | −0.525 |
| D | 44 | −21.0 | −0.477 |

acc: argmax raw = B ✓. acc_norm: argmax normalized = B ✓. Both score 1.0 for
this doc; the reported metric is the mean over all docs.

## Differences at a glance

| | ARC-Easy | HellaSwag |
|---|---|---|
| skill probed | grade-school science QA | commonsense next-sentence plausibility |
| continuation length | short phrase (3-8 words) | full sentence (10-30 words) |
| acc vs acc_norm gap | small | large (length bias matters) |
| docs scored | 2,376 (test) | 10,042 (validation) |
| forward passes | 4 per doc ≈ 9.5k | 4 per doc ≈ 40k |

Everything else — request shape, prefill likelihood computation
(`eval.py:170`, `generate.py:443`), argmax scoring — is byte-for-byte the same
code path.

## Where results land

- During training: milestone evals (`eval_milestones` in the config) append to
  `<dump_dir>/metrics.eval.jsonl` — keys `hellaswag` / `arc_easy`, field
  `acc_norm,none`.
- Standalone eval (`python -m apps.aunet.eval ...`): consolidates the FSDP
  checkpoint, then writes `<dump_dir>/eval/results.json`.

---

# Evaluation benchmark suite

## What we currently run

Our configs (`harness.tasks`) evaluate only **`hellaswag`** and **`arc_easy`**
— the two scores the AU-Net paper (Videau et al. 2025) reports for the 1.3B
model and our reproduction target (HS 64.2 / ARC-E 64.4). All scoring uses the
`output_type: multiple_choice` loglikelihood/acc_norm path documented above.

## Full byte-LM benchmark list (BLT, *Patches Scale Better Than Tokens*)

The BLT paper (Meta, byte-latent transformer) evaluates byte models on a wider
suite. These are standard `lm-eval` tasks and are the natural set to expand to
when comparing byte vs subword models. From **BLT Table 1** (8B, FLOP-matched):

| Task | lm-eval name | type | metric |
|---|---|---|---|
| Arc-E | `arc_easy` | multiple_choice | acc_norm |
| Arc-C | `arc_challenge` | multiple_choice | acc_norm |
| HellaSwag | `hellaswag` | multiple_choice | acc_norm |
| PIQA | `piqa` | multiple_choice | acc_norm |
| MMLU | `mmlu` | multiple_choice | acc |
| MBPP | `mbpp` | generate_until (code) | pass@1 |
| HumanEval | `humaneval` | generate_until (code) | pass@1 |

MBPP/HumanEval are **generative** (code execution), so they go through the
`generate_until` path (`eval.py:132`), not the loglikelihood path — and need
the byte→token gen-length scaling already handled there (`GEN_BYTES_PER_TOKEN`).

## Robustness & character-awareness suite (BLT §6, **Table 3**)

These probe what byte models are uniquely good at — robustness to input noise
and character-level manipulation. BLT applies them to **8B BLT vs 8B Llama 3**:

- **Noised HellaSwag** — the multiple-choice task above, with the text
  corrupted by 5 noise strategies (detailed in the next section).
- **Phonology – G2P** (grapheme-to-phoneme), 5-shot, from *Phonology Bench*
  (Suvarna et al. 2024): map a word's letters to its phoneme transcription.
- **CUTE** (Edman et al. 2024): 14 character-manipulation subtasks, grouped as
  (a) understanding composition, (b) orthographic similarity, (c) sequence
  manipulation — `Contains Char`, `Contains Word`, `Del Char`, `Del Word`,
  `Ins Char`, `Ins Word`, `Orthography`, `Semantic`, `Spelling`,
  `Spelling Inverse`, `Substitute Char`, `Substitute Word`, `Swap Char`,
  `Swap Word`. BLT-Entropy beats BPE Llama 3 by 25+ points here and hits 99.9%
  on both spelling tasks — the headline evidence that byte models "see" letters
  that BPE tokens hide.

---

# Noisy evaluation (BLT Table 3)

## Why add it

Our pipeline already produces clean HellaSwag/ARC-E numbers close to the AU-Net
paper. The **byte-model thesis** — that operating on raw bytes makes the model
robust to character-level corruption that shatters BPE tokenization — is only
visible under *noised* inputs. BLT Table 3 is exactly this probe: a BPE model's
HellaSwag drops from 79.1 → 56.9 under noise, while the byte model holds 80.6 →
64.3. Running it lets us claim the same robustness advantage for our AU-Net /
BPEByte runs rather than just matching clean accuracy.

## The 5 noise strategies (BLT §6.1, *Noisy Data*, quoted exactly)

Each is a character-level transformation applied to the HellaSwag text:

| Strategy | Definition (verbatim from the paper) |
|---|---|
| **AntSpeak** | "converts the entire text into uppercase, space-separated characters" |
| **Drop** | "Randomly removes 10% of the characters from the text" |
| **RandomCase** | "Converts 50% of the characters to uppercase and 50% to lowercase randomly throughout the text" |
| **Repeat** | "Repeats 20% of the characters up to a maximum of four times" |
| **UpperCase** | "Transforms all characters in the text to uppercase" |

Worked illustration on `"the man removes the snow"`:

```
original    : the man removes the snow
AntSpeak    : T H E   M A N   R E M O V E S   T H E   S N O W
Drop        : the mn rmoves the now            (~10% of chars deleted at random)
RandomCase  : tHe mAn ReMoVes THe snOw         (each char's case flipped w.p. ~0.5)
Repeat      : thhe man remmooves the snnow      (~20% of chars duplicated up to 4x)
UpperCase   : THE MAN REMOVES THE SNOW
```

## Application protocol (this is the subtle part)

From the paper: *"During evaluation, we apply each noising strategy to either
the **prompt**, **completion**, or **both** as separate tasks and report the
average scores."*

So for each of the 5 strategies there are **3 variants** (noise the context
only / the choices only / both) = **15 noised tasks**, and "HellaSwag Noise
Avg." in Table 3 is the **mean over all 15**. Everything downstream is
unchanged: the noised strings are fed through the *identical*
multiple-choice loglikelihood → acc_norm pipeline (Steps 2–3 above). Noisy eval
is purely a **preprocessing transform on the task's `doc_to_text` /
`doc_to_choice` strings** — no model or scoring change.

BLT's reported HellaSwag Noise Avg.: Llama 3 = 56.9, Llama 3.1 = 64.3,
BLT = 64.3 (per-strategy: AntSpeak 57.9, Drop 58.2, RandomCase 65.7,
Repeat 66.6, UpperCase 77.3 for BLT).

## Implementation in our stack (`apps/aunet/eval_noise.py`)

Implemented as a task-side transform — no change to `generate.py` or the
`eval.py:170` likelihood path, which is why it composes cleanly with the byte
tokenizer.

**Module `apps/aunet/eval_noise.py`:**
- One function per strategy (`_antspeak`, `_drop`, `_random_case`, `_repeat`,
  `_uppercase`) matching the verbatim definitions above. Rates:
  `DROP_RATE=0.10`, `REPEAT_RATE=0.20`, `REPEAT_MAX=4` (a repeated char appears
  2–4 times total).
- `build_noised_hellaswag_tasks(base_seed)` clones the stock `hellaswag`
  `ConfigurableTask` (via `config.to_dict()`), and for each
  `(strategy, target ∈ {prompt, completion, both})` swaps in a `process_docs`
  that runs the original HellaSwag preprocessing, then noises `query` (prompt)
  and/or `choices` (completion). → **15 task objects**.
- **Deterministic seeding** via `_seed(base, idx, strategy, field)` — pure
  integer arithmetic (no Python string hashing), so corruption is identical
  across checkpoints/processes regardless of `PYTHONHASHSEED`. `field=0` is the
  prompt, `field=1+j` the j-th choice, keeping prompt/choice noise independent.
- `expand_noise_tasks(tasks, base_seed)` replaces the `hellaswag_noise`
  sentinel in a task list with the 15 objects (no-op if absent).
- `summarize_noise(results)` computes `hellaswag_noise_avg` (mean acc_norm over
  all 15) and `hellaswag_noise_<strategy>` (mean over that strategy's 3
  targets).

**Wiring in `eval.py`** (`launch_eval`): before `simple_evaluate`, the task list
is run through `expand_noise_tasks`; after, `summarize_noise` injects the
aggregate rows into `results["results"]` (rank 0 only). lm-eval's
`get_task_dict` accepts pre-built `Task` objects directly (`tasks/__init__.py`),
so the variants flow through unchanged.

**How to run** — add the sentinel to a config's `harness.tasks` (already done in
`apps/aunet/configs/eval_aunet2_1.3B_b200.yaml`):

```yaml
harness:
  tasks: [hellaswag, arc_easy, hellaswag_noise]   # → 2 + 15 = 17 task rows
```

Then `results.json` carries `hellaswag`, `arc_easy`, the 15
`hellaswag_noise_<strategy>_<target>` rows, plus `hellaswag_noise_avg` and the 5
per-strategy means. Compare `hellaswag_noise_avg` against the clean `hellaswag`
score (the byte-robustness gap) and against BLT's 64.3.

> **Note**: kept out of the *training* milestone config
> (`aunet2_1.3B_b200.yaml`) on purpose — 15 extra tasks would ~8× the per-
> milestone eval cost. It lives in the standalone eval config only.

### CUTE (`apps/aunet/eval_cute.py` + `eval_tasks/gen_mc/cute_*.yaml`)
The character-awareness half of Table 3 is now wired in. CUTE (Edman et al. 2024,
the `leukas/cute` HF dataset) is **14 character-manipulation subtasks**, 1000
examples each, grouped as **composition** (`spell`, `spell_inverse`,
`contains_char`, `contains_word`), **orthographic similarity** (`orth`, `sem`),
and **sequence manipulation** (`ins`/`del`/`sub`/`swap` × `char`/`word`). Each
example is a **self-contained prompt** (its own 4 in-context demos + a `Question:`
line) with a gold `answer`; BLT used these original prompts with no prompt
engineering, so they run **0-shot** from lm-eval's view.

Implemented as plain `generate_until` YAML tasks in the gen-mc include dir (base
`_cute_common.yaml`): `doc_to_text` appends ` Answer:` as the generation cue (the
in-context demos render answers as `Answer: "<ans>"`), the `extract-quote` regex
filter pulls the first double-quoted span (surrounding spaces stripped — the CUTE
`" glad "` format — but inner spaces kept, since `spell` golds like `t h e` are
space-separated), and `exact_match` (ignore case + punctuation) scores it. The
**`cute`** sentinel in `harness.tasks` expands to the 14 `cute_<split>` tasks
(`eval.py:expand_cute_tasks`), and `summarize_cute` injects the aggregates
**`cute_avg`** plus the three category means (`cute_composition`,
`cute_orthographic`, `cute_sequence`) — mirroring the `hellaswag_noise_gen`
sentinel+summarize structure. Run via `apps/aunet/configs/eval_cute_b200.yaml`
(heavy: 14×1000 generations; `harness.limit` for a smoke run). This is the
headline byte-vs-BPE probe — BLT-Entropy beats BPE Llama 3 by 25+ points and hits
99.9% on both spelling tasks.

### Noisy-question HellaSwag — realistic typos (`apps/aunet/eval_typo.py`)
A third robustness probe, complementing the BLT-strategy suite above. Where that
suite uses *aggressive, unnatural* transforms (antspeak/drop/randomcase/repeat/
uppercase), this corrupts the **question (stem) only** with *realistic typos* —
misspellings and missing characters of the kind a human types — and leaves the
endings **clean**. It is the input-corruption analogue of CUTE: CUTE asks "can you
manipulate characters", this asks "are you robust when the input characters are
corrupted". For a byte/char model a typo changes a few bytes; for a BPE model it
shatters the token sequence — so the clean-vs-noisy acc_norm gap is the
byte-robustness signal.

Four single-character edit **ops** (grounded in Belinkov & Bisk 2018, "Synthetic
and Natural Noise Both Break NMT"): `delete` (missing char), `swap` (transpose two
adjacent — the most common real typo), `key` (substitute a QWERTY-adjacent key —
fat-finger misspelling), `insert` (insert an adjacent key). Each under two
**severity modes**: `word` (one edit on an **interior** char, first & last
preserved — the "Cmabrigde" effect — on a `WORD_RATE`≈0.30 fraction of eligible
words, len≥4) and `char` (each alpha char corrupted w.p. `CHAR_RATE`≈0.15,
anywhere). → **4 × 2 = 8 prompt-only variants**. Scoring is the **identical**
multiple_choice loglikelihood → acc_norm path as clean `hellaswag` (a pure
task-side transform on `query`), reusing the deterministic `_seed` from
`eval_noise.py`.

The **`hellaswag_typo`** sentinel expands to the 8 variants
(`eval.py:expand_typo_tasks`, cloned from stock `hellaswag` — no `include_path`),
and `summarize_typo` injects `hellaswag_typo_avg` plus per-op
(`hellaswag_typo_<delete|swap|key|insert>`) and per-mode
(`hellaswag_typo_<word|char>`) means. Kept on a **separate aggregate** so it does
**not** alter the BLT-exact `hellaswag_noise_avg`. Run via
`apps/aunet/configs/eval_hellaswag_typo_b200.yaml` (lists clean `hellaswag` for the
gap reference). Worked example on `"The man carefully removes the snow"`:

```
word/delete : The man carefully reoves the snow      (one interior char dropped/word)
word/swap   : The man carefully reomves the snow     (adjacent interior chars transposed)
char/key    : Hhe mab carefully remkvrs the snpw      (per-char QWERTY-adjacent subs)
```

### Not yet implemented (full Table 3)
Phonology-G2P (Suvarna et al. 2024) is documented above but not wired in — it
needs its own dataset/prompts. Add as a follow-up for the complete Table 3.

> Source: *BLT: Patches Scale Better Than Tokens*, Table 1 (benchmark suite),
> Table 3 + §6.1 (noise strategies & protocol), §6.1 CUTE/G2P paragraphs.

---

# Generation: prefix vs generated tokenization (`before_root:bt`)

The likelihood path above (Steps 2–3) is teacher-forced: the whole string is
known, so patching is trivial. The **generative** tasks (`generate_until`:
MBPP/HumanEval, the eval-gen suite) are different — the model emits bytes one at
a time, and for a BPEByte model **the patch boundaries can't be computed ahead
of generation.** This is why `apps/aunet/generate_bt.py` exists. AU-Net (BPEByte
mode) pools bytes into **patches**; a *boundary* (`level=1`) marks the last byte
of each patch — the same position llama3 BPE would end a token (`before_root`).
The whole prefix-vs-generated difference is **whether those boundaries are known
with full lookahead or have to be guessed and corrected as bytes stream out.**

## Shared representation

```
bytes :  h  e  l  l  o     w  o  r  l  d
level :  0  0  0  0  1  0  0  0  0  0  1     <- 1 = patch end (pooling point)
patches: └─ "hello" ─┘     └── " world" ──┘
         the model pools each patch's bytes into one vector
```

## Prefix (prompt) — boundaries are EXACT

The full prompt is known, so BPE runs over **all of it at once**
(`RegexPool.get_levels_mask(full_bytes)`) with complete lookahead. Every merge
that will ever happen is visible, so every boundary is final and immutable —
exactly how training data and the likelihood path cut text:

```
prompt = "hello world"            (known in full)
        ┌────────────── BPE the entire string once ──────────────┐
bytes :  h  e  l  l  o     w  o  r  l  d
level :  0  0  0  0  1  0  0  0  0  0  1        ✓ STABLE — never changes
```

## Generated — boundaries are SPECULATIVE (need backtracking)

Bytes are produced left→right with **no lookahead**, and BPE is greedy by
merge-rank, so a *future* byte can re-segment *earlier* bytes. The incremental
parser (`BPEIncrementalBTParser`) therefore holds the last `commit_margin = 2`
token-ends **speculative** and commits a boundary only once later bytes can't
move it. When a newly generated byte changes an already-committed boundary that
an earlier sample depended on, that sample was drawn under a now-wrong patch
mask (off-policy), so the loop **rolls back** — truncates to the corrected point,
`restore()`s the parser (O(1)), and re-samples (`generate_bt.py:online_bt_loop`).

## Why the naive cached path is wrong

The stock cached generator (`get_levels_mask_gen`) has no lookahead, so it marks
**every generated byte as a boundary** (the last byte of any current prefix is
trivially a BPE token-end):

```
naive generated:  ' '  w   o   r   l   d
level          :   1   1   1   1   1   1     ✗ every byte its own patch
```

That patch structure is nothing like what the model trained on → garbage
generation. The `bt` loop exists to make generated boundaries match the offline
ones.

## Real trace

Captured by driving the actual `online_bt_loop` (real llama3-BPE parser) with a
forced greedy forward over `prompt="hello"`, generating `" world"` — the
`on_event` hook in `generate_bt.py` emits these `sample`/`final` events.
`·` = space, `|` = committed patch end, `tok N` is the token index (0 = BOS).

```
feed '·'   t= 6  ACCEPT   hello·
feed 'w'   t= 7  ACCEPT   hello·w
feed 'o'   t= 8  ACCEPT   hello·wo
feed 'r'   t= 9  ACCEPT   hello·wor          <- "hello" boundary still speculative
feed 'l'   t=10  ROLLBACK first_change@tok5 keep=6 drop='·worl'
         before: hello|·worl
         after : hello|   <- "hello" boundary just committed; re-sample from here
feed '·'   t= 6  ACCEPT   hello|·
feed 'w'   t= 7  ACCEPT   hello|·w
feed 'o'   t= 8  ACCEPT   hello|·wo
feed 'r'   t= 9  ACCEPT   hello|·wor
feed 'l'   t=10  ACCEPT   hello|·worl
feed 'd'   t=11  ACCEPT   hello|·world
FINAL live mask : hello|·world      (trailing token still speculative) rollbacks=1
finalize()      : hello|·world|     (trailing boundary now committed)
MATCH finalize()==offline get_levels_mask: True
```

What happened: right after the prompt, even `hello`'s boundary is held
speculative (within `commit_margin`), so the first pass samples `·wor` while
token 5 has `level=0`. Feeding `l` gives the parser enough lookahead to commit
the `hello` boundary (`tok5: 0→1`) — a `first_change` strictly before the
frontier — so the `·wor` bytes were off-policy. The loop rolls back to `keep=6`
(just past the corrected boundary), drops `·worl`, and re-samples `·world` under
the right mask. The closing invariant holds: after `finalize()` the generated
boundaries equal the offline `get_levels_mask` of the full string — **the
generated text ends up tokenized exactly as if it had been a known prefix.**
(`rb` counts across more cases: `hello world` →1, `tokenization` →1,
`super`+`market` →1, `The quick brown fox` →2; cleanly-segmenting strings like
`for`+` forest` →0.)

## Summary

| | Prefix / prompt | Generated tokens |
|---|---|---|
| Lookahead | full (whole string known) | none (one byte at a time) |
| Boundaries | computed once, **immutable** | **speculative**, can move backward |
| Mechanism | `get_levels_mask(all bytes)` | incremental parser + rollback/re-sample |
| Tail token | prompt may end mid-token (left uncommitted) | grows until stable, then commits |
| **Backtracking occurs** | **no — boundaries fixed** | **yes — rollback + re-sample on a pre-frontier boundary change** |
| Failure if ignored | — | naive path: every byte its own patch → diverges from training |

> Trace reproducible on CPU (no GPU/checkpoint) via the machinery in
> `apps/aunet/test_generate_bt.py` (`RegexPool` + `online_bt_loop` with a forced
> forward and the `on_event` hook).

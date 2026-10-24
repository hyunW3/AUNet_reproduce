# NUMCoT on the 1B models (2026-10-24)

NUMCoT (Xu et al., ACL Findings 2024, [arXiv 2406.02864](https://arxiv.org/abs/2406.02864)) asks whether models can
convert numerals and units. We ran it on the 1.3B matched trio (Transformer `llama_1.8B_paper` 60k, AU-Net
`aunet2_1.3B` 180k, BPEByte `bpebyte_br_greedy_root_1.3B` 180k, paper ckpt md5 `3fbc9bbf…`) and on two external byte
models (BLT-1B, H-Net 1-stage XL).

Scripts are in `scripts/`. All tables are in `tables.md` (`scripts/analyze.py`). Per-item generations are in
`out/<model>.jsonl` and scored items in `scored/`.

## Setup

**Data.** We used the authors' repo (github.com/CAS-SIAT-ConsistencyAI/NUMCoT @ `437a60b6`). Its ChatGPT result CSVs
hold, for every condition, each question, its gold answer and the exact prompt. We kept only the English splits;
our models are English-only, so we dropped the Chinese splits and SuanJing. 4,557 questions:

| task | splits (n) |
|---|---|
| num2word: number → English words | easy 400 / medium 400 / hard fraction 200 / hard decimal 200 |
| word2num: English words → number | same four splits |
| units: unit conversion | easy / medium / hard, 200 each |
| MWP: math word problems | AddSub 395 / SingleOp 562 / MultiArith 600 |

**Conditions.** We used the paper's five: 0-shot, 0-shot CoT ("Let's think step by step"), few-shot (4 shots for
numerals; 3 for units and MWP), few-shot CoT, and few-shot CoT + unit knowledge (units only). That makes 18,828
generations per model.

**Prompts for base models.** The chat prompts are flattened into plain text: the system message, then each shot as
`<user text> <assistant text>`, with a blank line between shots. The prompt ends right after `Answer:` or
`step by step,`. Every model sees byte-identical prompts (`build_items.py`).

**Generation.** Greedy decoding stops at a newline or at the start of a next question (`Question`, `The following`).
The byte budget is fixed per split, never per item:
- without CoT: max(64, 1.5 × the split's longest gold + 32);
- with CoT: min(1024, 1.5 × the longest CoT demonstration).

Every output is cut to the same byte budget (`gen_items.py`). Backends:
- Trio: lingua generator + EvalHarnessLM. BPEByte uses its boundary-correct cached path without vocab_norm, with
  an eager block mask.
- BLT: official bytelatent + xformers, batch size 1, released threshold.
- H-Net: eval_suite harness, batch size 8.
- BLT and H-Net have no KV cache. The only change to their generation loops is an early stop once the stop string
  appears; the output is the same.

**Scoring.**
- `paper`: the repo's own scoring, ported. It matches the authors' stored ChatGPT scores on 18,828 of 18,828 items.
  It counts an item correct if the gold appears anywhere in the response.
- `strict`: only the first answer counts. That is the text after the last "answer is" if present, otherwise the
  whole response. For num2word, the normalized words must equal the gold. For the numeric tasks, the first number
  must equal the gold (units hard: the first two numbers).

  For 1.3B base models, `paper` gives false credit. For example, gold "eleven" is scored correct inside the wrong
  answer "One hundred and eleven", and in CoT any intermediate number counts. **The conclusions below use `strict`.**
- CIs come from an item bootstrap (B=10,000). Model differences are paired by item.

**External models: reduced set.** The full protocol would take tens to hundreds of GPU hours without a KV cache.
`items_ext.jsonl` (`make_ext_subset.py`) holds 5,407 generations:
- few-shot on every question (4,557);
- few-shot CoT and CoT + knowledge on 50 random questions per split (850);
- no 0-shot conditions.

The trio also has outputs for these keys, so section B of `tables.md` compares all five on them. In this subset,
250 few-shot-CoT items (5 numeral splits × 50) exceed BLT's 4,096-byte context, so BLT sees their prompts with the
start truncated.

**Hardware.** snu188 (RTX A5000) GPU0 and GPU1, in `/mnt/ssd2/hyunw3/AUNet`.

| run | items | time |
|---|---|---|
| Transformer | full | 3.7 h |
| BPEByte | full | 10.2 h |
| AU-Net | full | 9.5 h |
| H-Net | reduced | 7.1 h |
| BLT | reduced | 10.3 h |

## Results (strict, %, 95% CI)

**All five models on the shared subset** (few-shot n = full split; CoT n = 50 per split)

| task | cond | n | Transformer | AU-Net | BPEByte | BLT† | H-Net† |
|---|---|---|---|---|---|---|---|
| num2word | few-shot | 1200 | 0.3 | 3.7 | 7.8 [6.3, 9.4] | **16.8** | 1.5 |
| num2word | few-shot CoT | 200 | 0.0 | 0.0 | 0.0 | 3.0 | 0.0 |
| word2num | few-shot | 1200 | 0.1 | 1.1 | 2.8 [1.9, 3.8] | **7.2** | 0.8 |
| word2num | few-shot CoT | 200 | 1.5 | 4.0 | 1.0 | **11.5** | 0.0 |
| units | few-shot | 600 | 1.3 | 3.7 | 1.2 | 3.5 | 0.5 |
| units | few-shot CoT | 150 | 0.7 | 2.0 | 2.7 | 4.0 | 10.0 |
| units | CoT + knowledge | 150 | 0.0 | 0.7 | 3.3 | 2.0 | 4.0 |
| MWP | few-shot | 1557 | 7.4 [6.1, 8.7] | 3.1 | 2.6 [1.8, 3.3] | **23.4** | 2.4 |
| MWP | few-shot CoT | 150 | 8.0 | 6.7 | 5.3 | 13.3 | 4.0 |

† External models, trained on different data and compute. They are not matched to the trio.

**Trio, full protocol** (n = 1,200 / 1,200 / 600 / 1,557 per condition; tables.md section A)

| task | cond | Transformer | AU-Net | BPEByte | BPEByte − Transformer |
|---|---|---|---|---|---|
| num2word | few-shot | 0.3 | 3.7 | 7.8 | **+7.5 [+6.0, +9.1]** |
| word2num | 0-shot | 1.0 | 0.1 | 3.9 | +2.9 [+1.8, +4.0] |
| word2num | few-shot | 0.1 | 1.1 | 2.8 | +2.8 [+1.8, +3.8] |
| word2num | few-shot CoT | 0.6 | 4.7 | 1.3 | +0.8 [+0.0, +1.5] |
| units | few-shot | 1.3 | 3.7 | 1.2 | −0.2 [−1.0, +0.7] |
| units | few-shot CoT | 1.0 | 2.3 | 2.0 | +1.0 [−0.3, +2.3] |
| units | CoT + knowledge | 0.2 | 0.2 | 3.7 | +3.5 [+2.0, +5.2] |
| MWP | 0-shot CoT | 2.1 | 1.2 | 0.4 | −1.6 [−2.2, −1.0] |
| MWP | few-shot | 7.4 | 3.1 | 2.6 | **−4.8 [−6.2, −3.5]** |
| MWP | few-shot CoT | 6.4 | 5.2 | 4.2 | −2.1 [−3.5, −0.8] |

All num2word, units 0-shot, and units 0-shot CoT cells not listed here are 0.0–0.1% for every model.

## Findings

1. **All 1B models are near the floor.** Strict accuracy is mostly under 10%. BLT on MWP few-shot (23.4%) is the
   only exception. Hard splits (fractions, decimals, units hard) are about 0% for every model. As in the paper,
   accuracy falls as difficulty rises: num2word easy → medium → hard is 16.5 → 7.0 → 0 for BPEByte.

2. **Numeral conversion: BPEByte > AU-Net > Transformer, all significant.**
   - num2word few-shot: +7.5 pp over the Transformer and +4.2 pp over AU-Net. word2num few-shot: +2.8 and +1.8 pp.
   - **The Transformer copies the demonstrations.** All four demonstration answers start with "One hundred
     twenty-three…". The Transformer starts 99.8% of its answers with "One", so its first word is right only 10.8%
     of the time, which is exactly the share of golds that start with "one". 40.7% of its answers copy
     "twenty-three".
   - The byte models read the query's leading digit. Their first word is right 85.1% (AU-Net) and 88.2%
     (BPEByte) of the time, and only 6.2% of BPEByte's answers copy "twenty-three" (tables.md section C).
   - So part of this gap is how the models interact with NUMCoT's demonstration design (every shot starts with
     123…). It does not show that the Transformer cannot convert numbers at all.

3. **MWP: the Transformer is best (few-shot −4.8 pp vs BPEByte).** This matches the earlier NumericBench and
   ICL-demonstration results, where BPEByte was weaker. With few-shot CoT the gap shrinks to −2.1 pp. BPEByte and
   AU-Net are statistically tied.

4. **CoT effects partly reproduce the paper's direction.** Under `paper` scoring, adding CoT to num2word lowers
   accuracy for every model, as the paper found:

   | model | num2word few-shot → few-shot CoT |
   |---|---|
   | BPEByte | 9.1 → 0.2 |
   | AU-Net | 5.5 → 1.2 |
   | Transformer | 2.2 → 0.8 |

   For units and MWP, CoT raises accuracy. For example, MWP few-shot → few-shot CoT goes from 7.5 to 14.3 for the
   Transformer and from 2.5 to 10.0 for BPEByte. Under `strict` the gain shrinks a lot, because CoT outputs often
   contain the gold number only as an intermediate step. The Transformer's MWP gain disappears (7.4 → 6.4), while the
   byte models keep a small one (AU-Net 3.1 → 5.2, BPEByte 2.6 → 4.2).

5. **External models:**
   - BLT is best on numerals and MWP: num2word 16.8, word2num 7.2, MWP 23.4. It still starts 62.4% of num2word
     answers with "One", so its demonstration copying is weaker than the Transformer's but present.
   - H-Net is below BPEByte on num2word, word2num and MWP. Its only high cell is units few-shot CoT (10.0%, n=150,
     CI [5.3, 15.3]), and some of those hits are accidental: wrong reasoning that lands on the right number, or a
     copy of a demonstration question.
   - BPEByte vs BLT is −8.9 pp on num2word and −20.9 pp on MWP. BPEByte beats H-Net on num2word by +6.3 pp.

## Caveats

- **Prompt adaptation.** The paper's prompts are written for chat models. For base models we flattened them and
  added stop strings and fixed byte budgets. 0-shot outputs mostly ramble until the budget runs out, so the 0-shot
  conditions carry little information.
- **Small CoT cells for the external models.** The CoT cells have n = 150–200, so their CIs are wide.
- **BLT context.** BLT's 4,096-byte context left-truncates 250 CoT prompts.
- **Stop strings.** `Question` and `The following` are our additions. They cut off the next question a base model
  writes after its answer.
- **Environment.** snu188 used torch 2.8 cu126 on the conda env `aunet_leak`. The trio was not compared item by item
  against its NoLiMa/S-NIAH environment. The weights are the paper ckpts with optimizer state stripped; tensors were
  checked equal to the originals.

## Reproduce

```
python scripts/build_items.py <NUMCoT repo @437a60b6>    # items.jsonl (18,828)
python scripts/make_ext_subset.py 50                     # items_ext.jsonl (5,407)
# on snu188: bash scripts/queue_snu188.sh <gpu> 0 llama aunet bpebyte   /   ... blt hnet
python scripts/analyze.py                                # scored/, tables.md
```

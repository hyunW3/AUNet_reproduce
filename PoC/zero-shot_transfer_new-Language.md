# Zero-shot language transfer by swapping ONLY the patch tokenizer (no training)

**Question.** Can a byte-level BPEByte model (trained on English DCLM, patch boundaries from the
llama3 tokenizer) gain zero-shot ability in a *new* language purely by swapping the eval-time
patch-boundary tokenizer — **no weight updates**? And which language is the cleanest test bed?

**TL;DR — use Telugu.** It is the only language that satisfies all three requirements *and* is
technically clean with the greedy byte-trie: llama3 has **zero** Telugu vocab (pure byte-fallback),
a knowledge-free 2-choice benchmark exists (`xstorycloze_te`), and a byte-level BPE tokenizer that
covers Telugu is available (**BLOOM**, 6,694 Telugu tokens).

---

## 1. Why a *zero-vocab* language is the right test

Prior result (see `chinese-tokenizer-swap-eval` / `reports/`): swapping llama3→Qwen2 on **Chinese**
*hurt* BPB (+0.21). But that was confounded — llama3 already covers Chinese well (char-level patches),
so the model trained on meaningful Chinese segmentation and the swap **broke** it (train/eval mismatch).

For a language where **llama3 has no vocab at all**, the situation is different and cleaner:
- Under llama3 the language is pure **byte-fallback** → patches are byte-level noise; the model never
  learned any useful segmentation for it. There is **no good training segmentation to break.**
- Swapping to a real tokenizer gives genuine **word/morpheme patches** → maximal granularity contrast
  (byte-level → word-level).

So a zero-vocab language is the *fairest* and *sharpest* test of the swap hypothesis. Residual caveat:
the model's pooling layers were tuned to llama3's boundary **distribution** (mostly English word
patches), so the new tokenizer's boundaries are still somewhat out-of-distribution for pooling — the
swap may still not help. But it is no longer confounded by a broken pre-existing segmentation.

## 2. Requirements

1. **Benchmark** for the language available in the installed `lm_eval`.
2. **Tokenizer** for the language — must be **byte-level BPE** (ByteLevel decoder, like Qwen/BLOOM) so
   the greedy byte-trie recovers correct raw token bytes. A SentencePiece/`▁`-style tokenizer would
   mismatch raw input bytes (space vs `▁`) and degenerate to byte-level — do NOT use.
3. **llama3 has no vocab** for the language (pure byte-fallback → maximal contrast).

## 3. Evidence

### 3a. llama3 dedicated-vocab tokens per script (0 = pure byte-fallback)
```
Telugu 0   Kannada 0   Sinhala 0   Gujarati 0   Punjabi 0   Oriya 0
Myanmar(Burmese) 0   Lao 0   Georgian 0   Amharic 0   Tibetan 0   Mongolian 0
Malayalam 1   Khmer 1   Armenian 1   Tamil 3   Bengali 6   Hebrew 20
```
(For contrast, llama3 covered scripts: Latin 96,909; Cyrillic 6,510; CJK 4,387; Arabic 3,782;
Hangul 2,246; Thai 1,391; Devanagari 1,012.)

### 3b. Benchmark availability (`lm_eval`)
- `xstorycloze`: langs incl. **te (Telugu)**, **my (Burmese)**, eu, sw, hi, id … — 2-choice
  commonsense, chance 50%, knowledge-free (best for detecting transfer).
- `xcopa`: incl. **ta (Tamil)** — 2-choice.
- `belebele`: **122 languages** incl. `tel_Telu`, `kan_Knda`, `mya_Mymr`, `amh_Ethi`, `guj_Gujr`,
  `kat_Geor`, `khm_Khmr`, `lao_Laoo`, `ory_Orya`, `pan_Guru`, `sin_Sinh`, `tam_Taml` — 4-choice
  reading comprehension (chance 25%).
- `xnli`: no strictly-zero-vocab language (its non-Latin langs — ar, el, ru, th, ur, zh — are all covered).

### 3c. Tokenizer (BLOOM `tokenizer.json`, fetched to `tokenizer/bloom/`)
- `model.type = BPE`, `decoder.type = ByteLevel` → **trie-compatible** (same class as Qwen/polyglot).
- vocab 250,680. Script coverage: **Telugu 6,694**, Kannada 6,489, Tamil 6,216, **Burmese 2 (not covered)**.

## 4. Recommendation

| language | llama3 vocab | benchmark | byte-level tokenizer | verdict |
|----------|-------------:|-----------|----------------------|---------|
| **Telugu (te)** | **0** | `xstorycloze_te` (2-choice) + `belebele_tel_Telu` | **BLOOM** (6,694) | ★ best — all reqs + commonsense task |
| Kannada (kn) | 0 | `belebele_kan_Knda` only | BLOOM (6,489) | good backup; no commonsense task |
| Tamil (ta) | 3 (near-zero) | `xcopa_ta` + `belebele_tam_Taml` | BLOOM (6,216) | not *strictly* zero |
| Burmese (my) | 0 | `xstorycloze_my` + `belebele_mya_Mymr` | **none in BLOOM** (2) | appealing (zero + no-whitespace) but needs a Burmese byte-level tokenizer |

**Pick Telugu.** Runner-up Burmese is scientifically attractive (zero vocab **and** no word spacing →
maximizes the tokenizer's role, like Chinese) but blocked on tokenizer availability.

## 5. Experiment setup (two arms, no training)

Model: `bpebyte_br_greedy_root_1.3B/checkpoints/0000180000` (or, later, the Chinese-continue ckpts).
Both arms use the leak-free greedy+root scheme; only the patch tokenizer differs.

- **Arm A (baseline)** — llama3 patching (Telugu = byte-fallback): no override.
- **Arm B (swap)** — BLOOM patching: `regex_bpe_tokenizer_path=tokenizer/bloom/tokenizer.json
  force_bpe_online_mode=greedy` (the `force_*` is REQUIRED — a bare tokenizer override runs offline/leaky).

Metrics (before-vs-after style, both arms):
- **BPB on a Telugu corpus** (headline; per-byte likelihood, sensitive) — build a wiki-te `*.val.jsonl`
  like `data/zh_bpb/`, run via `eval_bpb_zh.yaml` pattern.
- **`xstorycloze_te`** (2-choice, chance 50%) and **`belebele_tel_Telu`** (4-choice, chance 25%),
  0-shot, mirroring `run_zh_easy.sh` / the ceval/cmmlu runners.

Success = Arm B (BLOOM) meaningfully **below** Arm A on Telugu BPB and/or **above** it on the tasks.
Given §1's caveat, a null is still informative: it would show that even under maximal granularity
contrast, an eval-only tokenizer swap cannot induce transfer without training.

## 6. RESULT (2026-07-06, ran on ece-agpu11, A100)
Telugu BPB, 250 wiki-te docs, leak-free greedy+root, llama3 (byte-fallback) vs BLOOM (word patches):

| metric | llama3 | BLOOM | Δ (BLOOM−llama3) |
|--------|-------:|------:|-----------------:|
| **BPB** (bits/byte) | **1.235** | 1.709 | **+0.474 worse** |
| bits/char | 3.25 | 4.49 | +1.24 worse |

**The swap HURTS — more than Chinese did (+0.47 vs +0.21). Hypothesis REFUTED.** Zero-vocab is NOT
where the swap helps. The model is bound to llama3's boundary *distribution* (English word-patches ~4 B),
not just its vocab: llama3-on-Telugu = fine byte-fallback patches → degrades gracefully toward a pure-byte
model; BLOOM = coarse word patches → forces the trunk to compress spans it never learned → more OOD → worse.
Consistent granularity law across all 3 runs: the further the swap moves patches from the trained ~4-B
granularity, the worse BPB (Chinese char→word +0.21; Telugu byte→word +0.47). An eval-time tokenizer swap
cannot induce transfer on a fixed ckpt; larger contrast = larger penalty. Only training-in the boundaries
(continue-pretrain, `plans/chinese_continue_pretrain_plan.md`) can help.

Notes for reproduction on ece: needed `uv pip install tokenizers` in the venv (HF loader); patch BOTH
`params.json` + `consolidated/params.json` (empty data.sources) or BPB's eval_on_val crashes on the baked
DCLM path (0-chunks ZeroDivisionError); keep BPB (validation) DECOUPLED from the downstream harness (a late
crash nukes 3 h of harness work). Downstream DONE (decoupled, limit50): xstorycloze_te (2-ch, chance 50%) llama3 52% / BLOOM 50%;
belebele_tel_Telu (4-ch, chance 25%) llama3 20% / BLOOM 20% — both arms AT CHANCE, swap changes nothing
downstream (only BPB is sensitive, and it got worse). Confirms: no eval-side lever helps a model with no
Telugu knowledge; the swap hurts BPB and is neutral-at-chance downstream.
- Related: `plans/chinese_continue_pretrain_plan.md`, memory `chinese-tokenizer-swap-eval`.

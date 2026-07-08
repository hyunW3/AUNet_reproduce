# Tokenizer-swap on a fixed checkpoint — all results

**Question.** A BPEByte model is byte-in/byte-out; the reference tokenizer only sets *where patches
pool*. Can we change a trained model's behavior — or transfer to a new domain/language — by swapping
ONLY the patch-boundary tokenizer at eval, with **no weight updates**?

**TL;DR — No. The swap ALWAYS hurts**, across every domain tested (Chinese, Telugu, code, English
reasoning). The cause is not tokenizer coverage or average granularity: the model is bound to llama3's
**specific boundary-position distribution**, and any other tokenizer places patches differently → the
trunk sees out-of-distribution pooling → loss/accuracy degrade. Magnitude tracks how far the boundary
distribution moves.

---

## Setup
- **Model.** `bpebyte_br_greedy_root_1.3B` @180k (DCLM/English, leak-free greedy+root). Byte-in/out.
- **Swap arms.** Same checkpoint, eval-time patch tokenizer overridden via
  `regex_bpe_tokenizer_path=<tok> force_bpe_online_mode=greedy` (the `force_*` is REQUIRED — a bare
  override runs offline/leaky).
- **Subword reference.** `llama_1.8B_paper` (subword; its tokenizer CANNOT be swapped — embeddings are
  bound to the vocab). Iso-byte (~283 GB) with the byte model; NOT iso-parameter (1.3B vs 1.8B).
- **BPB** = |nll|/bytes/ln2 (lower better). **acc / acc_norm** in % (higher better; acc_norm = length-normalized).
- Ran on ece-agpu11 (A100). Full record in memory `chinese-tokenizer-swap-eval`.

---

## 1. Chinese — llama3 → Qwen2 (Qwen2 = byte-level BPE, 151k, Chinese-dense)
| metric | llama3 | Qwen2 | Δ |
|--------|-------:|------:|---:|
| wiki-zh **BPB** | **2.679** | 2.891 | **+0.212 worse** |
| DCLM-EN ref BPB | 0.828 | 0.876 | +0.047 |
| ceval letter (acc) | 26.4 | 27.3 | ~chance(25) |
| cmmlu letter (acc) | 25.3 | 25.2 | ~chance |
| cmmlu cloze (acc_norm) | 27.2 | 27.2 | ~chance |
| ceval cloze (acc_norm) | 26.0 | 26.9 | ~chance |
| xcopa_zh (2-ch, 50%) | 50.0 | 48.6 | ~chance |
| xstorycloze_zh (2-ch, 50%) | 49.4 | 46.6 | ~chance |

Swap hurts BPB; all downstream at chance (model has ~no Chinese — DCLM is English). Cloze vs letter
didn't help (no latent knowledge to recover).

## 2. Telugu — llama3 → BLOOM (zero-vocab language; BLOOM = byte-level BPE, 250k)
llama3 has **0** Telugu tokens (pure byte-fallback) → the sharpest possible test.
| metric | llama3 | BLOOM | Δ |
|--------|-------:|------:|---:|
| wiki-te **BPB** | **1.235** | 1.709 | **+0.474 worse** |
| wiki-te bits/char | 3.25 | 4.49 | +1.24 |
| xstorycloze_te (2-ch, 50%) | 52 | 50 | ~chance |
| belebele_tel (4-ch, 25%) | 20 | 20 | ~chance |

**Hypothesis "zero-vocab is where the swap helps" REFUTED** — swap hurts MORE than Chinese. llama3
byte-fallback degrades gracefully to a near-pure-byte model; BLOOM's coarse word patches are more OOD.

## 3. Code — llama3 → StarCoder2 (SC2 = byte-level BPE, 49k, code-aware)
| metric | byte-llama3 | byte-SC2 | subword | Δ swap |
|--------|:-----------:|:--------:|:-------:|:------:|
| code-BPB Python (250 docs) | 0.808 | 1.145 | **0.786** | +0.337 |
| **code-BPB multi-lang** (8 langs, 95% CI) | 0.994 [.96,1.03] | 1.331 [1.29,1.38] | **0.935** [.90,.97] | **+0.337** [.32,.36] |
| MMLU-CS letter (acc) | 24.9 | 24.2 | 24.8 | ~chance |
| MMLU-CS cloze (acc_norm) | **31.9** | 30.4 | 29.6 | small |
| MMLU 57-subj cloze (acc_norm) | 30.7 | 30.9 | 29.9 | small |
| **HumanEval-MC** cloze (acc_norm) | **56.7** | 48.8 | 54.3 | **−7.9** |

- Multi-lang swap **+0.337** [.32,.36] (paired, significant); ordering `subword<byte-l3<byte-SC2` in all 8 langs.
- subword beats byte on BPB by **+0.059** [.05,.07] (significant).
- **HumanEval-MC** (correct-vs-distractor-solution likelihood) is the ONLY accuracy metric to clear chance
  (~50-57% vs 25%) — swap hurts here too (byte-SC2 lowest); byte-l3≈subword on discrimination.
- **pass@k NOT run** — byte-model generation O(n²)/timeout (HumanEval deliberately skipped in repo pipeline).
- Cloze lifts MMLU off the letter-format chance floor (24-25%→~30-32%) → letter floor was partly a
  symbol-binding artifact, but even cloze can't separate the arms (only BPB does).

## 4. English reasoning (home turf) — llama3 → Gemma (Gemma = SentencePiece, 256k)
Full sets, acc_norm %:
| task | byte-llama3 | byte-gemma | Δ swap | subword |
|------|:-----------:|:----------:|:------:|:-------:|
| HellaSwag (10k) | **62.4** | 57.8 | **−4.6** | 62.2 |
| ARC-Easy (2376) | **66.9** | 64.4 | −2.5 | 65.6 |
| ARC-Challenge (1172) | **37.5** | 36.3 | −1.2 | 35.2 |
| PIQA (1838) | **74.1** | 73.1 | −1.0 | 74.9 |
| **avg** | **60.2** | 57.9 | **−2.3** | 59.5 |

**Swap hurts all 4 (HellaSwag −4.6 ≈ 9 stderr, significant). MOST STRIKING RESULT:** Gemma & llama3
segment English at the **SAME average granularity** (16 patches / 5.12 B/patch), yet the swap still
degrades 2-5 pts → it is NOT granularity; the model is bound to llama3's **specific boundary POSITIONS**.
byte-l3 ≈ subword on downstream (60.2 vs 59.5) — byte holds its own despite worse BPB.

---

## 5. Chinese & Telugu downstream in CLOZE (answer-text) format — does cloze rescue them?
Chinese (acc_norm %, chance in parens): cmmlu-cloze llama3 27.2 / qwen2 27.2 · ceval-cloze 26.0 / 26.9 ·
belebele-cloze (reading comp) **28.1 (acc 30.8)** / 27.0 · xstorycloze_zh(50) 49.4 / 46.6 · xcopa_zh(50) 50.0 / 48.6.
Telugu: belebele-cloze **25.6 (acc 28.0)** / bloom 24.4 · xstorycloze_te(50) 52.0 / 50.0.
**Finding: cloze does NOT rescue Chinese/Telugu (all ~chance) — unlike English MMLU-cloze which lifted off
the floor (24→31%).** No latent Chinese/Telugu ability to surface. ONE nuance: Chinese belebele-cloze reaches
30.8% raw acc (~4 stderr above 25% on ~900Q) — weak-but-real reading-comprehension signal (answer is in the
passage); Telugu stays flat at chance. Gap tracks coverage EXACTLY: llama3 has 4,387 CJK tokens (slight
Chinese processing) but ZERO Telugu (none). Swap neutral-to-negative on accuracy (consistent).

## Cross-cutting: the swap always hurts BPB, magnitude tracks boundary-distribution shift
| swap | domain | avg-granularity contrast | ΔBPB (or Δacc) |
|------|--------|--------------------------|----------------|
| llama3→Qwen2 | Chinese | moderate (char→word) | +0.21 BPB |
| llama3→StarCoder2 | code | small avg, big whitespace shift | +0.34 BPB |
| llama3→BLOOM | Telugu | huge (byte-fallback→word) | +0.47 BPB |
| llama3→Gemma | English | ~zero (same rate) | −2.3 acc_norm downstream (still hurts) |

**Mechanistic conclusion.** The penalty is boundary-**distribution** binding, not granularity or coverage.
Even at identical average granularity (Gemma/English), moving *where* the boundaries fall degrades the
model. An eval-time tokenizer swap cannot improve or re-target a fixed checkpoint; only training/continue-
pretraining with the new boundaries could (see `plans/chinese_continue_pretrain_plan.md`).

**Byte vs subword (secondary).** subword compresses better (BPB, all domains); byte ≈ subword on task
accuracy (downstream, HumanEval-MC). The two axes disagree on byte-vs-subword.

---

## Reference: llama3 tokenizer coverage (128,256 vocab, tokens containing each script)
Latin 96,909 · Cyrillic 6,510 · Latin-ext 5,205 · **CJK 4,387** · Arabic 3,782 · **Hangul 2,246** ·
Thai 1,391 · Greek 1,376 · Devanagari 1,012 · JP-kana 1,082. **Zero-vocab (byte-fallback only):** Telugu,
Kannada, Sinhala, Gujarati, Punjabi, Oriya, Burmese, Lao, Georgian, Amharic, Tibetan, Mongolian.
Korean: llama3 (4.73 B/tok) ≈ polyglot-ko specialist (4.92 B/tok) — near-parity, weak swap lever.

## Byte-trie tokenizer support added this investigation (`lingua/lingua/tokenizer.py`)
`HFOffsetTokenizer.iter_token_bytes` now recovers RAW token bytes for all 3 families:
- **byte-level BPE** (GPT-2 style: Qwen2, BLOOM, StarCoder2, polyglot-ko) — invert gpt2 bytes↔unicode.
- **tiktoken** (llama3) — native `decode_single_token_bytes`.
- **SentencePiece metaspace** (Gemma) — `▁`→space, `<0xNN>`→raw byte, skip added_tokens (detect via `▁`
  in normalizer/decoder json, `ensure_ascii=False`).
Gotcha: a bare `regex_bpe_tokenizer_path` override runs OFFLINE/leaky → always pair with
`force_bpe_online_mode=greedy`. Tokenizers at `tokenizer/{qwen2,bloom,starcoder2,gemma}/`.

*Related: `PoC/code_tokenizer_swap_results.md`, `PoC/zero-shot_transfer_new-Language.md`,
`plans/chinese_continue_pretrain_plan.md`, memory `chinese-tokenizer-swap-eval`.*

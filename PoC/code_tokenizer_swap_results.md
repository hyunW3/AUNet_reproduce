# Tokenizer-swap on CODE — BPEByte vs subword, and llama3→StarCoder2 patch swap

**Setup.** Model arms (all eval on identical bytes, ran on ece-agpu11 A100):
- `byte-llama3` — BPEByte root_greedy 1.3B, native llama3 patch boundaries.
- `byte-SC2` — same checkpoint, StarCoder2 patch boundaries at eval (`force_bpe_online_mode=greedy`).
- `subword` — Llama 1.8B subword baseline (llama3 tokenizer; its tokenizer CANNOT be swapped — embeddings are bound to the vocab).
Byte arms are iso-byte (~283 GB) with the Llama; NOT iso-parameter (1.3B vs 1.8B).

## A. BPB — bits/byte (lower = better)
| benchmark | byte-llama3 | byte-SC2 | subword | best |
|-----------|:-----------:|:--------:|:-------:|:----:|
| code-BPB Python (codeparrot, 250 docs) | 0.808 | 1.145 | **0.786** | subword |
| **code-BPB multi-lang** (8 langs×40, rosetta, 95% CI) | 0.994 [.96,1.03] | 1.331 [1.29,1.38] | **0.935** [.90,.97] | subword |

Paired deltas (multi-lang, B=10k, significant): **StarCoder2 swap +0.337** [.320,.356] worse; **subword beats byte +0.059** [.051,.066]. Per-language ordering `subword < byte-llama3 < byte-SC2` holds in all 8 languages (Java easiest 0.66, Ruby hardest 1.26).

## B. MC-likelihood accuracy — % (higher = better; chance = 25%)
| benchmark (metric) | byte-llama3 | byte-SC2 | subword |
|--------------------|:-----------:|:--------:|:-------:|
| MMLU-CS **letter** (acc) | 24.9 | 24.2 | 24.8 |
| MMLU-CS **cloze** (acc_norm) | **31.9** | 30.4 | 29.6 |
| MMLU 57-subj cloze (acc_norm) | 30.7 | **30.9** | 29.9 |
| **HumanEval-MC** cloze (acc_norm) | **56.7** | 48.8 | 54.3 |
| HumanEval-MC cloze (raw acc) | 37.2 | 32.3 | **41.5** |

## C. pass@k (generation + execution) — NOT RUN
Infeasible for the byte model: O(n²) re-prefill (~30–120 min), and HumanEval was deliberately skipped in the repo's own pipeline (`llama18_eval_then_chain.sh`). All code-gen benchmarks (HumanEval/MBPP/cruxeval) are generate_until.

## Conclusions
1. **BPB → subword wins, StarCoder2 swap hurts** — both statistically firm (CIs). The only metric that cleanly ranks all three.
2. **Knowledge/letter-MC is dead** (MMLU-CS letter at chance for all). **Cloze lifts MMLU off the floor** (letter 24-25% → cloze acc_norm ~30-32%) but still doesn't separate arms → the letter floor was partly a symbol-binding artifact, not signal.
3. **HumanEval-MC is the only accuracy metric with real signal** — clears chance (~50-57% acc_norm), and the **swap hurts here too** (byte-SC2 lowest on both metrics), consistent with BPB. But **byte-llama3 ≈ subword** on this code-discrimination task even though subword won on code compression (BPB) → BPB and discrimination disagree on byte-vs-subword.
4. **Two consistent threads:** (a) the StarCoder2 swap ALWAYS hurts (BPB + HumanEval-MC agree); (b) byte-vs-subword depends on the axis (subword compresses better; comparable on discrimination).

Caveats: HumanEval-MC N=164 (unpaired CIs overlap; paired bootstrap via log_samples would firm the swap penalty); code corpora modest; pass@k (byte-model home turf) untested.

## HumanEval-MC construction (for reproducibility)
`data/humaneval_mc/` (parquet, seed 42): 164 HumanEval problems as 4-choice likelihood — context = prompt (signature+docstring), choices = correct canonical_solution + 3 random other-problem solution bodies (shuffled), score LL of body given prompt (`target_delimiter=""`). Task `eval_tasks/humaneval_mc/`.

## D. English downstream swap — llama3 → Gemma patches (acc_norm %, higher = better)
BPEByte 1.3B, llama3 vs Gemma patch boundaries, + Llama-1.8B subword reference. Full sets.
| task | byte-llama3 | byte-gemma | Δ swap | subword |
|------|:-----------:|:----------:|:------:|:-------:|
| HellaSwag (10k) | **62.4** | 57.8 | **−4.6** | 62.2 |
| ARC-Easy (2376) | **66.9** | 64.4 | −2.5 | 65.6 |
| ARC-Challenge (1172) | **37.5** | 36.3 | −1.2 | 35.2 |
| PIQA (1838) | **74.1** | 73.1 | −1.0 | 74.9 |
| **avg** | **60.2** | 57.9 | **−2.3** | 59.5 |

**Gemma swap HURTS English reasoning on all 4 tasks (avg −2.3 acc_norm, HellaSwag −4.6 ≈ 9 stderr, significant).**
STRIKING: Gemma & llama3 segment English at the SAME avg granularity (16 patches / 5.12 B/patch), yet the swap
still degrades 2-5 pts → it's NOT granularity, it's that the model is bound to llama3's SPECIFIC boundary
POSITIONS; Gemma puts boundaries in different places at the same rate. Strongest evidence for boundary-
distribution binding — the swap hurts even on the model's home turf (English) where granularity contrast ≈ 0.
byte-llama3 ≈ subword on downstream (60.2 vs 59.5; byte wins ARC, subword wins PIQA) — byte holds its own
despite worse BPB. NOTE: Gemma is SentencePiece (▁ metaspace + <0xNN> byte-fallback), NOT byte-level — needed
a new metaspace branch in HFOffsetTokenizer.iter_token_bytes (▁→space, <0xNN>→byte, skip specials).

*Related: `PoC/zero-shot_transfer_new-Language.md` (Chinese/Telugu language swaps), memory `chinese-tokenizer-swap-eval`.*

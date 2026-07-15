# 760M matched-scale results

All models 760M, same DCLM data + matched compute budget (byte models ~29.2k steps; llama ~12.9k steps, iso-text). **rg** = BPEByte root_greedy (current code); **aunet2** = original AU-Net (word-split, `regex.strategy word1:1@1`); **llama** = subword baseline.

## Bits-per-byte (BPB)

Byte models: BPB = train `loss/out` / ln2 (per-byte). **llama** is subword (per-token loss), converted with the measured **4.5376 bytes/token** for the llama3 tokenizer on DCLM: BPB = loss/(ln2·bytes_per_token) = 2.9044/(ln2·4.5376).

| model | BPB |
|---|---:|
| rg | 0.9256 |
| aunet2 | 0.9284 |
| llama | 0.9234 |

_Ranking: llama 0.9234 < rg 0.9256 < aunet2 0.9284._

## Downstream (0-shot)

| metric | rg | aunet2 | llama |
|---|---:|---:|---:|
| HellaSwag (acc_norm) | 0.4675 | 0.4680 | 0.4895 |
| ARC-Easy (acc_norm) | 0.5060 | 0.5285 | 0.5545 |
| ARC-Challenge 0-shot (acc_norm) | 0.2901 | 0.2944 | 0.2952 |
| BoolQ (acc) | 0.4547 | 0.4988 | 0.5841 |
| PIQA (acc_norm) | 0.6899 | 0.6899 | 0.7100 |
| WinoGrande (acc) | 0.5462 | 0.5556 | 0.5399 |
| MMLU (acc) | 0.2314 | 0.2303 | 0.2501 |
| CUTE avg (exact_match) | 0.1544 | 0.1538 | 0.1335 |
| HellaSwag-Noise avg (acc_norm) | 0.3512 | 0.3495 | 0.3394 |
| HellaSwag-Typo avg (acc_norm) | 0.4213 | 0.4199 | 0.4420 |
| BoolQ-Typo avg (acc) | 0.4219 | 0.4846 | 0.5419 |
| PBP cut-point ΔBPC (→0 ideal) | -0.0001 | 0.0010 | 0.5461 |
| PBP-MC ΔAcc (→0 ideal) | 0.0007 | 0.0003 | -0.0611 |

_**BoolQ + WinoGrande** are full-set 0-shot from `evals_5bench` (BoolQ 2510 docs, WinoGrande 1267). Byte models (rg/aunet2) evaluated in their native online-greedy leak-free regime (no tokenizer override). BoolQ ranking: llama 0.5841 ≫ aunet2 0.4988 > rg 0.4547 — subword prior helps this yes/no task; WinoGrande is near-chance (0.54–0.56) for all at this scale._

## ARC-Challenge few-shot (acc_norm)

| shots | rg | aunet2 | llama |
|---|---:|---:|---:|
| 0-shot | 0.2901 | 0.2944 | 0.2952 |
| 3-shot | 0.3097 | 0.3174 | 0.3097 |
| 5-shot | 0.3046 | 0.3072 | 0.3114 |

## Notes
- **PBP** is the headline: byte models cut-invariant (ΔBPC≈0, ΔAcc≈0); llama subword ΔBPC 0.55 / ΔAcc −0.06 = the Prompt Boundary Problem.
- Subword leads clean MC at this scale; byte models lead CUTE (char-level) + HellaSwag-Noise (robustness).
- Few-shot adds a small lift (0→3-shot ≈ +1–2 pts), plateaus by 5-shot.

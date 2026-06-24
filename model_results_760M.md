# 760M matched-scale results

All models 760M, same DCLM data + matched compute budget (byte models ~29.2k steps; llama ~12.9k steps, iso-text). **OLD rg** = BPEByte root_greedy on the old code point (commit 4ed607e, the 1.3B code); **NEW rg** = BPEByte root_greedy on the current code; **aunet2** = original AU-Net (word-split, `regex.strategy word1:1@1`); **llama** = subword baseline.

## Bits-per-byte (BPB)

Byte models: BPB = train `loss/out` / ln2 (per-byte). **llama** is subword (per-token loss), converted with the measured **4.5376 bytes/token** for the llama3 tokenizer on DCLM: BPB = loss/(ln2·bytes_per_token) = 2.9044/(ln2·4.5376).

| model | BPB |
|---|---:|
| OLD rg | 0.9275 |
| NEW rg | 0.9256 |
| aunet2 | 0.9284 |
| llama | 0.9234 |

_Ranking: llama 0.9234 < NEW rg 0.9256 < OLD rg 0.9275 < aunet2 0.9284._

## Downstream (0-shot)

| metric | OLD rg | NEW rg | aunet2 | llama |
|---|---:|---:|---:|---:|
| HellaSwag (acc_norm) | 0.4515 | 0.4675 | 0.4680 | 0.4895 |
| ARC-Easy (acc_norm) | 0.5020 | 0.5060 | 0.5285 | 0.5545 |
| ARC-Challenge 0-shot (acc_norm) | 0.3097 | 0.2901 | 0.2944 | 0.2952 |
| BoolQ (acc) | 0.3870 | 0.4485 | 0.4915 | 0.5805 |
| PIQA (acc_norm) | 0.6866 | 0.6899 | 0.6899 | 0.7100 |
| WinoGrande (acc) | 0.5359 | 0.5462 | 0.5556 | 0.5399 |
| MMLU (acc) | 0.2564 | 0.2314 | 0.2303 | 0.2501 |
| CUTE avg (exact_match) | 0.1556 | 0.1544 | 0.1538 | 0.1335 |
| HellaSwag-Noise avg (acc_norm) | 0.3485 | 0.3512 | 0.3495 | 0.3394 |
| HellaSwag-Typo avg (acc_norm) | 0.4133 | 0.4213 | 0.4199 | 0.4420 |
| BoolQ-Typo avg (acc) | 0.3805 | 0.4219 | 0.4846 | 0.5419 |
| PBP cut-point ΔBPC (→0 ideal) | -0.0009 | -0.0001 | 0.0010 | 0.5461 |
| PBP-MC ΔAcc (→0 ideal) | 0.0009 | 0.0007 | 0.0003 | -0.0611 |

## BPEByte root_greedy: new code vs old code (Δ = new − old)

| metric | OLD | NEW | Δ |
|---|---:|---:|---:|
| BPB | 0.9275 | 0.9256 | -0.0019 |
| HellaSwag (acc_norm) | 0.4515 | 0.4675 | +0.0160 |
| ARC-Easy (acc_norm) | 0.5020 | 0.5060 | +0.0040 |
| ARC-Challenge 0-shot (acc_norm) | 0.3097 | 0.2901 | -0.0196 |
| BoolQ (acc) | 0.3870 | 0.4485 | +0.0615 |
| PIQA (acc_norm) | 0.6866 | 0.6899 | +0.0033 |
| WinoGrande (acc) | 0.5359 | 0.5462 | +0.0103 |
| MMLU (acc) | 0.2564 | 0.2314 | -0.0250 |
| CUTE avg (exact_match) | 0.1556 | 0.1544 | -0.0012 |
| HellaSwag-Noise avg (acc_norm) | 0.3485 | 0.3512 | +0.0028 |
| HellaSwag-Typo avg (acc_norm) | 0.4133 | 0.4213 | +0.0079 |
| BoolQ-Typo avg (acc) | 0.3805 | 0.4219 | +0.0414 |
| PBP cut-point ΔBPC (→0 ideal) | -0.0009 | -0.0001 | +0.0007 |
| PBP-MC ΔAcc (→0 ideal) | 0.0009 | 0.0007 | -0.0001 |

**Mean Δ (new−old):** core-7 MC = +0.0072; all downstream scores = +0.0092 (mean |Δ| = 0.0175). Both versions are leak-free (PBP ΔBPC/ΔAcc ≈ 0). Net: small win for new code, mostly redistribution (BoolQ ↑, MMLU/ARC-C ↓).

## ARC-Challenge few-shot (acc_norm)

Few-shot run for the three new-code models (OLD rg is 0-shot only: 0.3097).

| shots | NEW rg | aunet2 | llama |
|---|---:|---:|---:|
| 0-shot | 0.2901 | 0.2944 | 0.2952 |
| 3-shot | 0.3097 | 0.3174 | 0.3097 |
| 5-shot | 0.3046 | 0.3072 | 0.3114 |

## Notes
- **PBP** is the headline: byte models cut-invariant (ΔBPC≈0, ΔAcc≈0); llama subword ΔBPC 0.55 / ΔAcc −0.06 = the Prompt Boundary Problem.
- Subword leads clean MC at this scale; byte models lead CUTE (char-level) + HellaSwag-Noise (robustness).
- Few-shot adds a small lift (0→3-shot ≈ +1–2 pts), plateaus by 5-shot.


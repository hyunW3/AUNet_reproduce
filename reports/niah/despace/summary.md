# S-NIAH despace robustness (RULER, 1.3B)

All prompt spaces stripped (`"I have a boy" -> "Ihaveaboy"`); the number/UUID value has no internal spaces so it survives and must still be retrieved verbatim. **Headline metric = `exact_match`** (teacher-forced greedy == model emits the value verbatim), matching the clean NIAH reports; `tok_frac` (fraction of value units that are argmax; per-byte for byte models, per-subword for Llama — a diagnostic, not strictly cross-family comparable) shown in parentheses.

## S-NIAH-1 — repeated-noise haystack + 7-digit number

| family | exact clean | exact despace | Δexact | tok_frac clean→despace |
|---|---|---|---|---|
| subword (Llama) | 0.920 | **0.000** | -0.920 | 0.980→0.750 |
| AU-Net (word) | 1.000 | **0.980** | -0.020 | 1.000→0.997 |
| BPEByte rg | 1.000 | **0.560** | -0.440 | 1.000→0.937 |

## S-NIAH-2 — natural-essay (DCLM) haystack + 7-digit number

| family | exact clean | exact despace | Δexact | tok_frac clean→despace |
|---|---|---|---|---|
| subword (Llama) | 0.640 | **0.000** | -0.640 | 0.910→0.745 |
| AU-Net (word) | 1.000 | **0.200** | -0.800 | 1.000→0.874 |
| BPEByte rg | 1.000 | **0.680** | -0.320 | 1.000→0.954 |

## S-NIAH-3 — natural-essay (DCLM) haystack + UUID

| family | exact clean | exact despace | Δexact | tok_frac clean→despace |
|---|---|---|---|---|
| subword (Llama) | 0.180 | **0.000** | -0.180 | 0.959→0.953 |
| AU-Net (word) | 0.480 | **0.080** | -0.400 | 0.977→0.917 |
| BPEByte rg | 1.000 | **0.460** | -0.540 | 1.000→0.982 |

## Takeaway

- **BPEByte-rg is the most despace-robust for verbatim copy** (exact retained 0.46–0.68 across all three; best on the hard UUID S3), and is byte-perfect clean everywhere.
- **Llama collapses to exact 0.000 on all three** — its subword output cannot reproduce the exact token alignment of the value embedded in a despaced run-on context (tok_frac 0.75–0.95 shows it recovers most pieces, so the 0.000 is the all-or-nothing metric plus subword misalignment).
- **AU-Net (word) is haystack-dependent**: near-immune on the repetitive noise haystack (S1 0.98) but collapses on natural essays (S2 0.20, S3 0.08), where despacing turns the whole essay into 16-char word-blobs that bury the needle.
- Opposite of the MCQ despace ranking (where AU-Net-word was worst): verbatim byte-copy favors byte-output models; semantic reasoning favors byte-granular ones.

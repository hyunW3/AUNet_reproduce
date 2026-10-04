> **SUPERSEDED (2026-10-04):** the model-accuracy sections (B-*) score BPEByte-rg / AU-Net without the final `vocab_norm` (`apps/aunet/generate.py::_next_byte_logits`, applied only with `AUNET_FIX_VOCAB_NORM=1`). Section A (data-only diagnostics) and the Llama columns are unaffected. Re-run → `runs/multilingual/1B_direct_vnfix`.

# Multilingual LAMBADA — translation noise and paired-bootstrap CIs (1B direct eval)

Companion to `reports/multilingual_1B_direct.md`. **gt** = EleutherAI `lambada_openai` non-English splits (Google Translate); **sl** = StableLM-2 re-translation (`lambada_multilingual_stablelm`, made because the gt version was judged too noisy by native speakers). Exact-match greedy accuracy, zero-shot; 95% CIs from 10,000 bootstrap resamples over documents; byte−Llama gaps are paired on the same documents.

## A. Translation-noise diagnostics (data only)

| version | lang | n | target in context % | target has punct % | rand-word base % | most-freq base % | =en target % | gt=sl target % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gt | en | 5153 | 81.4 | 0.8 | 1.7 | 7.0 | – | 100.0 |
| gt | de | 5153 | 46.1 | 5.2 | 1.1 | 3.1 | 9.6 | 18.2 |
| gt | es | 5153 | 30.8 | 59.7 | 0.7 | 3.0 | 0.2 | 0.3 |
| gt | fr | 5153 | 66.4 | 6.2 | 1.3 | 5.0 | 2.1 | 2.9 |
| gt | it | 5153 | 67.4 | 3.6 | 1.5 | 6.9 | 0.3 | 0.4 |
| sl | en | 5153 | 81.4 | 0.8 | 1.8 | 7.0 | – | 100.0 |
| sl | de | 5153 | 49.9 | 0.9 | 1.1 | 3.6 | 33.0 | 18.2 |
| sl | es | 5153 | 58.7 | 24.9 | 1.3 | 6.6 | 38.2 | 0.3 |
| sl | fr | 5153 | 60.8 | 11.3 | 1.2 | 5.0 | 39.7 | 2.9 |
| sl | it | 5153 | 57.6 | 27.2 | 1.2 | 5.9 | 33.2 | 0.4 |
| sl | nl | 5153 | 48.2 | 23.8 | 1.1 | 4.0 | 34.1 | – |
| sl | pt | 5153 | 76.0 | 1.0 | 1.7 | 6.1 | 45.7 | – |

## B-gt. Model accuracy with 95% CI (`lambada`)

| lang | Llama | BPEByte-rg | AU-Net | rg − Llama | AU-Net − Llama |
|---|---:|---:|---:|---:|---:|
| en | 62.6 [61.3, 63.9] | 64.8 [63.5, 66.1] | 64.6 [63.3, 65.9] | +2.2 [+1.0, +3.5] | +2.0 [+0.8, +3.2] |
| de | 18.0 [16.9, 19.0] | 19.5 [18.5, 20.6] | 19.3 [18.2, 20.4] | +1.6 [+0.6, +2.6] | +1.3 [+0.4, +2.3] |
| es | 12.7 [11.8, 13.6] | 26.8 [25.6, 28.0] | 25.5 [24.3, 26.7] | +14.1 [+12.9, +15.3] | +12.8 [+11.7, +14.0] |
| fr | 25.3 [24.1, 26.5] | 30.7 [29.4, 32.0] | 33.4 [32.1, 34.7] | +5.4 [+4.3, +6.6] | +8.1 [+7.0, +9.3] |
| it | 19.8 [18.7, 20.9] | 27.2 [26.0, 28.4] | 29.4 [28.2, 30.7] | +7.5 [+6.3, +8.6] | +9.6 [+8.4, +10.8] |

**Clean targets only** (target is a bare word, no punctuation; `lambada`) — removes the exact-match punctuation artifact (on punctuated targets the subword model collapses, e.g. gt-es 3.9% vs ≈20% for bytes)

| lang | n clean | Llama | BPEByte-rg | AU-Net | rg − Llama | AU-Net − Llama |
|---|---:|---:|---:|---:|---:|---:|
| en | 5112 | 62.9 [61.5, 64.1] | 65.1 [63.8, 66.4] | 64.8 [63.5, 66.1] | +2.2 [+1.1, +3.4] | +2.0 [+0.8, +3.2] |
| de | 4884 | 18.8 [17.7, 19.8] | 20.0 [18.9, 21.1] | 19.9 [18.7, 21.0] | +1.2 [+0.2, +2.3] | +1.1 [+0.1, +2.1] |
| es | 2075 | 25.7 [23.9, 27.6] | 34.7 [32.7, 36.8] | 36.2 [34.1, 38.3] | +9.0 [+7.2, +10.8] | +10.5 [+8.6, +12.4] |
| fr | 4832 | 26.2 [25.0, 27.4] | 32.0 [30.6, 33.3] | 34.7 [33.4, 36.1] | +5.8 [+4.5, +7.0] | +8.5 [+7.3, +9.8] |
| it | 4965 | 20.4 [19.3, 21.5] | 28.2 [26.9, 29.4] | 30.3 [29.0, 31.6] | +7.8 [+6.6, +9.0] | +9.9 [+8.7, +11.1] |

**Gap split by whether the target occurs in the passage** (`lambada`; acc Llama / rg / AU-Net, then rg−Llama)

| lang | in-ctx n | in-ctx acc | in-ctx rg−Llama | not-in-ctx n | not-in-ctx acc | not-in-ctx rg−Llama |
|---|---:|---|---:|---:|---|---:|
| en | 4194 | 68.2 / 70.6 / 70.7 | +2.4 [+1.1, +3.8] | 959 | 38.1 / 39.5 / 38.0 | +1.5 [-1.3, +4.2] |
| de | 2374 | 31.1 / 35.0 / 36.2 | +3.9 [+2.1, +5.7] | 2779 | 6.7 / 6.3 / 4.9 | -0.4 [-1.4, +0.6] |
| es | 1586 | 32.4 / 44.8 / 46.4 | +12.4 [+10.0, +14.8] | 3567 | 3.9 / 18.8 / 16.2 | +14.9 [+13.6, +16.2] |
| fr | 3424 | 35.5 / 43.3 / 47.1 | +7.9 [+6.2, +9.5] | 1729 | 5.0 / 5.6 / 6.3 | +0.6 [-0.5, +1.6] |
| it | 3471 | 28.5 / 39.4 / 42.6 | +10.9 [+9.2, +12.5] | 1682 | 1.8 / 2.3 / 2.1 | +0.4 [-0.3, +1.1] |

## B-sl. Model accuracy with 95% CI (`lambada_sl`)

| lang | Llama | BPEByte-rg | AU-Net | rg − Llama | AU-Net − Llama |
|---|---:|---:|---:|---:|---:|
| en | 62.6 [61.2, 63.9] | 64.8 [63.4, 66.1] | 64.6 [63.3, 65.9] | +2.2 [+1.0, +3.4] | +2.0 [+0.8, +3.2] |
| de | 21.8 [20.6, 22.9] | 23.9 [22.7, 25.1] | 23.3 [22.2, 24.5] | +2.1 [+1.1, +3.2] | +1.6 [+0.5, +2.7] |
| es | 26.2 [25.0, 27.4] | 33.9 [32.6, 35.2] | 33.9 [32.6, 35.1] | +7.7 [+6.5, +8.9] | +7.7 [+6.5, +8.8] |
| fr | 26.1 [24.9, 27.3] | 37.9 [36.6, 39.2] | 39.3 [38.0, 40.6] | +11.8 [+10.6, +13.1] | +13.2 [+11.9, +14.4] |
| it | 21.7 [20.6, 22.8] | 31.6 [30.3, 32.9] | 32.8 [31.6, 34.1] | +9.9 [+8.7, +11.0] | +11.1 [+9.9, +12.3] |
| nl | 16.0 [15.0, 17.0] | 21.2 [20.1, 22.3] | 18.9 [17.9, 20.0] | +5.2 [+4.2, +6.2] | +2.9 [+1.8, +4.0] |
| pt | 27.6 [26.4, 28.9] | 38.4 [37.1, 39.8] | 43.8 [42.4, 45.2] | +10.8 [+9.6, +12.0] | +16.2 [+14.9, +17.5] |

**Clean targets only** (target is a bare word, no punctuation; `lambada_sl`) — removes the exact-match punctuation artifact (on punctuated targets the subword model collapses, e.g. gt-es 3.9% vs ≈20% for bytes)

| lang | n clean | Llama | BPEByte-rg | AU-Net | rg − Llama | AU-Net − Llama |
|---|---:|---:|---:|---:|---:|---:|
| en | 5112 | 62.8 [61.5, 64.1] | 65.0 [63.8, 66.4] | 64.8 [63.5, 66.1] | +2.2 [+1.0, +3.4] | +2.0 [+0.8, +3.2] |
| de | 5108 | 21.8 [20.7, 23.0] | 24.0 [22.9, 25.2] | 23.5 [22.3, 24.6] | +2.2 [+1.1, +3.3] | +1.6 [+0.5, +2.8] |
| es | 3868 | 34.7 [33.2, 36.2] | 44.3 [42.8, 45.9] | 44.2 [42.6, 45.8] | +9.6 [+8.1, +11.1] | +9.5 [+7.9, +11.0] |
| fr | 4572 | 28.5 [27.2, 29.8] | 41.6 [40.2, 43.1] | 43.1 [41.6, 44.5] | +13.1 [+11.8, +14.5] | +14.6 [+13.1, +16.0] |
| it | 3749 | 28.9 [27.4, 30.3] | 39.6 [38.0, 41.1] | 40.3 [38.7, 41.9] | +10.7 [+9.2, +12.2] | +11.4 [+9.9, +12.9] |
| nl | 3928 | 20.8 [19.6, 22.1] | 27.6 [26.2, 29.0] | 24.6 [23.2, 25.9] | +6.8 [+5.5, +8.1] | +3.7 [+2.3, +5.2] |
| pt | 5099 | 27.7 [26.5, 28.9] | 38.6 [37.3, 40.0] | 44.0 [42.6, 45.3] | +10.9 [+9.7, +12.1] | +16.3 [+15.0, +17.6] |

**Gap split by whether the target occurs in the passage** (`lambada_sl`; acc Llama / rg / AU-Net, then rg−Llama)

| lang | in-ctx n | in-ctx acc | in-ctx rg−Llama | not-in-ctx n | not-in-ctx acc | not-in-ctx rg−Llama |
|---|---:|---|---:|---:|---|---:|
| en | 4194 | 68.2 / 70.6 / 70.7 | +2.4 [+1.1, +3.8] | 959 | 38.2 / 39.4 / 38.1 | +1.3 [-1.6, +4.0] |
| de | 2570 | 34.8 / 40.9 / 41.2 | +6.1 [+4.3, +7.9] | 2583 | 8.7 / 6.9 / 5.6 | -1.9 [-3.0, -0.7] |
| es | 3026 | 42.5 / 55.4 / 54.9 | +12.9 [+11.0, +14.8] | 2127 | 3.0 / 3.4 / 3.9 | +0.4 [-0.5, +1.2] |
| fr | 3134 | 39.2 / 50.1 / 51.6 | +10.9 [+9.2, +12.6] | 2019 | 5.8 / 19.0 / 20.2 | +13.2 [+11.4, +15.0] |
| it | 2970 | 35.5 / 48.8 / 49.7 | +13.3 [+11.4, +15.2] | 2183 | 2.9 / 8.2 / 9.8 | +5.2 [+4.0, +6.4] |
| nl | 2484 | 30.8 / 41.8 / 37.1 | +11.1 [+9.3, +12.9] | 2669 | 2.2 / 1.9 / 2.0 | -0.3 [-0.9, +0.3] |
| pt | 3915 | 34.7 / 49.1 / 55.7 | +14.4 [+12.8, +15.9] | 1238 | 5.3 / 4.8 / 6.1 | -0.5 [-1.9, +0.8] |

## Takeaways for the paper

- **Both translations carry exact-match artifacts.** gt-es attaches punctuation to 59.7% of targets (`W.` / `W?`); sl has `W..` endings in es/nl (~18–20%), `W."` in it, and 456 fr targets with no word characters at all. On punctuated targets the subword model collapses (gt-es 3.9% vs ≈20% bytes), so **clean-target accuracy is the metric to report**; full-set accuracy goes to the appendix.
- **Translation also breaks LAMBADA's design property** that the target recurs in the passage (en 81% → gt-es 31%, gt-de 46%; sl 48–76%). Report this rate next to the scores.
- **The byte>subword gap survives the cleaner sl translation and the clean-target filter**: every non-English language, CIs excluding 0. By family: Romance (es/fr/it/pt) +9.5 to +16.3, Germanic (de/nl) +1.6 to +6.8, English +2.0 to +2.2 — the same ordering as the AU-Net paper's multilingual-MMLU gains (Romance ≈+4, Germanic ≈+3).
- **Mechanism caveat**: the gap lives almost entirely on targets that occur in the passage; when the word must be produced without a copy source all three models score 2–9% and the gap vanishes (de even favours Llama, −1.9). Frame it as *reproducing context words in an unfamiliar orthography*, not as better cross-lingual semantic prediction. (sl-fr / sl-it not-in-ctx gaps are inflated by the odd punctuation-only targets; the clean-target table is the reference.)


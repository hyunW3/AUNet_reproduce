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

| lang | Llama | BPEByte-rg | AU-Net | AU-Net (fix) | rg − Llama | AU-Net − Llama | AU-Net − rg | AU-Net(fix) − Llama | AU-Net(fix) − rg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 62.6 [61.3, 63.9] | 62.8 [61.5, 64.1] | 64.6 [63.3, 65.9] | 62.8 [61.5, 64.2] | +0.2 [-1.0, +1.4] | +2.0 [+0.8, +3.2] | +1.8 [+0.7, +2.8] | +0.2 [-1.0, +1.4] | +0.0 [-1.0, +1.0] |
| de | 18.0 [16.9, 19.0] | 19.7 [18.7, 20.8] | 19.3 [18.2, 20.4] | 19.1 [18.0, 20.2] | +1.8 [+0.8, +2.8] | +1.3 [+0.3, +2.3] | -0.4 [-1.3, +0.5] | +1.1 [+0.1, +2.1] | -0.7 [-1.6, +0.2] |
| es | 12.7 [11.8, 13.6] | 23.7 [22.6, 24.9] | 25.5 [24.4, 26.7] | 22.5 [21.3, 23.6] | +11.0 [+10.0, +12.2] | +12.8 [+11.7, +14.0] | +1.8 [+0.8, +2.8] | +9.8 [+8.7, +10.9] | -1.3 [-2.2, -0.3] |
| fr | 25.3 [24.1, 26.5] | 28.6 [27.4, 29.8] | 33.4 [32.1, 34.7] | 29.8 [28.6, 31.1] | +3.3 [+2.2, +4.4] | +8.1 [+6.9, +9.3] | +4.8 [+3.7, +5.9] | +4.6 [+3.5, +5.7] | +1.2 [+0.3, +2.3] |
| it | 19.8 [18.7, 20.9] | 24.5 [23.4, 25.7] | 29.4 [28.2, 30.6] | 25.9 [24.7, 27.1] | +4.8 [+3.7, +5.8] | +9.6 [+8.4, +10.8] | +4.9 [+3.8, +6.0] | +6.1 [+5.0, +7.3] | +1.3 [+0.3, +2.3] |

**Clean targets only** (target is a bare word, no punctuation; `lambada`) — removes the exact-match punctuation artifact (on punctuated targets the subword model collapses, e.g. gt-es 3.9% vs ≈20% for bytes)

| lang | n clean | Llama | BPEByte-rg | AU-Net | AU-Net (fix) | rg − Llama | AU-Net − Llama | AU-Net − rg | AU-Net(fix) − Llama | AU-Net(fix) − rg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 5112 | 62.9 [61.6, 64.2] | 63.1 [61.8, 64.5] | 64.8 [63.6, 66.2] | 63.1 [61.8, 64.4] | +0.3 [-0.9, +1.4] | +2.0 [+0.8, +3.2] | +1.7 [+0.7, +2.8] | +0.2 [-1.0, +1.4] | -0.0 [-1.0, +1.0] |
| de | 4884 | 18.8 [17.7, 19.9] | 20.1 [19.0, 21.3] | 19.9 [18.7, 21.0] | 19.6 [18.5, 20.7] | +1.4 [+0.4, +2.4] | +1.1 [+0.1, +2.1] | -0.3 [-1.2, +0.7] | +0.9 [-0.1, +1.9] | -0.5 [-1.4, +0.4] |
| es | 2075 | 25.7 [23.8, 27.6] | 31.6 [29.6, 33.6] | 36.2 [34.1, 38.3] | 32.1 [30.1, 34.1] | +5.9 [+4.1, +7.7] | +10.5 [+8.6, +12.4] | +4.6 [+2.9, +6.3] | +6.5 [+4.7, +8.3] | +0.6 [-1.0, +2.2] |
| fr | 4832 | 26.2 [25.0, 27.4] | 29.6 [28.4, 30.9] | 34.7 [33.4, 36.1] | 31.0 [29.7, 32.3] | +3.4 [+2.3, +4.6] | +8.5 [+7.3, +9.7] | +5.1 [+4.0, +6.2] | +4.8 [+3.6, +5.9] | +1.3 [+0.3, +2.4] |
| it | 4965 | 20.4 [19.3, 21.6] | 25.3 [24.1, 26.5] | 30.3 [29.0, 31.6] | 26.7 [25.4, 28.0] | +4.9 [+3.8, +6.1] | +9.9 [+8.6, +11.1] | +5.0 [+3.8, +6.1] | +6.3 [+5.1, +7.5] | +1.3 [+0.3, +2.4] |

**Gap split by whether the target occurs in the passage** (`lambada`; acc Llama / rg / AU-Net / AU-Net(fix), then rg−Llama)

| lang | in-ctx n | in-ctx acc | in-ctx rg−Llama | not-in-ctx n | not-in-ctx acc | not-in-ctx rg−Llama |
|---|---:|---|---:|---:|---|---:|
| en | 4194 | 68.2 / 68.1 / 70.7 / 68.6 | -0.1 [-1.4, +1.2] | 959 | 38.1 / 39.7 / 38.0 / 37.9 | +1.7 [-1.0, +4.5] |
| de | 2374 | 31.1 / 34.3 / 36.2 / 34.8 | +3.2 [+1.4, +5.0] | 2779 | 6.7 / 7.3 / 4.9 / 5.6 | +0.5 [-0.5, +1.6] |
| es | 1586 | 32.4 / 40.5 / 46.4 / 41.1 | +8.1 [+5.8, +10.3] | 3567 | 3.9 / 16.3 / 16.2 / 14.2 | +12.4 [+11.1, +13.6] |
| fr | 3424 | 35.5 / 39.9 / 47.1 / 41.7 | +4.4 [+2.8, +6.0] | 1729 | 5.0 / 6.2 / 6.3 / 6.3 | +1.2 [+0.2, +2.3] |
| it | 3471 | 28.5 / 35.3 / 42.6 / 37.5 | +6.9 [+5.3, +8.4] | 1682 | 1.8 / 2.3 / 2.1 / 1.8 | +0.4 [-0.2, +1.1] |

## B-sl. Model accuracy with 95% CI (`lambada_sl`)

| lang | Llama | BPEByte-rg | AU-Net | AU-Net (fix) | rg − Llama | AU-Net − Llama | AU-Net − rg | AU-Net(fix) − Llama | AU-Net(fix) − rg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 62.6 [61.3, 63.9] | 62.9 [61.6, 64.2] | 64.6 [63.3, 65.9] | 62.7 [61.3, 64.0] | +0.3 [-0.9, +1.5] | +2.0 [+0.9, +3.2] | +1.7 [+0.7, +2.8] | +0.1 [-1.1, +1.3] | -0.2 [-1.2, +0.8] |
| de | 21.8 [20.6, 22.9] | 23.8 [22.6, 24.9] | 23.3 [22.2, 24.5] | 22.5 [21.4, 23.7] | +2.1 [+1.0, +3.1] | +1.6 [+0.5, +2.7] | -0.5 [-1.5, +0.5] | +0.8 [-0.3, +1.8] | -1.3 [-2.2, -0.3] |
| es | 26.2 [25.0, 27.4] | 31.2 [29.9, 32.5] | 33.9 [32.6, 35.1] | 30.5 [29.2, 31.7] | +5.0 [+3.9, +6.2] | +7.7 [+6.5, +8.8] | +2.6 [+1.6, +3.6] | +4.3 [+3.1, +5.4] | -0.8 [-1.8, +0.2] |
| fr | 26.1 [24.9, 27.3] | 36.0 [34.6, 37.3] | 39.3 [37.9, 40.6] | 36.4 [35.1, 37.7] | +9.9 [+8.6, +11.1] | +13.2 [+11.9, +14.5] | +3.3 [+2.2, +4.3] | +10.3 [+9.1, +11.6] | +0.5 [-0.5, +1.4] |
| it | 21.7 [20.6, 22.8] | 29.4 [28.2, 30.7] | 32.8 [31.5, 34.1] | 30.9 [29.6, 32.1] | +7.7 [+6.6, +8.9] | +11.1 [+9.9, +12.3] | +3.4 [+2.3, +4.4] | +9.2 [+8.0, +10.3] | +1.4 [+0.4, +2.4] |
| nl | 16.0 [15.0, 17.0] | 19.2 [18.1, 20.2] | 18.9 [17.9, 20.0] | 15.8 [14.8, 16.8] | +3.2 [+2.3, +4.1] | +2.9 [+1.8, +4.0] | -0.3 [-1.2, +0.7] | -0.2 [-1.2, +0.8] | -3.4 [-4.3, -2.5] |
| pt | 27.6 [26.4, 28.9] | 35.6 [34.2, 36.9] | 43.8 [42.5, 45.2] | 38.6 [37.2, 39.8] | +7.9 [+6.7, +9.2] | +16.2 [+14.9, +17.5] | +8.2 [+7.1, +9.4] | +10.9 [+9.7, +12.2] | +3.0 [+1.9, +4.1] |

**Clean targets only** (target is a bare word, no punctuation; `lambada_sl`) — removes the exact-match punctuation artifact (on punctuated targets the subword model collapses, e.g. gt-es 3.9% vs ≈20% for bytes)

| lang | n clean | Llama | BPEByte-rg | AU-Net | AU-Net (fix) | rg − Llama | AU-Net − Llama | AU-Net − rg | AU-Net(fix) − Llama | AU-Net(fix) − rg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 5112 | 62.8 [61.5, 64.2] | 63.2 [61.8, 64.5] | 64.8 [63.6, 66.2] | 62.9 [61.6, 64.3] | +0.3 [-0.9, +1.5] | +2.0 [+0.8, +3.2] | +1.7 [+0.6, +2.7] | +0.1 [-1.1, +1.3] | -0.2 [-1.2, +0.8] |
| de | 5108 | 21.8 [20.7, 22.9] | 23.9 [22.8, 25.1] | 23.5 [22.3, 24.6] | 22.6 [21.5, 23.8] | +2.1 [+1.1, +3.1] | +1.6 [+0.5, +2.8] | -0.5 [-1.5, +0.5] | +0.8 [-0.3, +1.9] | -1.3 [-2.3, -0.4] |
| es | 3868 | 34.7 [33.2, 36.2] | 40.9 [39.4, 42.5] | 44.2 [42.6, 45.8] | 40.0 [38.4, 41.5] | +6.2 [+4.7, +7.7] | +9.5 [+7.9, +11.0] | +3.3 [+1.9, +4.6] | +5.3 [+3.8, +6.8] | -0.9 [-2.2, +0.3] |
| fr | 4572 | 28.5 [27.2, 29.8] | 39.5 [38.1, 40.9] | 43.1 [41.6, 44.5] | 40.1 [38.7, 41.5] | +11.0 [+9.7, +12.4] | +14.6 [+13.2, +16.0] | +3.5 [+2.4, +4.7] | +11.6 [+10.3, +13.0] | +0.6 [-0.5, +1.7] |
| it | 3749 | 28.9 [27.4, 30.4] | 36.8 [35.2, 38.3] | 40.3 [38.8, 41.9] | 37.0 [35.5, 38.6] | +7.9 [+6.4, +9.4] | +11.4 [+9.9, +13.0] | +3.5 [+2.2, +4.9] | +8.2 [+6.7, +9.6] | +0.3 [-1.0, +1.5] |
| nl | 3928 | 20.8 [19.6, 22.1] | 25.0 [23.6, 26.3] | 24.6 [23.2, 25.9] | 20.5 [19.3, 21.8] | +4.1 [+3.0, +5.3] | +3.7 [+2.3, +5.2] | -0.4 [-1.7, +0.9] | -0.3 [-1.7, +1.0] | -4.5 [-5.7, -3.3] |
| pt | 5099 | 27.7 [26.6, 29.0] | 35.7 [34.4, 37.0] | 44.0 [42.7, 45.3] | 38.7 [37.4, 40.1] | +8.0 [+6.8, +9.2] | +16.3 [+14.9, +17.6] | +8.3 [+7.1, +9.5] | +11.0 [+9.7, +12.2] | +3.0 [+1.9, +4.1] |

**Gap split by whether the target occurs in the passage** (`lambada_sl`; acc Llama / rg / AU-Net / AU-Net(fix), then rg−Llama)

| lang | in-ctx n | in-ctx acc | in-ctx rg−Llama | not-in-ctx n | not-in-ctx acc | not-in-ctx rg−Llama |
|---|---:|---|---:|---:|---|---:|
| en | 4194 | 68.2 / 68.2 / 70.7 / 68.4 | +0.0 [-1.3, +1.3] | 959 | 38.2 / 39.8 / 38.1 / 37.7 | +1.7 [-1.0, +4.4] |
| de | 2570 | 34.8 / 39.7 / 41.2 / 38.4 | +4.9 [+3.2, +6.6] | 2583 | 8.7 / 8.0 / 5.6 / 6.8 | -0.8 [-1.9, +0.3] |
| es | 3026 | 42.5 / 51.0 / 54.9 / 49.6 | +8.5 [+6.6, +10.3] | 2127 | 3.0 / 3.1 / 3.9 / 3.3 | +0.1 [-0.8, +0.9] |
| fr | 3134 | 39.2 / 45.8 / 51.6 / 46.5 | +6.6 [+5.0, +8.2] | 2019 | 5.8 / 20.7 / 20.2 / 20.9 | +14.9 [+13.1, +16.7] |
| it | 2970 | 35.5 / 45.3 / 49.7 / 45.7 | +9.8 [+8.0, +11.6] | 2183 | 2.9 / 7.9 / 9.8 / 10.6 | +4.9 [+3.8, +6.1] |
| nl | 2484 | 30.8 / 37.8 / 37.1 / 30.8 | +7.0 [+5.3, +8.9] | 2669 | 2.2 / 1.8 / 2.0 / 1.8 | -0.4 [-1.0, +0.2] |
| pt | 3915 | 34.7 / 45.0 / 55.7 / 48.8 | +10.3 [+8.8, +11.8] | 1238 | 5.3 / 5.8 / 6.1 / 6.1 | +0.6 [-0.8, +2.0] |

## Takeaways for the paper

- **Both translations carry exact-match artifacts.** gt-es attaches punctuation to 59.7% of targets (`W.` / `W?`); sl has `W..` endings in es/nl (~18–20%), `W."` in it, and 456 fr targets with no word characters at all. On punctuated targets the subword model collapses (gt-es 3.9% vs ≈20% bytes), so **clean-target accuracy is the metric to report**; full-set accuracy goes to the appendix.
- **Translation also breaks LAMBADA's design property** that the target recurs in the passage (en 81% → gt-es 31%, gt-de 46%; sl 48–76%). Report this rate next to the scores.
- **Scorer policy** (lingua `apps/aunet/SCORING_POLICY.md`, main 6237a7b): BPEByte-rg = `generate_bpebyte.BPEByteGenerator`, vocab_norm applied; AU-Net = official `generate.py`, no vocab_norm; AU-Net (fix) = same checkpoint with vocab_norm (sensitivity only). Llama is unaffected.
- **The byte>subword gap survives the cleaner sl translation and the clean-target filter** (sl, clean): BPEByte-rg − Llama is +2.1 (de), +4.1 (nl), +6.2 to +11.0 (Romance es/it/pt/fr), CIs excluding 0; English +0.3 (n.s.). AU-Net (official) − Llama is larger, +1.6 to +16.3, with the same family ordering (Germanic small, Romance large), matching the direction of the AU-Net paper's multilingual-MMLU gains.
- **BPEByte vs AU-Net is decided by the scorer, not the architecture, on this task.** Under the policy (rg fixed, AU-Net official) AU-Net leads by +1.7 (en) and +3.3 to +8.3 (Romance). Like-for-like (both with vocab_norm) the two are tied in en/es/fr/it (|Δ| ≤ 0.9, CIs covering 0), rg leads in de (−1.3) and nl (−4.5), and AU-Net leads only in pt (+3.0). The official no-norm head scores *higher* on exact-match LAMBADA than the normed one (same AU-Net checkpoint: +1.7 to +5.3), so any rg-vs-AU-Net LAMBADA claim must state this asymmetry.
- **Mechanism caveat**: the gap lives on targets that occur in the passage (rg − Llama +4.9 to +10.3 in the non-English sl languages); when the word must be produced without a copy source all models score 2–9% and the gap is ≈0 (de −0.8, nl −0.4). Frame it as *reproducing context words in an unfamiliar orthography*, not as better cross-lingual semantic prediction. (sl-fr / sl-it not-in-ctx gaps are inflated by the punctuation-only targets; the clean-target table is the reference.)


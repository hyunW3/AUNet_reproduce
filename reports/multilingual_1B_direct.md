# Direct multilingual eval — 1B (English-trained, no further training)

Zero-shot loglikelihood eval of the DCLM-trained 1B checkpoints on non-English tasks — the multilingual counterpart of `reports/zh_cloze_1B.md` (same configs, only the task list changes). **transfer** = (non-en mean − chance) / (en − chance): the share of the English above-chance margin that survives in other languages.

**Scorer policy** (lingua main): BPEByte-rg is scored with the final `vocab_norm` (`1B_direct_vnfix`), AU-Net with the official generator, i.e. without it (`1B_direct`); Llama is unaffected. *AU-Net word (fix)* re-scores the same AU-Net checkpoint with `vocab_norm` as a sensitivity check. AU-Net char = `aunet2_1.3B` re-pooled per codepoint at eval (no-space scripts only).

## xstorycloze  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| xstorycloze_en | en | 70.9 | 71.3 | 72.1 | 71.9 |
| xstorycloze_ar | ar | 47.1 | 48.0 | 45.7 | 46.7 |
| xstorycloze_es | es | 52.9 | 53.2 | 53.1 | 52.6 |
| xstorycloze_eu | eu | 50.6 | 51.0 | 51.4 | 50.2 |
| xstorycloze_hi | hi | 49.6 | 47.6 | 47.7 | 48.2 |
| xstorycloze_id | id | 50.2 | 49.6 | 49.0 | 49.2 |
| xstorycloze_my | my | 46.3 | 46.8 | 47.3 | 47.3 |
| xstorycloze_ru | ru | 48.1 | 48.2 | 45.6 | 46.3 |
| xstorycloze_sw | sw | 49.7 | 48.8 | 48.1 | 47.8 |
| xstorycloze_te | te | 52.2 | 52.0 | 51.3 | 52.3 |
| xstorycloze_zh | zh | 53.3 | 47.6 | 47.2 | 48.2 |
| **non-en mean** | | **50.0** | **49.3** | **48.6** | **48.9** |
| **transfer** | | 0.00 | -0.03 | -0.06 | -0.05 |

## xcopa  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| copa | en | 78.0 | 78.0 | 78.0 | 75.0 |
| xcopa_et | et | 48.8 | 47.6 | 50.6 | 48.6 |
| xcopa_ht | ht | 49.6 | 50.6 | 50.6 | 49.6 |
| xcopa_id | id | 52.6 | 51.6 | 50.2 | 49.8 |
| xcopa_it | it | 49.8 | 51.8 | 50.2 | 50.6 |
| xcopa_qu | qu | 51.6 | 50.2 | 50.6 | 50.4 |
| xcopa_sw | sw | 53.0 | 55.6 | 56.0 | 54.4 |
| xcopa_ta | ta | 56.4 | 55.8 | 54.6 | 55.4 |
| xcopa_th | th | 53.4 | 53.6 | 53.0 | 54.8 |
| xcopa_tr | tr | 50.8 | 53.6 | 53.6 | 51.8 |
| xcopa_vi | vi | 50.8 | 50.2 | 50.2 | 52.2 |
| xcopa_zh | zh | 52.8 | 51.8 | 47.8 | 49.4 |
| **non-en mean** | | **51.8** | **52.0** | **51.6** | **51.5** |
| **transfer** | | 0.06 | 0.07 | 0.06 | 0.06 |

## xwinograd  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| xwinograd_en | en | 83.1 | 84.2 | 83.3 | 84.9 |
| xwinograd_fr | fr | 60.2 | 56.6 | 57.8 | 56.6 |
| xwinograd_jp | jp | 53.1 | 48.0 | 52.0 | 49.7 |
| xwinograd_pt | pt | 54.0 | 57.0 | 56.7 | 56.7 |
| xwinograd_ru | ru | 53.7 | 53.0 | 48.6 | 49.2 |
| xwinograd_zh | zh | 60.7 | 52.6 | 48.6 | 49.2 |
| **non-en mean** | | **56.3** | **53.4** | **52.7** | **52.3** |
| **transfer** | | 0.19 | 0.10 | 0.08 | 0.07 |

## pawsx  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| paws_en | en | 59.2 | 56.2 | 58.8 | 56.0 |
| paws_de | de | 55.2 | 51.3 | 52.7 | 50.8 |
| paws_es | es | 55.5 | 58.1 | 56.6 | 58.5 |
| paws_fr | fr | 51.3 | 49.9 | 52.0 | 49.4 |
| paws_ja | ja | 55.9 | 55.0 | 46.2 | 47.6 |
| paws_ko | ko | 54.8 | 55.5 | 44.8 | 45.0 |
| paws_zh | zh | 55.8 | 54.7 | 45.6 | 46.9 |
| **non-en mean** | | **54.7** | **54.1** | **49.7** | **49.7** |
| **transfer** | | 0.52 | 0.65 | -0.04 | -0.05 |

## xnli  (`acc`, chance 33.3)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| xnli_en | en | 51.4 | 53.3 | 51.7 | 53.0 |
| xnli_ar | ar | 33.3 | 33.6 | 33.9 | 33.3 |
| xnli_bg | bg | 33.8 | 32.1 | 33.3 | 33.4 |
| xnli_de | de | 38.7 | 34.3 | 35.8 | 34.8 |
| xnli_el | el | 37.1 | 33.2 | 33.3 | 33.6 |
| xnli_es | es | 38.1 | 38.1 | 39.1 | 40.2 |
| xnli_fr | fr | 42.4 | 38.0 | 38.8 | 36.9 |
| xnli_hi | hi | 33.6 | 33.1 | 34.5 | 33.8 |
| xnli_ru | ru | 35.7 | 33.6 | 34.2 | 33.6 |
| xnli_sw | sw | 32.9 | 33.9 | 33.3 | 33.6 |
| xnli_th | th | 33.4 | 32.5 | 36.1 | 33.5 |
| xnli_tr | tr | 33.1 | 33.5 | 32.2 | 32.5 |
| xnli_ur | ur | 33.4 | 33.4 | 33.4 | 33.3 |
| xnli_vi | vi | 35.9 | 33.4 | 33.3 | 33.5 |
| xnli_zh | zh | 32.9 | 33.5 | 33.3 | 33.3 |
| **non-en mean** | | **35.3** | **34.0** | **34.6** | **34.2** |
| **transfer** | | 0.11 | 0.04 | 0.07 | 0.05 |

## lambada  (`acc`, chance 0)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| lambada_openai_mt_en | en | 62.6 | 62.8 | 64.6 | 62.8 |
| lambada_openai_mt_de | de | 18.0 | 19.7 | 19.3 | 19.1 |
| lambada_openai_mt_es | es | 12.7 | 23.7 | 25.5 | 22.5 |
| lambada_openai_mt_fr | fr | 25.3 | 28.6 | 33.4 | 29.8 |
| lambada_openai_mt_it | it | 19.8 | 24.5 | 29.4 | 25.9 |
| **non-en mean** | | **18.9** | **24.1** | **26.9** | **24.3** |
| **transfer** | | 0.30 | 0.38 | 0.42 | 0.39 |

## lambada_sl  (`acc`, chance 0)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| lambada_openai_mt_stablelm_en | en | 62.6 | 62.9 | 64.6 | 62.7 |
| lambada_openai_mt_stablelm_de | de | 21.8 | 23.8 | 23.3 | 22.5 |
| lambada_openai_mt_stablelm_es | es | 26.2 | 31.2 | 33.9 | 30.5 |
| lambada_openai_mt_stablelm_fr | fr | 26.1 | 36.0 | 39.3 | 36.4 |
| lambada_openai_mt_stablelm_it | it | 21.7 | 29.4 | 32.8 | 30.9 |
| lambada_openai_mt_stablelm_nl | nl | 16.0 | 19.2 | 18.9 | 15.8 |
| lambada_openai_mt_stablelm_pt | pt | 27.6 | 35.6 | 43.8 | 38.6 |
| **non-en mean** | | **23.2** | **29.2** | **32.0** | **29.1** |
| **transfer** | | 0.37 | 0.46 | 0.50 | 0.46 |

## arc  (`acc_norm`, chance 25)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| arc_challenge | en | 35.1 | 37.1 | 36.2 | 36.2 |
| arc_ar | ar | 22.4 | 23.3 | 22.7 | 22.4 |
| arc_de | de | 24.2 | 22.8 | 21.8 | 21.4 |
| arc_es | es | 23.4 | 24.5 | 22.8 | 23.7 |
| arc_fr | fr | 25.8 | 27.1 | 25.0 | 25.6 |
| arc_hi | hi | 23.3 | 24.2 | 23.5 | 23.8 |
| arc_id | id | 21.9 | 22.4 | 23.0 | 23.2 |
| arc_it | it | 23.5 | 24.5 | 23.2 | 23.1 |
| arc_ru | ru | 23.3 | 23.1 | 22.8 | 23.8 |
| arc_vi | vi | 23.5 | 22.7 | 22.6 | 23.2 |
| arc_zh | zh | 25.0 | 26.0 | 23.6 | 24.6 |
| **non-en mean** | | **23.6** | **24.1** | **23.1** | **23.5** |
| **transfer** | | -0.14 | -0.08 | -0.17 | -0.14 |

## hellaswag  (`acc_norm`, chance 25)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| hellaswag | en | 55.8 | 57.5 | 57.6 | 58.6 |
| hellaswag_ar | ar | 26.9 | 26.4 | 27.2 | 27.4 |
| hellaswag_de | de | 30.0 | 29.7 | 28.7 | 28.7 |
| hellaswag_es | es | 33.2 | 35.9 | 34.8 | 34.8 |
| hellaswag_fr | fr | 33.6 | 35.8 | 34.8 | 34.7 |
| hellaswag_hi | hi | 28.0 | 26.8 | 28.7 | 28.0 |
| hellaswag_id | id | 28.7 | 29.8 | 30.0 | 29.6 |
| hellaswag_it | it | 29.3 | 31.9 | 31.4 | 31.4 |
| hellaswag_ru | ru | 28.5 | 28.0 | 27.9 | 27.5 |
| hellaswag_vi | vi | 28.1 | 29.1 | 29.8 | 29.8 |
| **non-en mean** | | **29.6** | **30.4** | **30.4** | **30.2** |
| **transfer** | | 0.15 | 0.17 | 0.16 | 0.15 |

## gmmlu  (`acc_norm`, chance 25)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---|---:|---:|---:|---:|
| gmmlu_cloze_en | en | 34.1 | 34.8 | 34.5 | 34.7 |
| gmmlu_cloze_ar | ar | 23.9 | 23.9 | 24.1 | 24.2 |
| gmmlu_cloze_bn | bn | 24.1 | 24.5 | 24.7 | 24.9 |
| gmmlu_cloze_cs | cs | 25.3 | 25.5 | 26.1 | 25.9 |
| gmmlu_cloze_de | de | 25.6 | 24.8 | 25.3 | 25.3 |
| gmmlu_cloze_el | el | 25.0 | 25.1 | 25.8 | 25.3 |
| gmmlu_cloze_es | es | 25.4 | 25.8 | 25.7 | 25.3 |
| gmmlu_cloze_fa | fa | 26.1 | 25.5 | 25.2 | 25.0 |
| gmmlu_cloze_fr | fr | 27.3 | 26.1 | 26.4 | 26.6 |
| gmmlu_cloze_hi | hi | 24.9 | 24.9 | 24.8 | 24.6 |
| gmmlu_cloze_id | id | 27.3 | 26.9 | 26.2 | 25.7 |
| gmmlu_cloze_it | it | 24.1 | 24.9 | 24.8 | 24.2 |
| gmmlu_cloze_ja | ja | 26.3 | 25.1 | 25.6 | 25.7 |
| gmmlu_cloze_ko | ko | 25.9 | 26.0 | 25.7 | 25.4 |
| gmmlu_cloze_nl | nl | 24.6 | 24.0 | 24.7 | 24.0 |
| gmmlu_cloze_pl | pl | 25.2 | 24.3 | 24.6 | 25.0 |
| gmmlu_cloze_pt | pt | 27.1 | 26.4 | 26.8 | 26.4 |
| gmmlu_cloze_ro | ro | 24.2 | 24.9 | 25.1 | 25.3 |
| gmmlu_cloze_ru | ru | 26.7 | 26.7 | 26.3 | 26.3 |
| gmmlu_cloze_sv | sv | 26.2 | 26.2 | 25.7 | 25.1 |
| gmmlu_cloze_sw | sw | 24.9 | 24.6 | 25.0 | 25.5 |
| gmmlu_cloze_te | te | 23.7 | 25.0 | 24.3 | 24.4 |
| gmmlu_cloze_tr | tr | 24.4 | 27.8 | 26.2 | 26.2 |
| gmmlu_cloze_vi | vi | 26.7 | 25.1 | 26.4 | 26.2 |
| gmmlu_cloze_zh | zh | 25.3 | 26.0 | 25.3 | 25.0 |
| **non-en mean** | | **25.4** | **25.4** | **25.4** | **25.3** |
| **transfer** | | 0.05 | 0.04 | 0.05 | 0.03 |

## nospace  (`acc`)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) | AU-Net char |
|---|---|---:|---:|---:|---:|---:|
| paws_ja | ja | 55.9 | 55.0 | 46.2 | 47.6 | 45.6 |
| xwinograd_jp | jp | 53.1 | 48.0 | 52.0 | 49.7 | 51.7 |
| paws_ko | ko | 54.8 | 55.5 | 44.8 | 45.0 | 44.5 |
| xstorycloze_my | my | 46.3 | 46.8 | 47.3 | 47.3 | 47.5 |
| xnli_th | th | 33.4 | 32.5 | 36.1 | 33.5 | 36.2 |
| xcopa_th | th | 53.4 | 53.6 | 53.0 | 54.8 | 52.2 |
| paws_zh | zh | 55.8 | 54.7 | 45.6 | 46.9 | 45.5 |
| xwinograd_zh | zh | 60.7 | 52.6 | 48.6 | 49.2 | 48.6 |
| xnli_zh | zh | 32.9 | 33.5 | 33.3 | 33.3 | 32.8 |
| xstorycloze_zh | zh | 53.3 | 47.6 | 47.2 | 48.2 | 46.5 |
| xcopa_zh | zh | 52.8 | 51.8 | 47.8 | 49.4 | 48.0 |
| arc_zh | zh | 25.0 | 26.0 | 23.6 | 24.6 | 22.1 |

## Summary (en → non-en mean · transfer)

| suite | chance | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net word (fix) |
|---|---:|---:|---:|---:|---:|
| xstorycloze | 50 | 70.9 → 50.0 · 0.00 | 71.3 → 49.3 · -0.03 | 72.1 → 48.6 · -0.06 | 71.9 → 48.9 · -0.05 |
| xcopa | 50 | 78.0 → 51.8 · 0.06 | 78.0 → 52.0 · 0.07 | 78.0 → 51.6 · 0.06 | 75.0 → 51.5 · 0.06 |
| xwinograd | 50 | 83.1 → 56.3 · 0.19 | 84.2 → 53.4 · 0.10 | 83.3 → 52.7 · 0.08 | 84.9 → 52.3 · 0.07 |
| pawsx | 50 | 59.2 → 54.7 · 0.52 | 56.2 → 54.1 · 0.65 | 58.8 → 49.7 · -0.04 | 56.0 → 49.7 · -0.05 |
| xnli | 33.3 | 51.4 → 35.3 · 0.11 | 53.3 → 34.0 · 0.04 | 51.7 → 34.6 · 0.07 | 53.0 → 34.2 · 0.05 |
| lambada | 0 | 62.6 → 18.9 · 0.30 | 62.8 → 24.1 · 0.38 | 64.6 → 26.9 · 0.42 | 62.8 → 24.3 · 0.39 |
| lambada_sl | 0 | 62.6 → 23.2 · 0.37 | 62.9 → 29.2 · 0.46 | 64.6 → 32.0 · 0.50 | 62.7 → 29.1 · 0.46 |
| arc | 25 | 35.1 → 23.6 · -0.14 | 37.1 → 24.1 · -0.08 | 36.2 → 23.1 · -0.17 | 36.2 → 23.5 · -0.14 |
| hellaswag | 25 | 55.8 → 29.6 · 0.15 | 57.5 → 30.4 · 0.17 | 57.6 → 30.4 · 0.16 | 58.6 → 30.2 · 0.15 |
| gmmlu | 25 | 34.1 → 25.4 · 0.05 | 34.8 → 25.4 · 0.04 | 34.5 → 25.4 · 0.05 | 34.7 → 25.3 · 0.03 |

## Notes

- **Classification suites do not separate the families**: all four columns sit near chance outside English on xstorycloze/xcopa/xwinograd/xnli/arc/hellaswag, whichever scorer is used.
- **Global-MMLU does not reproduce the AU-Net paper's multilingual-MMLU gains at this scale**: English 34.1–34.8 and non-English mean 25.3–25.4 (chance 25) for every model, per-language spread within ±2 pt. The paper's 1B (370B tokens) sat at 30–41% outside English, which is where its Romance ≈+4 / Germanic ≈+3 gaps live; our checkpoints have no above-chance margin to transfer. (The paper does not name its multilingual MMLU; Global-MMLU covers 24 of its 27 languages.)
- **LAMBADA is the one suite with a byte>subword signal**; see `reports/lambada_multilingual_analysis.md` for the translation-noise controls, paired CIs and the scorer asymmetry between BPEByte and AU-Net.
- **PAWS-X is label bias, not ability**: the test sets are 44–45% positive in every language, so always-'No' scores ≈55 and always-'Yes' ≈45.
- Anchors: xcopa's English anchor is SuperGLUE COPA **validation** (100 items, noisy); xnli uses lm-eval's default validation split (2490/lang); hellaswag_* capped at 2000 docs/lang. lambada = EleutherAI Google-Translate splits, lambada_sl = StableLM-2 re-translation (see `reports/lambada_multilingual_analysis.md`). gmmlu = Global-MMLU in cloze form (as our English `mmlu_text`), 2044 parallel items, the 24 AU-Net Table-3 languages Global-MMLU has (fi/hu/th missing). Llama is the `llama_1.8B_paper` step-60000 checkpoint (same as the Chinese run).

_Sources: `/mnt/ssd2/hyun2/AUNet/runs/multilingual/1B_direct` and `/mnt/ssd2/hyun2/AUNet/runs/multilingual/1B_direct_vnfix` `/<model>/<suite>/results.json`; launcher `scripts/multilingual/run_1b_direct.sh`._

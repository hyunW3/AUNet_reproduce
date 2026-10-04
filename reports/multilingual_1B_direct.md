> **SUPERSEDED (2026-10-04):** every BPEByte-rg / AU-Net number below was scored without the final `vocab_norm` (`apps/aunet/generate.py::_next_byte_logits`, applied only with `AUNET_FIX_VOCAB_NORM=1`), which corrupts loglikelihood for the byte models. Llama columns are unaffected. Re-run → `runs/multilingual/1B_direct_vnfix`.

# Direct multilingual eval — 1B (English-trained, no further training)

Zero-shot loglikelihood eval of the DCLM-trained 1B checkpoints on non-English tasks — the multilingual counterpart of `reports/zh_cloze_1B.md` (same configs, only the task list changes). **transfer** = (non-en mean − chance) / (en − chance): the share of the English above-chance margin that survives in other languages. AU-Net char = the same `aunet2_1.3B` re-pooled per codepoint at eval (no-space scripts only).

## xstorycloze  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| xstorycloze_en | en | 70.9 | 71.6 | 72.1 |
| xstorycloze_ar | ar | 47.1 | 46.3 | 45.7 |
| xstorycloze_es | es | 52.9 | 53.1 | 53.1 |
| xstorycloze_eu | eu | 50.6 | 51.4 | 51.4 |
| xstorycloze_hi | hi | 49.6 | 46.7 | 47.7 |
| xstorycloze_id | id | 50.2 | 48.8 | 49.0 |
| xstorycloze_my | my | 46.3 | 46.7 | 47.3 |
| xstorycloze_ru | ru | 48.1 | 46.7 | 45.6 |
| xstorycloze_sw | sw | 49.7 | 48.8 | 48.1 |
| xstorycloze_te | te | 52.2 | 51.3 | 51.3 |
| xstorycloze_zh | zh | 53.3 | 46.9 | 47.2 |
| **non-en mean** | | **50.0** | **48.7** | **48.6** |
| **transfer** | | 0.00 | -0.06 | -0.06 |

## xcopa  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| copa | en | 78.0 | 76.0 | 78.0 |
| xcopa_et | et | 48.8 | 48.8 | 50.6 |
| xcopa_ht | ht | 49.6 | 50.4 | 50.6 |
| xcopa_id | id | 52.6 | 51.4 | 50.2 |
| xcopa_it | it | 49.8 | 51.2 | 50.2 |
| xcopa_qu | qu | 51.6 | 50.8 | 50.6 |
| xcopa_sw | sw | 53.0 | 56.8 | 56.0 |
| xcopa_ta | ta | 56.4 | 57.8 | 54.6 |
| xcopa_th | th | 53.4 | 51.0 | 53.0 |
| xcopa_tr | tr | 50.8 | 55.0 | 53.6 |
| xcopa_vi | vi | 50.8 | 49.6 | 50.2 |
| xcopa_zh | zh | 52.8 | 50.0 | 47.8 |
| **non-en mean** | | **51.8** | **52.1** | **51.6** |
| **transfer** | | 0.06 | 0.08 | 0.06 |

## xwinograd  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| xwinograd_en | en | 83.1 | 83.2 | 83.3 |
| xwinograd_fr | fr | 60.2 | 59.0 | 57.8 |
| xwinograd_jp | jp | 53.1 | 48.9 | 52.0 |
| xwinograd_pt | pt | 54.0 | 56.7 | 56.7 |
| xwinograd_ru | ru | 53.7 | 53.0 | 48.6 |
| xwinograd_zh | zh | 60.7 | 53.0 | 48.6 |
| **non-en mean** | | **56.3** | **54.1** | **52.7** |
| **transfer** | | 0.19 | 0.12 | 0.08 |

## pawsx  (`acc`, chance 50)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| paws_en | en | 59.2 | 56.8 | 58.8 |
| paws_de | de | 55.2 | 51.8 | 52.7 |
| paws_es | es | 55.5 | 55.9 | 56.6 |
| paws_fr | fr | 51.3 | 53.6 | 52.0 |
| paws_ja | ja | 55.9 | 48.0 | 46.2 |
| paws_ko | ko | 54.8 | 44.8 | 44.8 |
| paws_zh | zh | 55.8 | 46.3 | 45.6 |
| **non-en mean** | | **54.7** | **50.1** | **49.7** |
| **transfer** | | 0.52 | 0.01 | -0.04 |

## xnli  (`acc`, chance 33.3)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| xnli_en | en | 51.4 | 52.2 | 51.7 |
| xnli_ar | ar | 33.3 | 33.2 | 33.9 |
| xnli_bg | bg | 33.8 | 32.2 | 33.3 |
| xnli_de | de | 38.7 | 36.0 | 35.8 |
| xnli_el | el | 37.1 | 33.7 | 33.3 |
| xnli_es | es | 38.1 | 36.2 | 39.1 |
| xnli_fr | fr | 42.4 | 39.0 | 38.8 |
| xnli_hi | hi | 33.6 | 33.2 | 34.5 |
| xnli_ru | ru | 35.7 | 33.6 | 34.2 |
| xnli_sw | sw | 32.9 | 33.7 | 33.3 |
| xnli_th | th | 33.4 | 34.6 | 36.1 |
| xnli_tr | tr | 33.1 | 33.3 | 32.2 |
| xnli_ur | ur | 33.4 | 33.5 | 33.4 |
| xnli_vi | vi | 35.9 | 33.4 | 33.3 |
| xnli_zh | zh | 32.9 | 33.3 | 33.3 |
| **non-en mean** | | **35.3** | **34.2** | **34.6** |
| **transfer** | | 0.11 | 0.05 | 0.07 |

## lambada  (`acc`, chance 0)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| lambada_openai_mt_en | en | 62.6 | 64.8 | 64.6 |
| lambada_openai_mt_de | de | 18.0 | 19.5 | 19.3 |
| lambada_openai_mt_es | es | 12.7 | 26.8 | 25.5 |
| lambada_openai_mt_fr | fr | 25.3 | 30.7 | 33.4 |
| lambada_openai_mt_it | it | 19.8 | 27.2 | 29.4 |
| **non-en mean** | | **18.9** | **26.1** | **26.9** |
| **transfer** | | 0.30 | 0.40 | 0.42 |

## arc  (`acc_norm`, chance 25)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| arc_challenge | en | 35.1 | 37.4 | 36.2 |
| arc_ar | ar | 22.4 | 22.3 | 22.7 |
| arc_de | de | 24.2 | 22.5 | 21.8 |
| arc_es | es | 23.4 | 25.4 | 22.8 |
| arc_fr | fr | 25.8 | 27.1 | 25.0 |
| arc_hi | hi | 23.3 | 22.9 | 23.5 |
| arc_id | id | 21.9 | 22.1 | 23.0 |
| arc_it | it | 23.5 | 24.3 | 23.2 |
| arc_ru | ru | 23.3 | 23.4 | 22.8 |
| arc_vi | vi | 23.5 | 23.2 | 22.6 |
| arc_zh | zh | 25.0 | 24.7 | 23.6 |
| **non-en mean** | | **23.6** | **23.8** | **23.1** |
| **transfer** | | -0.14 | -0.10 | -0.17 |

## hellaswag  (`acc_norm`, chance 25)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---|---:|---:|---:|
| hellaswag | en | 55.8 | 57.2 | 57.6 |
| hellaswag_ar | ar | 26.9 | 26.7 | 27.2 |
| hellaswag_de | de | 30.0 | 29.7 | 28.7 |
| hellaswag_es | es | 33.2 | 35.9 | 34.8 |
| hellaswag_fr | fr | 33.6 | 35.0 | 34.8 |
| hellaswag_hi | hi | 28.0 | 28.0 | 28.7 |
| hellaswag_id | id | 28.7 | 30.2 | 30.0 |
| hellaswag_it | it | 29.3 | 31.4 | 31.4 |
| hellaswag_ru | ru | 28.5 | 27.9 | 27.9 |
| hellaswag_vi | vi | 28.1 | 28.7 | 29.8 |
| **non-en mean** | | **29.6** | **30.4** | **30.4** |
| **transfer** | | 0.15 | 0.17 | 0.16 |

## nospace  (`acc`)

| task | lang | Llama (subword) | BPEByte-rg | AU-Net word | AU-Net char |
|---|---|---:|---:|---:|---:|
| paws_ja | ja | 55.9 | 48.0 | 46.2 | 45.6 |
| xwinograd_jp | jp | 53.1 | 48.9 | 52.0 | 51.7 |
| paws_ko | ko | 54.8 | 44.8 | 44.8 | 44.5 |
| xstorycloze_my | my | 46.3 | 46.7 | 47.3 | 47.5 |
| xcopa_th | th | 53.4 | 51.0 | 53.0 | 52.2 |
| xnli_th | th | 33.4 | 34.6 | 36.1 | 36.2 |
| xnli_zh | zh | 32.9 | 33.3 | 33.3 | 32.8 |
| xstorycloze_zh | zh | 53.3 | 46.9 | 47.2 | 46.5 |
| arc_zh | zh | 25.0 | 24.7 | 23.6 | 22.1 |
| paws_zh | zh | 55.8 | 46.3 | 45.6 | 45.5 |
| xwinograd_zh | zh | 60.7 | 53.0 | 48.6 | 48.6 |
| xcopa_zh | zh | 52.8 | 50.0 | 47.8 | 48.0 |

## Summary (en → non-en mean · transfer)

| suite | chance | Llama (subword) | BPEByte-rg | AU-Net word |
|---|---:|---:|---:|---:|
| xstorycloze | 50 | 70.9 → 50.0 · 0.00 | 71.6 → 48.7 · -0.06 | 72.1 → 48.6 · -0.06 |
| xcopa | 50 | 78.0 → 51.8 · 0.06 | 76.0 → 52.1 · 0.08 | 78.0 → 51.6 · 0.06 |
| xwinograd | 50 | 83.1 → 56.3 · 0.19 | 83.2 → 54.1 · 0.12 | 83.3 → 52.7 · 0.08 |
| pawsx | 50 | 59.2 → 54.7 · 0.52 | 56.8 → 50.1 · 0.01 | 58.8 → 49.7 · -0.04 |
| xnli | 33.3 | 51.4 → 35.3 · 0.11 | 52.2 → 34.2 · 0.05 | 51.7 → 34.6 · 0.07 |
| lambada | 0 | 62.6 → 18.9 · 0.30 | 64.8 → 26.1 · 0.40 | 64.6 → 26.9 · 0.42 |
| arc | 25 | 35.1 → 23.6 · -0.14 | 37.4 → 23.8 · -0.10 | 36.2 → 23.1 · -0.17 |
| hellaswag | 25 | 55.8 → 29.6 · 0.15 | 57.2 → 30.4 · 0.17 | 57.6 → 30.4 · 0.16 |

## Notes

- **Same picture as Chinese**: every family is strong in English and near chance elsewhere; between-family gaps on the classification suites are within noise. Direct transfer from an English-only 1B does not separate the architectures, except on LAMBADA.
- **LAMBADA is the one clear signal**: byte models beat the subword model on es/fr/it (non-en mean ≈26–27 vs 18.9), i.e. last-word prediction in Latin-script languages transfers better when the model is not bound to an English subword vocab.
- **PAWS-X is label bias, not ability**: the test sets are 44–45% positive in every language, so always-'No' scores ≈55 and always-'Yes' ≈45. Llama's ≈55 and the byte models' 44–46 on ja/ko/zh are those two degenerate answers.
- **AU-Net char vs word** (nospace table): the eval-time per-codepoint re-pool does not help an English-trained checkpoint on CJK/Thai/Burmese either (same caveat as `zh_cloze_1B.md`: train/eval boundary mismatch). char and word agree within ±1.5 pt on every task.
- **Llama's small CJK edge**: xwinograd_zh 60.7 vs 48.6–53.0 (n=504, ~±2.2 pt s.e.) and xstorycloze_zh 53.3 vs ≈47 — plausibly the llama3 vocab's CJK tokens; elsewhere the families are tied.
- Anchors: xcopa's English anchor is SuperGLUE COPA **validation** (100 items, noisy); xnli uses lm-eval's default validation split (2490/lang); hellaswag_* capped at 2000 docs/lang. Llama is the `llama_1.8B_paper` step-60000 checkpoint (same as the Chinese run).

_Sources:`/mnt/ssd2/hyun2/AUNet/runs/multilingual/1B_direct/<model>/<suite>/results.json`; launcher `scripts/multilingual/run_1b_direct.sh`._

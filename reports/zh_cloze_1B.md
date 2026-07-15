# Chinese CLOZE — 1B scale (Llama · BPEByte-rg · AU-Net word/char)

_Auto-generated · 4/4 evals in · updated 2026-07-14 22:28 KST_

**Eval-only** on the existing English-trained 1.3B checkpoints (DCLM) → this is **cross-lingual transfer**, expected near chance (25); the signal is the *ordering*, esp. AU-Net word-vs-char. `acc_norm` (`acc` in parens). AU-Net's two variants are the **same `aunet2_1.3B` checkpoint re-pooled at eval**: word1 (its trained boundary) vs char1 (per-codepoint boundary swap) — boundaries live in the data path, not the weights.

| Family | Boundary | bytes/patch | cmmlu-cloze | ceval-cloze | **mean** |
|---|---|---|---:|---:|---:|
| Llama (subword) | llama3 vocab · 1.8B | ~3.15 B/tok | 27.3 (24.3) | 26.6 (23.7) | **26.9** |
| BPEByte-rg | online greedy · 1.3B | ~3 B/patch | 27.0 (23.8) | 26.3 (22.8) | **26.6** |
| AU-Net word | regex-cutting (native) | ~15 B/patch | 27.0 (23.7) | 26.2 (23.8) | **26.6** |
| AU-Net char | per-codepoint (boundary-swap) | ~3 B/patch | 25.7 (23.5) | 27.3 (23.6) | **26.5** |

## Notes

- **AU-Net word vs char** = pure boundary ablation on one checkpoint. char re-pools per-codepoint (~3 B/patch), word keeps the trained regex (~15 B/patch). NB char is an eval-time boundary swap on a word-TRAINED model, so it carries a train/eval boundary-distribution mismatch (cf. the qwen2 swap on rg).
- Compare with the 100M **21 GB from-scratch** run (`reports/zh_cloze_100M_21G.md`), where char/word were each TRAINED — there char>word cleanly.
- Sources: `runs/zh/1B_cloze/{llama,rg,aunet_word,aunet_char}/results.json`.

# Chinese CLOZE — 300M, 21 GB data (from scratch)

_Auto-generated · 4/4 evals in · updated 2026-07-15 16:16 KST_

Full **21 GB** (40,054 byte-steps / ~16,990 llama-steps, iso-data) on `zh_wiki_train` (1.3 GB → ~16 epochs; NB training loss ~0.06–0.7 = heavy overfit of the small corpus). `acc_norm` (`acc` in parens); chance ≈ 25. Each model uses its own native patching. **AU-Net has two variants**: *word* (original regex-cutting, coarse ~15 B/patch) and *char* (per-codepoint, ~3 B/patch — Chinese-appropriate, matches byte granularity).

| Family | Boundary | bytes/patch | cmmlu-cloze | ceval-cloze | **mean** |
|---|---|---|---:|---:|---:|
| Llama (subword) | llama3 vocab | ~3.15 B/tok | 27.0 (24.8) | 24.7 (22.4) | **25.9** |
| BPEByte-rg | online greedy, llama3 | ~3 B/patch | 26.5 (24.2) | 26.1 (24.5) | **26.3** |
| AU-Net word | regex-cutting | ~15 B/patch | 26.8 (24.6) | 25.6 (22.7) | **26.2** |
| AU-Net char | per-codepoint | ~3 B/patch | 27.0 (24.8) | 27.8 (23.6) | **27.4** |

## Notes

- **AU-Net word vs char** is the key new axis: word regex pools up to 16 CJK chars into one patch (tiny trunk workload); char gives one patch per character (~3 B/patch), so the trunk actually does work — expect char ≫ word if granularity matters on Chinese.
- **BPEByte-rg** (online greedy+root, causal) vs **AU-Net char** (offline per-char, before-root) isolates online-causal vs offline at the same ~3 B/patch granularity.
- 4 GB pilot (near-chance) is in `reports/zh_cloze_100M.md`; this is the 21 GB run you asked for.
- Sources: `runs/zh/300M/scratch/*/eval_zh_cloze_*/results.json`; configs `runs/poc/portable_aunetlaw/zh/`.

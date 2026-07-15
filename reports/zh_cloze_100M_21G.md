# Chinese CLOZE — 100M, FULL 21 GB (ratio-10), from scratch

_Auto-generated · 5/5 evals in · updated 2026-07-14 22:03 KST_

Full **21 GB** (53,504 byte-steps / ~16,990 llama-steps, iso-data) on `zh_wiki_train` (1.3 GB → ~16 epochs). `acc_norm` (`acc` in parens); chance ≈ 25. Each model uses its own native patching. **AU-Net has two variants**: *word* (original regex-cutting, coarse ~15 B/patch) and *char* (per-codepoint, ~3 B/patch — Chinese-appropriate, matches byte granularity).

| Family | Boundary | bytes/patch | cmmlu-cloze | ceval-cloze | **mean** |
|---|---|---|---:|---:|---:|
| Llama (subword) | llama3 vocab | ~3.15 B/tok | 26.9 (24.2) | 27.1 (23.6) | **27.0** |
| BPEByte-rg | online greedy, llama3 | ~3 B/patch | 26.8 (24.6) | 27.9 (26.4) | **27.3** |
| BPEByte-rg | online greedy, qwen2 | ~4 B/patch | 26.5 (24.4) | 27.9 (23.1) | **27.2** |
| AU-Net word | regex-cutting | ~15 B/patch | 25.6 (23.8) | 26.9 (23.6) | **26.2** |
| AU-Net char | per-codepoint | ~3 B/patch | 27.1 (25.0) | 28.4 (24.5) | **27.7** |

## Notes

- **AU-Net word vs char** is the key new axis: word regex pools up to 16 CJK chars into one patch (tiny trunk workload); char gives one patch per character (~3 B/patch), so the trunk actually does work — expect char ≫ word if granularity matters on Chinese.
- **BPEByte-rg** (online greedy+root, causal) vs **AU-Net char** (offline per-char, before-root) isolates online-causal vs offline at the same ~3 B/patch granularity.
- 4 GB pilot (near-chance) is in `reports/zh_cloze_100M.md`; this is the 21 GB run you asked for.
- Sources: `runs/zh/100M_21G/scratch/*/eval_zh_cloze_*/results.json`; configs `runs/poc/portable_aunetlaw/zh/`.

# Chinese CLOZE — 100M 3-way (Llama · AU-Net-BPE · BPEByte-rg), zh_wiki pilot

_Auto-generated · 10/10 evals in · updated 2026-07-14 11:27 KST_

**Metric** = group `acc_norm` (primary for cloze), `acc` in parentheses. Chance ≈ 25 (4-choice). Cloze = score the answer TEXT, not the A/B/C/D letter. Each model evaluated with its OWN native patch scheme (llama3/qwen2, online-greedy for rg / offline-BPE for AU-Net); Llama is subword. **~4 GB pilot** (byte 10k steps / llama 3150, iso-data) — first-read, not converged.

## From scratch on Chinese

| Family | Patch tok | cmmlu-cloze | ceval-cloze | **mean** |
|---|---|---:|---:|---:|
| Llama (subword) | — | 27.4 (24.7) | 28.1 (24.4) | **27.7** |
| BPEByte-rg | llama3 | 27.1 (24.6) | 29.2 (25.1) | **28.1** |
| BPEByte-rg | qwen2 | 26.3 (24.3) | 26.1 (23.3) | **26.2** |
| AU-Net-BPE | llama3 | 26.9 (24.7) | 27.1 (22.6) | **27.0** |
| AU-Net-BPE | qwen2 | 26.8 (24.5) | 26.7 (23.9) | **26.8** |

## Continual (warm-start from English 100M + zh0.7/dclm0.3 replay, lr 1e-4)

| Family | Patch tok | cmmlu-cloze | ceval-cloze | **mean** |
|---|---|---:|---:|---:|
| Llama (subword) | — | 26.2 (23.5) | 27.2 (23.0) | **26.7** |
| BPEByte-rg | llama3 | 23.7 (23.2) | 25.0 (23.9) | **24.4** |
| BPEByte-rg | qwen2 | 24.8 (23.3) | 25.6 (22.1) | **25.2** |
| AU-Net-BPE | llama3 | 26.4 (23.8) | 26.7 (23.7) | **26.6** |
| AU-Net-BPE | qwen2 | 26.3 (24.1) | 25.9 (23.5) | **26.1** |

## Notes

- **Patch axis** (llama3 char-level ~3 B/patch vs qwen2 word-level ~4 B/patch) applies to the byte models; Llama's vocab is fixed. **Boundary axis**: BPEByte-rg = online greedy+root (causal, leak-free); AU-Net-BPE = offline real-BPE before-root (leaky, full-lookahead).
- English-only reference (the original swap study, 1.3B): all Chinese cloze ~chance (25). Signal here = whether 4 GB of zh training lifts any family off that floor, and whether qwen2 (Chinese-aware) patching beats llama3 for the byte models.
- Sources: `runs/zh/100M/{scratch,continual}/*/eval_zh_cloze_*/results.json`; configs in `runs/poc/portable_aunetlaw/zh/`. Trainers: `apps.aunet.train` (byte) / `apps.main.train` (llama).

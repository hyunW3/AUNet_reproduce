# LAMBADA (English) — 1B main-table rows

`lambada_openai`, 0-shot, all 5,153 items, `acc` = greedy exact match of the whole last word (the same
string for subword and byte-level models). Each model was run alone (`TASKS=lambada_openai`) so the lingua
packing of the default 7-task runs is untouched. ± = half-width of the paired item bootstrap 95% CI
(B=2000, seed 0, as in the main table); Δ = paired difference vs the Transformer.

| model (registry name) | acc | ± | Δ vs Transformer [95% CI] |
|---|---:|---:|---:|
| Transformer (`llama_1.3b`) | 62.66 | 1.33 | – |
| AUNet (`aunet_1.3b`) | 64.62 | 1.30 | +1.96 [+0.68, +3.18] |
| BPEByte (`bpebyte_1.3b`) | 64.78 | 1.27 | +2.12 [+0.85, +3.34] |
| BLT θ=1.34 (`blt_1b`) | 66.33 | 1.26 | +3.67 [+2.43, +4.87] |
| BLT θ=1.61 (`blt_1b_t1609`) | 65.46 | 1.27 | +2.79 [+1.49, +4.04] |
| H-Net 1-stage XL (`hnet_1stage_XL`) | 48.50 | 1.35 | −14.17 [−15.49, −12.79] |

Checks:
- The three matched models equal the earlier `lambada_openai_mt_en` numbers from the multilingual run
  (`reports/multilingual_1B_direct.md` on branch `worktree-multilingual-1b-direct`: 62.6 / 64.6 / 64.8).
- H-Net 48.50 matches the H-Net paper (Hwang et al., Table 2: 1-stage XL LAMBADA 48.4; FineWeb-Edu training).
  Batch size is not the cause: the first 500 items at batch 1 vs 8 give 46.8 vs 46.6 (3 items differ).

Reproduce one row: `TASKS=lambada_openai SUFFIX=lambada GPU=2 bash scripts/probes/std_bench/run_std_bench.sh <model>`
→ `reports/std_bench/<model>/ds0_seed1234_lambada.json`. Per-item bits are in `<model>.json` here.

In the paper (Overleaf `main`): LAMBADA column of `tables/1B_table.tex` (`tab:main_13b`, commit d608779), and the
protocol + in-context / not-in-context breakdown in the appendix (`app:lambada`, `table_appendix/lambada.tex`,
commit 03c78ed). "In context" = the target occurs as a whole word in the passage (4,195 of 5,153 items):

| model | in context | not in context |
|---|---:|---:|
| Transformer | 68.3 | 37.9 |
| AUNet | 70.7 (+2.4 [+1.0, +3.7]) | 38.1 (+0.2 [−2.5, +2.7]) |
| BPEByte | 70.6 (+2.2 [+1.0, +3.6]) | 39.5 (+1.6 [−1.2, +4.5]) |
| BLT θ=1.34 | 71.4 (+3.1) | 44.1 (+6.2) |
| BLT θ=1.61 | 70.4 (+2.1) | 43.9 (+6.1) |
| H-Net | 54.5 (−13.8) | 22.2 (−15.7) |

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

## Few-shot (2026-10-06) — main-table Downstream Avg6

`SHOT=3|5 TASKS=lambada_openai SUFFIX=lambada bash scripts/probes/std_bench/run_std_bench.sh <model>` on gpusvr0908
(fewshot seed 1234 = the seed of the five multiple-choice tasks; lm-eval draws the demonstrations from the test split,
excluding the scored item; every 5-shot prompt is < 2.6 KB, so BLT's 4096-byte context never truncates).
Full outputs: `<model>_ds{3,5}.json` here (= `reports/std_bench/<model>/ds{3,5}_seed1234_lambada.json`).

| model | 0-shot | 3-shot | 5-shot |
|---|---:|---:|---:|
| Transformer | 62.66 | 57.21 | 57.05 |
| AUNet | 64.62 | 60.53 | 60.88 |
| BPEByte | 64.78 | 61.46 | 61.34 |
| BLT θ=1.34 | 66.33 | 61.91 | 61.69 |
| BLT θ=1.61 | 65.46 | 61.07 | 61.30 |
| H-Net | 48.50 | 43.10 | 43.10 |

Demonstrations lower LAMBADA for every model (the harness format has no fill-in-the-blank cue); the Transformer loses
the most. H-Net 3- and 5-shot are equal in total but differ on 572 items.

Paper main table (`tab:main_robust`) Downstream Avg = mean of HS/ARC-E/ARC-C/PIQA/WG + LAMBADA at the same shot count:
`python scripts/probes/ci_downstream6.py [--blt_prefix t1609_]` → `reports/ci_main_table/downstream6{,_t1609}.{json,md}`.
Holm tests with these columns (8 columns × 3 pairs): `python scripts/probes/matched_holm.py --B 10000 --tex <sig_holm.tex>`
→ `reports/sig_holm_avg6/`.

### Why few-shot lowers LAMBADA (analysis 2026-10-06, scripts/probes/std_bench/lambada_fewshot/)

Not context length: the exact prompts (rebuilt with the same lm-eval call, `dump_prompts.py`) are at most 2,003 B
(3-shot) / 2,727 B (5-shot), far inside every model's window (Transformer 4,096 tokens, byte models >= 8,192 B, BLT 4,096 B),
and the drop is flat across prompt-length quartiles (`flip.py`; Transformer 3-shot 5.9/4.8/6.1/5.0 points).

The drop sits on targets that already occur in the passage (81% of items; Transformer -6.2 vs -2.2 points). lm-eval
prepends 3-5 unrelated test passages joined by blank lines with no task marker, so the model reads one document whose
earlier stories offer competing referents. Greedy continuations of the items right at 0-shot but wrong at 3-shot
(`gen_flips.py`, `classify.py`; Transformer 483, BPEByte 397 items): 72% / 62% of these targets are capitalized (mostly
names), but only 31% / 33% of the predictions are; about 15-18% of the flips copy a name that appears only in a
demonstration (Kate->Edward, Ares->Alcander, Gregory->Hardy), and the rest switch to another passage word or a
generic phrase (Hercules->"white one", Cooper->"the man"). Same pattern for every model; the Transformer loses most.

## GPT-3 fill-in-the-blank format (2026-10-06)

Prompt per item: `<passage without its last word> ____. ->` + ` <word>`, demonstrations in the same form joined by
blank lines (Brown et al. 2020). Task `lambada_openai_gpt3` (`scripts/probes/std_bench/tasks/lambada_openai_gpt3.yaml`):
`AUNET_TASKS=<that dir> SHOT=k TASKS=lambada_openai_gpt3 SUFFIX=gpt3 bash scripts/probes/std_bench/run_std_bench.sh <model>`.
lm-eval's own `lambada_openai_cloze_yaml` is unusable few-shot: its demonstrations read `->  word` (two spaces; the
target_delimiter is added only to demonstrations) while the scored continuation is ` word`, so every model scores
0.00 at 3/5 shots. The fixed task sets target_delimiter "" (all 5,153 prompts checked: no double space). The 0-shot
column below is the lm-eval cloze task (no demonstrations, so unaffected). Files: `<model>_gpt3_ds{0,3,5}.json`.

| model | std 0 | std 3 | std 5 | GPT-3 0 | GPT-3 3 | GPT-3 5 |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 62.7 | 57.2 | 57.1 | 5.4 | 59.3 | 66.1 |
| AUNet | 64.6 | 60.5 | 60.9 | 24.2 | 61.7 | 65.5 |
| BPEByte | 64.8 | 61.5 | 61.3 | 28.5 | 66.1 | 70.8 |
| BLT θ=1.34 | 66.3 | 61.9 | 61.7 | 12.5 | 48.0 | 50.2 |
| BLT θ=1.61 | 65.5 | 61.1 | 61.3 | 13.0 | 46.4 | 48.4 |
| H-Net | 48.5 | 43.1 | 43.1 | 12.3 | 39.2 | 43.5 |

Paired differences (item bootstrap B=2000, 95% CI), GPT-3 format: 3-shot BPEByte−Transformer +6.8 [5.6, 8.1],
AUNet−Transformer +2.4 [1.1, 3.8], BPEByte−AUNet +4.4 [3.1, 5.7]; 5-shot +4.7 [3.5, 6.0], −0.6 [−1.9, 0.7], +5.4 [4.2, 6.5].
With the format cue, 5-shot beats the standard 0-shot for the three matched models; BLT and H-Net lose instead
(BLT 50 vs 62 standard 5-shot) — not yet diagnosed.

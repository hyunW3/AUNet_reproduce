# Format / whitespace / punctuation robustness (2026-10-02)

Tables: [`summary.md`](summary.md) (acc, primary), [`summary_acc_norm.md`](summary_acc_norm.md);
machine-readable `summary*.json`; per-run outputs with per-item bits in `raw/`; frozen prompts in `items/`.

## Protocol

- **Items**: the paper robustness protocol — HellaSwag / ARC-E / ARC-C / PIQA / BoolQ, first 2000 docs
  (ARC-C 1172, PIQA 1838 in full), lm-eval `doc_to_text` context + `" <option>"` continuation,
  loglikelihood scoring. acc = argmax total logprob (as in despace_mc / brittle_mc), acc_norm = per byte.
  Clean bits were checked against the paper's despace runs on the first 20 PIQA/BoolQ items: BLT and
  AU-Net 40/40 identical, H-Net and BPEByte 39/40 (one bf16 batch-composition flip each).
- **Models**: AU-Net 2 1.3B (`main/main/1.3B/aunet2_1.3B` step 180000), BPEByte br-greedy-root 1.3B
  (`main/main/1.3B/bpebyte_br_greedy_root_1.3B` step 180000), BLT-1B (official bytelatent, xformers,
  released threshold, bs 1), H-Net 1-stage XL (eval_suite harness, bs 8). Run on ece-agpu18 GPU 5/6.
- **What is perturbed**: only the question block. The final `Answer:` cue line and the options are
  byte-identical to clean. HellaSwag has no cue line, so its whole `label: ctx` is perturbed and no
  trailing pad is added.
- **Rates**: p ∈ {0.5, 1.0}; every eligible site (word gap / punctuation char / symbol) is hit
  independently with prob p from a per-item seed (1234 + idx); p=0.5 sites ⊂ p=1.0 sites.

| family | variant | operation |
|---|---|---|
| whitespace | `space_run` | word gap → 2–4 spaces |
| | `pad_space` | 4 spaces before/after the block + word gap → 5 spaces |
| | `pad_newline` | `\n\n` before/after the block + word gap → `\n\n` |
| | `punct_space` | space inserted before `, . ; : ? !` |
| removal | `punct_drop` | delete `. , ; : ? ! ' "` (+ curly quotes, ellipsis) |
| | `symbol_drop` | delete every other non-alphanumeric non-space char (`( ) - / & % $` …) |
| ReCode-style | `line_split` | word gap → `\n` (LineSplit) |
| | `tab` | word gap → `\t` (Tab-Indent) |
| | `nl_insert` | blank line at line breaks / sentence ends / before the cue (NewlineInsert) |
| NL-Augmenter | `nla_*` | upstream code + default params + fixed seeds: whitespace_perturbation, underscore_trick, butter_fingers, change_char_case, swap_characters, leet_letters |
| FormatSpread | `fs00`–`fs09` | 10 formats sampled (seed 42) from upstream grammar lists: descriptor casing × separator × field joiner |

ReCode itself is a code-generation benchmark (HumanEval/MBPP pass@k), which 1B base models barely solve,
so its format transformations are applied to the MC prompts instead. NL-Augmenter's model-based
transformations (paraphrase, punctuation restoration) were not run.

## Reads (acc, 5-task macro Δ in pt)

1. **More frequent whitespace hurts every model, roughly linearly in p.** At p=1.0: `space_run`
   AU-Net −2.7 / BPEByte −2.9 / BLT −1.4 / H-Net −4.3; `pad_space` −4.2 / −4.8 / −2.6 / −4.2.
   BLT is the most robust to space runs. Extra spaces cost far less than removing spaces did
   (despace all100 cost 7–22 pt, AU-Net worst at 22).
2. **H-Net collapses when every word gap becomes `\n\n`** (`pad_newline_p100` −13.7; ARC-E −26.5,
   BoolQ −15.1, ARC-C −13.3). The other three lose 3.8–4.4. With a single `\n` (`line_split_p100`)
   H-Net loses only 2.5, so the collapse comes from the doubled newline. A `\t` per gap is H-Net's
   next worst (−4.9). acc_norm agrees (−12.5).
3. **AU-Net vs BPEByte**: similar on whitespace, but AU-Net is more robust to `tab_p100`
   (−2.1 vs −3.6), `punct_drop_p100` (−0.7 vs −2.3), `punct_space_p100` (−0.3 vs −1.2) and
   `nla_change_char_case` (−0.1 vs −1.6). AU-Net is never clearly worse than BPEByte; the cells
   where it trails (`space_run_p50` −1.9 vs −1.7, `pad_newline_p50` −3.1 vs −2.9) are within CI.
4. **Removing punctuation or symbols is nearly free.** `punct_drop_p100` costs 0.7–2.3; `symbol_drop`
   is ≈0 for everyone, though only 22–25 % of items contain a symbol to drop.
5. **ReCode-style**: `line_split_p100` −2.5…−4.1 (BLT worst), `tab_p100` −2.1…−4.9 (H-Net worst),
   `nl_insert` ≈0 (−0.2…−0.6): blank lines between sentences don't matter.
6. **NL-Augmenter**: `leet_letters` dominates (−11.8…−15.3, BLT worst). The rest cost 1–5 pt.
   `change_char_case` (10 % of letters flipped) leaves AU-Net untouched (−0.1) but costs
   BLT 2.6 and H-Net 2.2.
7. **FormatSpread spread** (max−min over default + 10 formats): AU-Net 4.6, H-Net 4.0,
   BPEByte 7.8, BLT 7.8. The bad formats are the ones that put whitespace-control separators
   between descriptor and text (`' \n\t'`, `'\n    '`, `'\n '`: fs01/04/05/06/09). The
   colon-style variants (`':::'`, `':'`, `'::'`, `'- '`) are within ±1 pt of default.

## Caveats

- BoolQ often *gains* under AU-Net whitespace perturbations (+1…+2.6 pt). It is a 2-way yes/no
  task where prompt shifts move the yes/no prior, so treat BoolQ deltas as prior shifts, not
  robustness.
- H-Net (bs 8) and the lingua models batch requests, so ~1/40 items can flip from batch composition
  alone (see clean-bit check above). BLT runs at bs 1 and is deterministic (its shard A/B clean bits
  are identical).
- Side finding: `ece-agpu18:~/AUNet_lc/ckpt/bpebyte/consolidated.pth` (used by the long-context suite)
  symlinks to `runs/bpebyte_br_greedy_root_1.3B_a100x4/.../0000180000`, whose weights differ from the paper
  checkpoint (first-50 MB md5 `b4a3207f…` vs `378ded14…`). This run used a verified copy of the paper
  checkpoint (`~/AUNet_fmt/ckpt/bpebyte`).

## Reproduce

```
# items (once, needs lm-eval + lingua):  PYTHONPATH=lingua python scripts/probes/format_mc/dump_items.py --items_dir reports/format_robustness/items
# on ece-agpu18 (~/AUNet_fmt = scripts/format_mc, items/, ckpt/, hnet_repo/):
bash scripts/format_mc/run_ece.sh <gpu> aunet|bpebyte|hnet ~/AUNet_fmt/full
bash scripts/format_mc/run_ece.sh <gpu> blt ~/AUNet_fmt/full 2000 <task> A|B     # 10 shards
python scripts/probes/format_mc/report.py --in_dir reports/format_robustness/raw --out_dir reports/format_robustness [--metric acc_norm]
```

# FFLM state-tracking probe

Does a byte-level transformer track state as well as a subword one? This probes
frozen **1.3B** checkpoints on the **Flip-Flop Language Modeling** task
(Liu et al. 2023, *Exposing Attention Glitches with Flip-Flop Language Modeling*,
[arXiv:2306.00946](https://arxiv.org/pdf/2306.00946)).

## The task

Sequences over `{w, r, i, 0, 1}`, alternating instruction/bit, begin `w`, end `r`:

```
w 0 i 1 i 0 r 0 w 1 i 0 r 1 ...
```

A 1-bit memory holds the **last written** bit. `w b` writes, `i b` ignores
(state unchanged, `b` is a distractor), `r b` reads — `b` **must** equal the
current state. We score the model's prediction of the bit right after each `r`.

Regimes (via `p_i` = fraction of ignores; `p_w=p_r=(1-p_i)/2`):

| regime  | `p_i` | character |
|---------|-------|-----------|
| dense   | 0.10  | short-range, many reads |
| in-dist | 0.80  | the paper's train distribution |
| sparse  | 0.98  | long-range dep — exposes "attention glitches" |

## What we compare

Three checkpoint families, each at byte or subword granularity:

| family            | checkpoint                       | granularity |
|-------------------|----------------------------------|-------------|
| `subword_llama`   | `llama_1.8B_paper`               | flat BPE (128k) |
| `aunet_static`    | `aunet2_1.3B`                    | byte → word static pooling |
| `byte_greedyroot` | `bpebyte_br_greedy_root_1.3B`    | byte, greedy-root pooling |

> There is **no flat 1-level byte** 1.3B checkpoint; `aunet2` and `br_greedy_root`
> are the byte-input models (both 2-level hierarchical, differing only in the
> pooling rule). The three are trained **iso-byte** (matched on training bytes /
> compute, not steps): Llama's `@60k` and the byte models' `@180k` reflect
> bytes-per-token, not a shorter budget — so this is a *fair* iso-byte comparison,
> not a step-mismatched one.

## Metrics

- **greedy_acc** — the model's argmax next token equals the correct bit. This is
  the paper's strict metric (they call any run below 100% a *reasoning error*).
  Chance = 0.50.
- **binary_acc** — `P(correct bit) > P(wrong bit)`: pure 0-vs-1 discrimination,
  ignores whether the model would even emit a bit as its top token.
- **margin** — mean `logprob(correct) − logprob(wrong)`.

Scoring reuses each codebase's tested `generator.generate` loglikelihood path, so
it is tokenization-agnostic across the three families.

## Run

```bash
bash scripts/fflm/run_fflm.sh          # picks the freest GPU, writes reports/fflm/
```

Env knobs: `GPU`, `NUM_INSTR` (T=2·NUM_INSTR), `N`, `N_SPARSE`, `MAX_READS`,
`SHOTS` (few-shot demos, default 0 = in-sequence context only), `BATCH`,
`REQUIRE_IDLE=1` (wait for an idle GPU before starting).

Outputs: `reports/fflm/results.jsonl`, `reports/fflm/summary.md`,
`reports/fflm/fflm_read_accuracy.png`.

## Files

- `fflm_data.py` — FFL(p) generator (`python fflm_data.py` self-tests invariants).
- `fflm_probe.py` — loads a checkpoint, scores reads (`--family subword|aunet`).
- `run_fflm.sh` — runs all three families × three regimes.
- `report_fflm.py` — table + grouped bar chart.

## Notes / caveats

- **In-context, no training.** The paper *trains* small models on FFLM; here we
  ask whether pretrained 1.3B LMs state-track zero-/few-shot on an OOD format.
  Low absolute numbers (esp. sparse) are expected and are themselves the finding.
- `SHOTS>0` prepends demo sequences; default 0 relies on the pattern being
  established within the single test sequence.
- `/tmp` is noexec on this box; the launcher redirects triton/inductor caches to
  `scripts/fflm/.cache/`.

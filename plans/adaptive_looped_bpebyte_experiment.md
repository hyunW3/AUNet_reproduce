# Adaptive Looped Byte Transformer — OnlineBPE-guided recursive depth

Implementation of `brainstorming/adaptive_looped_bpebyte_codex_plan.md`, adapted to this repo.
Launched 2026-09-09.

## The model

`embed → one shared Transformer stack → vocab`, over raw bytes. Bytes that the OnlineBPE parse
marks get **3** passes through the *same* parameters; every other byte gets **1**.

```
h1, KV1 = F(x)                  # ordinary pass, keeps its k/v
h2      = F(h1, kv_in=KV1)      # q from h1, memory from pass 1
h3      = F(h2, kv_in=KV1)
h       = where(boundary, h3, h1)
```

Two properties make this adaptive **depth** rather than a 3× deeper model:

- K/V come from recursion 1 only, so recurrent passes add no attention memory. The KV cache is the
  base model's — 42 KB/token at 100M, against 57–60 KB/token for the FLOPs-matched deeper Llamas.
- Non-boundary positions output `h1`, and nothing any position *reads* is written by recursion 2/3.
  So a decoder can skip recursions 2/3 at non-boundary positions outright.

## Three things that changed from the plan, and why

**1. Backbone: flat byte transformer, not the hierarchical BPEByte.** The plan is written for a
byte-level model with one shared stack (§1, §3, §8.1). In this repo "BPEByte" is a hierarchical
AU-Net (byte encoder → pool at patch starts → wide trunk → up → byte decoder), where boundary
positions *already* get the extra compute — that is what the trunk is. Grafting recursion there
would blur what the experiment isolates. Instead the arms are all **flat** byte models, which
`HierarchicalTransformer` already builds for free: a single-entry `dimensions: [d]` gives zero
encoders and zero transitions, and the stack *is* `trunk`. Crucially `level_mask` — the OnlineBPE
boundary mask — is already an argument to `forward()` and simply unused on a flat config, so the
routing signal needs no data-pipeline change at all.

**2. The boundary is the patch START, not the patch end.** Verified in
`apps/aunet/data/regex_cutting.py`: with `bpe_online_placement: root` (what every current BPEByte
run uses), `_PLACEMENT_OFF["root"] == 0` and `online_byte_boundaries` returns `min(e - off, n-1)`
for end-exclusive token ends `e`, so `level_mask[t] == 1` marks the **first byte of the next
patch**.

The more intuitive place to spend compute would be the *last* byte of a patch — that byte is the
one that has to predict the first byte of a new word, and it carries the high loss. That is the
`before_root` convention, and it is **not available to us**: deciding that byte `e-1` ends a token
requires seeing byte `e`, a future-byte leak. `root` is the leak-free convention (marking byte `e`
needs only bytes ≤ `e`), and it is what the causal model can actually route on. So: **patch starts
get the extra depth**, and the story is "the extra passes let a byte integrate the token that just
closed", not "the extra passes help predict the next word".

**3. p ≈ 0.125 / E[R] ≈ 1.25 does not hold here.** Measured on 400 DCLM docs (2.17 MB) through the
actual pipeline (llama3 root_greedy):

```
boundary rate  0.2179      bytes/patch  4.589      mean recursion  1.436
```

Training confirms it: the 100M looped arm logs 7168/32768 = 0.2188 per batch. Every FLOPs-matched
baseline depth below is derived from the **measured** rate, not the plan's nominal one.

## Arms

`python adaloop_gen_configs.py` writes all of them from one template, so a diff between two YAMLs
shows exactly the axis under test and nothing else.

| scale | arm | L | params | mean R | inference FLOPs | dense-training FLOPs | KV KB/tok |
|---|---|---:|---:|---:|---:|---:|---:|
| 30M | base | 8 | 27.54M | 1.000 | 1.000× | 1.000× | 16.0 |
| 30M | deep11 | 11 | 37.76M | 1.000 | 1.374× | 1.374× | 22.0 |
| 30M | looprnd | 8 | 27.54M | 1.436 | 1.393× | 2.804× | 16.0 |
| 30M | **loopbpe** | 8 | 27.54M | 1.436 | 1.393× | 2.804× | 16.0 |
| 100M | base | 14 | 99.51M | 1.000 | 1.000× | 1.000× | 42.0 |
| 100M | deep19 | 19 | 134.91M | 1.000 | 1.357× | 1.357× | 57.0 |
| 100M | deep20 | 20 | 141.99M | 1.000 | 1.428× | 1.428× | 60.0 |
| 100M | looprnd | 14 | 99.51M | 1.436 | 1.385× | 2.767× | 42.0 |
| 100M | **loopbpe** | 14 | 99.51M | 1.436 | 1.385× | 2.767× | 42.0 |
| 30M/100M | loopres | 8/14 | +2 params | 1.436 | 1.393/1.385× | 2.804/2.767× | 16.0/42.0 |

`deep19` (1.357×) and `deep20` (1.428×) bracket `loopbpe` (1.385×), so the compute–performance
curve can be interpolated exactly at the matched point rather than compared to a rounded depth.

`looprnd` is **count-matched per sequence**: it takes exactly as many depth-3 positions as
OnlineBPE would, chosen uniformly at random, so it has the same parameters, the same mean recursion
count, the same FLOPs and the same KV design. The only thing that differs is *which* positions were
picked. That is the arm that isolates whether OnlineBPE boundaries mean anything.

`loopres` is `loopbpe` **plus a zero-init residual gate** on each recurrent update (added
2026-09-10, prompted by arXiv 2605.05222 "Adaptive Computation Depth via Learned Token Routing",
whose TSA is a learned per-token gate on residual updates between blocks):

```
h_r = h_{r-1} + g_{r-1} * (F(h_{r-1}) - h_{r-1}),    g init 0
```

The ungated loop is *already* a residual update — `F` is pre-norm, so
`out = h1 + boundary*(Δ(h1) + Δ(h2))` — what it lacks is a scale. Measured at init on the real
100M shape, the residual stream grows RMS **3.26 → 6.13 → 9.16** across the loop and a routed byte
leaves the stack at **||h3−h1||/||h1|| = 1.97**, cos(h1,h3) = 0.89. So the ungated model hands the
head a sequence in two regimes — 22% of bytes at one scale, 78% at another — and the first thing it
must learn is to undo that; `base` carries no such handicap, which makes it a confound in the
`loopbpe` vs `base` comparison. With `g = 0` the looped model **is** the base model at step 0
(confirmed in training: `loopres` and `base` both log loss 3.2717 at step 20, while `loopbpe` logs
3.2757). Same principle as `nonboundary.py`'s zero-init `out_proj` and `patch_lag.py`'s zero-init
updater. Costs 2 parameters. `loopbpe` is deliberately kept ungated, or the gate's effect could not
be measured.

> **Measured 2026-09-10, and the reasoning above is wrong.** 30M complete: ungated `loopbpe`
> **1.2635**, gated `loopres` **1.2750**, base 1.2757. The gate does not merely fail to help — it
> removes the loop's entire gain (−0.0122 → −0.0006), at 2.9× the noise floor.
>
> The initial discontinuity is not a handicap to be undone; it is the condition under which the
> loop gets trained. With `g = 0` we have `∂out/∂h_D = 0`, so no gradient reaches the recurrent
> path early, and by the time `g` opens the rest of the network has already settled into lowering
> the loss without the loop. Zero-init suits `nonboundary.py` / `patch_lag.py` because those are
> auxiliary heads *beside* the main path; this gate sits *on* it.

`loopunif` (every byte depth 3) is generated but not launched — a pure-recurrence upper bound for a
spare GPU.

**Dense training costs 2.8× inference.** Recursions 2/3 are computed at *every* position and masked
at the output (plan §3: correctness before sparse kernels). This is a property of the current
training path, not of the model — the adaptive inference cost really is 1.39×. Both numbers are
reported separately everywhere (`flops_per_token()` returns the dense one; `looped.py:flops_report`
returns the adaptive one). The practical consequence is that the looped arms take ~2.8× the
wall-clock of the base arm for the same number of bytes.

## Everything held fixed

Same data, same `data.seed` and world size (so the same batches in the same order), same bytes seen,
optimizer, LR, cosine schedule, warmup, batch, sequence length, weight decay, bf16, grad clipping,
init. Every arm runs the OnlineBPE data pipeline, including the arms that ignore `level_mask`, so
the dataloader and its cache are identical across arms.

Full causal attention (no sliding window) at `seq_len 4096`, identical in every arm. 4096 rather
than the repo's usual 8192: attention is ~50% of a flat byte model's FLOPs at that length, and
halving it buys back the dense-training overhead. Absolute BPB is therefore **not** comparable to
the repo's 8192-context BPEByte runs — only the arms are comparable to each other.

## Where things live

```
lingua/apps/aunet/looped.py             routing masks, the recursion driver, FLOPs/KV accounting
lingua/apps/aunet/test_looped.py        7 correctness tests (see below)
lingua/apps/aunet/adaloop_profile.py    params / KV / FLOPs / latency / peak memory per arm
lingua/adaloop_gen_configs.py           config generator + the summary table above
lingua/adaloop_run_local.sh             30M launcher (gpusvr0908, one arm per GPU)
lingua/adaloop_run_ece.sh               100M launcher (run ON ece-agpu11 / ece-agpu18)
lingua/adaloop_status.sh                one line per arm: step, loss, BPB, boundary rate, ETA
lingua/adaloop_watch.sh                 emits only ERROR / DEAD / DONE state changes
lingua/apps/aunet/configs/adaloop/*.yaml
```

Two upstream files carry small opt-in additions, both default-off and bit-identical when unused:
`lingua/transformer.py` gained `kv_in` / `return_kv` on `Attention`, `TransformerBlock` and
`BaseTransformer`; `apps/aunet/hierarchical.py` gained the `loop_*` args, the recursion branch in
`forward`, and a `flops_per_token()` that no longer crashes on a zero-encoder (flat) config.

## Tests

`.venv/bin/python -m apps.aunet.test_looped` — all pass on A5000 and A100.

1. `loop_routing=""` and a no-boundary looped model are **bit-identical** to the baseline (max|Δ| = 0).
2. Strict sharing: same parameter count as the base model, and a base checkpoint loads `strict=True`.
3. Causality: rewriting every byte *and every routing decision* after position 30 moves nothing at
   ≤ 30 (max|Δ| = 0) and does move > 30.
4. Shared KV: 3 q projections per layer, exactly **1** k and **1** v.
5. Dense masking really is `where(boundary, h3, h1)`.
6. Routing: `online_bpe` fires exactly on `level_mask > 0`; `random` is count-matched per sequence.
7. The routing mask is read **before** `get_pool_mask(force_first=True)`, which writes
   `level_mask[:, 0]` in place and would otherwise silently clear position 0 on a flat config.

## Where it is running

| host | GPU | arm | status |
|---|---|---|---|
| gpusvr0908 (A5000) | 0/1/2/3 | 30M base / deep11 / looprnd / loopbpe | **done** |
| gpusvr0908 (A5000) | 0/1/2/3 | 30M loopres / base·loopbpe·looprnd @ seed 778 | running |
| ece-agpu11 (A100) | 0/1/2/3 | 100M base / deep19 / looprnd / loopbpe | running |
| ece-agpu18 (A100) | 7 | 100M deep20 | running |
| ece-agpu18 (A100) | 5 | 100M loopres | running |

The seed-778 repeats exist because the 30M deltas below are 0.01–0.03 BPB with no error bar. They
re-run three arms with a different `seed` and the **same `data.seed`**, so every arm still sees the
same bytes in the same order and the only thing that moves is initialisation/optimisation noise.
Shared accounts: on ece-agpu11 GPUs 5–7 and on ece-agpu18 GPUs 0–4 belong to other users — check
`nvidia-smi --query-compute-apps` ownership before claiming a card.

Budget: 30M = 10,000 steps × 262,144 B = **2.62 GB**; 100M = 38,000 steps × 262,144 B = **9.96 GB**.
Measured throughput: 30M base 1.26e5 B/s, 30M looped ~6e4 B/s; 100M looped 4.67e4 B/s @ 37% MFU.
So the 30M sweep finishes in ~6–12 h and the 100M baselines in ~21–29 h, while the 100M **looped**
arms need ~2.5 days — the dense-training overhead, not a slower model.

## Results so far

**30M, complete** (2.62 GB, 10,000 steps; BPB = tail-mean `loss/out` over the last 200 steps / ln 2;
widening the tail to 4,000 steps leaves the ordering and the gaps essentially unchanged):

| arm | L | params | inf FLOPs | KV/tok | BPB | vs base |
|---|---:|---:|---:|---:|---:|---:|
| base | 8 | 27.5M | 1.000× | 16 KB | 1.2757 | — |
| **deep11** | 11 | 37.8M | 1.374× | 22 KB | **1.2493** | **−0.0264** |
| loopbpe | 8 | 27.5M | 1.393× | 16 KB | 1.2635 | −0.0122 |
| looprnd | 8 | 27.5M | 1.393× | 16 KB | 1.2779 | +0.0022 |

**100M, in flight**, every arm scored at the common step 11,060 (2.90 GB — near the 30M budget), mean
over the preceding 2,000 steps: deep20 1.2429 (−0.0245), deep19 1.2456 (−0.0217),
loopbpe 1.2631 (−0.0043), base 1.2674, looprnd 1.2699 (+0.0025). **Same ordering as 30M.**

- **Q3 — the boundaries carry real signal.** Count-matched random routing lands at **+0.0025 at both
  scales**: it buys nothing, marginally worse than base. Same parameters, same FLOPs, same *number*
  of depth-3 positions — change only *which* positions, to the OnlineBPE boundaries, and 30M gains
  0.0144 and 100M gains 0.0068. The win comes from where the compute goes, not from recurrence.
- **Q2 — yes.** loopbpe beats base by 0.0122 at identical parameters and identical KV cache.
- **Q1 — no, at this budget.** `deep11` beats `loopbpe` by 0.0142 at matched inference FLOPs, and
  deep19/deep20 lead at 100M. But `deep11` pays for it with **+37% parameters and +37% KV cache**;
  `loopbpe` gets about half the gain for **zero** of either. If the binding constraint is memory
  rather than FLOPs the comparison inverts, which is what Q4 was asking.

Caveat: no error bars yet — that is what the seed-778 repeats are for. Do not quote these deltas as
significant until the repeats land.

## Reading the result

Primary metric is `loss/out` from each run's `metrics.jsonl` (tail-mean over the last 200 steps);
for byte models **BPB = loss/out / ln 2**, directly comparable across arms without any extra eval.
`nb_toks/enc_0` and `nb_toks/enc_1` in the same file are the depth-3 byte count and the total byte
count — their ratio is the empirical boundary rate.

The questions the arms answer:

- **Q1** at matched inference FLOPs, is OnlineBPE-routed recursion better than uniformly more
  layers? → `loopbpe` vs the `deep19`/`deep20` interpolation at 1.385×.
- **Q2** at matched parameters, does selective recursion beat the base model? → `loopbpe` vs `base`.
- **Q3** does OnlineBPE routing beat random at the same rate? → `loopbpe` vs `looprnd`. **This is
  the load-bearing comparison**; if it is flat, the boundaries carry no signal and any gain over
  `base` is just recurrence.
- **Q4** without growing the KV cache? → already answered by construction: 42 vs 57–60 KB/token.

## Not done yet, deliberately

- **Sparse inference** (plan §14/§25.14): recursion is dense-only. The plan puts this after dense
  validation, and it is what would turn the 2.8× training cost into ~1.4×.
- **Decode latency** (plan §22): the looped model has no incremental decoder —
  `apps/aunet/generate.py` re-implements the forward pass for its cached path without recursion.
  `adaloop_profile.py` reports prefill latency and peak memory and says so rather than timing the
  dense path and calling it decode.
- **Learned router** (plan §19): future work, as specified.
- **Diagnostics** (plan §17): loss split by boundary/non-boundary, and `L^(1)` vs `L^(2)` vs `L^(3)`
  at boundary positions. `looped_forward(..., collect_hidden=True)` already returns `[h1, h2, h3]`
  for this; the scoring script is not written yet. Worth running on the finished checkpoints.

# iso-FLOP-365 comparison — byte model vs FLOP-matched llama

A **compute-matched** (not param-matched) comparison of the byte AU-Net against a subword llama,
both at **~365 M FLOP/byte**. Answers: *at equal training compute, does byte-level representation
beat a subword transformer* — which here carries ~2.4× more parameters for the same FLOPs.

## The two arms (both ≈365 M FLOP/byte, same total FLOPs)

| arm | config | shape | non-embed | FLOP/byte | steps | total FLOP |
|-----|--------|-------|-----------|-----------|-------|------------|
| byte  | `apps/aunet/configs/bpebyte_100M_v4_root_greedy.yaml` | `[512,768]/[3,10]` | 98 M | 365.2 M | 1672 | 3.842e18 |
| llama | `apps/main/configs/llama_iso365.yaml` | dim1024 / 18L / 16h | 231 M | 363.7 M | 2218 | 3.842e18 |

Both train on the same **~10.5 B bytes of text** (byte: 10.52 B bytes; llama: 2.33 B subword
tokens × 4.541 B/tok = 10.56 B). Total training FLOPs match to **−0.02 %**.

Contrast with `llama_100M_cmp.yaml`, which is iso-**param** (99 M non-embed both) and therefore
iso-**text** but *half* the FLOPs — that's the iso-param comparison; this bundle is the iso-FLOP one.

## Run it
```bash
# from the repo root (parent of lingua/)
bash scripts/isoflop365/run_isoflop365.sh 4      # 4 GPUs; trains both arms then evals BPB
```
The driver: prints the FLOP-match check, trains the byte arm, trains the llama arm, then scores
**BPB = |nll_per_byte| / ln2** on the same held-out DCLM `*.val.jsonl` for both (via each app's
`eval_on_val`, which normalizes NLL by encoded **bytes**, not tokens — so subword vs byte is
apples-to-apples). Lower BPB at equal FLOPs wins.

## Re-derive the sizing
If you change GPUs / batch / data budget, re-solve the llama step count:
```bash
cd lingua && PYTHONPATH=$PWD python ../scripts/isoflop365/isoflop365_plan.py --gpus 8
```
It prints the byte reference FLOPs, a candidate grid, and the matched steps + residual error.

## Files
- `apps/main/configs/llama_iso365.yaml` — grown llama arm.
- `apps/main/configs/eval_bpb_llama_iso365.yaml` — llama BPB eval.
- `apps/aunet/configs/eval_bpb_bpebyte_v4.yaml` — byte BPB eval.
- `scripts/isoflop365/isoflop365_plan.py` — sizing / FLOP-match calculator.
- `scripts/isoflop365/run_isoflop365.sh` — train-both + BPB driver.

## Caveats
- **HP:** llama's LR is inherited from `llama_100M_cmp` (3e-3). For a 2.3× larger model it should
  ideally be re-tuned (short sweep or muP) — otherwise the larger arm may be under-served and the
  comparison confounded. Decide whether you're matching HP *protocol* or tuning per arm.
- **FLOP convention:** the byte arm's FLOP/byte uses `hierarchical.flops_per_token`; llama's uses
  `get_num_flop_per_token`. Both are dominated by 6·N·D; the attention terms use slightly different
  constants (a ~1–2 % difference), folded into the step-matched total.
- **One point ≠ a frontier.** A single iso-FLOP point can favor either arm at an architecture-specific
  sweet spot. For a real verdict, run this at ≥3 compute budgets and compare BPB-vs-FLOP frontiers
  (slots into the existing root_greedy compute-frontier — just add llama points).
- **FLOP ≠ wall-clock:** at iso-FLOP the byte arm is actually a bit *faster* in wall-clock (higher MFU);
  a throughput-matched variant is the stricter test.

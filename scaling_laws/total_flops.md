# Total training FLOPs per scale — BPEByte root_greedy (rg)

Total training compute for the rg (leak-free byte) ladder, used as the x-axis of the
compute-frontier scaling figure (`scaling_rg_bpb.png`).

## Method
Compute is recovered **from the training logs**, not assumed: per run,
`flops_per_token = median(speed/FLOPS ÷ speed/wps)` over all logged steps (the same recovery as
`runs/plot_bpb_compare.py`), then **total FLOPs = flops_per_token × total_tokens**. For byte
models `total_tokens` is in bytes, so `flops_per_token` is FLOP/byte. 760M and 1.3B are measured;
100M and 300M are extrapolated from the fitted `fpt(N) = 46 · N^0.866` FLOP/byte (anchored on the
two measured points) and will be confirmed from their live metrics as the runs complete.

## Per-scale totals (rg, byte)

| scale | N (non-embed trunk) | budget | **total training FLOPs** | ZFLOP (1e21) | source |
|---|---|---|---|---|---|
| 100M | 98.6M | 42 GB | **~1.6 × 10¹⁹** | 0.016 | est |
| 300M | 296M  | 42 GB | **~4.2 × 10¹⁹** | 0.042 | est |
| 760M | 679.6M | 69 GB | **1.41 × 10²⁰** | 0.141 | measured |
| 1.3B | 1.30B | 283 GB | **1.02 × 10²¹** | 1.017 | measured |

`fpt` (FLOP/byte): 100M 3.85e8 · 300M 9.98e8 · 760M 2.05e9 (meas) · 1.3B 3.59e9 (meas).

## Notes
- **Span ≈ 63×** in total compute across the ladder (1.0e21 / 1.6e19), vs 13× in params and
  6.7× in bytes — both N and D grow together.
- **`6·N·D` overestimates byte models by ~1.5–2.2×** (6ND = 2.5e19 / 7.5e19 / 2.8e20 / 2.2e21 for
  100M/300M/760M/1.3B). The AU-Net hierarchy pools bytes into far fewer trunk patches, so per-byte
  FLOPs is well below the dense `6N`; the discount grows with scale (measured÷6ND: 0.65 → 0.56 →
  0.50 → 0.46) as the trunk dominates.
- **Llama (subword) is much cheaper per byte** — measured 0.088 ZFLOP (760M) and 0.676 ZFLOP
  (1.3B) — since it processes ~4.5× fewer tokens than bytes (bytes/token ≈ 4.55).
- Budgets are the allotted per-run budgets (not iso-data): 42/42/69/283 GB → bytes/param
  426/142/102/218 (see `scaling_laws_plan.md`).

_Regenerate the measured numbers anytime with the `fpt × total_tokens` recovery over each run's
`metrics.jsonl`; estimates refresh automatically once 100M/300M finish._

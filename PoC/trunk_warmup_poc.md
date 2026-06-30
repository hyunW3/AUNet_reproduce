# Trunk warm-start PoC — token/byte-LM → AU-Net global transformer

**Question.** Does initializing the AU-Net **trunk** (the "global" transformer over pooled patches)
from a *pretrained flat transformer* speed up / improve AU-Net training? Tested at **100M** and
**500M**, for two boundary schemes, with a granularity control.

## Method
- The AU-Net trunk (`CausalTransformer` over pooled patches) and a flat `LMTransformer` both extend
  `BaseTransformer`, so per-layer block keys/shapes match when block hyperparameters match.
- `apps/bytepoc/transplant_into_trunk.py`: build a fresh AU-Net, copy a pretrained flat model's
  `layers.{i}.*` → `trunk.layers.{i}.*` (shape-checked), leave encoders/decoders/transitions/
  tok_embeddings at fresh init, save a full AU-Net dcp checkpoint for stage-B `init_ckpt_path`.
- **Two warmups** (same block dims as the trunk): `subA` = subword token-LM (tiktoken 128k);
  `subA_byte` = byte-LM control (isolates *granularity* vs the transplant itself).
- **Two boundary schemes**: `br_bt` (before_root, online bt — the *strong* scheme) and
  `root_greedy` (root placement, online greedy — the *weak* scheme).
- Metric: train **BPB** = `loss/out / ln2`. Iso-budget framing (~3.1B bytes stage-B); warm runs
  additionally see the warmup compute. No seed repeats.

## Results (final BPB)

**100M** (trunk dim768/10L):

| scheme | scratch | subword-warm | byte-warm |
|--------|--------:|-------------:|----------:|
| br_bt (strong)   | 1.118 | 1.131 | 1.145 |
| root_greedy(weak)| 1.357 | 1.250 | 1.256 |

Δ vs scratch: br_bt subword **+0.014 / byte +0.027 (WORSE)**; rg subword **−0.107 / byte −0.101 (BETTER)**.

**500M** (trunk dim1536/24L, 12k steps ~3.1B bytes):

| scheme | scratch | subword-warm | byte-warm |
|--------|--------:|-------------:|----------:|
| br_bt (strong)   | 1.324 | 1.155 | 1.153 |
| root_greedy(weak)| 1.403 | 1.341 | 1.295 |

Δ vs scratch: br_bt subword **−0.169 / byte −0.171**; rg subword **−0.062 / byte −0.108** (all BETTER).

## Findings
1. **Scale flips the verdict for the strong scheme.** At 100M, warm-start *hurts* br_bt (+0.02) and
   only helps the weak root_greedy (−0.10). At **500M it helps BOTH, most of all br_bt (−0.17)** —
   the largest gain in the study.
2. **Why the flip — undertraining regime.** At a fixed ~3.1B-byte budget, 500M sees ~6 tokens/param
   vs 100M's ~31; its from-scratch trunk is far from converged, so a pretrained trunk is a big
   head-start. br_bt's scratch trunk was "good enough" only because 100M was near its budget-optimal
   point. Read as: **warm-start's benefit grows as the model is more undertrained at fixed data** —
   not a clean "scales to compute-optimal 500M" claim.
3. **Granularity edge shifts to byte at scale.** 100M: subword-warm slightly beats byte-warm in both
   schemes (−0.006 to −0.014). 500M: they tie on br_bt and **byte-warm beats subword on root_greedy
   by 0.046**. The "token-granularity matches the trunk" intuition no longer dominates at 500M.

## Caveats
Train BPB only; single run per cell (no seed repeats, ~±0.005 noise — the 100M granularity gap is
near noise; scheme and scale effects are well above it). 500M is heavily undertrained at this budget,
so its absolute BPB sits *above* 100M (expected) — **only within-scale scratch-vs-warm deltas are the
headline**; cross-scale absolute BPB is not comparable (different seq/batch + undertraining). Byte-warm
matches compute but sees ~4.5× less source text.

## Artifacts
- Code: `apps/bytepoc/transplant_into_trunk.py`; configs `apps/bytepoc/configs/{tokwarm_subA*,
  aunet_*}.yaml` (100M) and `{tokwarm500_subA*,aunet_*_500}.yaml` (500M). `apps/aunet/train.py:293`
  patched to reset nested rope buffers for AU-Net `init_ckpt_path`.
- Runs: `runs/tokwarm/` (100M, ece) and `ece:/home/hwbae/AUNet/runs/tokwarm500/` (500M).
- Detailed log: `runs/tokwarm/RESULT.md`.

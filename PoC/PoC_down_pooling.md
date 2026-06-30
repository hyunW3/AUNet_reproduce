# PoC: content-aware down-pooling for BPEByte (richer trunk patch tokens)

Status: **A0/A1 ablation RUN (2026-06-29) — see §7.** `last_byte` showed a large BPB drop (−0.062)
that **matches the §2 leak magnitude almost exactly**, and the leak probe was never run → most likely
the future-byte leak this PoC was designed to avoid, not genuine enrichment. **L1 down-pool code dropped
from the working tree** (hierarchical.py reverted, A1 config deleted) pending a leak audit; never
committed. Code references verified against `lingua/apps/aunet/hierarchical.py` @ commit 5b28e31.

## 1. The observation: "pooling" is a single-byte gather

In the AU-Net hierarchy the trunk does not run on a *summary* of each patch — it runs on the
encoder state at **one byte per patch**. `SimpleTransition.down` (`hierarchical.py:333-354`):

```python
idx = self.idx = self.max_sum_mask(mask)                       # boundary positions
x = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.dim_in)) # ONE byte per patch
out = self.trans_down(x)                                       # 512 -> trunk dim linear
```

`MaxSumMask` (`hierarchical.py:271-285`) returns the positions where `mask==True`. Under **root**
placement those `True` positions are the patch **starts** (the first byte of each patch, index `e` =
the next token's first byte; `regex_cutting.py` `off=0`). So:

> **trunk input token `k` = encoder hidden state at `start_k`, the FIRST byte of patch `k`.**

The encoder is a 3-layer transformer with a 512 sliding window, so this state is *contextualized*
over the previous ~512 bytes — it is not literally one byte of information. But it provably contains
**nothing about the interior of patch `k` itself**: every byte in `(start_k, start_{k+1})` is in the
strict future of `start_k`, so the causal encoder state at `start_k` cannot have attended to them.

**Consequence — a one-patch content lag.** Each patch's own interior content first reaches the trunk
only at token `k+1` (whose start `start_{k+1}` = the end of patch `k`, so `encoder@start_{k+1}` has
attended over all of patch `k`). The global model therefore reasons over a sequence in which "what
patch `k` actually contained" arrives one trunk step late. This is a plausible reason the hierarchy
extracts little advantage from BPE-placed boundaries: at 1.3B, leak-free `root_greedy` train BPB
(0.860) is *worse* than subword Llama (0.839) and only ties whitespace patches (AU-Net2, 0.866) — the
trunk barely sees inside its own patches.

## 2. Why you cannot just pool harder (the leak nobody mentioned)

The obvious fix — gather the patch's **last** byte, or **mean**-pool its interior — is **NOT
leak-free** under the current read path, and this is the trap to avoid.

Read path (`up`, `hierarchical.py:393-411`): `repeat_idx = mask.cumsum(1)` (base-0), so the hidden at
output position `t` (which predicts byte `t+1`) reads trunk token `cumsum(mask)[t]`. For `t` in the
interior of patch `k` (`t ≥ start_k`) that index is `k` — **the byte reads its own patch's token.**

Now suppose trunk token `k` pooled the *whole* of patch `k` (up to `start_{k+1}-1`). Predicting byte
`t+1` for an interior `t < start_{k+1}-1` would then consume information from bytes up to
`start_{k+1}-1 > t` — **future-byte leak.** The leaky `bt` variants buy ~0.07 BPB exactly this way;
content-pooling-without-care silently reintroduces it.

**The clean result.** Under root placement with zero read delay, the trunk token a byte reads is
indexed by `cumsum(mask)[t]`, and the smallest interior `t` that reads token `k` is `t = start_k`.
So trunk input token `k` may depend on **bytes ≤ `start_k` only**. The first-byte gather is therefore
already the *maximal-information causal pooling* for the current read scheme — you genuinely cannot
add patch-`k` interior content to token `k` without either a leak **or** a read delay.

## 3. The proposal: content pool + one-patch read shift

To feed the trunk a real per-patch summary while staying leak-free, pool each **completed** patch and
consume it with the natural one-patch shift: when predicting inside patch `k`, read the full summary
of patch `k-1` (which is entirely in the past), instead of the encoder state at `start_k`.

Two equivalent ways to implement the shift; pick whichever is cleaner:

- **(a) Shift the input.** Set trunk input token `k` = `pool(bytes of patch k-1)`. Keep the existing
  `cumsum` read. Byte in patch `k` reads token `k` = summary of patch `k-1` ⊆ bytes `< start_k ≤ t`.
  Leak-free by construction; reuses all existing `repeat_idx` machinery.
- **(b) Keep input aligned, delay the read.** Set trunk input token `k` = `pool(bytes of patch k)`,
  and read with `patch_read_delay = 1` (`hierarchical.py:396-402`, already implemented) so byte in
  patch `k` reads token `k-1`. Same effect.

Pooling operator variants (ranked by cost):

| variant | trunk input token `k` | params | note |
|---|---|---|---|
| **last-byte** | `encoder@(start_k - 1)` (last byte of patch `k-1` under shift (a)) | 0 | cheapest; a 1-line index change |
| **mean** | `mean(encoder[start_{k-1} : start_k])` via segment-sum on `repeat_idx` | 0 | order-washing; cheap |
| **attention-pool** | learned query attends over patch `k-1`'s byte states | small | richest; `up_attn_norm` hooks already scaffolded at `hierarchical.py:405-406` |

What you trade: the decoder gives up the *immediacy* of patch `k`'s start byte (gone with the shift)
in exchange for a *complete* summary of patch `k-1`. This is a real tradeoff, not a free lunch — the
experiment decides whether the richer-but-staler trunk input is net-positive.

## 4. Implementation sketch

- `SimpleTransition.down` (`hierarchical.py:333-354`): replace the single `gather` at `idx` with the
  chosen pooling op. For **mean**: build per-byte `repeat_idx = mask.cumsum(1)` (same expression used
  in `up`), `segment_sum(x, repeat_idx) / counts`. For **last-byte under shift (a)**: gather at
  `idx - 1` (clamp ≥ 0; first patch falls back to its own first byte). For **attention-pool**: add a
  tiny cross-attn block with one learned query per patch over its byte states.
- Read shift: prefer route **(b)** — set `model.patch_read_delay: 1` in the config; the delayed read
  is already implemented and is exactly the one-patch shift required. Then the down-pool can align to
  the byte's own patch with no input reindexing.
- **Generation parity**: the streaming decoder must pool only *committed* bytes. Mean/last-byte over a
  completed (committed) patch is trivially streamable; the `committed_patch_idx` view
  (`hierarchical.py:381-392`, `regex_cutting.online_committed_patch_idx`) already provides the settled
  frontier to pool against. Attention-pool needs the same committed-only restriction.
- **Leak probe**: reuse `lingua/probe_root_causal.py` to confirm 0% byte leak after the change (the
  whole point of §2).
- Default off → bit-identical to current `root_greedy`.

## 5. Impact estimate

**Honest priors.** This is a representational enrichment of trunk inputs, partly offset by a staler
read. Reference points: hourglass/BLT-style pooling-operator choices typically move LM loss by low
single-digit percent, not step changes; and the encoder's 512-window already leaks *prior*-patch
context into the first-byte gather, so the marginal content added is "patch `k-1`'s interior beyond
what the window already encoded" — real but bounded.

The bar at 1.3B: `root_greedy` 0.860 vs Llama 0.839 (−0.021) and AU-Net2 0.866. Seed noise on 100M
BPB deltas is **±0.005** (`review_260622.md` D.9), so anything smaller needs ≥3 seeds to call.

| outcome | est. train-BPB Δ | rough P | what it would mean |
|---|---|---|---|
| **win** | −0.008 to −0.020 | ~30% | closes most of the gap to Llama; makes BPEByte competitive on the headline metric, not just the robustness basket |
| **modest** | −0.002 to −0.008 | ~40% | real but seed-noise-adjacent; worth keeping, needs seeds + the attention-pool variant to amplify |
| **null/neg** | −0.002 to +0.005 | ~30% | the read-delay cost cancels the richer input (the 512-window already carried most of it); first-byte gather was near-optimal after all |

Expected value is positive and the **information** is high regardless of sign: a null result *proves*
the first-byte gather is already near-optimal (a clean architectural finding); a win is a direct
attack on the one metric BPEByte currently loses. Either way it is decisive about whether the
"content-starved trunk" hypothesis explains the BPB gap.

Downstream: if BPB moves, expect a fractional-point lift on reasoning (HellaSwag/ARC); robustness/PBP
axes should be unchanged (they are byte-level properties, orthogonal to trunk pooling).

**Cost.** last-byte/mean: ~10–40 lines, zero new params. attention-pool: a small block, lands on the
cheap trunk side (the frozen dim-512 byte enc/dec dominate wall-clock, so pooling/trunk richness is
nearly free in throughput). One 100M / ratio-5 run per variant (~2229 steps, a few hours on 4×B200
per `PoC_BPEByte.md` §3).

## 6. Experiment

Grid on `apps/aunet/configs/bpebyte_100M_v4_root_greedy.yaml`, identical model/data/steps, only the
`down` representative changes:

| arm | down pool | read |
|---|---|---|
| **A0 (baseline)** | patch-start gather (current) | cumsum, delay 0 |
| **A1** | last-byte | shift (a) / delay 1 |
| **A2** | mean | shift (a) / delay 1 |
| **A3** | attention-pool | shift (a) / delay 1 |

Read out: held-out BPB (primary), HellaSwag/ARC-Easy mc + `*_gen`, leak probe (must stay 0%),
tokens/sec. Run A0–A2 first (cheap, no params); promote A3 only if A1/A2 show a positive trend. If
the best arm beats A0 by > the ±0.005 seed floor, repeat at 3 seeds before believing it, then carry
the winner to the 300M/760M configs.

## 7. Result — A0 vs A1, 100M ratio-5 (RUN 2026-06-29, ece-agpu18)

Ran arms **A0** (start gather, delay 0) and **A1** (last-byte, route (b): `down_pool=last_byte` +
`patch_read_delay=1`). Matched: 100M, English DCLM, ratio-5 / 1672 steps, seed 777, bs24·ga8 (≡ bs96·ga2),
identical except the down representative + its required read shift. A2 (mean) / A3 (attention) not run.
Driver: `runs/ablation_100M/A_downpool.sh`.

| arm | down pool | read | final `loss/out` | **train BPB** |
|---|---|---|---:|---:|
| **A0** | patch-start gather | delay 0 | 0.9297 | **1.3413** |
| **A1** | last-byte | delay 1 | 0.8865 | **1.2789** |

**ΔBPB = A1 − A0 = −0.0624** (Δloss −0.043). Taken at face value, last-byte pooling is a large win.

### ⚠️ This is almost certainly the §2 leak, not enrichment
§2 (line 46) states the leaky `bt` variants "buy **~0.07 BPB** exactly this way" by letting a byte
consume its own patch's future. The observed gain is **−0.062 BPB — the same magnitude.** It is also
**~3× larger than this doc's own optimistic "win" prior** (§5: −0.008 to −0.020). When an ablation
beats its pre-registered best case by 3× *and* lands precisely on the known-leak value, the prior
should be: the gain is the leak. Corroborating:
- **The leak probe was never run.** §4 requires `probe_root_causal.py` to confirm 0% byte leak; the
  `A_downpool.sh` driver only trained + logged BPB. So leak-freeness of the `last_byte + delay 1` path
  is **unverified** — exactly the verification §2 says is mandatory.
- ratio-5 is undertrained; large BPB deltas there are the regime where leaks are most inflated (cf.
  the before_root "83%" artifact in this project's history).

### Decision & what would change it
- **L1 down-pool dropped** from the working tree (not committed) — the unverified-and-leak-shaped
  result is not a basis to keep the code.
- **Re-pursue only if** `probe_root_causal.py` reports **0% leak** for `last_byte + patch_read_delay≥1`.
  If leak-free and a (smaller, seed-checked) gain persists at ratio-20/40, it's real and worth carrying
  to 300M/760M. If the gain evaporates once the leak is closed, that confirms first-byte gather was
  already near-optimal (§5's "null" outcome — itself a clean architectural finding).
- A2 (mean) / A3 (attention-pool) remain unrun; only worth it after the leak question is settled.

---

*Companion: `BPEByte_root_greedy_method.md` (the boundary scheme this builds on), `review_blt.md`
(BLT local-decoder pooling for the attention-pool variant). Origin: 3-perspective improvement review,
architecture lens.*

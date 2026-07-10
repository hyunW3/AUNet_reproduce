# 100M Leaderboard — ratio-10.4, matched 21B-byte budget

All models trained at **~100M params** (exactly **98,591,488** params for the trunk-matched byte
models) on the **matched 21.04 GB byte budget** (3344 steps × 768 global batch × 8192 seq). This is a
**data-to-model ratio of γ = 10.4** (= **213.4 bytes/param**) — the project standard, and the *same*
per-param budget the entire AU-Net-law ladder (100M→1.3B) trains at. It's anchored to the paper's 1.3B
γ10 point (60B LLaMa tokens ≈ 273 GB): the ladder's 1.3B run is 283 GB = 62.25B LLaMa-tok = 1.04 × that
60B γ10 → γ10.4. (The bare "ratio-10 / γ=10" label in the configs is a round-down; the earlier "γ10.2"
used a stricter nominal-1.3e9 anchor.) Note the batch-768 3344-step budget here is *identical* to the
AU-Net-law batch-48 53504-step budget — same 21.04 GB, just a larger batch. Metric is the
**final-window training loss** `loss/out` at **step 3340** (the ratio-10.4 endpoint), converted to
**BPB** (bits-per-byte).

- Byte models: `loss/out` is per-byte cross-entropy in **nats** → `BPB = loss/out ÷ ln2`.
- Llama (subword): `loss/out` is per-token CE → `BPB = loss/out ÷ (ln2 · BPT)`, BPT = 4.5483 bytes/token,
  taken at its own ratio-10 endpoint (step 4400).

> Caveat: this is a *training-loss* proxy for BPB (final-window CE), **not** held-out validation BPB.
> Valid for a same-budget cross-model ranking; do not mix with the held-out numbers in
> `runs/poc/campaign/campaign_results.log`.

## Ranking (lower BPB = better)

| # | Model | boundary rule | loss/out | **BPB** | notes |
|---|---|---|---:|---:|---|
| 1 | BPEByte root_greedy — **pure offline** | greedy BPE-merge, whole-doc | 0.7306 | **1.054** | ⚠ leaky upper bound (non-causal) |
| 2 | Llama (subword) | fixed BPE vocab | 3.400 (tok) | **1.079** | |
| 3 | Hybrid **leaf_full** | offline-leaf prefill · full boundaries | 0.7787 | 1.123 | |
| 4 | Hybrid **leaf_mid** | offline-leaf prefill · mid boundaries | 0.7793 | 1.124 | |
| 5 | Hybrid **bt_full** | before-root backtracking prefill · full boundaries | 0.7812 | 1.127 | |
| 6 | Hybrid **bt_half** | before-root backtracking prefill · half boundaries | 0.7822 | 1.128 | |
| 7 | Hybrid **bt_mid** | before-root backtracking prefill · mid boundaries | 0.7826 | 1.129 | |
| 8 | Hybrid **leaf_half** | offline-leaf prefill · half boundaries | 0.7831 | 1.130 | |
| 9 | **BPEByte root_greedy** (v4) | greedy BPE-merge, causal | 0.8055 | **1.162** | 0-leak, deployable |
| 10 | **AU-Net (word)** | whitespace / word | 0.8065 | **1.164** | |
| 11 | **ByteFlow global_topk** (K=3200) | coding-rate top-K (eq 29) | 0.8227 | **1.187** | K=3200 ≈ eff. ratio 2.56 |
| 11 | Hybrid **root-offline_full** | offline-root prefill · full boundaries | 0.8227 | 1.187 | |
| 13 | Hybrid **root-offline_mid** | offline-root prefill · mid boundaries | 0.8257 | 1.191 | |
| 14 | Hybrid **root-offline_half** | offline-root prefill · half boundaries | 0.8270 | 1.193 | |
| 15 | BPEByte root_greedy — **pure online** | greedy BPE-merge, causal streaming | 0.8589 | 1.239 | 0-leak |
| 16 | **Entropy / BLT (low, 5×)** | entropy patching | 0.8889 | **1.282** | |
| 17 | **Entropy / BLT (high, 20×)** | entropy patching | 0.8920 | **1.287** | |
| — | Entropy / BLT (high-global, *partial*) | entropy + global-topk | 0.9396 | *1.356* | ⚠ only step 1510, still training |

## Takeaways

- **Ordering of boundary rules at 100M:** lexical/learned (offline BPE, Llama, word) **>** coding-rate
  (ByteFlow) **>** entropy (BLT). ByteFlow's coding-rate top-K is faithful to the paper's math but the
  AU-Net port is missing Canon layers and uses E=3 vs paper E=6, so it trails by ~0.025 BPB.
- **Hybrid grid — prefill dominates, boundary density barely matters.** The `leaf` and `bt`
  prefill families cluster tight at **1.123–1.130 BPB** (the best deployable band), while `root-offline`
  prefill is a clear step worse at **1.187–1.193**. Within a prefill family the full/half/mid boundary
  choice moves loss by only ~0.004 BPB. Notably **leaf_mid (1.124) beats leaf_half (1.130)** — the mid
  boundary density recovers most of the full-density result at lower boundary count.
- **Entropy/BLT is weakest** among completed methods (~0.10 behind ByteFlow), and **entropy-model
  budget barely matters**: LOW (5× Chinchilla) 1.282 ≈ HIGH (20×) 1.287 — patch quality saturates early.
- **Pure offline is the strongest number but a leaky upper bound** (whole-doc, non-causal → not
  deployable); treat it as an oracle reference. Its deployable causal twin is pure online at 1.239.

## Downstream evaluation — 4-bench, 0-shot acc_norm

Full **4-bench** (HellaSwag, ARC-Easy, ARC-Challenge, PIQA), 0-shot `acc_norm`, `limit=1000`. All rows are
the **ratio-10.4** (step 3344) checkpoints — matching the BPB table — **except ByteFlow**, whose ratio-10.4
checkpoint is still training (dedicated 3344-step run on ece18), so it is reported at **ratio-40** (⚠ ~4× more training; its
downstream is therefore optimistic vs the rest). ByteFlow ran on ece-agpu18; the ratio-10 hybrid variants
were re-evaluated on ece-agpu18 (GPU 2) for this table.

Ranked by 4-bench mean (↓):

| Model | BPB | HS | ARC-E | ARC-C | PIQA | **3-bench** | **4-bench** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama (subword) | 1.079 | 31.3 | 42.1 | 23.7 | 64.3 | 45.9 | **40.4** |
| BPEByte hybrid **leaf · offline** (matched) | 1.123 | 37.2 | 33.1 | 20.9 | 57.0 | 42.4 | **37.1** |
| Entropy / BLT (high, 20×) | 1.287 | 34.4 | 35.3 | 21.5 | 56.5 | 42.1 | **36.9** |
| Entropy / BLT (low, 5×) | 1.282 | 34.9 | 33.8 | 20.7 | 57.5 | 42.1 | **36.7** |
| AU-Net (word) | 1.164 | 28.5 | 33.7 | 23.3 | 58.3 | 40.2 | **36.0** |
| BPEByte root_greedy (v4 causal) | 1.162 | 28.6 | 32.6 | 22.6 | 58.1 | 39.8 | **35.5** |
| **ByteFlow global_topk** ⚠r40 | 1.187 | 38.3 | 29.3 | 20.5 | 52.8 | 40.1 | **35.2** |
| BPEByte root_greedy — **online** | 1.239 | 35.2 | 29.3 | 24.1 | 50.6 | 38.4 | **34.8** |
| BPEByte hybrid **root · offline** | 1.187 | 35.5 | 27.4 | 20.4 | 50.1 | 37.7 | **33.4** |

**Notation (downstream hybrids):** `BPEByte hybrid {prefill} · {question-tokenization}` — the **prefill**
axis is **leaf** (`offline_leaf`) or **root** (`offline_root`); the **question-tokenization** axis at eval
is **offline** (`leaf-offline`/`root-offline`) or **online** (`root-online-gr`, causal greedy). Each hybrid is scored
under the **question mode matched to its prefill regime**: *leaf · offline* = leaf-prefill model scored with
offline-leaf question tokenization (`offline_question`, its native regime); *root · offline* = root-prefill
model with offline-root question tokenization. `BPEByte root_greedy — online` is the non-hybrid pure-online
model (`c_online`), scored root-online-gr (its native regime). Note: for the leaf model the matched *leaf · offline*
eval is **bit-identical** to the non-native *leaf · online* (root-online-gr) — on these MC prompts offline-leaf BPE
and greedy-online segmentation coincide (0/4000 argmax flips), so no leaf row is lost by reporting only the
matched one.

**Key finding — BPB rank ≠ downstream rank.**
- **Entropy/BLT is worst on BPB (1.28) but 3rd–4th on downstream (36.7–36.9)** — it beats AU-Net and
  root_greedy despite ~0.12 higher BPB, driven by **PIQA (56–58)** and the **strongest HS (34–35)**; it is
  weak on ARC-E. Training loss does not predict its task accuracy.
- **Llama leads by a wide margin** (40.4), carried by ARC-E (42.1) and PIQA (64.3). The best byte-side
  model is **hybrid leaf · offline** (37.1, prefill-matched).
- **ByteFlow, even with 4× more training (ratio-40), only reaches 35.2** — tying root_greedy and *below*
  AU-Net; at an iso-budget ratio-10 it would be lower still. It posts the **highest HellaSwag (38.3)** but
  the weakest ARC-E of the byte models (29.3, ≈ chance). This confirms the BPB story: the coding-rate
  boundary rule does not beat the existing boundary methods at 100M.
- **ARC-Challenge is noise at 100M** — every model sits 20.4–24.1 around the 25% 4-way chance line, so it
  drags all 4-bench means down uniformly and adds little signal (the top ARC-C, online's 24.1, is within
  noise). The 3-bench column (HS/ARC-E/PIQA) is the higher-signal read.

**Caveats.** (1) **ByteFlow is ratio-40, all others ratio-10.4** — not iso-budget (⚠r40); the matched
ratio-10.4 ByteFlow is training and will replace this row. (2) ARC-C ≈ chance
at this scale. (3) Hybrids scored under the **prefill-matched** Q-mode (leaf · offline = leaf-offline, root ·
offline = root-offline, pure online = root-online-gr); for the leaf model leaf-offline ≡ root-online-gr on this data. (4) All 0-shot `acc_norm`, `limit=1000`. (5) Pure
offline (c_offline) is a non-deployable training-loss oracle → no downstream row.

## Legend — hybrid naming: `{prefill}_{boundary}`

Genuine **hybrid** runs (`bpe_hybrid: true`) split tokenization into a **prefill** policy (teacher-forced
context) and a **decode boundary** density. Notation is `{prefill}_{boundary}`:

**prefill** (`bpe_hybrid_prefill`):
- **leaf** = `offline_leaf` — offline *leaf*-level BPE segmentation during prefill
- **root-offline** = `offline_root` — offline *root*-level segmentation during prefill
- **bt** = `bt` — before-root **backtracking** streaming prefill (causal parser that allows
  re-merging; distinct from pure-online *greedy* streaming)

**boundary** (`bpe_hybrid_boundary`, decode-time density):
- **full** = `uniform_full` — every candidate is a boundary (full density)
- **half** = `static_half` — static half-density boundary set
- **mid** = `uniform_mid` — uniform mid-density

So **leaf_mid** = offline-leaf prefill + mid-density decode boundaries (old code name `p1b3`).

**Pure offline / pure online are NOT hybrid** (`bpe_hybrid: false`) — the two endpoints of BPEByte
root_greedy: pure offline segments the whole document non-causally (lowest loss but **leaky**), pure
online is fully causal streaming (0-leak, deployable).

### old → new name map

| old | new | old | new | old | new |
|---|---|---|---|---|---|
| p1b1 | leaf_full | p1b2 | leaf_half | p1b3 | leaf_mid |
| rootp_B1 | root-offline_full | rootp_B2 | root-offline_half | rootp_B3 | root-offline_mid |
| p3b1 | bt_full | p3b2 | bt_half | p3b3 | bt_mid |

## Sources

- g10 baselines: `runs/small/cmp_g10/{v4_root_greedy,aunet_100M,llama_100M}/metrics.jsonl`
- hybrid PoC grid: `runs/small/poc/campaign/{p1b2,p1b3,p3b1,p3b2,p3b3,hybrid_rootp_B2,hybrid_rootp_B3,hybrid_p1b1_rootp}_r10_s777/`,
  `runs/small/poc/{hybrid_p1b1,c_online_g10}/`, `runs/small/poc/campaign/c_offline_r10_s777/`
- ByteFlow: `ece-agpu11:/home/hwbae/AUNet/runs/byteflow_100M_r10_paperK/` (global_topk, topk=3200; step-3340
  checkpoint — the run's *final* 13376-step/0.7468 point is **ratio-40**, not ratio-10)
- Entropy/BLT: `snu30:/home/hyunw3/AUNet/runs/aunet_100M_entropy_{low,high,high_global}_r10/`

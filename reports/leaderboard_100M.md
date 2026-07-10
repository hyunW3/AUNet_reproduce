# 100M Leaderboard — AU-Net scaling-law recipe, γ10.4, full-dataset downstream

All models trained at **~100M params** (exactly **98,591,488** for the trunk-matched byte models) on
the matched **21.04 GB** byte budget (**53,504 steps × global batch 48 × 8192 seq**), using the
**AU-Net scaling-law HP recipe** (Videau 2025 §2.3: small global batch + high LR — fixes small-model
over-batching). This is a data-to-model ratio **γ = 10.4** (213.4 bytes/param). The batch-48 /
53,504-step budget here is the *same 21.04 GB* as the ad-hoc batch-768 / 3,344-step budget in
[`leaderboard_100M_adhoc.md`](leaderboard_100M_adhoc.md) — only the recipe differs.

- **BPB** = final-window training loss (`loss/out`, mean of last 20 logged steps). Byte:
  `BPB = loss/out ÷ ln2`. Llama (subword): `BPB = loss/out ÷ (ln2 · 4.5483)`, BPT = 4.5483 bytes/token.
  Training-loss proxy for a same-budget cross-model ranking (not held-out validation BPB).
- **Downstream** = **full-dataset** (no `limit`) 6-bench, **0-shot**, `acc_norm` where defined
  (`acc` for BoolQ/WinoGrande). Native regime per model (see notes). Two means: **HS/ARC-E/PIQA**
  (ARC-C excluded — near-chance noise at 100M) and **all-6**.

## Ranking (primary metric = BPB, lower = better)

| # | Model | boundary rule | **BPB** | HS/ARC-E/PIQA | all-6 |
|---|---|---|---:|---:|---:|
| 1 | **BPEByte hybrid leaf_mid** | offline-leaf prefill · mid boundaries | **1.041** | 42.62 | 42.91 |
| 2 | **Llama (subword)** | fixed BPE vocab | 1.053 | **47.43** | **45.27** |
| 3 | **BPEByte root_greedy** | online greedy, causal (0-leak) | 1.079 | 41.59 | 39.60 |
| 4 | **AU-Net (word)** | whitespace / word | 1.082 | 42.32 | 42.63 |
| 5 | **ByteFlow global_topk** (K=3200) | coding-rate top-K (eq 29) | 1.108 | 37.24 | 37.44 |
| — | BPEByte hybrid **bt** (online-bt prefill) | before-root backtracking prefill · mid | *training* | — | — |
| — | Entropy / BLT (low, 5×) | entropy patching | *training* | — | — |

## Downstream — full-dataset 6-bench, 0-shot

| Model | HS | ARC-E | ARC-C | BoolQ | PIQA | Wino | **HS/ARC-E/PIQA** | **all-6** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama (subword) | 33.2 | 45.1 | 23.8 | 54.9 | 64.0 | 50.6 | **47.43** | **45.27** |
| BPEByte hybrid leaf_mid | 31.6 | 34.9 | 24.4 | 53.6 | 61.4 | 51.6 | 42.62 | 42.91 |
| AU-Net (word) | 31.8 | 36.0 | 24.4 | 54.8 | 59.1 | 49.6 | 42.32 | 42.63 |
| BPEByte root_greedy | 30.7 | 34.6 | 23.5 | 39.1 | 59.4 | 50.2 | 41.59 | 39.60 |
| ByteFlow global_topk | 30.2 | 28.4 | 20.4 | 41.3 | 53.2 | 51.2 | 37.24 | 37.44 |

(HellaSwag 10,042 · ARC-E 2,376 · ARC-C 1,172 · BoolQ 3,270 · PIQA 1,838 · WinoGrande 1,267.)

## Takeaways

- **BPB rank ≠ downstream rank.** Hybrid leaf_mid wins BPB (1.041) but **Llama leads downstream**
  (47.43 / 45.27). Hybrid is the best byte model on both means; it edges AU-Net (all-6 42.91 vs 42.63).
- **Llama's downstream lead is concentrated in ARC-E (+9) and PIQA (+3)** — knowledge/commonsense
  benches where subword tokenization compresses content words better at iso-byte budget. On ARC-C and
  WinoGrande (near-chance for all at 100M) Llama does *not* lead. HellaSwag is a near-tie.
- **rg's weak spot is BoolQ (39.1)** — well below the offline-leaf/word models (53–55). Real, not a
  sampling artifact; it drops rg's all-6 despite a mid-pack BPB.
- **ByteFlow trails on both metrics** — the coding-rate boundary rule does not beat lexical/word/entropy
  at 100M, iso-budget.
- **Law recipe vs ad-hoc:** every byte model gains ~0.08 BPB and +3–4 downstream pts vs the batch-768
  ad-hoc recipe (same 21 GB budget). See [`leaderboard_100M_adhoc.md`](leaderboard_100M_adhoc.md).

## Notes — eval regime (native per model)

- **rg** — online root_greedy (byte): `greedy_question_loglikelihood` (root-online) + `force_bpe_online_mode=greedy`. Mandatory; a bare tokenizer override runs offline and tanks online byte models.
- **hybrid leaf_mid** — `offline_question` (leaf-offline, native offline-leaf) + `force_bpe_online_mode=greedy`. (leaf-offline ≡ root-online-gr here — 0 argmax flips.)
- **AU-Net (word)**, **ByteFlow** — plain aunet eval (no online-BPE; no force, no question flag).
- **Llama** — `apps.main.eval`, `eval_downstream_main.yaml` (subword).
- **bt / BLT-low (pending)** — bt scored both leaf-offline (matches leaf_mid) and root-online-gr (bt online-native), force-greedy; BLT-low plain native (online-entropy pool, no force). Filled by `runs/watch_bt_blt_law100M.sh` on completion.

## Sources

- Configs + launchers: `runs/poc/portable_aunetlaw/{bpebyte_rg,aunet,llama,bpebyte_rg_hybrid_leaf,byteflow,blt}_100M.yaml`
- Trained dumps (ece, `/home/hwbae/AUNet/runs/poc/portable_aunetlaw/`): `lb_{rg,llama,hybrid,aunet,byteflow,hybrid_bt,blt_low}_100M`
- Full-dataset downstream: `.../eval_law100M_full/{model}/results.json` (driver `eval_law100M_full_ece11.sh`)
- limit=1000 downstream (for comparison): `.../eval_law100M/`
- BPB: `.../lb_*_100M/metrics.jsonl` (final-window `loss/out`)

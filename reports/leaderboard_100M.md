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
| 2 | **BPEByte hybrid bt** | online-bt (backtracking) prefill · mid | **1.043** | 41.33 | 43.33 |
| 3 | **Llama (subword)** | fixed BPE vocab | 1.053 | **47.43** | **45.27** |
| 4 | **BPEByte root_greedy** | online greedy, causal (0-leak) | 1.079 | 41.59 | 39.60 |
| 5 | **AU-Net (word)** | whitespace / word | 1.082 | 42.32 | 42.63 |
| 6 | **Entropy / BLT (low, 5×)** | entropy patching | 1.102 | 42.33 | 41.63 |
| 7 | **ByteFlow global_topk** (K=3200) | coding-rate top-K (eq 29) | 1.108 | 37.24 | 37.44 |

## Downstream — full-dataset 6-bench, 0-shot

| Model | HS | ARC-E | ARC-C | BoolQ | PIQA | Wino | **HS/ARC-E/PIQA** | **all-6** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
Each hybrid shown at its **native** question regime (leaf_mid → leafQ, bt → btQ); see the regime matrix below.

| Model | HS | ARC-E | ARC-C | BoolQ | PIQA | Wino | **HS/ARC-E/PIQA** | **all-6** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama (subword) | 33.2 | 45.1 | 23.8 | 54.9 | 64.0 | 50.6 | **47.43** | **45.27** |
| BPEByte hybrid leaf_mid (leafQ) | 31.6 | 34.9 | 24.4 | 53.6 | 61.4 | 51.6 | 42.62 | 42.91 |
| Entropy / BLT (low, 5×) | 31.3 | 35.4 | 24.0 | 47.5 | 60.3 | 51.3 | 42.33 | 41.63 |
| AU-Net (word) | 31.8 | 36.0 | 24.4 | 54.8 | 59.1 | 49.6 | 42.32 | 42.63 |
| BPEByte root_greedy | 30.7 | 34.6 | 23.5 | 39.1 | 59.4 | 50.2 | 41.59 | 39.60 |
| BPEByte hybrid bt (btQ) | 31.2 | 34.9 | 23.5 | 61.7 | 57.9 | 50.8 | 41.33 | 43.33 |
| ByteFlow global_topk | 30.2 | 28.4 | 20.4 | 41.3 | 53.2 | 51.2 | 37.24 | 37.44 |

(HellaSwag 10,042 · ARC-E 2,376 · ARC-C 1,172 · BoolQ 3,270 · PIQA 1,838 · WinoGrande 1,267.)

### Hybrid eval-regime matrix (all-6 / HS-ARC-E-PIQA)

Each hybrid scored under three question tokenizations (answer=greedy throughout, full dataset). **Bold =
each model's own native regime** (leaf_mid = leafQ offline-BPE prefill; bt = btQ online-bt prefill).

| Model | leafQ (offline-BPE) | btQ (online-bt) | greedyQ (online-greedy) |
|---|---:|---:|---:|
| leaf_mid | **42.91 / 42.62** | 43.15 / 41.87 | 43.12 / 41.90 |
| bt | 40.88 / 40.70 | **43.33 / 41.33** | 43.33 / 41.30 |

(bold = each model's own native prefill regime.)

**Findings.**
- **leaf_mid is robust across all three regimes** (all-6 42.9–43.2, tri 41.9–42.6) — it does *not* drop under
  bt's btQ. **bt is robust across the two *causal* regimes** (btQ ≈ greedyQ = 43.33) but **collapses under the
  offline-BPE leafQ** (40.88). So the penalty is specifically **offline-BPE question tokenization hurting bt**,
  not a symmetric "own-regime" effect. (Corrects the earlier "leaf-offline ≡ greedy, 0 flips" note — leaf_mid's
  regimes differ modestly, and neither hybrid is best strictly in its own regime.)
- **Matched-regime (fairer than native-vs-native):** under *leafQ* leaf_mid 42.91/42.62 ≫ bt 40.88/40.70; under
  *btQ* and *greedyQ* the two are ~tied — bt +0.2 on noisy all-6, leaf_mid +0.5–0.6 on the clean tri. **On
  HS/ARC-E/PIQA, leaf_mid ≥ bt in every matched regime.**
- **Net:** bt ties leaf_mid on BPB (1.043 vs 1.041) and matches it on causal evals, but is fragile under
  offline scoring and ~0.5 behind on the clean metric. offline-BPE prefill (leaf_mid) is the safer hybrid;
  bt trades a little robustness + clean-bench quality for a 37%-cheaper parser.

## Takeaways

- **BPB rank ≠ downstream rank.** Hybrid leaf_mid wins BPB (1.041) but **Llama leads downstream**
  (47.43 / 45.27). Hybrid is the best byte model on both means; it edges AU-Net (all-6 42.91 vs 42.63).
- **hybrid bt ties leaf_mid on BPB (1.043 vs 1.041)** — the byte-trie prefill (backtracking, ~37% cheaper
  to parse than offline real-BPE, verified-equivalent patch-length distribution) matches the offline-leaf
  prefill on loss. **Downstream, native-vs-native** (bt=btQ 43.33/41.33, leaf_mid=leafQ 42.91/42.62): bt
  edges leaf_mid on all-6 (BoolQ-driven) but trails **~1.3 on the clean HS/ARC-E/PIQA**. bt is regime-robust
  (btQ ≈ greedyQ), leaf_mid is regime-sensitive — see the eval-regime matrix. Net: a real cost/quality
  tradeoff for the cheaper parser, not a free win.
- **Entropy/BLT-low: weak BPB (1.102, 2nd-worst) but mid-pack downstream (all-6 41.63).** It beats
  root_greedy and ByteFlow downstream despite worse loss — but costs the extra entropy-precompute FLOPs and
  still trails the trie/word schemes, consistent with the ratio-40 finding that entropy patching < trie.
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

## Checkpoints used (verified)

All under `ece:/home/hwbae/AUNet/runs/poc/portable_aunetlaw/`. BPB read from `<dump>/metrics.jsonl`
(final-window `loss/out`); downstream from `eval_law100M_full/<dump>/results.json`, which evaluated
exactly the checkpoint below (`ckpt_dir` in `eval_law100M_full_ece11.sh`).

| model | dump / checkpoint step | % |
|---|---|---|
| BPEByte hybrid leaf_mid | `lb_hybrid_100M/checkpoints/0000053504` | 100% |
| Llama (subword) | `lb_llama_100M/checkpoints/0000011891` | 100% |
| BPEByte root_greedy | `lb_rg_100M/checkpoints/0000053504` | 100% |
| AU-Net (word) | `lb_aunet_100M/checkpoints/0000053504` | 100% |
| ByteFlow (K=3200) | `lb_byteflow_100M/checkpoints/0000053504` | 100% |
| BPEByte hybrid bt | `lb_hybrid_bt_100M/checkpoints/0000053504` | 100% (BPB + downstream leafQ/greedyQ done) |
| Entropy/BLT (low) | `lb_blt_low_100M/checkpoints/0000053504` | 100% |

(Byte runs = 53,504 steps; Llama = 11,891 steps — both the γ10.4 endpoint. Downstream numbers above
were confirmed to match each `results.json` cell.)

## Sources

- Configs + launchers: `runs/poc/portable_aunetlaw/{bpebyte_rg,aunet,llama,bpebyte_rg_hybrid_leaf,byteflow,blt}_100M.yaml`
- Trained dumps (ece, `/home/hwbae/AUNet/runs/poc/portable_aunetlaw/`): `lb_{rg,llama,hybrid,aunet,byteflow,hybrid_bt,blt_low}_100M`
- Full-dataset downstream: `.../eval_law100M_full/{model}/results.json` (driver `eval_law100M_full_ece11.sh`)
- limit=1000 downstream (for comparison): `.../eval_law100M/`
- BPB: `.../lb_*_100M/metrics.jsonl` (final-window `loss/out`)

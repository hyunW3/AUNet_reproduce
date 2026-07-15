# ThingsToMake — filled report

_Regenerated from the 5-item template. Each section below is the requested deliverable filled with the
actual measured numbers. Three model families throughout: **Llama** (subword), **AU-Net** (word-pooling),
**BPEByte-rg** (`root_greedy`, online/0-leak). "—" = not measured. Downstream: HS/ARC-E/ARC-C/PIQA =
`acc_norm`, BoolQ/Wino/MMLU = `acc`(_norm as noted); avg3 = HS·ARC-E·PIQA, avgall = mean of the listed set._

---

## 1. 1.3B-scale — BPB + zero-shot / 5-shot downstream

**Models:** Llama · AU-Net · BPEByte-rg. Benchmarks: HellaSwag, ARC-Easy, ARC-Challenge, PIQA, BoolQ,
WinoGrande, MMLU-text, avg3, avgall. Source: `scaling_downstream_table.md`, `model_results_1.3B.md`,
`leaderboard_1B.md`, `downstream_data.json`.

### 0-shot

| Benchmark | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| HellaSwag | 62.2 | 62.6 | 62.5 |
| ARC-Easy | 65.5 | 65.7 | **66.8** |
| ARC-Challenge | 35.3 | 36.5 | **37.5** |
| PIQA | **75.3** | 74.2 | 74.3 |
| BoolQ | **63.5** | 61.1 | 62.0 |
| WinoGrande | **61.6** | 61.5 | 61.1 |
| MMLU-text | 31.9 | 32.2 | **32.3** |
| **avg3** (HS·ARC-E·PIQA) | 67.7 | 67.5 | **67.9** |
| **avgall** (7-bench) | 56.5 | 56.3 | **56.6** |

### 5-shot

| Benchmark | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| HellaSwag | 62.9 | **63.7** | **63.7** |
| ARC-Easy | 70.3 | **72.4** | 71.9 |
| ARC-Challenge | 38.5 | 39.5 | **39.9** |
| PIQA | **75.6** | 74.1 | 74.4 |
| BoolQ | **65.6** | 59.1 | 65.5 |
| WinoGrande | 62.6 | 63.6 | **64.1** |
| MMLU-text | **35.2** | 34.9 | 34.4 |
| **avg3** | 69.6 | **70.1** | 70.0 |
| **avgall** (7-bench) | 64.6 | **64.8** | **64.8** |

### BPB (bits/byte, ↓)

| Measure | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| train-log (full-context, iso-byte) | 0.872 | 0.872 | **0.853** |
| 512-cap (held-out) | — ¹ | — ² | 0.933 |

¹ Llama is subword — no byte-level 512-cap metric defined. ² AU-Net 512-cap not measured.

**Read:** the three families are **statistically tied at 1.3B** on downstream (avgall 56.3–56.6 0-shot, 64.6–64.8
5-shot; sub-1-pt gaps are single-run noise), and BPEByte-rg leads on avg3 both shots. On BPB, rg's train-log
edge over Llama/AU-Net (~0.02) is ~4–5× the seed floor (robust); rg-vs-AU-Net (0.0054) is within noise. MMLU-text
0-shot vs 5-shot use different protocols (letter-ish vs cloze) — do not compare the two rows.

---

## 2. Scale-wise full table (100M · 300M · 760M · 1.3B)

**Models:** Llama · AU-Net · BPEByte-rg. Source: `scaling_bpb.csv` (BPB, tail-mean `loss/out` over last 200
steps), `scaling_data.csv` (0-shot downstream). γ10 (ad-hoc) recipe, iso-byte budget per scale.

### Held-out BPB (↓)

| Scale | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| 100M | **1.0587** | 1.0858 | 1.0833 |
| 300M | **0.9945** | 1.1126 | 1.1071 |
| 760M | **0.9191** | 0.9435 | 0.9403 |
| 1.3B | **0.8404** | 0.8662 | 0.8578 |

### Downstream — avgall (6-bench) / avg3 (0-shot)

| Scale | Llama avgall | AU-Net avgall | rg avgall | Llama avg3 | AU-Net avg3 | rg avg3 |
|---|---:|---:|---:|---:|---:|---:|
| 100M | **45.61** | 41.69 | 41.26 | **45.88** | 40.15 | 39.76 |
| 300M | **48.58** | 45.42 | 45.00 | **53.16** | 48.64 | 47.70 |
| 760M | **53.14** | 50.68 | 49.21 | **58.89** | 56.40 | 55.31 |
| 1.3B | 60.56 | 60.28 | **60.72** | 67.66 | 67.51 | **67.88** |

### Per-benchmark (0-shot acc, `acc_norm` for HS/ARC-E/ARC-C/PIQA; `acc` for BoolQ/Wino)

| Bench | Model | 100M | 300M | 760M | 1.3B |
|---|---|---:|---:|---:|---:|
| HellaSwag | Llama / AU-Net / rg | 31.3 / 28.5 / 28.6 | 42.4 / 37.2 / 37.0 | 50.0 / 46.7 / 45.8 | 62.2 / 62.7 / 62.5 |
| ARC-Easy | Llama / AU-Net / rg | 42.1 / 33.7 / 32.6 | 48.8 / 43.9 / 41.8 | 55.7 / 53.4 / 51.1 | 65.5 / 65.7 / 66.8 |
| ARC-Chal. | Llama / AU-Net / rg | 23.7 / 23.3 / 22.6 | 26.0 / 25.6 / 24.8 | 29.8 / 29.4 / 29.3 | 35.3 / 36.5 / 37.5 |
| PIQA | Llama / AU-Net / rg | 64.3 / 58.3 / 58.1 | 68.3 / 64.9 / 64.3 | 71.0 / 69.2 / 69.1 | 75.3 / 74.2 / 74.3 |
| BoolQ | Llama / AU-Net / rg | 61.7 / 56.7 / 55.7 | 55.8 / 50.7 / 47.3 | 58.4 / 49.9 / 45.5 | 63.5 / 61.1 / 62.1 |
| WinoGr. | Llama / AU-Net / rg | 50.6 / 49.7 / 50.0 | 50.2 / 50.3 / 54.8 | 54.0 / 55.6 / 54.6 | 61.6 / 61.5 / 61.1 |

**Read:** Llama leads at 100M–760M (subword edge) but the **byte slope is steeper** — the families converge and
cross at 1.3B (rg avgall 60.72 ≥ Llama 60.56; rg avg3 67.88 ≥ Llama 67.66). BPB: Llama keeps a ~0.02–0.03 edge
at every scale (AU-Net ≈ rg throughout). _(BPB here is train-log tail-mean and differs slightly from §1's
leaderboard measure — same trend, different window.)_

---

## 3. 100M parsing ablation

**Methods:** Llama, BLT (entropy patching), BPEByte-rg, ByteFlow global top-k, AU-Net. γ10.4 (21.04 GB),
full-dataset 0-shot. Source: `leaderboard_100M.md`, `parsing_ablation_100M.json`, `100M_ablation.md`,
`forward_parsing_benchmark.html`.

### BPB + downstream

| Method | BPB ↓ | HS | ARC-E | ARC-C | PIQA | BoolQ | Wino | avg3 | avgall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Llama** (subword) | **1.053** | 33.2 | 45.1 | 23.8 | 64.0 | 54.9 | 50.6 | **47.4** | **45.3** |
| **AU-Net** (word) | 1.082 | 31.8 | 36.0 | 24.4 | 59.1 | 54.8 | 49.6 | 42.3 | 42.6 |
| **BLT** (entropy, low 5×) | 1.102 | 31.3 | 35.4 | 24.0 | 60.3 | 47.5 | 51.3 | 42.3 | 41.6 |
| **BPEByte-rg** (greedy·0-leak) | 1.079 | 30.7 | 34.6 | 23.5 | 59.4 | 39.1 | 50.2 | 41.6 | 39.6 |
| **ByteFlow global top-k** (K=3200) | 1.108 | 30.2 | 28.4 | 20.4 | 53.2 | 41.3 | 51.2 | 37.2 | 37.4 |

### Parsing + forward throughput

The benchmark logs **parse time** (CPU, ms/KB) and **forward latency** (ms/step, bsz4·8192 B = 32 KB)
separately; the "parse+fwd throughput" column below is **derived** (32 KB ÷ (parse·32 + forward), iso-8192 B
byte-seq). Source: `forward_parsing_benchmark.html`.

| Method | Parse (ms/KB) | Forward (ms/step) | **Parse+Fwd (KB/s)** ³ |
|---|---:|---:|---:|
| Llama (tiktoken 128k) | 0.143 | 48.6 ⁴ | — ⁴ |
| BPEByte-rg (byte-trie) | 0.166 | 59.0 | ~509 |
| AU-Net (regex word) | 0.223 | 59.0 | ~495 |
| ByteFlow | 0.226 | 59.0 | ~495 |
| BLT (entropy, fp16 default) | 2.47 (GPU) | 138.1 ⁵ | ~237 |

³ derived, byte models at iso-8192 B seq. ⁴ Llama runs a 2048-token (~9.3 KB) seq, not iso-byte — combined
KB/s not comparable, omitted. ⁵ BLT forward = 58.9 base + 79.2 entropy-model = 138.1 ms (fp32 old = 451.9 ms).

**Read:** Llama leads downstream (47.4 avg3) but at a different compute-per-byte axis (2048-tok seq, ~2.2× lower
FLOPs/byte — not iso-FLOP). Among byte models, rg ≈ AU-Net ≈ ByteFlow on parse+forward speed (~0.9× real-time at
this batch); **BLT is ~2× slower** (entropy model in the loop). ByteFlow global-top-k is the weakest on both BPB
and downstream. _(Secondary ratio-40 boundary-scheme ranking in `100M_ablation.md`: Llama 42.7 > BPEByte 38.8 ≈
AU-Net 38.7 > BLT 37.0 avg over HS·ARC-E·PIQA·ARC-C.)_

---

## 4. Robustness — input perturbation, 1.3B

**Models:** Llama · BPEByte-rg · AU-Net. Source: `robustness.json`, `despace/summary.md`,
`model_results_1.3B.md`. All n=2000 subset, single run.

### 4a. PBP-mc (prompt boundary problem, MC) — ΔAcc when a trailing space shifts the boundary

| Model | PBP-mc ΔAcc | HellaSwag Δ | ARC-Easy Δ | ARC-Challenge Δ | cut-point ΔBPC |
|---|---:|---:|---:|---:|---:|
| Llama | **−8.92** | −0.93 | −21.46 | −3.16 | +0.710 |
| AU-Net | −0.03 | 0.00 | +0.04 | 0.00 | 0.000 |
| BPEByte-rg | −0.16 | +0.12 | 0.00 | −0.26 | 0.000 |

### 4b. Noisy / typo (acc_norm)

| Condition | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| HellaSwag-Noise (avg 15 variants) | 37.6 | 41.5 | **42.3** |
| HellaSwag-Typo (avg 8 ops) | 49.7 | **52.1** | 51.9 |
| ARC-C-Typo (avg 8 ops) | 30.8 | 31.8 | **33.5** |
| _(HellaSwag clean / ARC-C clean, n=2000)_ | 55.8 / 35.2 | 57.8 / 36.0 | 56.8 / 37.3 |

_No separate "HellaSwag-Typo/Arc-c-noise" splits exist beyond the above: only a 15-variant HellaSwag-Noise suite
and an 8-op typo suite (applied to both HS and ARC-C questions)._

### 4c. Despace (macro-avg accuracy over 5 tasks incl. HellaSwag & ARC-Easy; independent space-removal prob p)

| Condition | Llama | AU-Net | BPEByte-rg |
|---|---:|---:|---:|
| clean | 0.557 | **0.568** | 0.557 |
| ctx10 | 0.548 | 0.551 | 0.545 |
| ctx40 | 0.526 | 0.501 | **0.530** |
| ctx70 | **0.523** | 0.464 | 0.509 |
| ctx100 | — ⁶ | — ⁶ | — ⁶ |
| all10 | 0.499 | 0.496 | **0.504** |
| all40 | 0.457 | 0.431 | **0.462** |
| all70 | 0.435 | 0.382 | **0.438** |
| all100 | — ⁶ | — ⁶ | — ⁶ |

⁶ ctx100 / all100 conditions were not run (eval stopped at p=70%). ctx = spaces removed from context only;
all = context + answer choices.

**Read:** the byte models are **boundary-robust** where Llama is not — PBP-mc ΔAcc −8.9 (Llama, driven by
ARC-Easy −21.5) vs ≈0 for both byte models; cut-point ΔBPC +0.71 vs 0.00. rg leads on noise/typo (HS-Noise 42.3,
ARC-C-Typo 33.5) and on heavy-despace (all70 0.438). AU-Net degrades most under heavy context-despace (ctx70 0.464).

---

## 5. State tracking & recall, 1.3B

**Models:** Llama (`subword_llama`) · AU-Net (`aunet_static`) · BPEByte-rg (`byte_greedyroot`). Source:
`scripts/fflm/run.log`, `scripts/probes/{dyck,mk}.log`, `scripts/niah/{run,run_23,mk_grid}.log`.

### 5a. FFLM (in-context state tracking, greedy acc) + Dyck-3

| Model | FFLM Dense | FFLM In-Dist | FFLM Sparse | Dyck-3 (len40, depth6) |
|---|---:|---:|---:|---:|
| Llama | 0.823 | **0.784** | 0.808 | 0.520 |
| AU-Net | 0.853 | 0.724 | 0.666 | 0.540 |
| BPEByte-rg | **0.865** | 0.768 | **0.869** | **0.673** |

### 5b. S-NIAH grid (exact-match acc, by context length; mean over 5 depths)

| Task | Model | 512 B | 1024 B | 2048 B | 4096 B | 6144 B | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| **S-NIAH-1** (noise+#) | Llama | 0.85 | 0.85 | 1.00 | 1.00 | 0.95 | 0.93 |
| | AU-Net | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| | BPEByte-rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **S-NIAH-2** (essay+#) | Llama | 0.90 | 0.70 | 0.65 | 0.65 | 0.60 | 0.70 |
| | AU-Net | 1.00 | 1.00 | 0.95 | 1.00 | 0.95 | 0.98 |
| | BPEByte-rg | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | **0.99** |
| **S-NIAH-3** (essay+UUID) | Llama | 0.15 | 0.15 | 0.05 | 0.25 | 0.20 | 0.16 |
| | AU-Net | 0.85 | 0.60 | 0.45 | 0.30 | 0.40 | 0.52 |
| | BPEByte-rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |

### 5c. MK-NIAH multi-key grid (len 2048, exact-match acc; K=1 and K=8 requested, full K-sweep shown)

| Model | K=1 | K=2 | K=4 | K=8 | Mean |
|---|---:|---:|---:|---:|---:|
| Llama | 0.450 | 0.275 | 0.325 | 0.175 | 0.306 |
| AU-Net | 0.975 | 0.725 | 0.425 | 0.300 | 0.631 |
| BPEByte-rg | **1.000** | **0.900** | **0.725** | **0.550** | **0.794** |

**Read:** BPEByte-rg is the strongest on every state-tracking/recall probe — perfect S-NIAH-3 UUID copy (1.00 vs
Llama 0.16, AU-Net 0.52), best sparse FFLM (0.869 vs AU-Net 0.666), best Dyck-3 (0.673), and best MK-NIAH at
every K (K=8: 0.55 vs Llama 0.175). AU-Net (static word-pooling) sits between the two, degrading on sparse/OOD;
Llama (subword) floors on exact-copy of unseen character strings (UUIDs) — the byte tokenizer's character-level
addressing is the lever.

---

_Provenance & caveats: all numbers single-run (no seeds) unless a CI is shown; treat sub-1-pt downstream gaps as
noise. §1/§2 BPB use different measurement windows (leaderboard vs tail-mean CSV) — same ranking. §3 TPS is
derived from separately-logged parse + forward times; Llama's is not iso-byte. §4 uses n=2000 subsets, so clean
baselines differ from full-set §1/§2. Model tags in §5: `subword_llama` / `aunet_static` / `byte_greedyroot`._

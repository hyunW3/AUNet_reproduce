# Structured-locality suite (`whybpebyte`)

Implements `reports_afterAAAIsub/whybpebyte/bpebyte_structured_locality_experiment_design.md`
for the three production segmentation schemes:

| arm | what it is | where it comes from |
|---|---|---|
| `llama` | conventional subword tokenizer | Llama-3 tiktoken token starts |
| `aunet` | AU-Net word/regex pooling | `RegexPool(strategy={"word1": "1@1"})` |
| `bpebyte` | OnlineBPE causal byte-trie walk | `RegexPool(bpe_br, bpe_online, greedy, root)` |

All three come from `scripts/probes/compression_stability.build_parser`, so this suite and the
compression-stability sweep can never disagree about what a boundary is. Controls
(`fixed_k`, `random_matched`, `whitespace`) live in `segmenters.py`.

## Pipeline

```
dump.py                 stage 1+2  boundary masks + cached per-byte probe surprisal -> cache/
analyze_structural.py   exp A + B  surprisal gap, coherence, peaks, retrieval, controls
analyze_classes.py      exp C      what kind of byte becomes a boundary
analyze_perturb.py      exp D      single-edit boundary locality (uses perturb.py)
exp_downstream.py       exp F      clean/perturbed tail BPB per example, one model per run
analyze_downstream.py   exp F      Spearman + OLS over those rows
repr_extract.py         exp E      hooks the eval path for the GLOBAL transformer's input
analyze_repr.py         exp E      Levenshtein + centred cosine, --regime single | suite
report_repr.py          exp E      the E1/E2 tables
join_robustness.py      exp F2     representation similarity vs reports/robust_aligned/ drops
axis_decomposition.py   --         where each arm's macro robustness gap actually comes from
plots.py                figures A-E
report.py               the design's section-14 tables, as markdown
```

Run everything for English:

```bash
export PYTHONPATH=$PWD/lingua AUNET_ROOT=$PWD \
       AUNET_TOK=$PWD/tokenizer/llama3/tokenizer.model TORCHDYNAMO_DISABLE=1
PY=/home/hyunwoong/miniconda3/envs/aunet_eval_cu121/bin/python
S=scripts/probes/whybpebyte
$PY $S/dump.py                --domain en --probe blt_entropy      # needs a GPU
$PY $S/analyze_structural.py  --domain en --probe blt_entropy --n_rand 200
$PY $S/analyze_classes.py     --domain en
$PY $S/analyze_perturb.py     --domain en --sites 16
$PY $S/plots.py               --domain en --probe blt_entropy
$PY $S/report.py              --domain en --probe blt_entropy
```

## Two things worth knowing before reading any number

**The probe.** `probe_lm.py` loads a frozen byte LM and returns REALIZED surprisal
`s_i = -log2 p(x_i | x_<i)`, which `apps.aunet.data.entropy_patch.EntropyModel` does not (it
returns the predictive entropy `H_i`). It also fixes two things that make that class unusable as
a probe for a foreign checkpoint: its CPU path hardcodes lingua's `BOS=256 / offset=0` convention
(Meta's BLT entropy model wants `bos_id=1 / byte_offset=4` -- 7.12 vs 1.45 bits/byte), and
nothing applies the checkpoint's sliding window, because `apps.main.create_causal_mask` honours
`sliding_window` only under `attn_impl="fmha"` while the default path is full-causal sdpa. The
BLT entropy model is trained with a 512-byte local window and costs 3.79 bits/byte without it.

Primary probe: `blt_entropy` (Meta's, 99.5M params, external to every arm here).
Secondary: `byte_50M` (trained in this repo) as an independence check.

**Segmentation is not the same question as representation.** Experiment D says a single edit moves
BPEByte's boundaries about as far as Llama's (p99 radius 8 vs 6 bytes). That is a fact about
segmentation and it does not settle whether "the damage stays confined", which is a claim about the
vector the global transformer reads. Experiment E measures that directly — `tok_embeddings` output
for the subword arm, `transitions[-1].down(...)` for the byte arms — in raw byte coordinates, with
one slot-lookup rule for both families, and centred so the two architectures' cosines are on the
same scale. There the byte arms do keep the trunk input closer to clean on every typo and noise
family, and that similarity (not the boundary movement) is what correlates with the measured
accuracy drops. Quote E, not D, for the robustness claim.

**BPEByte's robustness margin is the absence of a failure axis, not a uniform gain.**
`axis_decomposition.py` splits the canonical 1.3B table: 67% of BPEByte's macro margin over Llama is
the PBP axis (structural for any byte model), 88% of its margin over AU-Net is despace. Remove those
two axes and the three arms are Llama -15.77, AU-Net -12.67, BPEByte -12.09 -- the byte models
within 0.6 pp of each other. Do not write "BPEByte is more robust" without saying which axis.

**Boundary placement.** Every production scheme here is space-leading: a llama3 / AU-Net /
OnlineBPE unit is `" word"`, not `"word "`. So the byte at offset 0 of a unit is usually the
separating space -- the most predictable byte in English text -- and the design's Delta S, read
literally, measures the valley rather than the peak (it comes out NEGATIVE for all three arms).
The suite therefore reports the literal Delta S, the offset-swept `dS_shift` / `dS_gap_vs_random`
(whose maximum lands at +1), and a whitespace-excluded `dS_nws`, and the report shows all three.

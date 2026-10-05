# State tracking & recall — in-context probes of frozen 1.3B checkpoints

Synthetic probes run **in-context (no training)** on the frozen 1.3B
checkpoints, teacher-forced greedy exact-match. Three model families, all
token-budget matched (iso-byte):

| tag | checkpoint | family / pooling rule |
|---|---|---|
| **Llama** | `llama_1.8B_paper` | subword — flat BPE (128k) |
| **AUNet** | `aunet2_1.3B` | byte → **static word** pooling |
| **BPEByte rg** | `bpebyte_br_greedy_root_1.3B` | byte → **greedy-root BPE** boundaries |

`byte (bt)` (`bpebyte_br_bt_1.3B`, before-root BPE boundaries) is also run for the
pooling-rule comparisons in the first battery, but omitted from the headline tables.
All three "byte" models are **byte-input hierarchical** models (byte encoder +
pooling). There is no flat/unpooled byte transformer in the probe registry
(`scripts/probes/common.py:20`), so nothing here speaks to that architecture.

---

## Headline

**No family does state tracking in any general way.** Every model sits at or near
chance on S5 permutation composition (the one probe whose answer *requires*
maintaining a running state through a non-solvable group), scores **exactly
0.000** on whole-completion unbalanced Dyck at every bracket count k=2…8, and
collapses on ≥2-hop Variable Tracking. Depth-controlled S5 (D=1/2/3, equal n per
depth) confirms this is not a bucketing artifact.

The one place a real signal exists is bracket matching, and it is **narrow**: the
mechanism probe (§"Dyck mechanism") isolates the reads where a stack and a
recency heuristic disagree, and on ASCII brackets all three families beat chance
there (0.56–0.80). Respell the same brackets as letter pairs and all three drop
to at-or-below chance. So the stack-like behaviour is tied to ASCII bracket
glyphs — plausibly code exposure in pretraining — not to a general mechanism.

**This is not a notation artifact.** Re-running the same abstract tasks as
English prose (§"Natural-language state tracking": BIG-bench
tracking_shuffled_objects, temporal_sequences, navigate) leaves all three
families on the chance floor in all 15 cells. `tracking_shuffled_objects` is
composition of transpositions written out in fluent English — the natural-
language S5 — and scores 0.187–0.192 against a 0.200 floor, matching the
synthetic probe's 0.166–0.212. Prose framing buys nothing.

**The byte models' advantage is on surface/character access, not state.** BPEByte
rg leads on recall/copy (S-NIAH, MK-NIAH) and on the easy end of bracket
prediction, but that lead is **conditional**: it holds for small k with ASCII
bracket glyphs and **reverses** when the bracket inventory grows (k≥7) or the
delimiters become letters, and it is the *worst* of the three on the real
BIG-bench Dyck benchmark. A model that had acquired a stack would not be this
sensitive to how the brackets are spelled. Note this is **not** the usual
"subword merges the brackets" story — the probe space-joins every symbol
(`dyck.py:39`), so Llama sees one token per bracket and still loses; whatever the
byte models are exploiting on ASCII brackets, it is not relief from token
merging. The mechanism probe pins this down: BPEByte rg's Dyck lead lives almost
entirely in the reads where copying the most recent opener's closer is already
correct, and on the reads where it isn't, BPEByte rg is beaten by AUNet (k=3) or
tied with Llama (k=4).

The pooling rule still matters, but it trades off by axis: BPEByte rg's
content-adaptive boundaries win on copy/recall, AUNet's static word pooling wins
on the harder bracket regimes and on multi-query associative recall.

---

## First battery (2026-07-14)

Bold = best of the three; *(chance)* marks metrics with a random floor.

| regime | benchmark | metric | chance | Llama | AUNet | **BPEByte rg** |
|---|---|---|---|---|---|---|
| State tracking | FFLM dense | read acc (T=512) | 0.50 | 0.90 | 0.91 | **0.92** |
| | FFLM in-dist | read acc | 0.50 | **0.74** | 0.68 | 0.73 |
| | FFLM sparse-OOD | read acc | 0.50 | 0.77 | 0.70 | **0.80** |
| | S5 permutation | read acc | 0.20 | 0.23 | 0.23 | 0.25 |
| | Dyck-3 brackets | close-pred acc | 0.33 | 0.52 | 0.54 | **0.67** |
| Recall / copy | S-NIAH-1 noise+num | exact (mean len) | — | 0.93 | **1.00** | **1.00** |
| | S-NIAH-2 essay+num | exact (mean len) | — | 0.70 | 0.98 | **0.99** |
| | S-NIAH-3 essay+**UUID** | exact (mean len) | — | 0.16 | 0.52 | **1.00** |
| | MK-NIAH K=1 | exact (2 KB) | — | 0.45 | 0.98 | **1.00** |
| | MK-NIAH K=8 | exact (2 KB) | — | 0.18 | 0.30 | **0.33** |
| | Var. Tracking 1-hop | exact | — | **1.00** | 0.35 | 0.75 |
| | Var. Tracking 2-hop | exact | — | **0.20** | 0.03 | 0.18 |

### Base vs. hard — does the advantage survive escalation?

↗ improved · ↘ dropped · → flat.

| benchmark | base → hard | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| S5 permutation | 40ev/2-shot → 60ev/4-shot | 0.23→0.23 | 0.23→0.22 | 0.25↗0.28 |
| Dyck brackets | k3/depth6 → k4/depth10 | 0.52↘0.27 | 0.54↘0.35 | 0.67↘0.52 |
| MK-NIAH K=8 | 2 KB → 4 KB | 0.18↘0.10 | 0.30→0.28 | 0.33↗0.50 |
| Var. Tracking 2-hop | 3 → 6 chains | 0.20↘0.13 | 0.03→0.03 | 0.18↘0.10 |

Read at the time as "the byte advantage holds under deeper Dyck and more
needles." The extended battery below shows the Dyck half of that claim does
**not** generalize past k=4 / ASCII brackets — see §"Extended Dyck".

---

## Extended battery (2026-07-22 → 08-03)

Probes added after the first battery. Same frozen checkpoints, same
teacher-forced greedy protocol. These are what qualify the headline.

### Extended Dyck — the byte lead is condition-dependent

`scripts/probes/dyck.py`, `--length 40 --maxdepth 6 --n 150` (6000 scored
close-predictions per cell), 2-shot. **Balanced** = predict the single correct
closer at each close position (chance = 1/k).

> ⚠️ **This sweep is confounded — read it with the letters sweep below.** The
> default alphabet is `OPEN = "([{<ABCD"` (`dyck.py:24`): k≤4 is pure ASCII
> bracket glyphs, and k=5…8 progressively mix in letter pairs (A/a, B/b, …). So
> moving down this table changes the *delimiter inventory* as well as k. It is
> **not** a clean k-manipulation. The `--letters` sweep holds the alphabet fixed
> across k and is the one to cite for a k effect.

| k | chance | Llama | AUNet | BPEByte rg | winner |
|---|---|---|---|---|---|
| 2 | 0.500 | 0.509 | 0.485 | **0.650** | byte |
| 3 | 0.333 | 0.520 | 0.540 | **0.673** | byte |
| 4 | 0.250 | 0.286 | 0.356 | **0.576** | byte |
| 5 | 0.200 | 0.443 | 0.555 | **0.567** | byte (tie-ish) |
| 6 | 0.167 | 0.450 | 0.509 | **0.517** | byte (tie-ish) |
| 7 | 0.143 | 0.504 | **0.507** | 0.464 | **AUNet** |
| 8 | 0.125 | 0.494 | **0.511** | 0.395 | **AUNet** |

BPEByte rg's margin over the best other family is +0.14/+0.13/+0.22 at k=2/3/4
(pure ASCII brackets), collapses to +0.01/+0.01 at k=5/6, and goes negative
(−0.04/−0.12) at k=7/8. Because of the confound above, the right reading is not
"the byte advantage decays with depth of the bracket inventory" but **"the byte
advantage tracks the fraction of delimiters that are ASCII bracket glyphs"** —
it is largest where the alphabet is 100% brackets (k≤4) and gone once half the
pairs are letters (k=8: 4/8 letter pairs). The letters sweep, where that fraction
is 0 at every k, confirms it: byte has no advantage at any k there.

Note also that *every* model is flat-ish in absolute terms (~0.3–0.65) while
chance falls 0.50→0.125 — the models are using a weak positional/recency
heuristic, not a stack, and the apparent "improvement vs chance" at large k is
mostly the floor dropping.

**Letter delimiters** (same task, brackets respelled as letter pairs;
`run_dyck_letters.sh`):

| k | Llama | AUNet | BPEByte rg |
|---|---|---|---|
| 2 | 0.278 | **0.573** | 0.442 |
| 3 | 0.427 | **0.433** | 0.423 |
| 4 | 0.356 | **0.505** | 0.360 |
| 5 | 0.369 | **0.476** | 0.367 |
| 6 | 0.384 | **0.430** | 0.354 |
| 7 | 0.337 | **0.415** | 0.311 |
| 8 | 0.304 | **0.390** | 0.345 |

AUNet wins **every** cell once the brackets stop being ASCII bracket glyphs (though
k=3 is a three-way tie within 0.01), and BPEByte rg drops to Llama's level. Same
abstract task, same k, same depth, uniform alphabet across the whole sweep — only
the spelling changed. This is the clearest evidence that the byte lead in the
first battery was a surface-form effect, and it is the sweep to cite for any
claim about k, since the default-alphabet sweep above confounds k with delimiter
type.

Caveat in the other direction: the letters variant is plausibly out-of-
distribution for *all three* models (letter pairs as brackets are rare in
pretraining), so it establishes "the byte advantage is glyph-specific" but does
not by itself establish "AUNet is better at bracket state." Absolute accuracies
there (0.30–0.57) are low for everyone.

**Unbalanced Dyck** (BIG-bench style: given an unbalanced prefix, emit the whole
closing suffix; whole-completion exact-match, n=150/cell):

| k | Llama exact / tok-frac | AUNet exact / tok-frac | BPEByte rg exact / tok-frac |
|---|---|---|---|
| 2 | 0.000 / 0.262 | 0.000 / **0.660** | 0.000 / 0.629 |
| 3 | 0.000 / 0.205 | 0.000 / 0.544 | 0.000 / **0.548** |
| 4 | 0.000 / 0.124 | 0.000 / **0.458** | 0.000 / 0.450 |
| 5 | 0.000 / 0.084 | 0.000 / **0.429** | 0.000 / 0.400 |
| 6 | 0.000 / 0.081 | 0.000 / **0.422** | 0.000 / 0.420 |
| 7 | 0.000 / 0.059 | 0.000 / 0.404 | 0.000 / **0.412** |
| 8 | 0.000 / 0.074 | 0.000 / 0.413 | 0.000 / **0.424** |

**Exact-match is 0.000 in all 21 cells.** Not one model closes a multi-bracket
suffix correctly even once. The partial-credit column (per-closer token accuracy)
is the only place a signal exists, and there the two byte models are
indistinguishable from each other (Δ ≤ 0.04 at every k) while both hold a large,
k-stable margin over Llama (0.39–0.66 vs 0.06–0.26) — i.e. the byte/subword gap
on this probe is about emitting bracket characters at all, not about depth.

### Dyck mechanism — is it a stack, or copying? (2026-09-07)

`scripts/probes/dyck_mech.py` + `run_dyck_mech.sh`. Accuracy alone cannot tell a
stack from "emit the closer of the most recent opener you saw", because the two
policies **agree** on most reads. So every read is labelled by whether the
recency answer equals the gold answer, and we report accuracy separately on the
**diagnostic** reads where they disagree. Prediction is **forced choice** over
the k closers (argmax of answer log-likelihood), which also decouples "which
closer" from "is a close even due".

**Headline table — accuracy on diagnostic reads only.** This is the closest
thing in this whole report to a clean state-tracking measure.

| config | chance | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| Dyck-3, ASCII brackets | 0.333 | 0.561 | **0.797** | 0.735 |
| Dyck-4, ASCII brackets | 0.250 | 0.562 | **0.651** | 0.575 |
| Dyck-4, letter delimiters | 0.250 | 0.224 | 0.199 | 0.177 |

Three findings, in order of how much they change the rest of this report.

**(1) The free-argmax metric was measuring the wrong thing.** Under forced choice
Llama goes 0.520 → 0.838 at k=3, AUNet 0.540 → 0.877, BPEByte rg 0.673 → 0.894.
The `dyck.py` numbers everywhere above conflate "picks the wrong closer" with
"does not predict a closer at all", and the second effect dominates. The
`chance ~ 1/k (if the model knows a close is due)` caveat in `dyck.py`'s
docstring turns out to be the main story, not a footnote.

**(2) On ASCII brackets there IS real, partial stack behaviour — and BPEByte rg
is not the one that has it.** All three beat chance on diagnostic reads, so the
earlier "no family does state tracking" reading was too strong for this probe.
But the ranking inverts relative to `dyck.py`: BPEByte rg's overall lead comes
almost entirely from the reads where recency already gives the right answer
(k=3: 0.956, the best of the three; k=4: 0.946, likewise), while on diagnostic
reads it sits at 0.735 (k=3, below AUNet) and 0.575 (k=4, statistically tied with
Llama's 0.562). Its errors are biased toward the recency answer at both k
(recency-share 0.568 / 0.525 vs 0.500 / 0.333 for unbiased errors) and its
distance-matched copy bias is the largest of the three at k=4 (+0.163). **On the
Dyck probe BPEByte rg is a better copier, not a better state tracker** — which is
what the letters and BIG-bench results above already implied, now shown directly.

**(3) The residual stack signal is glyph-specific for every family.** Swap the
brackets for letter pairs and all three fall to **at or below chance** on
diagnostic reads (0.224 / 0.199 / 0.177 vs 0.250) while their copy bias rises to
its maximum (+0.201 / +0.290 / +0.326). So AUNet's diagnostic lead on brackets is
*also* not a general stack; and AUNet winning every cell of the `--letters` sweep
above is likewise a copy effect (its letters accuracy is carried by the
recency-agrees reads, 0.832). Nothing here survives respelling the delimiters.

Note on the distance confound: the raw agree/disagree split is not distance-
matched — a read whose previous token is an opener has its answer at distance 1
and necessarily has recency agreeing, so `agree` is dominated by short
dependencies (mean distance 3.1 vs 11.3). The **distance-matched** tables below
fix this by keeping only reads whose previous token is a closer; there
"agree by luck" and "disagree" match on distance (11.3 vs 11.3) and depth (2.78
vs 2.77). In that regime the recency answer is literally the previous token ~71%
of the time, so the matched Δ measures a copy-the-previous-token bias. AUNet's
negative Δ at k=3 (−0.121) means it is *penalised* when the answer repeats the
previous token — repetition avoidance, the opposite of a recency lure.

#### Full tables

**Dyck-3, ASCII brackets** — k=3, 6000 scored reads/model, **forced choice** over the 3 closers (argmax of answer log-likelihood), so `acc` here is not comparable to `dyck.py`'s free-argmax exact-match — see the note below the table.

`recency-share` = of the ERRORS on diagnostic reads, the fraction that are exactly the recency answer. Unbiased errors give 1/(k-1) = 0.500; pure recency copying gives 1.0. (The raw lure rate P(pred==recency) is in the jsonl; it is not the right comparison because it is bounded by the error rate.)

| model | acc all | acc (recency agrees) | acc (recency **disagrees**) | Δ | recency-share of errors |
|---|---|---|---|---|---|
| Llama | 0.838 | 0.946 [0.939,0.952] | 0.561 [0.537,0.585] | +0.385 | 0.687 [0.653,0.719] |
| AUNet | 0.877 | 0.909 [0.900,0.917] | 0.797 [0.777,0.815] | +0.112 | 0.434 [0.383,0.487] |
| BPEByte rg | 0.894 | 0.956 [0.949,0.961] | 0.735 [0.714,0.756] | +0.221 | 0.568 [0.522,0.613] |

n: agree=4312, disagree=1688 (28.1% of reads). Answer token counts seen: [1] (uniform => the log-likelihood argmax is not length-biased).

Distance-matched (both subsets have a **closer** as the previous token, so dependency distance and depth are matched):

| model | acc, recency right by luck | acc, recency wrong | copy-bias Δ |
|---|---|---|---|
| Llama | 0.739 [0.709,0.767] (n=858) | 0.561 [0.537,0.585] (n=1688) | +0.178 |
| AUNet | 0.676 [0.644,0.706] (n=858) | 0.797 [0.777,0.815] (n=1688) | -0.121 |
| BPEByte rg | 0.782 [0.753,0.808] (n=858) | 0.735 [0.714,0.756] (n=1688) | +0.047 |

Forced-choice chance on both subsets = 1/3 = 0.333.

**Dyck-4, ASCII brackets** — k=4, 6000 scored reads/model, **forced choice** over the 4 closers (argmax of answer log-likelihood), so `acc` here is not comparable to `dyck.py`'s free-argmax exact-match — see the note below the table.

`recency-share` = of the ERRORS on diagnostic reads, the fraction that are exactly the recency answer. Unbiased errors give 1/(k-1) = 0.333; pure recency copying gives 1.0. (The raw lure rate P(pred==recency) is in the jsonl; it is not the right comparison because it is bounded by the error rate.)

| model | acc all | acc (recency agrees) | acc (recency **disagrees**) | Δ | recency-share of errors |
|---|---|---|---|---|---|
| Llama | 0.790 | 0.896 [0.886,0.905] | 0.562 [0.540,0.584] | +0.333 | 0.451 [0.418,0.485] |
| AUNet | 0.824 | 0.903 [0.894,0.912] | 0.651 [0.629,0.672] | +0.253 | 0.338 [0.303,0.375] |
| BPEByte rg | 0.829 | 0.946 [0.939,0.953] | 0.575 [0.552,0.597] | +0.372 | 0.525 [0.491,0.560] |

n: agree=4102, disagree=1898 (31.6% of reads). Answer token counts seen: [1] (uniform => the log-likelihood argmax is not length-biased).

Distance-matched (both subsets have a **closer** as the previous token, so dependency distance and depth are matched):

| model | acc, recency right by luck | acc, recency wrong | copy-bias Δ |
|---|---|---|---|
| Llama | 0.626 [0.588,0.662] (n=663) | 0.562 [0.540,0.584] (n=1898) | +0.064 |
| AUNet | 0.627 [0.590,0.663] (n=663) | 0.651 [0.629,0.672] (n=1898) | -0.023 |
| BPEByte rg | 0.738 [0.703,0.770] (n=663) | 0.575 [0.552,0.597] (n=1898) | +0.163 |

Forced-choice chance on both subsets = 1/4 = 0.250.

**Dyck-4, letter delimiters (control)** — k=4, 6000 scored reads/model, **forced choice** over the 4 closers (argmax of answer log-likelihood), so `acc` here is not comparable to `dyck.py`'s free-argmax exact-match — see the note below the table.

`recency-share` = of the ERRORS on diagnostic reads, the fraction that are exactly the recency answer. Unbiased errors give 1/(k-1) = 0.333; pure recency copying gives 1.0. (The raw lure rate P(pred==recency) is in the jsonl; it is not the right comparison because it is bounded by the error rate.)

| model | acc all | acc (recency agrees) | acc (recency **disagrees**) | Δ | recency-share of errors |
|---|---|---|---|---|---|
| Llama | 0.582 | 0.748 [0.734,0.761] | 0.224 [0.206,0.243] | +0.524 | 0.521 [0.495,0.546] |
| AUNet | 0.632 | 0.832 [0.821,0.843] | 0.199 [0.181,0.217] | +0.634 | 0.567 [0.542,0.591] |
| BPEByte rg | 0.623 | 0.829 [0.818,0.841] | 0.177 [0.160,0.194] | +0.653 | 0.593 [0.569,0.617] |

n: agree=4102, disagree=1898 (31.6% of reads). Answer token counts seen: [1] (uniform => the log-likelihood argmax is not length-biased).

Distance-matched (both subsets have a **closer** as the previous token, so dependency distance and depth are matched):

| model | acc, recency right by luck | acc, recency wrong | copy-bias Δ |
|---|---|---|---|
| Llama | 0.425 [0.388,0.463] (n=663) | 0.224 [0.206,0.243] (n=1898) | +0.201 |
| AUNet | 0.489 [0.451,0.527] (n=663) | 0.199 [0.181,0.217] (n=1898) | +0.290 |
| BPEByte rg | 0.502 [0.464,0.540] (n=663) | 0.177 [0.160,0.194] (n=1898) | +0.326 |

Forced-choice chance on both subsets = 1/4 = 0.250.

---

### BIG-bench Dyck languages (real benchmark)

`bigbench_dyck_languages_multiple_choice`, limit=1000, via lm-eval-harness.
Choice count varies per item, so there is no single chance floor.

| model | acc | stderr |
|---|---|---|
| Llama | 0.150 | 0.011 |
| **AUNet** | **0.226** | 0.013 |
| BPEByte rg | 0.115 | 0.010 |

BPEByte rg is **last**, 6.8σ below AUNet. The synthetic Dyck-3 result
(byte 0.67 vs AUNet 0.54) does not transfer to the benchmark it was meant to
proxy.

### S5 permutation at controlled depth

`s5_depth.py`, exactly D composed transpositions per scored read, n=300/depth,
2-shot, chance = 1/5 = 0.20. This removes the swap-count bucketing confound in
the original S5 run.

| depth | Llama | AUNet | BPEByte rg |
|---|---|---|---|
| 1 | 0.150 | 0.210 | 0.203 |
| 2 | 0.163 | 0.170 | 0.240 |
| 3 | 0.183 | 0.197 | 0.193 |

Everything is inside ±0.05 of the 0.20 floor, with no depth trend and no family
separation. Llama is *below* chance at D=1. Confirms the original reading: S5 is
a wall for all three, and the first battery's byte "lead" (0.25 vs 0.20) is not
reproduced at controlled depth.

### Natural-language state tracking (2026-09-08)

Every state-tracking probe above is symbolic (`w 1 0 r 1`, `( [ ] >`,
`start A B C D E . swap`, `VAR X1234 = 5678.`). That leaves an obvious
alternative explanation open: maybe the models track state fine but cannot parse
the synthetic notation. This battery removes that explanation by running the
same abstract tasks **as English prose**, from BIG-bench, via lm-eval-harness.

`scripts/probes/run_nlstate.sh`. Multiple choice scored by option
log-likelihood — the `generate_until` / CoT variants of these tasks need
instruction-following that 1.3B base checkpoints do not have, same reasoning as
the BIG-bench Dyck run above.

**`tracking_shuffled_objects` is the natural-language S5.** An item reads:

> Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game,
> they are each holding a ball: Alice has a pink ball, Bob has a white ball, …
> As the game progresses, pairs of players trade balls. First, Alice and Dave
> swap balls. Then, Claire and Eve swap balls. … At the end of the game, Alice
> has the ___

That is composition of transpositions — structurally identical to the `s5.py`
probe, dressed in prose. It ships as a 3/5/7-object mix, which we split into
three tasks (`scripts/probes/nlstate_tasks/`) so each is read against its own
floor (1/3, 1/5, 1/7) and the object count works as a difficulty ladder, the way
depth does in the synthetic S5 sweep.

Accuracy ±stderr, and (in parentheses) how many stderr **above the chance floor**. Multiple choice scored by option log-likelihood; the generate_until / CoT variants of these tasks are not meaningful for 1.3B base checkpoints.

| task | chance | n | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|---|
| Tracking shuffled objects, 3 | 0.333 | 750 | 0.329 ±0.017 (-0.2σ) | 0.315 ±0.017 (-1.1σ) | 0.323 ±0.017 (-0.6σ) |
| Tracking shuffled objects, 5 | 0.200 | 1250 | 0.186 ±0.011 (-1.2σ) | 0.186 ±0.011 (-1.2σ) | 0.182 ±0.011 (-1.7σ) |
| Tracking shuffled objects, 7 | 0.143 | 1750 | 0.137 ±0.008 (-0.8σ) | 0.136 ±0.008 (-0.8σ) | 0.134 ±0.008 (-1.1σ) |
| Temporal sequences | 0.250 | 1000 | 0.260 ±0.014 (+0.7σ) | 0.221 ±0.013 (-2.2σ) | 0.270 ±0.014 (+1.4σ) |
| Navigate | 0.500 | 1000 | 0.500 ±0.016 (+0.0σ) | 0.505 ±0.016 (+0.3σ) | 0.533 ±0.016 (+2.1σ) |

**Pooled tracking_shuffled_objects** (750+1250+1750 = 3750 items). The 3/5/7 mix has a mean chance floor of exactly **0.2000**, the same floor as the synthetic S5 permutation probe — so these two numbers measure the same abstract task (composition of transpositions) in symbols vs in English.

| model | NL tracking_shuffled_objects | synthetic S5 (depth-controlled) | chance |
|---|---|---|---|
| Llama | 0.192 | 0.166 | 0.200 |
| AUNet | 0.189 | 0.192 | 0.200 |
| BPEByte rg | 0.187 | 0.212 | 0.200 |

(S5 column = mean of the D=1/2/3 exact-match rows in §"S5 permutation at controlled depth", n=900/model.)

**Read: the synthetic result was not a notation artifact.** All 15 cells sit on
the chance floor. On the pooled tracking-shuffled-objects task all three models
are *below* the 0.200 floor (0.187–0.192), matching their synthetic S5 scores
(0.166–0.212) at the same floor. Raising the object count 3 → 5 → 7 does not
change where any model sits relative to chance, because there is no signal for
the ladder to degrade.

Two cells looked like they might be exceptions — BPEByte rg on Navigate (+2.1σ)
and AUNet on Temporal sequences (−2.2σ). The scoring-rule ablation below resolves
both: the first is noise (it vanishes entirely once the decision rule is
length-normalised), the second is a robust *below*-chance effect.

Two things follow. First, the S5 wall is a property of the task, not of the
symbolic encoding — putting it in fluent English, in-distribution for
pretraining, buys exactly nothing. Second, this is the one regime where the three
families are genuinely indistinguishable: no tokenization effect, no pooling-rule
effect, no surface effect, because none of them is above the floor to begin with.
Contrast Dyck, where a real (if glyph-bound) signal existed and the families
separated.

---

#### Scoring rule ablation (snu20, 2026-09-09)

The run above decides each item by `np.argmax(lls)` — the **raw summed**
log-likelihood of each option (`lm_eval/api/task.py:1510`), which is what
`metric: acc` means. That rule is biased toward short options, and **99.5% of TSO
items have choices of unequal length** (mean within-item spread 6.9 characters,
max 20): `'red ball.'` vs `'purple present.'`. This mattered, because all nine
TSO cells landed *below* chance — a systematic-bias signature, not noise. The
same trap is documented in `scripts/probes/listmax_eval.py`: "raw argmax lands
below chance", which is why that probe reports PMI alongside.

So the five tasks were re-run declaring all three decision rules the harness can
apply to the **same forward passes**:

| rule | decision | cost |
|---|---|---|
| `acc` | argmax `lls` | — (what the first run used) |
| `acc_norm` | argmax `lls / len(choice)` | free, same passes |
| `acc_mutual_info` | argmax `log P(c\|ctx) − log P(c)` (PMI) | ~2× (unconditional passes) |

**acc (raw sum)** — accuracy ±stderr (σ above chance)

| task | chance | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| TSO, 3 objects | 0.333 | 0.331 ±0.017 (-0.2σ) | 0.317 ±0.017 (-0.9σ) | 0.323 ±0.017 (-0.6σ) |
| TSO, 5 objects | 0.200 | 0.186 ±0.011 (-1.3σ) | 0.188 ±0.011 (-1.1σ) | 0.186 ±0.011 (-1.2σ) |
| TSO, 7 objects | 0.143 | 0.136 ±0.008 (-0.8σ) | 0.137 ±0.008 (-0.7σ) | 0.133 ±0.008 (-1.2σ) |
| Temporal sequences | 0.250 | 0.258 ±0.014 (+0.6σ) | 0.220 ±0.013 (-2.3σ) | 0.270 ±0.014 (+1.4σ) |
| Navigate | 0.500 | 0.500 ±0.016 (+0.0σ) | 0.503 ±0.016 (+0.2σ) | 0.525 ±0.016 (+1.6σ) |

**acc_norm (length)** — accuracy ±stderr (σ above chance)

| task | chance | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| TSO, 3 objects | 0.333 | 0.344 ±0.017 (+0.6σ) | 0.325 ±0.017 (-0.5σ) | 0.327 ±0.017 (-0.4σ) |
| TSO, 5 objects | 0.200 | 0.186 ±0.011 (-1.2σ) | 0.187 ±0.011 (-1.2σ) | 0.187 ±0.011 (-1.2σ) |
| TSO, 7 objects | 0.143 | 0.142 ±0.008 (-0.1σ) | 0.134 ±0.008 (-1.1σ) | 0.136 ±0.008 (-0.8σ) |
| Temporal sequences | 0.250 | 0.249 ±0.014 (-0.1σ) | 0.207 ±0.013 (-3.4σ) | 0.270 ±0.014 (+1.4σ) |
| Navigate | 0.500 | 0.485 ±0.016 (-0.9σ) | 0.507 ±0.016 (+0.4σ) | 0.500 ±0.016 (+0.0σ) |

**PMI** — accuracy ±stderr (σ above chance)

| task | chance | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| TSO, 3 objects | 0.333 | 0.336 ±0.017 (+0.2σ) | 0.323 ±0.017 (-0.6σ) | 0.328 ±0.017 (-0.3σ) |
| TSO, 5 objects | 0.200 | 0.197 ±0.011 (-0.3σ) | 0.191 ±0.011 (-0.8σ) | 0.198 ±0.011 (-0.2σ) |
| TSO, 7 objects | 0.143 | 0.138 ±0.008 (-0.6σ) | 0.141 ±0.008 (-0.2σ) | 0.138 ±0.008 (-0.6σ) |
| Temporal sequences | 0.250 | 0.243 ±0.014 (-0.5σ) | 0.217 ±0.013 (-2.5σ) | 0.255 ±0.014 (+0.4σ) |
| Navigate | 0.500 | 0.504 ±0.016 (+0.3σ) | 0.506 ±0.016 (+0.4σ) | 0.500 ±0.016 (+0.0σ) |

**Pooled tracking_shuffled_objects** (n=3750, mean chance exactly 0.2000 — the same floor as the synthetic S5 probe)

| model | acc (raw sum) | acc_norm (length) | PMI | synthetic S5 | chance |
|---|---|---|---|---|---|
| Llama | 0.191 | 0.197 | 0.197 | 0.166 | 0.200 |
| AUNet | 0.190 | 0.190 | 0.194 | 0.192 | 0.200 |
| BPEByte rg | 0.189 | 0.191 | 0.196 | 0.212 | 0.200 |

**Read: the below-chance pattern was a scoring artifact; the absence of signal
was not.** Length normalisation and PMI both remove the systematic sag — pooled
TSO goes 0.191/0.190/0.189 (raw) → 0.197/0.194/0.196 (PMI) against the 0.200
floor. But no signal appears under any rule: across all 45 cells nothing clears
+2σ upward. BPEByte rg's Navigate result, the only apparent positive in the first
run, collapses from +1.6σ (raw) to **+0.0σ under both `acc_norm` and PMI** — it
was option-length preference, not navigation.

One cell is genuinely anomalous: **AUNet on Temporal sequences sits below chance
under every rule** (−2.3σ raw, −3.4σ norm, −2.5σ PMI). Treating the 15 cells as
the family, Bonferroni α=0.05 needs |z| > 2.94, which −3.4σ clears. Below-chance
is a systematic preference for a *wrong* option — a distractor bias — so it is
not evidence of ability in either direction; it is worth a follow-up on what that
distractor is, not a state-tracking finding.

**Provenance and cross-machine reproducibility.** This ablation ran on **snu20**
(4× A5000, verified idle with no compute apps before launch, per the node
policy); the first run was local. The `acc` column should therefore reproduce the
first run exactly — same checkpoints (md5-verified after a 51 GB transfer), same
lm_eval 0.4.12, same pre-seeded dataset cache. It very nearly does: **29 of
17,250 item decisions flip (0.168%)**, the largest cell discrepancy being 0.51×
that cell's stderr (BPEByte rg on Navigate, 8 items). This is ordinary
cross-device floating-point nondeterminism flipping near-tied options, and it is
an order of magnitude below the noise floor — but it does mean **cell-level
comparisons between the local and snu20 runs are only meaningful to ~±0.5σ**.

---

### Variable Tracking with digit values

`vt.py --hop_counts 1 2 --value_digits 1..6 --num_chains 3 --n 50`. Exact-match
on the retrieved value; chance ≈ 10⁻ᵈ (negligible).

| hops | Llama | AUNet | BPEByte rg |
|---|---|---|---|
| 1 (mean over d=1…6) | **0.96** | 0.46 | 0.83 |
| 2 (mean over d=1…6) | **0.32** | 0.07 | 0.20 |

Per value-length, 1-hop: Llama 0.92/0.96/0.94/0.98/0.96/1.00 · AUNet
0.40/0.40/0.52/0.48/0.44/0.52 · BPEByte rg 0.84/0.78/0.76/0.84/0.84/0.94.
Llama leads at both hop counts. Llama and BPEByte rg each lose ~0.64 absolute going
1→2 hops; AUNet starts too low (0.46) to fall as far and lands effectively at zero. The ordering here is the reverse of the
copy/recall probes — VT chains are short numeric tokens, which is exactly where
subword tokenization is not a handicap.

### MQRAR — multi-query associative recall

`mqrar_eval.py --vocab 256 --seq_len 192 --n_seq 30 --shots 2`. top-1 chance =
1/256 ≈ 0.004.

| n_kv | queries | Llama | AUNet | BPEByte rg |
|---|---|---|---|---|
| 4 | 122 | 0.205 | **0.369** | 0.344 |
| 8 | 246 | 0.146 | **0.272** | 0.256 |
| 16 | 509 | 0.132 | **0.173** | 0.165 |
| 32 | 1076 | 0.098 | 0.099 | 0.099 |

Both byte models beat Llama at low KV load and the three converge by n_kv=32.
AUNet edges BPEByte rg at every load — the one recall-family probe where static
word pooling wins.

---

## Per-benchmark detail (first battery)

### FFLM (flip-flop, read accuracy, T=512)
| model | dense | in-dist | sparse-OOD |
|---|---|---|---|
| Llama | 0.904 | 0.735 | 0.772 |
| AUNet | 0.908 | 0.679 | 0.695 |
| BPEByte rg | 0.924 | 0.728 | 0.796 |
| byte (bt) | 0.931 | 0.702 | 0.774 |

Recency-glitch (acc when nearest distractor agrees vs disagrees): AUNet +0.125
(worst), BPEByte rg +0.093, byte(bt) +0.054, Llama +0.044.

### S-NIAH exact-match by context length (bytes)
| task | model | 512 | 1024 | 2048 | 4096 | 6144 |
|---|---|---|---|---|---|---|
| S1 noise+num | Llama | 0.85 | 0.85 | 1.00 | 1.00 | 0.95 |
| | AUNet | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S2 essay+num | Llama | 0.90 | 0.70 | 0.65 | 0.65 | 0.60 |
| | AUNet | 1.00 | 1.00 | 0.95 | 1.00 | 0.95 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| S3 essay+UUID | Llama | 0.15 | 0.15 | 0.05 | 0.25 | 0.20 |
| | AUNet | 0.85 | 0.60 | 0.45 | 0.30 | 0.40 |
| | BPEByte rg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

S-NIAH-3 verified real by an absent-needle control (BPEByte rg exact 0.00 with
needle removed vs 1.00 with needle).

### MK-NIAH exact-match by #needles K (2 KB)
| model | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| Llama | 0.45 | 0.28 | 0.33 | 0.18 |
| AUNet | 0.98 | 0.73 | 0.43 | 0.30 |
| BPEByte rg | 1.00 | 0.90 | 0.73 | 0.33 |

Hard (4 KB): K=4/8/16 → Llama 0.05/0.10/0.10 · AUNet 0.38/0.28/0.20 · BPEByte rg 0.60/0.50/0.40.

**Pressure-test grid (depth × context, to 8 KB)** — `../niah/mk_grid.png`. Within
window, BPEByte rg is the most uniform, AUNet shows the "lost-in-the-middle" dip,
Llama is weakest and collapses past 4 KB. At **8 KB body every family is
out-of-window** → those cells are **N/A**: the byte models cap at **8192 bytes**
and Llama at **2048 tokens** (an 8k-body prompt is ~8.6 KB / ~2.0–2.2 k tokens,
exceeding both). The largest cleanly-testable body is **~6 KB**. Note: the naive
over-window score was a truncation artifact (empty graded slice → vacuous pass),
caught by an absent-needle control and now guarded in `score()` (empty ⇒ miss);
`mk_grid.py` marks a cell N/A if any sample exceeds `WINDOW` (aunet 8192 B,
subword 2048 tok). The Llama checkpoint's `params.json` records `max_seqlen 4096`,
but the effective window is **2048 tokens** (RoPE θ=10000).

### Variable Tracking exact-match by #hops (3 chains, word values)
| model | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Llama | 1.00 | 0.20 | 0.20 | 0.13 |
| AUNet | 0.35 | 0.03 | 0.05 | 0.00 |
| BPEByte rg | 0.75 | 0.18 | 0.15 | 0.03 |

### S5 / Dyck (base & hard)
| task | Llama | AUNet | BPEByte rg | byte(bt) | chance |
|---|---|---|---|---|---|
| S5 base (40ev/2-shot) | 0.230 | 0.229 | 0.252 | 0.232 | 0.20 |
| S5 hard (60ev/4-shot) | 0.234 | 0.222 | 0.276 | 0.248 | 0.20 |
| Dyck-3 base (depth6) | 0.520 | 0.540 | 0.673 | 0.690 | 0.33 |
| Dyck-4 hard (depth10) | 0.274 | 0.354 | 0.523 | 0.558 | 0.25 |

---

## What these probes do and do not license

**Supported.** (1) At iso-byte budget, byte-input hierarchical models are far
better than subword at verbatim character-level recall (S-NIAH-3 UUID: 1.00 vs
0.16) and at emitting bracket characters at all (unbalanced-Dyck token accuracy).
(2) The pooling rule is a real, measurable knob, and it trades off by axis:
content-adaptive (rg) for copy/recall, static-word (AUNet) for hard bracket
regimes and multi-query recall. (3) No family in this suite performs *general*
state tracking: S5 is at chance, unbalanced Dyck exact is 0.000, multi-hop VT
collapses, the natural-language versions of the same tasks are at chance in all
15 cells, and the one probe with a real signal (bracket matching) loses it as
soon as the delimiters are respelled. (5) The failure is not a notation or
parsing artifact — English-prose framing of the same tasks changes nothing. (4) On the Dyck probe specifically,
BPEByte rg's lead is a copying advantage, not a state-tracking one — shown
directly by the diagnostic-read split, not inferred from accuracy alone.

**Not supported.** (1) "Byte models are better at state tracking" — the Dyck
evidence that motivated this reverses under delimiter changes, does not survive
contact with BIG-bench Dyck, and is attributed to copying by the mechanism probe.
(2) "AUNet is better at bracket state" — its diagnostic-read lead on ASCII
brackets is real but vanishes (to below chance) on letters, so it is glyph-bound
too. (3) Anything about flat/unpooled byte transformers — none were run. (4) Any
claim about trained-on-task performance; everything here is in-context on frozen
LM checkpoints.

**Open.** (1) S5 being at chance for all three is consistent with the TC⁰
expressivity argument (S5 is non-solvable, so no fixed-depth transformer composes
it), in which case no tokenization change would help and the probe is a floor
rather than a discriminator. The natural-language battery rules out *notation* as
the explanation but not this one: distinguishing "expressivity wall" from "not
learned at 1.3B" still needs either a fine-tuned control, a scratchpad/CoT arm
(which would lift a fixed-depth limit but not a learning failure), or a scale
ladder; none was run. (2) *Why* ASCII brackets specifically carry the residual stack signal is
untested. The obvious hypothesis is pretraining exposure to code, but the probe
that would show it — bracket-matching accuracy on real code, or a rare-Unicode
delimiter arm — has not been run. (3) The mechanism probe covers k=3/k=4 only;
whether the diagnostic-read ranking holds across the full k=2…8 sweep is unknown.

---

## Figures
- `../fflm/fflm_read_accuracy.png` — FFLM read accuracy by regime
- `../fflm/fflm_acc_vs_distance.png` — U-shaped accuracy vs dependency distance
- `../fflm/recency_glitch.md` — recency-copy error analysis
- `../fflm/segmentation_viz.png` / `_sparse.png` — how each model chunks FFLM
- `../niah/niah_exact_vs_length_S3.png` — UUID copy vs context length
- `../niah/mk_grid.png` — MK-NIAH pressure-test grid (depth × context)
- `statetrack_overview.png` — S5 / Dyck / MK / VT battery (first battery only)
- `statetrack_base_vs_hard.png` — base vs hard, all four tasks (first battery only)

No figures exist yet for the extended battery (extended Dyck, BIG-bench Dyck,
S5-depth, VT-digits, MQRAR), the Dyck mechanism probe, or the natural-language
battery; those sections are tables only.

## Method
Frozen 1.3B checkpoints, in-context (no fine-tuning). Each read/answer is scored
by teacher-forced greedy: the model's argmax must reproduce the answer token(s)
verbatim (== it would greedily generate them). Tokenization-agnostic across
subword / byte families. Modest sample counts (n=40–2000 for the first battery;
n=150/cell × 6000 scored reads for extended balanced Dyck, n=150 for unbalanced,
n=300/depth for S5-depth, n=50/cell for VT-digits, 1000 items for BIG-bench
Dyck) — scale for tighter CIs.

Probe code and runners:

| battery | code | runner |
|---|---|---|
| FFLM | `scripts/fflm/` | — |
| S-NIAH / MK-NIAH | `scripts/niah/`, `scripts/probes/mk_niah.py` | — |
| S5 / Dyck / VT (first) | `scripts/probes/{s5,dyck,vt}.py` | — |
| Extended Dyck | `scripts/probes/dyck.py` | `run_dyck_ext.sh`, `run_dyck_k58.sh`, `run_dyck_letters.sh`, `run_dyck_unbal_rerun.sh` |
| Dyck mechanism | `scripts/probes/dyck_mech.py`, report via `report_dyck_mech.py` | `run_dyck_mech.sh` (`GPUS="0 1"`) |
| BIG-bench Dyck | lm-eval-harness | `scripts/probes/run_bigbench_dyck.sh` |
| NL state tracking | lm-eval-harness + `scripts/probes/nlstate_tasks/`, report via `report_nlstate.py` | `run_nlstate.sh` (`GPUS="0 1"`) |
| NL scoring ablation | same tasks, three metrics; report via `report_nlstate_pmi.py` | `run_nlstate_pmi.sh` (local) / `run_nlstate_pmi_snu20.sh` (snu20) |
| S5 depth | `scripts/probes/s5_depth.py` | `run_s5_depth.sh` |
| VT digits | `scripts/probes/vt.py` | `run_vt_digits.sh` |
| MQRAR | `scripts/probes/mqrar_eval.py` | `run_mqrar.sh` |

Raw results: `reports/statetrack/{dyck_ext,dyck_k58,dyck_letters,dyck_mech,bigbench_dyck,nlstate,nlstate_pmi,vt_digits,mqrar}/`,
`s5_depth_results.jsonl`, `{s5,dyck,vt,mkniah}_*results.jsonl`.

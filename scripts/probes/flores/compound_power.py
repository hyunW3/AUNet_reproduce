#!/usr/bin/env python3
"""Three ways to get more power out of the compound hypothesis than slicing thinner strata.

(a) thinner strata (>=3, >=4 long tokens) -- the obvious move, and why it does not help
(b) sentence chrF instead of sentence BLEU -- same data, far lower variance
(c) regression of the per-sentence gap on a continuous compound predictor, pooled over the
    five Latin languages -- uses every sentence and needs no threshold
"""
import json, math, statistics as st
from pathlib import Path

import sacrebleu

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
RUNS = {"AU-Net2": "aunet2_1.3B", "BPEByte": "bpebyte_br_greedy_root_1.3B"}
LANGS = {"de": "sentence_deu_Latn", "fr": "sentence_fra_Latn", "es": "sentence_spa_Latn",
         "it": "sentence_ita_Latn", "nl": "sentence_nld_Latn"}
THRESH = 13


def pairs(run, task):
    d = json.load(open(ROOT / "runs" / run / "eval_flores_dev" / "results.json"))
    return {s["doc_id"]: s for s in d["samples"][task]}


def sent(metric, pair):
    ref, hyp = pair[0], pair[1]
    ref = ref[0] if isinstance(ref, list) else ref
    hyp = hyp[0] if isinstance(hyp, list) else hyp
    f = sacrebleu.sentence_bleu if metric == "bleu" else sacrebleu.sentence_chrf
    return f(hyp, [ref]).score


def collect():
    out = []
    for lg, col in LANGS.items():
        task = f"flores_dev_{lg}-en"
        A, B = pairs(RUNS["AU-Net2"], task), pairs(RUNS["BPEByte"], task)
        for i in sorted(set(A) & set(B)):
            src = A[i]["doc"][col]
            toks = [t.strip(".,;:!?\"'()«»") for t in src.split()]
            bl = [len(t.encode()) for t in toks]
            nb = sum(bl) or 1
            out.append({"lang": lg, "n_long": sum(x >= THRESH for x in bl),
                        "long_frac": sum(x for x in bl if x >= THRESH) / nb,
                        "d_bleu": sent("bleu", B[i]["bleu"]) - sent("bleu", A[i]["bleu"]),
                        "d_chrf": sent("chrf", B[i]["chrf"]) - sent("chrf", A[i]["chrf"])})
    return out


def summarize(rows, key, label):
    d = [r[key] for r in rows]
    if len(d) < 5:
        return f"n={len(d):4d}  (too few)"
    m, se = st.mean(d), st.pstdev(d) / math.sqrt(len(d))
    return f"n={len(d):4d}  Δ {m:+6.2f} ± {se:4.2f}   t={m/se:5.2f}"


def main():
    rows = collect()
    de = [r for r in rows if r["lang"] == "de"]
    print(f"(a) thinner strata — German only, sentence BLEU (n shrinks as fast as the effect grows)")
    for k in (0, 1, 2, 3, 4):
        sel = [r for r in de if r["n_long"] >= k]
        print(f"    >={k} long tokens   {summarize(sel,'d_bleu','')}")
    print(f"\n(b) same strata, sentence chrF — much lower variance")
    for k in (0, 1, 2, 3, 4):
        sel = [r for r in de if r["n_long"] >= k]
        print(f"    >={k} long tokens   {summarize(sel,'d_chrf','')}")

    def ols(xs, ys):
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        b = sxy/sxx; a = my - b*mx
        resid = [y - (a + b*x) for x, y in zip(xs, ys)]
        s2 = sum(r*r for r in resid)/(n-2)
        se = math.sqrt(s2/sxx)
        return b, se, b/se, n

    print(f"\n(c) regression of the per-sentence gap on compound load, pooled over de/fr/es/it/nl")
    for metric in ("d_bleu", "d_chrf"):
        for pred in ("n_long", "long_frac"):
            b, se, t, n = ols([r[pred] for r in rows], [r[metric] for r in rows])
            print(f"    {metric:7s} ~ {pred:10s}  slope {b:+7.3f} ± {se:5.3f}   t={t:5.2f}  n={n}")
    print("\n    German only:")
    for metric in ("d_bleu", "d_chrf"):
        b, se, t, n = ols([r["n_long"] for r in de], [r[metric] for r in de])
        print(f"    {metric:7s} ~ n_long      slope {b:+7.3f} ± {se:5.3f}   t={t:5.2f}  n={n}")


if __name__ == "__main__":
    main()

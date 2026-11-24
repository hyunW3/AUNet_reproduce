#!/usr/bin/env python3
"""Does AU-Net lose to BPEByte specifically on sentences with long compounds?

The cross-language correlation (patch_stats.json) is only 5 points and confounded by everything
that varies between languages. This tests the same claim WITHIN German: split the 200 dev
sentences by whether the source contains a long whitespace token -- which is what AU-Net pools
into a single patch and BPEByte splits -- and compare per-sentence BLEU in each stratum.

Paired by sentence, so translation difficulty cancels; only the stratum assignment differs.
"""
import json, statistics as st, sys

import sacrebleu
from pathlib import Path

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
RUNS = {"AU-Net2": "aunet2_1.3B", "BPEByte": "bpebyte_br_greedy_root_1.3B", "Llama": "llama_1.8B_paper"}
TASK = "flores_dev_de-en"
SRC = "sentence_deu_Latn"
THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 13   # AU-Net's p95 patch length on German


def sent_bleu(pair):
    """lm-eval stores the metric's (reference, hypothesis) arguments, not a score."""
    ref, hyp = pair[0], pair[1]
    ref = ref[0] if isinstance(ref, list) else ref
    hyp = hyp[0] if isinstance(hyp, list) else hyp
    return sacrebleu.sentence_bleu(hyp, [ref]).score


def load(run):
    d = json.load(open(ROOT / "runs" / run / "eval_flores_dev" / "results.json"))
    return {s["doc_id"]: s for s in d["samples"][TASK]}


def main():
    data = {k: load(v) for k, v in RUNS.items()}
    ids = sorted(set.intersection(*[set(v) for v in data.values()]))
    rows = []
    for i in ids:
        src = data["AU-Net2"][i]["doc"][SRC]
        toks = [t.strip(".,;:!?\"'()«»") for t in src.split()]
        longest = max((len(t.encode()) for t in toks), default=0)
        rows.append({"id": i, "longest_token": longest, "n_long": sum(len(t.encode()) >= THRESH for t in toks),
                     **{m: sent_bleu(data[m][i]["bleu"]) for m in RUNS}})

    def report(name, sel):
        g = [r for r in rows if sel(r)]
        if len(g) < 5:
            print(f"  {name:28s} n={len(g):3d}  (too few)"); return
        a, b = st.mean(r["AU-Net2"] for r in g), st.mean(r["BPEByte"] for r in g)
        dl = [r["BPEByte"] - r["AU-Net2"] for r in g]
        se = st.pstdev(dl) / (len(dl) ** 0.5)
        print(f"  {name:28s} n={len(g):3d}  AU-Net2 {a:5.2f}  BPEByte {b:5.2f}  "
              f"Δ {b-a:+5.2f} ± {se:.2f}   (BPEByte ahead in {sum(d>0 for d in dl)}/{len(g)})")

    print(f"German dev, n={len(rows)}, sentence-level BLEU. Long token = >= {THRESH} bytes "
          f"(AU-Net's p95 patch on German).\n")
    print("Stratified by whether the SOURCE contains a long whitespace token:")
    report("no long token", lambda r: r["n_long"] == 0)
    report("has >=1 long token", lambda r: r["n_long"] >= 1)
    report("has >=2 long tokens", lambda r: r["n_long"] >= 2)
    print("\nBy the longest token in the source:")
    for lo, hi in [(0, 10), (10, 13), (13, 16), (16, 99)]:
        report(f"longest token {lo}-{hi}B", lambda r, lo=lo, hi=hi: lo <= r["longest_token"] < hi)
    # control: same split, Llama (subword) vs AU-Net -- is the stratum just "harder sentences"?
    print("\nControl — Llama (subword) on the same strata, to separate 'compound sentences are")
    print("just harder' from 'AU-Net specifically suffers on them':")
    for name, sel in [("no long token", lambda r: r["n_long"] == 0),
                      ("has >=1 long token", lambda r: r["n_long"] >= 1)]:
        g = [r for r in rows if sel(r)]
        l_, a_, b_ = (st.mean(r[m] for r in g) for m in ("Llama", "AU-Net2", "BPEByte"))
        print(f"  {name:28s} n={len(g):3d}  Llama {l_:5.2f}  AU-Net2 {a_:5.2f}  BPEByte {b_:5.2f}"
              f"   (Llama-AUNet {l_-a_:+5.2f})")
    json.dump(rows, open(ROOT / "reports_afterAAAIsub/flores/compound_stratified.json", "w"), indent=1)
    print("\nWROTE reports_afterAAAIsub/flores/compound_stratified.json")


if __name__ == "__main__":
    main()

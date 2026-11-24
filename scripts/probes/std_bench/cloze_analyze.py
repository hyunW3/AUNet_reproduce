#!/usr/bin/env python3
"""Analyze fmt_robust.py --mode cloze (LAMBADA, blank marker x answer cue x shots) for the matched models.

Prints, per shot count: the blank x cue accuracy grid per model, the marginal effect of each blank marker
(mean over cues) and each cue (mean over blanks), and paired item-bootstrap CIs for the contrasts of the
original observation, relative to the GPT-3 frame "____. ->":
  cue swap    "____. ->"  -> "____. Answer:"      blank swap  "____. ->" -> "BLANK. ->"
  no blank    "____. ->"  -> "->"                  no cue      "____. ->" -> "____."
  standard    "____. ->"  -> plain passage (none/none)
For each contrast: per-model change and the BPEByte - Transformer / BPEByte - AU-Net difference of changes.

  python3 cloze_analyze.py --root reports/fmt_robust --n 1000 [--mode clozefull] [--out summary_cloze.json]
"""
import argparse, json, random
from collections import defaultdict

M = ["llama_1.3b", "aunet_1.3b", "bpebyte_1.3b"]
S = {"llama_1.3b": "Transf", "aunet_1.3b": "AUNet", "bpebyte_1.3b": "BPEByte"}
BLANKS = ["____.", "___", "BLANK.", "[blank].", "none"]
CUES = ["->", "Answer:", "=>", "=", "Word:", "none"]
REF = ("____.", "->")
CONTRASTS = {"cue ->  =>  Answer:": ("____.", "Answer:"), "cue ->  =>  Word:": ("____.", "Word:"),
             "cue ->  =>  =>": ("____.", "=>"), "cue ->  =>  =": ("____.", "="), "blank ____.  =>  BLANK.": ("BLANK.", "->"),
             "blank ____.  =>  [blank].": ("[blank].", "->"), "blank ____.  =>  ___": ("___", "->"),
             "drop blank (-> only)": ("none", "->"), "drop cue (____. only)": ("____.", "none"),
             "standard format (none/none)": ("none", "none")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/fmt_robust")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--mode", default="cloze", help="cloze (fixed demos) or clozefull (per-item demos, seeds averaged)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {}
    for m in M:
        acc_ = defaultdict(list)
        for l in open(f"{a.root}/{m}/{a.mode}_n{a.n}.jsonl"):
            r = json.loads(l)
            acc_[(r["blank"], r["cue"], r["shots"])].append([x["correct"] for x in r["rows"]])
        # per-item correctness averaged over seeds (a single run for the fixed-demo mode)
        R[m] = {key: [sum(col) / len(col) for col in zip(*runs)] for key, runs in acc_.items()}
    SHOTS = sorted({key[2] for key in R[M[0]]})
    n = a.n
    rng = random.Random(0)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(a.B)]
    acc = lambda v, idx=None: 100 * (sum(v) / n if idx is None else sum(v[i] for i in idx) / len(idx))
    out = defaultdict(dict)
    for k in SHOTS:
        print(f"\n######## {k}-shot   (rows: blank marker, cols: cue; cells Transf / AUNet / BPEByte)")
        print(f"{'':10s}" + "".join(f"{c:>20s}" for c in CUES) + f"{'mean':>20s}")
        for b in BLANKS:
            cells = ["/".join(f"{acc(R[m][(b, c, k)]):.0f}" for m in M) for c in CUES]
            mean = "/".join(f"{sum(acc(R[m][(b, c, k)]) for c in CUES) / len(CUES):.1f}" for m in M)
            print(f"{b:10s}" + "".join(f"{x:>20s}" for x in cells) + f"{mean:>20s}")
        means = ["/".join(f"{sum(acc(R[m][(b, c, k)]) for b in BLANKS) / len(BLANKS):.1f}" for m in M) for c in CUES]
        print(f"{'mean':10s}" + "".join(f"{x:>20s}" for x in means))
        out[k]["grid"] = {f"{b}|{c}": {m: acc(R[m][(b, c, k)]) for m in M} for b in BLANKS for c in CUES}
        ref = {m: R[m][(*REF, k)] for m in M}
        print(f"\n  contrasts vs '____. ->' (change in accuracy, paired 95% CI)")
        print(f"  {'':30s}" + "".join(f"{S[m]:>22s}" for m in M) + f"{'BPE-Transf':>24s}{'BPE-AUNet':>24s}")
        for name, cell in CONTRASTS.items():
            v = {m: R[m][(*cell, k)] for m in M}
            d = lambda m, idx: (sum(v[m][i] for i in idx) - sum(ref[m][i] for i in idx)) * 100 / len(idx)
            pt = {m: d(m, range(n)) for m in M}
            ci = {}
            for m in M:
                s = sorted(d(m, b) for b in boots); ci[m] = (s[int(.025 * a.B)], s[int(.975 * a.B) - 1])
            dd = {}
            for r in ("llama_1.3b", "aunet_1.3b"):
                s = sorted(d("bpebyte_1.3b", b) - d(r, b) for b in boots)
                dd[r] = (pt["bpebyte_1.3b"] - pt[r], s[int(.025 * a.B)], s[int(.975 * a.B) - 1])
            print(f"  {name:30s}" + "".join(f"{pt[m]:+7.1f} [{ci[m][0]:+5.1f},{ci[m][1]:+5.1f}]" for m in M)
                  + "".join(f"{dd[r][0]:+7.1f} [{dd[r][1]:+5.1f},{dd[r][2]:+5.1f}]" for r in dd))
            out[k][name] = {"change": pt, "ci": ci, "bpe_minus": dd}
        # the cue swap "->" => "Answer:" inside every blank-marker row (is the cue effect blank-specific?)
        print(f"\n  cue swap '->' => 'Answer:' within each blank row")
        for b in BLANKS:
            v0 = {m: R[m][(b, "->", k)] for m in M}; v1 = {m: R[m][(b, "Answer:", k)] for m in M}
            d = lambda m, idx: (sum(v1[m][i] for i in idx) - sum(v0[m][i] for i in idx)) * 100 / len(idx)
            pt = {m: d(m, range(n)) for m in M}
            s_ = sorted(d("bpebyte_1.3b", bb) - d("llama_1.3b", bb) for bb in boots)
            print(f"  {b:10s}" + "".join(f"{S[m]} {pt[m]:+6.1f}  " for m in M)
                  + f"BPE-Transf {pt['bpebyte_1.3b'] - pt['llama_1.3b']:+5.1f} [{s_[int(.025 * a.B)]:+5.1f},{s_[int(.975 * a.B) - 1]:+5.1f}]")
            out[k].setdefault("rowwise_cue_swap", {})[b] = {"change": pt, "bpe_minus_transf": [
                pt["bpebyte_1.3b"] - pt["llama_1.3b"], s_[int(.025 * a.B)], s_[int(.975 * a.B) - 1]]}
        # spread over all 30 frames and over the 24 frames that keep at least one cue element
        for m in M:
            vals = [acc(R[m][(b, c, k)]) for b in BLANKS for c in CUES]
            out[k].setdefault("spread", {})[m] = (min(vals), max(vals))
        print("  range over the 30 frames: " + "  ".join(f"{S[m]} {out[k]['spread'][m][0]:.1f}-{out[k]['spread'][m][1]:.1f}" for m in M))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

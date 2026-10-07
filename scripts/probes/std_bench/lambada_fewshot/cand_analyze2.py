#!/usr/bin/env python3
"""Shot curves of referent choice (cand_<model>.json + cand_<model>_k03.json) and the bias of wrong choices.

(1) choice / greedy accuracy at 0/3/5 shots, standard vs cloze format (5-shot also with "Answer:").
(2) Wrong choices on items with >= 2 other names: share that is the most frequent other name / the most recently
    mentioned other name, against the share expected if the wrong name were drawn uniformly from the others.
"""
import json, re
import numpy as np

T = "/home/hyunw3/.claude/jobs/003c5e28/tmp"
P0 = json.load(open(f"{T}/lambada_prompts.json"))["0"]
MODELS = [("Transformer", "llama_1.3b"), ("AUNet", "aunet_1.3b"), ("BPEByte", "bpebyte_1.3b"), ("BLT", "blt_1b"), ("H-Net", "hnet_1stage_XL")]
COLS = [("std0", "std 0"), ("std3", "std 3"), ("std", "std 5"), ("cloze0", "cloze 0"), ("cloze3", "cloze 3"), ("cloze", "cloze 5"), ("answer", "Ans. 5")]


def occ(text, w):
    return [m.start() for m in re.finditer(r"\b" + re.escape(w) + r"\b", text)]


def load(reg):
    D = json.load(open(f"{T}/cand_{reg}.json"))
    K = json.load(open(f"{T}/cand_{reg}_k03.json"))
    for d, k in zip(D, K):
        assert d["i"] == k["i"]
        d["ll"].update(k["ll"]); d["greedy"].update(k["greedy"])
    return D


def main():
    out = {}
    print("(1) name choice / greedy accuracy, %")
    print(f"{'model':12s}" + "".join(f"{c[1]:>14s}" for c in COLS))
    for name, reg in MODELS:
        D = load(reg)
        cells = []
        for f, _ in COLS:
            ch = np.mean([max(d["ll"][f], key=d["ll"][f].get) == d["tgt"] for d in D])
            gr = np.mean([d["greedy"][f][d["tgt"]] for d in D])
            out[f"{reg}|{f}"] = {"choice": ch, "greedy": gr}
            cells.append(f"{100*ch:5.1f} / {100*gr:4.1f}")
        print(f"{name:12s}" + "".join(f"{c:>14s}" for c in cells))

    print("\n(2) wrong choices, items with >= 2 other names: most-frequent / most-recent other name (expected if uniform)")
    for name, reg in MODELS:
        D = load(reg)
        for f in ("std", "cloze", "answer"):
            n = fr = rc = efr = erc = 0
            for d in D:
                others = [c for c in d["cands"] if c != d["tgt"]]
                if len(others) < 2:
                    continue
                best = max(d["ll"][f], key=d["ll"][f].get)
                if best == d["tgt"]:
                    continue
                passage = P0[d["i"]]["ctx"]
                cnt = {c: len(occ(passage, c)) for c in others}
                last = {c: max(occ(passage, c), default=-1) for c in others}
                top_f = [c for c in others if cnt[c] == max(cnt.values())]
                top_r = max(others, key=last.get)
                n += 1
                fr += best in top_f; rc += best == top_r
                efr += len(top_f) / len(others); erc += 1 / len(others)
            if n:
                print(f"  {name:12s} {f:7s} n={n:3d}  most frequent {100*fr/n:5.1f}% (uniform {100*efr/n:5.1f}%)   "
                      f"most recent {100*rc/n:5.1f}% (uniform {100*erc/n:5.1f}%)")
                out[f"{reg}|{f}|bias"] = {"n": n, "freq": fr / n, "freq_uniform": efr / n, "recent": rc / n, "recent_uniform": erc / n}
    json.dump(out, open(f"{T}/cand_summary2.json", "w"), indent=1)


if __name__ == "__main__":
    main()

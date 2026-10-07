#!/usr/bin/env python3
"""Summaries of cand_<model>.json (cand_score.py).

choice : argmax over the passage's names of log P(" name" | prompt) is the target (referent selection)
greedy : the target is the greedy continuation (the harness acc on these items)
lost   : choice right but greedy wrong (the model ranks the right name first among names, yet would generate
         something else: format/calibration loss rather than a wrong referent)
recent / frequent : among wrong choices, the chosen name is the last-mentioned / most frequent other name
margin : mean log P(target) - max log P(other name), nats
"""
import json, re, sys
import numpy as np

T = "/home/hyunw3/.claude/jobs/003c5e28/tmp"
P0 = json.load(open(f"{T}/lambada_prompts.json"))["0"]
MODELS = [("Transformer", "llama_1.3b"), ("AUNet", "aunet_1.3b"), ("BPEByte", "bpebyte_1.3b"), ("BLT", "blt_1b"), ("H-Net", "hnet_1stage_XL")]


def last_pos(text, w):
    m = [x.start() for x in re.finditer(r"\b" + re.escape(w) + r"\b", text)]
    return max(m) if m else -1


def main():
    rows = {}
    print(f"{'model':12s} {'format':7s} {'choice':>7s} {'greedy':>7s} {'lost':>6s} | wrong: {'recent':>7s} {'frequent':>8s} | margin")
    for name, reg in MODELS:
        try:
            D = json.load(open(f"{T}/cand_{reg}.json"))
        except FileNotFoundError:
            continue
        for f in ("std", "cloze", "answer"):
            ch = gr = lost = 0; wrong = rec = freq = 0; margins = []
            for d in D:
                ll, tgt = d["ll"][f], d["tgt"]
                best = max(ll, key=ll.get)
                others = [c for c in d["cands"] if c != tgt]
                margins.append(ll[tgt] - max(ll[c] for c in others))
                c_ok, g_ok = best == tgt, d["greedy"][f][tgt]
                ch += c_ok; gr += g_ok; lost += c_ok and not g_ok
                if not c_ok:
                    passage = P0[d["i"]]["ctx"]
                    wrong += 1
                    rec += best == max(others, key=lambda c: last_pos(passage, c))
                    cnt = {c: len(re.findall(r"\b" + re.escape(c) + r"\b", passage)) for c in others}
                    freq += cnt[best] == max(cnt.values())
            n = len(D)
            r = {"n": n, "choice": ch / n, "greedy": gr / n, "lost": lost / n, "wrong": wrong,
                 "recent": rec / max(wrong, 1), "frequent": freq / max(wrong, 1), "margin": float(np.mean(margins))}
            rows[f"{reg}|{f}"] = r
            print(f"{name:12s} {f:7s} {100*r['choice']:6.1f}% {100*r['greedy']:6.1f}% {100*r['lost']:5.1f}% | "
                  f"{100*r['recent']:6.1f}% {100*r['frequent']:7.1f}%  (n={wrong:3d}) | {r['margin']:+.2f}")
    json.dump(rows, open(f"{T}/cand_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summaries of starts_<model>.json (patch_starts.py): standard vs GPT-3-format LAMBADA 5-shot prompts.

Columns: B/patch over the whole prompt+target; share of 1-byte patches; for the GPT-3 format, patches spanned
by the 9-byte cue " ____. ->" of the scored item and its longest patch; share of items whose target
(" word") starts a new patch, i.e. is not merged with the bytes before it.
  python patch_compare.py [--tmp DIR] [--tex OUT]
"""
import argparse, json
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_starts import texts  # noqa: E402

NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte", "blt": "BLT ($\\theta{=}1.34$)", "hnet": "H-Net"}
CUE = len(" ____. ->".encode())


def stats(starts, n_bytes, tgt):
    s = sorted(set(starts) | {0})
    edges = s + [n_bytes]
    lens = np.diff(edges)
    return lens, s, tgt in set(s)


def cue_patches(s, tgt):
    a, b = tgt - CUE, tgt                     # cue bytes [a, b)
    inside = [x for x in s if a < x < b]
    first = max([x for x in s if x <= a], default=0)
    cuts = [first] + inside + [min([x for x in s if x >= b], default=b)]
    return len(inside) + 1, int(max(np.diff(cuts)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="/home/hyunw3/.claude/jobs/003c5e28/tmp")
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    X = texts()
    rows = {}
    for m in NAME:
        R = json.load(open(f"{a.tmp}/starts_{m}.json"))
        r = {}
        for f in ("std", "gpt3"):
            L, one, tstart, cue_n, cue_max = [], 0, 0, [], []
            for k, st in enumerate(R[f]):
                tgt = R["tgt_byte"][f][k]
                assert R["ids"] == X["ids"] and tgt == X["tgt_byte"][f][k]
                lens, s, ts = stats(st, len(X[f][k].encode()), tgt)
                L += list(lens); one += int((lens == 1).sum()); tstart += ts
                if f == "gpt3":
                    c, mx = cue_patches(s, tgt); cue_n.append(c); cue_max.append(mx)
            r[f] = {"bpp": float(np.mean(L)), "one": one / len(L), "tgt_start": tstart / len(R[f]),
                    "cue_n": float(np.mean(cue_n)) if cue_n else None, "cue_max": float(np.mean(cue_max)) if cue_max else None}
        rows[m] = r
    print(f"{'model':12s} | std: B/patch 1-byte tgt-starts | gpt3: B/patch 1-byte tgt-starts  cue-patches cue-maxlen")
    for m, r in rows.items():
        s, g = r["std"], r["gpt3"]
        print(f"{m:12s} | {s['bpp']:5.2f} {s['one']:6.1%} {s['tgt_start']:6.1%} | {g['bpp']:5.2f} {g['one']:6.1%} {g['tgt_start']:6.1%}  {g['cue_n']:5.2f} {g['cue_max']:5.2f}")
    json.dump(rows, open(f"{a.tmp}/patch_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()

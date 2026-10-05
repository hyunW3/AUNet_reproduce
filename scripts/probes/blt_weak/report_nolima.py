#!/usr/bin/env python3
"""NoLiMa-lite summary for the four models: exact match per (cond, length), the paired NoLiMa effect
(literal - paraphrase on the same item), and paired-bootstrap CIs for BLT vs each model on paraphrase.

  python scripts/probes/blt_weak/report_nolima.py --root reports/blt_weak > reports/blt_weak/NOLIMA.md
"""
import argparse
import glob
import json
import math
import random
from collections import defaultdict

MODELS = [("subword_llama", "Llama"), ("aunet_static", "AU-Net"), ("byte_greedyroot", "BPEByte"),
          ("blt_1b", "BLT-1B")]


def ci(v):
    p = sum(v) / len(v)
    return f"{p:.3f} ±{1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(v)):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    a = ap.parse_args()
    S = defaultdict(dict)
    for f in glob.glob(f"{a.root}/results/**/nolima*.jsonl", recursive=True):
        for l in open(f):
            r = json.loads(l)
            if r.get("score") is not None:
                S[r["tag"]][r["id"] if r["pos"] is not None else "old:" + r["id"]] = r
    for name, pref, Ls in (("NoLiMa-lite, n=200/cell (new, ≤4096 B)", "nolima-", [1024, 2048, 3200]),):
        print(f"## {name}\n")
        print("| cond / length | " + " | ".join(m for _, m in MODELS) + " |")
        print("|---|" + "---:|" * len(MODELS))
        for c in ("literal", "paraphrase"):
            for L in Ls + ["all"]:
                cells = []
                for t, _ in MODELS:
                    v = [r["score"] for i, r in S[t].items() if r["cond"] == c and i.startswith(pref)
                         and (L == "all" or r["length"] == L) and r["pos"] is not None]
                    cells.append(ci(v) if v else "—")
                print(f"| {c} {L} | " + " | ".join(cells) + " |")
        print("\nPaired NoLiMa effect (literal − paraphrase, same item):\n")
        print("| length | " + " | ".join(m for _, m in MODELS) + " |")
        print("|---|" + "---:|" * len(MODELS))
        for L in Ls + ["all"]:
            cells = []
            for t, _ in MODELS:
                d = [S[t][f"nolima-literal-{r['length']}-{r['pos']}"]["score"] - r["score"]
                     for i, r in S[t].items() if r["cond"] == "paraphrase" and r["pos"] is not None
                     and (L == "all" or r["length"] == L) and f"nolima-literal-{r['length']}-{r['pos']}" in S[t]]
                cells.append(f"{sum(d) / len(d):+.3f}" if d else "—")
            print(f"| {L} | " + " | ".join(cells) + " |")
        print("\nPaired bootstrap, paraphrase accuracy BLT − model (all lengths, 2000 iters):\n")
        print("| vs | diff | 95% CI |\n|---|---:|---:|")
        rng = random.Random(0)
        for t, m in MODELS[:-1]:
            ids = [i for i, r in S["blt_1b"].items() if r["cond"] == "paraphrase" and r["pos"] is not None and i in S[t]]
            f = lambda xs: sum(S["blt_1b"][i]["score"] - S[t][i]["score"] for i in xs) / len(xs)
            b = sorted(f([rng.choice(ids) for _ in ids]) for _ in range(2000))
            print(f"| {m} | {f(ids):+.3f} | [{b[50]:+.3f}, {b[1949]:+.3f}] |")
    # old 50/cell set (the prompts in reports/niah/probe_pairs.jsonl), 4096 cell dropped for BLT
    print("\n## Original probe_pairs NoLiMa-lite (n=50/cell; BLT skips the 4096 cell: > 4096 B)\n")
    print("| cond / length | " + " | ".join(m for _, m in MODELS) + " |")
    print("|---|" + "---:|" * len(MODELS))
    for c in ("literal", "paraphrase"):
        for L in (1024, 2048, 4096):
            cells = []
            for t, _ in MODELS:
                v = [r["score"] for i, r in S[t].items() if r["cond"] == c and r["length"] == L and r["pos"] is None]
                cells.append(ci(v) if v else "—")
            print(f"| {c} {L} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

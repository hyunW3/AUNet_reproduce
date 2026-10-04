#!/usr/bin/env python3
"""Text-or-letter scoring on the position-balanced listing probe (same prompts, scored both ways).

  OR      correct if the text-scored prediction OR the letter-scored prediction is right
          (two guesses -> chance 1-(1-1/k)^2: 0.4375 for 4 options, 0.75 for PIQA)
  logsum  per option log(P(text) + P(letter)): the probability of naming that option in either form;
          one prediction -> chance 1/k
Reported for 4-option items (ARC-E / ARC-C / HellaSwag) by gold position, and for all 2,000 items.

  python scripts/probes/blt_weak/report_pos4_combined.py --root reports/blt_weak >> reports/blt_weak/POSITION_BALANCED.md
"""
import argparse
import glob
import json
import math
from collections import defaultdict

MODELS = [("subword_llama", "Llama"), ("aunet_static", "AU-Net"), ("byte_greedyroot", "BPEByte"),
          ("hnet_1stage_XL", "H-Net"), ("blt_1b", "BLT-1B")]
L = "ABCD"


def lse(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    a = ap.parse_args()
    R = defaultdict(dict)
    for f in glob.glob(f"{a.root}/results/snu55/pos4_*.jsonl"):
        if "pos4_ref" in f:
            continue
        for l in open(f):
            r = json.loads(l)
            R[r["tag"]][(r["cond"], r["length"], r["pos"])] = r
    meta = {}
    for l in open(f"{a.root}/data/pos4.jsonl"):
        r = json.loads(l)
        meta[(r["length"], r["pos"])] = (r["gold_pos"], r["n_opt"])
    out = ["\n## Text OR letter (same prompt scored both ways)\n",
           "`OR` = right if either the text-scored or the letter-scored prediction is right (two guesses: chance 0.4375",
           "for 4 options, 0.75 for PIQA). `logsum` = per option log(P(text)+P(letter)), one prediction (chance 0.25 / 0.5).",
           "Single-format rows repeated for reference.\n"]
    for fmt, title in (("before", "options BEFORE the question"), ("after", "standard MCQ, options AFTER the question")):
        res = {}
        for t, _ in MODELS:
            rows = []
            for (c, m, i), rt in R[t].items():
                if c != f"{fmt}_text" or (f"{fmt}_letter", m, i) not in R[t]:
                    continue
                rl = R[t][(f"{fmt}_letter", m, i)]
                g, k = meta[(m, i)]
                comb = [lse(x, y) for x, y in zip(rt["lls"], rl["lls"])]
                rows.append({"m": m, "g": g, "k": k, "text": rt["score"], "letter": rl["score"],
                             "or": float(rt["score"] or rl["score"]),
                             "logsum": float(max(range(k), key=comb.__getitem__) == g)})
            res[t] = rows
        out.append(f"### {title}\n")
        out.append("| metric | subset | chance | " + " | ".join(n for _, n in MODELS) + " |")
        out.append("|---|---|---:|" + "---:|" * len(MODELS))
        for met in ("text", "letter", "or", "logsum"):
            for sub, pred, ch in (("all 2,000", lambda r: True, "≈0.31" if met != "or" else "≈0.49"),
                                  ("4-option", lambda r: r["k"] == 4, "0.25" if met != "or" else "0.4375")):
                cells = []
                for t, _ in MODELS:
                    v = [r[met] for r in res[t] if pred(r)]
                    cells.append(f"{sum(v) / len(v):.3f}" if v else "—")
                out.append(f"| {met} | {sub} | {ch} | " + " | ".join(cells) + " |")
        out.append("\nBy gold position (4-option items):\n")
        out.append("| metric | gold | " + " | ".join(n for _, n in MODELS) + " |")
        out.append("|---|---|" + "---:|" * len(MODELS))
        for met in ("or", "logsum"):
            for p in range(4):
                cells = []
                for t, _ in MODELS:
                    v = [r[met] for r in res[t] if r["k"] == 4 and r["g"] == p]
                    cells.append(f"{sum(v) / len(v):.3f}" if v else "—")
                out.append(f"| {met} | {L[p]} | " + " | ".join(cells) + " |")
            spread = []
            for t, _ in MODELS:
                acc = [sum(r[met] for r in res[t] if r["k"] == 4 and r["g"] == p) /
                       max(1, sum(1 for r in res[t] if r["k"] == 4 and r["g"] == p)) for p in range(4)]
                spread.append(f"{max(acc) - min(acc):.3f}")
            out.append(f"| {met} | max − min | " + " | ".join(spread) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()

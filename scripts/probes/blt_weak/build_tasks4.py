#!/usr/bin/env python3
"""Position-balanced listing probe (follow-up to POSITION.md).

Each item's options are permuted so the gold lands at position (idx mod k): A/B/C/D equally often per
task. Two listing formats, each scored three ways (separately -- never OR-ed together):
  before  "Options: (A) x (B) y ..\\nQuestion: q\\nAnswer:"        (= echo_all, balanced)
  after   "Question: q\\nA. x\\nB. y\\n..\\nAnswer:"                 (standard letter-MCQ layout)
  scoring text   continuation " x"        (the option text, as in every earlier probe)
          letter continuation " A"        (the letter label)
          full   continuation " (A) x" / " A. x"  (label + text)
HellaSwag keeps its own context (no "Question:"/"Answer:" cue) in `before`; `after` appends the list
and "Answer:" to it. Rows: task "pos4", cond "<format>_<scoring>", length = MC task, pos = idx,
plus gold_pos. Clean text baseline = echo/clean (the option order does not enter a clean prompt).

  python scripts/probes/blt_weak/build_tasks4.py --mc reports/blt_weak/data/mc_items.jsonl --out reports/blt_weak/data/pos4.jsonl
"""
import argparse
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tasks2 import split_q, rank_row  # noqa: E402

L = "ABCDEFGH"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for line in open(a.mc):
        it = json.loads(line)
        ch, g, k = it["choices"], it["gold"], len(it["choices"])
        rng = random.Random(zlib.crc32(repr(("pos4", it["task"], it["idx"])).encode()))
        gp = it["idx"] % k                                   # balanced gold position
        others = [c for i, c in enumerate(ch) if i != g]
        rng.shuffle(others)
        perm = others[:gp] + [ch[g]] + others[gp:]           # permuted option texts, gold at gp
        ctx = it["context"]
        q, suf = split_q(it)
        before = "Options: " + " ".join(f"({L[i]}) {c}" for i, c in enumerate(perm)) + "\n" + ctx
        lst = "\n".join(f"{L[i]}. {c}" for i, c in enumerate(perm))
        after = (f"{q}\n{lst}\nAnswer:" if suf == "\nAnswer:" else f"{ctx}\n{lst}\nAnswer:")
        for fmt, prompt, lab in (("before", before, lambda i, c: f" ({L[i]}) {c}"),
                                 ("after", after, lambda i, c: f" {L[i]}. {c}")):
            for sc, opts in (("text", [" " + c for c in perm]),
                             ("letter", [" " + L[i] for i in range(k)]),
                             ("full", [lab(i, c) for i, c in enumerate(perm)])):
                r = rank_row(f"pos4-{fmt}_{sc}-{it['task']}-{it['idx']}", "pos4", f"{fmt}_{sc}", it["task"],
                             it["idx"], prompt, opts, gp)
                r["gold_pos"], r["n_opt"] = gp, k
                rows.append(r)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows;", dict(Counter(r["cond"] for r in rows)))
    print("gold position balance (4-option items, before_text):",
          dict(Counter(r["gold_pos"] for r in rows if r["cond"] == "before_text" and r["n_opt"] == 4)))


if __name__ == "__main__":
    main()

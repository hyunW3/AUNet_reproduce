#!/usr/bin/env python3
"""List-maximum probe (a la Wallace et al. 2019, "Do NLP Models Know
Numbers?") on frozen 1.3B LMs, as a value cloze -- the easy dial of
NumericBench's comparison ability.

Each item: a list of K distinct 4-digit integers (same digit length, so the
task is pure digit-level magnitude comparison with zero option-length bias),
rendered exactly like our NumericBench context ("List (index:value): 0:v, ...").
Question: which is the largest number. We score 8 candidate values (the max +
7 other list values) as continuations of "The answer is". K in {8, 20, 100}
dials list length from Wallace-easy up to NumericBench-hard while holding
chance fixed at 1/8.

Raw option loglikelihood is confounded by each value's *prior* (gold = max
correlates with magnitude/leading digit, and LM priors over numerals are far
from uniform -- raw argmax lands below chance). We therefore also report PMI
scoring (Holtzman et al. 2021, surface-form competition): argmax of
ll(option | list + question) - ll(option | "The answer is" alone).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L


def gen_items(k, n, seed=0):
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        vals = rng.sample(range(1000, 10000), k)
        gold = max(vals)
        opts = [gold] + rng.sample([v for v in vals if v != gold], 7)
        rng.shuffle(opts)
        items.append((vals, opts, gold))
    return items


def run_model(tag, ks, n, batch, out):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 4096)
    rows = []
    for k in ks:
        items = gen_items(k, n)
        pairs, prior_pairs = [], []
        for vals, opts, _gold in items:
            lst = ", ".join(f"{i}:{v}" for i, v in enumerate(vals))
            ctx = (f"List (index:value): {lst}\n\n"
                   f"Which is the largest number in the list?\nThe answer is")
            for o in opts:
                pairs.append(fmt_pair(ctx, str(o), fam))
                prior_pairs.append(fmt_pair("The answer is", str(o), fam))
        res = score(gen, tok, pairs, batch)
        pri = score(gen, tok, prior_pairs, batch)
        raw_ok = pmi_ok = 0
        for j, (_vals, opts, gold) in enumerate(items):
            lls = [res[j * 8 + i][3] for i in range(8)]
            pmi = [res[j * 8 + i][3] - pri[j * 8 + i][3] for i in range(8)]
            raw_ok += int(opts[max(range(8), key=lambda i: lls[i])] == gold)
            pmi_ok += int(opts[max(range(8), key=lambda i: pmi[i])] == gold)
        acc, acc_pmi = raw_ok / max(1, len(items)), pmi_ok / max(1, len(items))
        rows.append({"tag": tag, "task": f"listmax_k{k}", "n": len(items),
                     "acc": acc, "acc_pmi": acc_pmi, "chance": 0.125})
        print(f"  {tag:16s} listmax_k{k:<8d} acc={acc:.3f} acc_pmi={acc_pmi:.3f} (chance 0.125) n={len(items)}",
              flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 20, 100])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/listmax_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"List-max cloze | ks={args.ks} n={args.n}")
    for tag in args.models:
        run_model(tag, args.ks, args.n, args.batch_size, args.out)


if __name__ == "__main__":
    main()

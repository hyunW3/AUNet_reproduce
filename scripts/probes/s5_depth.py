#!/usr/bin/env python3
"""S5 permutation composition at CONTROLLED depth.

Unlike s5.py (which buckets reads by however many swaps happened to accumulate),
here every scored read depends on EXACTLY D composed transpositions:

    start A B C D E .  swap ..  (x D)  get k =

so accuracy at D=1,2,3 is measured with a full, equal sample per depth. Greedy
exact-match of the queried symbol; chance = 1/5 = 0.20. Uses the same 2-shot
convention as s5.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

SYMS = "ABCDE"


def gen_depth(D: int, rng: random.Random):
    """start + exactly D random transpositions + a single 'get k =' read."""
    arr = list(SYMS)
    text = "start " + " ".join(SYMS) + " ."
    for _ in range(D):
        i, j = rng.sample(range(5), 2)
        arr[i], arr[j] = arr[j], arr[i]
        text += f" swap {i} {j} ."
    k = rng.randint(0, 4)
    ans = arr[k]
    return text + f" get {k} =", ans


def fewshot(shots: int, depths, seed: int) -> str:
    if shots <= 0:
        return ""
    rng = random.Random(seed + 555)
    demos = []
    for _ in range(shots):
        ctx, ans = gen_depth(rng.choice(depths), rng)
        demos.append(ctx + f" {ans} .")
    return "\n".join(demos) + "\n"


def run_model(tag, n, depths, shots, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rng = random.Random(seed)
    prefix = fewshot(shots, depths, seed)
    rows = []
    for D in depths:
        # score ALL 5 candidate symbols per read so we can report BOTH metrics:
        #   exact  = greedy argmax over the full vocab == gold  (the s5.py metric)
        #   rank5  = argmax_{A..E} log p(symbol) == gold        (5-way choice, chance 0.20)
        layout = []
        pairs = []
        for _ in range(n):
            ctx, gold = gen_depth(D, rng)
            base = len(pairs)
            for c in SYMS:
                pairs.append(fmt_pair(prefix + ctx, c, fam))
            layout.append((base, gold))
        res = score(gen, tok, pairs, batch)
        exact = rank5 = 0
        for base, gold in layout:
            gi = SYMS.index(gold)
            exact += int(res[base + gi][0])                       # gold's greedy flag
            lls = [res[base + ci][3] for ci in range(5)]          # ll_sum per symbol
            rank5 += int(SYMS[max(range(5), key=lambda ci: lls[ci])] == gold)
        rows.append({"tag": tag, "depth": D, "n": n, "shots": shots,
                     "exact_acc": exact / n, "rank5_acc": rank5 / n})
        print(f"  {tag:16s} depth={D}  exact={exact/n:.3f}  rank5={rank5/n:.3f}  n={n}",
              flush=True)
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n", type=int, default=300, help="reads per depth")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/s5_depth_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"S5-depth | depths={args.depths} shots={args.shots} n={args.n}/depth "
          f"(chance=0.20)")
    for tag in args.models:
        for r in run_model(tag, args.n, args.depths, args.shots, args.seed,
                           args.batch_size):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired item bootstrap for the difference of perturbation effects between two models:
  DiD = [acc_A(variant) - acc_A(clean)] - [acc_B(variant) - acc_B(clean)]
over the echo / insert MC items (resampling items, the same draw for both models).

  python scripts/probes/blt_weak/delta_ci.py --root reports/blt_weak --a blt_1b --b byte_greedyroot
"""
import argparse
import glob
import json
import random
from collections import defaultdict

MC = ["arc_easy", "arc_challenge", "piqa", "hellaswag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/blt_weak")
    ap.add_argument("--a", default="blt_1b")
    ap.add_argument("--b", nargs="+", default=["byte_greedyroot", "subword_llama", "aunet_static"])
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    sc = defaultdict(dict)
    for f in glob.glob(f"{a.root}/results/**/*.jsonl", recursive=True):
        for line in open(f):
            r = json.loads(line)
            if r.get("task") in ("echo", "insert") and r.get("score") is not None:
                sc[r["tag"]][(r["cond"], r["length"], r["pos"])] = r["score"]
    rng = random.Random(0)
    variants = ["echo_all", "echo_gold", "echo_wrong", "repeat_q", "zwsp", "shy", "nbsp", "emoji", "homoglyph"]
    print("| variant | vs | DiD (A−B) | 95% CI | per-task DiD (ARC-E / ARC-C / PIQA / HS) |")
    print("|---|---|---:|---:|---|")
    for v in variants:
        for b in a.b:
            items = [(m, i) for (c, m, i) in sc[a.a] if c == v
                     and all(k in sc[t] for t in (a.a, b) for k in ((v, m, i), ("clean", m, i)))]

            def did(its):
                da = sum(sc[a.a][(v, m, i)] - sc[a.a][("clean", m, i)] for m, i in its)
                db = sum(sc[b][(v, m, i)] - sc[b][("clean", m, i)] for m, i in its)
                return (da - db) / len(its)
            pt = did(items)
            boots = sorted(did([rng.choice(items) for _ in items]) for _ in range(a.iters))
            lo, hi = boots[int(0.025 * a.iters)], boots[int(0.975 * a.iters)]
            per = " / ".join(f"{did([x for x in items if x[0] == m]):+.3f}" for m in MC)
            sig = " **" if lo > 0 or hi < 0 else ""
            print(f"| {v} | {b} | {pt:+.3f}{sig} | [{lo:+.3f}, {hi:+.3f}] | {per} |")


if __name__ == "__main__":
    main()

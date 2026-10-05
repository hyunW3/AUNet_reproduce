#!/usr/bin/env python3
"""Row-paired comparison of BPEByte decoding paths on the same items.

Tags: byte_greedyroot (bt loop; boundaries committed 1 byte late, prompt's last byte always 0),
byte_greedyroot_btctl (same bt loop on another host/GPU: hardware / numerics noise floor),
byte_greedyroot_inc (cached AUNET_INC_PARSE decode + prefill last-byte fix: exact causal masks).
Per cell: mean score per path, mean paired difference with a bootstrap 95% CI, and the share of
rows whose generated answer window is byte-identical.
"""
import argparse
import collections
import glob
import json
import random


def load(res_dir, tag):
    out = {}
    for f in glob.glob(f"{res_dir}/*_{tag}*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["tag"] == tag and r.get("score") is not None:
                out.setdefault(r["id"], r)
    return out


def cell_of(r):
    pre = "4K-fit " if r["id"].startswith("f4k-") else ""
    if r["task"] == "needle_types":
        return f"{pre}needle {r['cond']}"
    if r["task"] == "kv_litm":
        return f"{pre}KV k{r['length']}"
    return f"{pre}{r['task']} {r['cond']}"


def boot(d, n=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/longctx/results")
    ap.add_argument("--a", default="byte_greedyroot")
    ap.add_argument("--b", default="byte_greedyroot_inc")
    ap.add_argument("--pooled", nargs="*", default=["needle", "4K-fit"],
                    help="also print pooled rows over cells starting with these prefixes")
    a_ = ap.parse_args()
    A, B = load(a_.results, a_.a), load(a_.results, a_.b)
    ids = sorted(set(A) & set(B))
    cells = collections.defaultdict(list)
    for i in ids:
        cells[cell_of(A[i])].append(i)
    for p in a_.pooled:
        cells[f"[pooled] {p}"] = [i for i in ids if cell_of(A[i]).startswith(p)]
    print(f"paired rows: {len(ids)}  ({a_.a} vs {a_.b})\n")
    print(f"| cell | n | {a_.a} | {a_.b} | diff (b−a) [95% CI] | identical window |")
    print("|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: (c.startswith("[pooled]"), c)):
        v = cells[c]
        if not v:
            continue
        sa = [A[i]["score"] for i in v]
        sb = [B[i]["score"] for i in v]
        d = [y - x for x, y in zip(sa, sb)]
        lo, hi = boot(d)
        same = sum(A[i].get("gen") == B[i].get("gen") for i in v) / len(v)
        print(f"| {c} | {len(v)} | {100 * sum(sa) / len(v):.1f} | {100 * sum(sb) / len(v):.1f} | "
              f"{100 * sum(d) / len(v):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {100 * same:.0f}% |")


if __name__ == "__main__":
    main()

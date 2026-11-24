#!/usr/bin/env python3
"""Compare cfqa_bench.py outputs: per condition and pooled over the format-instruction conditions,
correct / format / joint with paired item-bootstrap CIs of the differences against reference models.

  python3 cfqa_compare.py --ref llama_1.3b --ref aunet_1.3b --target bpebyte_1.3b \
    llama_1.3b=reports/cfqa/llama_1.3b/cfqa_n500.json aunet_1.3b=... bpebyte_1.3b=... [--out X.json]
"""
import argparse, json, random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        name, path = r.split("=", 1)
        runs[name] = {(x["task"], x["cond"], x["shots"], x["i"]): x for x in json.load(open(path))["rows"]}
    keys = sorted(set.intersection(*(set(v) for v in runs.values())))
    groups = defaultdict(list)
    for k in keys:
        groups[f"{k[0]}/{k[1]}/{k[2]}"].append(k)
        if k[1] != "plain":
            groups[f"POOLED-format/{k[2]}shot"].append(k)
        else:
            groups[f"POOLED-plain/{k[2]}shot"].append(k)
    rng = random.Random(0)
    lo, hi = int(0.025 * a.B), int(0.975 * a.B) - 1
    res = {}

    def mean(name, ks, m):
        return 100 * sum(runs[name][k][m] for k in ks) / len(ks)

    names = list(runs)
    print(f"{'group':30s} {'metric':7s} " + " ".join(f"{n:>13s}" for n in names) + "   "
          + "   ".join(f"{a.target}-{r}" for r in a.ref))
    for g, ks in groups.items():
        metrics = ["correct"] if "plain" in g else ["correct", "format", "joint"]
        # resample items within each (task, cond, shots) stratum
        strata = defaultdict(list)
        for k in ks:
            strata[k[:3]].append(k)
        boots = []
        for _ in range(a.B):
            bk = []
            for s in strata.values():
                bk += [s[rng.randrange(len(s))] for _ in s]
            boots.append(bk)
        res[g] = {}
        for m in metrics:
            row = {n: mean(n, ks, m) for n in names}
            diffs = {}
            for r in a.ref:
                d = sorted(mean(a.target, b, m) - mean(r, b, m) for b in boots)
                diffs[r] = (row[a.target] - row[r], d[lo], d[hi])
            res[g][m] = {"score": row, "diff": diffs}
            print(f"{g:30s} {m:7s} " + " ".join(f"{row[n]:13.1f}" for n in names) + "   "
                  + "   ".join(f"{v[0]:+5.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]" for v in diffs.values()))
    if a.out:
        json.dump({"n_items": len(keys), "target": a.target, "refs": a.ref, "groups": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

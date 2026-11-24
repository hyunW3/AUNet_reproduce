#!/usr/bin/env python3
"""Analyze fmt_robust.py (grid / promptsource / demos) and surface_bench.py outputs for the matched models.

Per (mode, task[, shots]) and model, over the formats (templates, demo configs): mean accuracy, spread
(max - min format), worst format, std, and item consistency (share of items whose correctness is the same
under every format). 95% CIs and the BPEByte - Transformer / BPEByte - AU-Net differences come from a paired
item bootstrap (items resampled jointly for all formats and models). Surface forms: form cost
correct(form) - correct(as-is), split by whether the Transformer's subword tokens of the gold word survive.

  python3 fmt_analyze.py --root reports/fmt_robust --surface reports/surface --n 500 --out reports/fmt_robust/summary.json
"""
import argparse, json, random, statistics
from collections import defaultdict

M = ["llama_1.3b", "aunet_1.3b", "bpebyte_1.3b"]
SHORT = {"llama_1.3b": "Transf", "aunet_1.3b": "AUNet", "bpebyte_1.3b": "BPEByte"}
TARGET, REFS = "bpebyte_1.3b", ["llama_1.3b", "aunet_1.3b"]


def stats(mat, items, gold=None):
    """mat[f][i] in {0,1}; -> dict over the formats, restricted to `items`. With gold (binary labels),
    accuracy is balanced (mean of the per-class recalls), which removes the majority-label bonus a
    model gets from a format-induced yes/no prior."""
    if gold is None:
        accs = [100 * sum(col[i] for i in items) / len(items) for col in mat]
    else:
        pos = [i for i in items if gold[i] == 1]; neg = [i for i in items if gold[i] == 0]
        accs = [50 * (sum(col[i] for i in pos) / max(1, len(pos)) + sum(col[i] for i in neg) / max(1, len(neg)))
                for col in mat]
    cons = 100 * sum(1 for i in items if len({col[i] for col in mat}) == 1) / len(items)
    return {"mean": statistics.mean(accs), "spread": max(accs) - min(accs), "worst": min(accs),
            "std": statistics.pstdev(accs), "consistency": cons}


def compare(mats, n, B, seed=0, gold=None):
    """mats[model] = list over formats of item vectors. -> point stats + CIs + paired diffs."""
    rng = random.Random(seed)
    idx = list(range(n))
    point = {m: stats(mats[m], idx, gold) for m in mats}
    boots = [[rng.randrange(n) for _ in idx] for _ in range(B)]
    bs = {m: [stats(mats[m], b, gold) for b in boots] for m in mats}
    lo, hi = int(0.025 * B), int(0.975 * B) - 1
    out = {}
    for m in mats:
        out[m] = {}
        for k, v in point[m].items():
            s = sorted(x[k] for x in bs[m])
            out[m][k] = {"v": v, "ci": [s[lo], s[hi]]}
    for r in REFS:
        for k in point[TARGET]:
            d = sorted(bs[TARGET][j][k] - bs[r][j][k] for j in range(B))
            out[TARGET].setdefault("diff", {}).setdefault(r, {})[k] = {
                "v": point[TARGET][k] - point[r][k], "ci": [d[lo], d[hi]]}
    return out


def show(title, res, nf):
    print(f"\n== {title}  ({nf} formats)")
    print(f"   {'':12s} " + " ".join(f"{SHORT[m]:>15s}" for m in M) + "   BPE-Transf            BPE-AUNet")
    for k in ("mean", "worst", "spread", "consistency"):
        row = " ".join(f"{res[m][k]['v']:15.1f}" for m in M)
        d = res[TARGET]["diff"]
        row += "   " + "   ".join(f"{d[r][k]['v']:+5.1f} [{d[r][k]['ci'][0]:+5.1f},{d[r][k]['ci'][1]:+5.1f}]" for r in REFS)
        print(f"   {k:12s} {row}")


def load_fmt(root, model, mode, n):
    out = {}
    for l in open(f"{root}/{model}/{mode}_n{n}.jsonl"):
        r = json.loads(l)
        if "rows" in r:
            out[r["chunk"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/fmt_robust")
    ap.add_argument("--surface", default="reports/surface")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--modes", nargs="+", default=["grid", "promptsource", "demos", "surface"])
    ap.add_argument("--balanced", action="store_true",
                    help="BoolQ: balanced accuracy (gold labels rebuilt with fmt_robust.load_data; lingua venv)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    summary = {}
    gold_bq = None
    if a.balanced:
        import fmt_robust
        gold_bq = [x["gold"] for x in fmt_robust.load_data(a.n, 1234, ["boolq"])["boolq"]["test"]]
    for mode in [m for m in a.modes if m != "surface"]:
        try:
            runs = {m: load_fmt(a.root, m, mode, a.n) for m in M}
        except FileNotFoundError as e:
            print(f"[skip {mode}] {e}"); continue
        common = set.intersection(*(set(v) for v in runs.values()))
        groups = defaultdict(list)
        for cid in sorted(common):
            p = cid.split("/")
            key = f"{mode}/{p[1]}" + (f"/{p[3]}" if mode == "grid" else "")
            if mode == "demos":
                groups[f"{key}/sets"].append(cid) if p[3] == "order012" else None
                groups[f"{key}/orders"].append(cid) if p[2] == "set0" else None
            else:
                groups[key].append(cid)
        for g, cids in groups.items():
            mats = {m: [[r["correct"] for r in runs[m][c]["rows"]] for c in cids] for m in M}
            gold = gold_bq if (gold_bq is not None and "/boolq" in g) else None
            res = compare(mats, len(mats[M[0]][0]), a.B, gold=gold)
            summary[g] = {"n_formats": len(cids), "formats": cids, "balanced": gold is not None, "res": res}
            show(g + (" [balanced acc]" if gold else ""), res, len(cids))
        if mode == "grid":  # canonical-format accuracy for reference
            for cid in sorted(common):
                if "/0-0-0/" in cid:
                    print(f"   canonical {cid}: " + " ".join(f"{SHORT[m]} {100 * runs[m][cid]['acc']:.1f}" for m in M))
    if "surface" in a.modes:
        try:
            S = {m: json.load(open(f"{a.surface}/{m}/surface_n{a.n}.json"))["rows"] for m in M}
        except FileNotFoundError as e:
            print(f"[skip surface] {e}"); S = None
        if S:
            idx = {m: {(r["task"], r["form"], r["i"]): r for r in S[m]} for m in M}
            kept = {k: r.get("core_kept") for k, r in idx["llama_1.3b"].items()}
            forms = sorted({k[1] for k in kept if k[1] != "asis"})
            rng = random.Random(0)
            print("\n== surface forms, 3-shot: form cost = correct(form) - correct(as-is); joint = format & correct")
            print(f"   {'task/form':18s} {'kept%':>6s} " + " ".join(f"{SHORT[m]+' cost':>13s}" for m in M)
                  + " " + " ".join(f"{SHORT[m]+' joint':>13s}" for m in M) + "   BPE-Transf cost")
            for task in ("boolq", "lambada"):
                for f in forms + ["POOLED-kept", "POOLED-changed", "POOLED-all"]:
                    if f.startswith("POOLED"):
                        units = [(task, ff, i) for ff in forms for i in range(a.n)
                                 if f == "POOLED-all" or (kept[(task, ff, i)] == 1) == (f == "POOLED-kept")]
                    else:
                        units = [(task, f, i) for i in range(a.n)]
                    if not units:
                        continue
                    cost = lambda m, us: 100 * sum(idx[m][u]["correct"] - idx[m][(u[0], "asis", u[2])]["correct"] for u in us) / len(us)
                    joint = lambda m, us: 100 * sum(idx[m][u]["joint"] for u in us) / len(us)
                    pt = {m: cost(m, units) for m in M}
                    bys = defaultdict(list)
                    for u in units:
                        bys[u[2]].append(u)
                    keys = list(bys)
                    ds = []
                    for _ in range(a.B):
                        us = [u for _ in keys for u in bys[keys[rng.randrange(len(keys))]]]
                        ds.append(cost(TARGET, us) - cost("llama_1.3b", us))
                    ds.sort()
                    kp = 100 * sum(kept[u] for u in units) / len(units)
                    print(f"   {task + '/' + f:18s} {kp:6.0f} " + " ".join(f"{pt[m]:13.1f}" for m in M) + " "
                          + " ".join(f"{joint(m, units):13.1f}" for m in M)
                          + f"   {pt[TARGET] - pt['llama_1.3b']:+5.1f} [{ds[int(.025 * a.B)]:+5.1f},{ds[int(.975 * a.B) - 1]:+5.1f}]")
                    summary[f"surface/{task}/{f}"] = {"kept_pct": kp, "cost": pt, "joint": {m: joint(m, units) for m in M},
                                                      "diff_bpe_transf": [pt[TARGET] - pt["llama_1.3b"], ds[int(.025 * a.B)], ds[int(.975 * a.B) - 1]]}
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

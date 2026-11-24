#!/usr/bin/env python3
"""Merge ifeval_bench.py outputs (shards allowed) per model and compare models on the same prompts:
score with 95% item-bootstrap CI, and paired-bootstrap CI of each difference vs a reference model.
Instruction-level metrics resample prompts (keeps the instructions of a prompt together).
Needs lm-eval only with --cut (run it with the lingua venv + runs/ifeval/extra_site, NLTK_DATA set).

  python3 ifeval_merge.py --ref llama_1.3b \
    llama_1.3b=reports/ifeval/llama_1.3b/ifeval_gen640_cut2700.json \
    bpebyte_1.3b=reports/ifeval/bpebyte_1.3b/ifeval_gen450_cut2700_shard0of2.json,..._shard1of2.json
"""
import argparse, json, random

KEYS = ["prompt_level_strict_acc", "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc"]


def load(paths, cut=None):
    """ifeval_bench.py outputs (.json) or resume logs (.jsonl). With cut, every response (the untruncated
    one when stored) is cut to `cut` UTF-8 bytes and rescored with lm-eval's IFEval checker."""
    items = {}
    for p in paths.split(","):
        rows = ([json.loads(l) for l in open(p) if l.strip()] if p.endswith(".jsonl")
                else json.load(open(p))["items"])
        for it in rows:
            items[it["doc"]["key"]] = it
    if cut:
        from lm_eval.tasks.ifeval.utils import process_results
        for it in items.values():
            full = it.get("response_full", it["response"])
            it["response"] = full.encode()[:cut].decode("utf-8", errors="ignore")
            it.update(process_results(it["doc"], [it["response"]]))
    return items


def stat(items, keys, k):
    if k.startswith("prompt"):
        return 100 * sum(int(items[q][k]) for q in keys) / len(keys)
    v = [int(b) for q in keys for b in items[q][k]]
    return 100 * sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path[,path...]")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--cut", type=int, default=None, help="rescore all responses cut to this many bytes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = {r.split("=", 1)[0]: load(r.split("=", 1)[1], a.cut) for r in a.runs}
    common = sorted(set.intersection(*(set(v) for v in runs.values())))
    rng = random.Random(0)
    boots = [[rng.choice(common) for _ in common] for _ in range(a.B)]
    lo, hi = int(0.025 * a.B), int(0.975 * a.B) - 1
    res = {"n_prompts": len(common), "ref": a.ref, "models": {}}
    print(f"IFEval, n={len(common)} prompts common to all runs; [95% CI]; diff vs {a.ref} (paired)")
    for name, it in runs.items():
        row = {}
        for k in KEYS:
            pt = stat(it, common, k)
            bs = sorted(stat(it, b, k) for b in boots)
            row[k] = {"score": pt, "ci": [bs[lo], bs[hi]]}
            if name != a.ref:
                d = sorted(stat(it, b, k) - stat(runs[a.ref], b, k) for b in boots)
                row[k]["diff"] = pt - stat(runs[a.ref], common, k)
                row[k]["diff_ci"] = [d[lo], d[hi]]
        row["mean_response_bytes"] = sum(len(it[q]["response"].encode()) for q in common) / len(common)
        res["models"][name] = row
        print(f"== {name}  (mean response {row['mean_response_bytes']:.0f} B)")
        for k in KEYS:
            r = row[k]
            s = f"  {k:24s} {r['score']:6.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]"
            if "diff" in r:
                s += f"   diff {r['diff']:+6.2f} [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}]"
            print(s)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

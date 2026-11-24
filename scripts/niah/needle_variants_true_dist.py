#!/usr/bin/env python3
"""Adds "tbins" to reports/niah/needle_variants/summary.json: exact match by the TRUE answer-to-needle distance
(answer start - needle value start, in bytes of the scored text; <256, 256-512, >=512, plus <512 / >=512_).
The "bins" of needle_variants_summary.py use (1-depth) x cell length, the distance to the end of the haystack; the
question and the value itself add 120-170 B, so ~60% of its nominal 256-512 bin lies beyond the 512-byte byte window
(reports/niah/force_byte/summary.md). Asserts that "bins" is reproduced, then writes tbins for every key in place.
  python scripts/niah/needle_variants_true_dist.py"""
import json, collections
L = "/mnt/ssd2/hyun2/AUNet"; D = f"{L}/reports/niah/needle_variants"
S = json.load(open(f"{D}/summary.json"))
items = {}
for f in ("items.jsonl", "items_v2.jsonl"):
    for q in map(json.loads, open(f"{D}/{f}")):
        items[q["key"]] = q
cnt = collections.Counter()
for q in map(json.loads, open(f"{L}/reports/niah/sniah123_n250_final_pairs.jsonl")):
    if q["probe"] in ("sniah2", "sniah3"):
        t = {"sniah2": "num7", "sniah3": "uuid"}[q["probe"]]; Ln = int(q["cell"].split("/")[1])
        i = cnt[q["cell"]]; cnt[q["cell"]] += 1
        items[f"{t}/{Ln}/{i}"] = {"task": t, "length": Ln, "depth": (i + 0.5) / 250, "prompt": q["prompt"], "value": q["values"][0]}
sc = {}
for m in ("aunet", "bpebyte", "llama"):
    for f in (f"out/{m}.jsonl", f"out/{m}_v2.jsonl"):
        for r in map(json.loads, open(f"{D}/{f}")):
            sc[(m, r["key"])] = r["exact"]
    for r in map(json.loads, open(f"{L}/reports/niah/final/{m}_items.jsonl")):
        if r["cell"].startswith(("sniah2/", "sniah3/")):
            t = {"sniah2": "num7", "sniah3": "uuid"}[r["cell"].split("/")[0]]
            sc[(m, f"{t}/{int(r['cell'].split('/')[1])}/{round(r['depth'] * 250 - 0.5)}")] = r["exact"]
B = lambda d: "<256" if d < 256 else "256-512" if d < 512 else ">=512"
for key in S:
    t, m = key.split("|"); nb, tb = collections.defaultdict(list), collections.defaultdict(list)
    for k, q in items.items():
        if q["task"] != t:
            continue
        e = sc[(m, k)]; p = q["prompt"].encode(); v = q["value"].encode()
        assert p.count(v) == 1, k
        td = len(p) + 1 - p.index(v)                   # the scored answer is " " + value right after the prompt
        nb[B((1 - q["depth"]) * q["length"])].append(e)
        tb[B(td)].append(e); tb["<512" if td < 512 else ">=512_"].append(e)
    for x in ("<256", "256-512", ">=512"):
        assert len(nb[x]) == S[key]["bins"][x][1] and abs(sum(nb[x]) / len(nb[x]) - S[key]["bins"][x][0]) < 1e-9, (key, x)
    S[key]["tbins"] = {x: [sum(v) / len(v), len(v)] for x, v in tb.items()}
    print(key, {x: round(100 * v[0], 1) for x, v in S[key]["tbins"].items()})
json.dump(S, open(f"{D}/summary.json", "w"), indent=1)

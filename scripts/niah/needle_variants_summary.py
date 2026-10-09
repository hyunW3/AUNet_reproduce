#!/usr/bin/env python3
"""Summary of the needle-value variants (dump_needle_variants.py) + the S-NIAH-2 (7-digit) and S-NIAH-3 (UUID) rows of the
final S-NIAH set: exact match by distance to the query ((1-depth) x cell length; <256, 256-512, >=512 bytes) and by cell
length, and B/patch on the value (value bytes / parsing units covering it; every 5th item).
  PYTHONPATH=lingua lingua/.venv/bin/python scripts/niah/needle_variants_summary.py -> reports/niah/needle_variants/summary.json"""
import json, sys, collections, statistics as st
L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, f"{L}/scripts/probes"); sys.path.insert(0, f"{L}/scripts/probes/patch_stats")
from compression_stability import build_parser
MODELS = ("aunet", "bpebyte", "llama")
ORDER = ["num7", "uuid", "uuid_space", "hex32_sp4", "digits32", "letters32", "tok4", "rand4", "rand4x4", "rand4x5", "tok4x11"]
P = {m: build_parser(m) for m in MODELS}
items = {json.loads(l)["key"]: json.loads(l) for f in ("items.jsonl", "items_v2.jsonl")  # v2: tok4/rand4 + controls
         for l in open(f"{L}/reports/niah/needle_variants/{f}")}
cnt = collections.Counter()
for q in map(json.loads, open(f"{L}/reports/niah/sniah123_n250_final_pairs.jsonl")):
    if q["probe"] in ("sniah2", "sniah3"):
        t = {"sniah2": "num7", "sniah3": "uuid"}[q["probe"]]; Ln = int(q["cell"].split("/")[1])
        i = cnt[q["cell"]]; cnt[q["cell"]] += 1
        items[f"{t}/{Ln}/{i}"] = {"task": t, "length": Ln, "depth": (i + 0.5) / 250, "prompt": q["prompt"], "value": q["values"][0]}
sc = {}
for m in MODELS:
    for f in (f"{m}.jsonl", f"{m}_v2.jsonl"):
        for r in map(json.loads, open(f"{L}/reports/niah/needle_variants/out/{f}")):
            sc[(m, r["key"])] = r["exact"]
    for r in map(json.loads, open(f"{L}/reports/niah/final/{m}_items.jsonl")):
        if r["cell"].startswith(("sniah2/", "sniah3/")):
            t = {"sniah2": "num7", "sniah3": "uuid"}[r["cell"].split("/")[0]]
            sc[(m, f"{t}/{int(r['cell'].split('/')[1])}/{round(r['depth'] * 250 - 0.5)}")] = r["exact"]
out = {}
for t in ORDER:
    for m in MODELS:
        bins, lens, bpp = collections.defaultdict(list), collections.defaultdict(list), []
        for k, q in items.items():
            if q["task"] != t:
                continue
            e = sc[(m, k)]; d = (1 - q["depth"]) * q["length"]
            bins["<256" if d < 256 else "256-512" if d < 512 else ">=512"].append(e); bins["all"].append(e)
            lens[str(q["length"])].append(e)
            if int(k.split("/")[-1]) % 5 == 0:
                text = q["prompt"] + " " + q["value"]; b = text.encode(); vs = len(b) - len(q["value"].encode())
                bpp.append(len(q["value"].encode()) / (1 + sum(1 for s in P[m](text) if vs < s < len(b))))
        out[f"{t}|{m}"] = {"bins": {k: [sum(v) / len(v), len(v)] for k, v in bins.items()},
                           "len": {k: sum(v) / len(v) for k, v in sorted(lens.items(), key=lambda x: int(x[0]))},
                           "bpp": st.mean(bpp)}
        print(t, m, {k: round(v[0], 3) for k, v in out[f"{t}|{m}"]["bins"].items()}, {k: round(v, 3) for k, v in out[f"{t}|{m}"]["len"].items()}, round(out[f"{t}|{m}"]["bpp"], 2))
json.dump(out, open(f"{L}/reports/niah/needle_variants/summary.json", "w"), indent=1)

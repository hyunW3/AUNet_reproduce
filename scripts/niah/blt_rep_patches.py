#!/usr/bin/env python3
"""BLT patch structure (released threshold, 512-byte entropy window) on original vs repeated-haystack S-NIAH
prompts, 2k and 4k cells, every 5th item (50 per cell). -> reports/niah/rep64/blt_patches.json

  source scripts/probes/patch_stats/env.sh; cd lingua; CUDA_VISIBLE_DEVICES=2 .venv/bin/python ../scripts/niah/blt_rep_patches.py
"""
import json, statistics as st, sys, torch
from collections import defaultdict
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes/patch_stats")
from measure_blt import load_patcher, entropies, byte_starts, BOS, OFFSET
from patch_common import lengths_from_starts
patcher, thr = load_patcher(torch.float32)
S = defaultdict(lambda: defaultdict(list))
items = [json.loads(l) for l in open("/mnt/ssd2/hyun2/AUNet/reports/niah/rep64/sniah_rep64_items.jsonl")]
orig = [json.loads(l) for l in open("/mnt/ssd2/hyun2/AUNet/reports/niah/sniah123_n250_final_pairs.jsonl")]
def meas(cell, text):
    b = text.encode()
    e = entropies(patcher, [[BOS] + [x + OFFSET for x in b]], window=512)[0]
    lens = lengths_from_starts(byte_starts(e, len(b), thr), len(b))
    S[cell]["n"].append(len(lens)); S[cell]["max"].append(max(lens)); S[cell]["long"].append(sum(x for x in lens if x > 64) / len(b))
for q in items:
    if q["length"] in (2048, 4096) and int(q["key"].split("/")[-1]) % 5 == 0:
        meas(q["cell"], q["prompt"] + " " + q["value"])
for i, q in enumerate(orig):
    if q["cell"].endswith(("/2048", "/4096")) and (i % 250) % 5 == 0:
        meas(q["cell"], q["prompt"] + " " + q["values"][0])
out = {c: {"patches": st.mean(v["n"]), "max_med": st.median(v["max"]), "max": max(v["max"]), "long_pct": 100 * st.mean(v["long"]), "n": len(v["n"])} for c, v in sorted(S.items())}
json.dump(out, open("/mnt/ssd2/hyun2/AUNet/reports/niah/rep64/blt_patches.json", "w"), indent=1)
for c, v in out.items(): print(c, {k: round(x, 1) for k, x in v.items()})

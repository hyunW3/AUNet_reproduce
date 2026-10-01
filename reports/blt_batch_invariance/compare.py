#!/usr/bin/env python3
"""Summarise out/<variant>_bs<k>.json: task metric per batch size, item flips vs that variant's bs=1."""
import glob, json, os, re, statistics as st
D = os.path.dirname(os.path.abspath(__file__)) + "/out"
METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm",
          "winogrande": "acc", "boolq": "acc"}
runs = {}
for f in glob.glob(f"{D}/*_bs*.json"):
    v, b = re.match(r"(.+)_bs(\d+)\.json", os.path.basename(f)).groups()
    runs[(v, int(b))] = json.load(open(f))
ref = runs.get(("fixed", 1))
for v in ("fixed", "fixed_entfp32", "unfixed"):
    bss = sorted(b for (vv, b) in runs if vv == v)
    if not bss: continue
    print(f"\n### {v}")
    print("| task | metric | " + " | ".join(f"bs{b}" for b in bss) + " | flips vs own bs1 (" + ", ".join(f"bs{b}" for b in bss) + ") |")
    print("|---|---|" + "---|" * len(bss) + "---|")
    for t, m in METRIC.items():
        cells, flips = [], []
        for b in bss:
            r = runs[(v, b)]
            if "error" in r: cells.append("ERR"); flips.append("-"); continue
            s = r["samples"][t][m]; cells.append(f"{100*st.mean(s):.1f}")
            r0 = runs.get((v, 1))
            if r0 and "samples" in r0:
                s0 = r0["samples"][t][m]; flips.append(str(sum(x != y for x, y in zip(s, s0))))
        print(f"| {t} | {m} | " + " | ".join(cells) + " | " + ", ".join(flips) + " |")
    avg = []
    for b in bss:
        r = runs[(v, b)]
        avg.append("ERR" if "error" in r else f"{100*st.mean(st.mean(r['samples'][t][m]) for t, m in METRIC.items()):.2f}")
    print("| **avg** | | " + " | ".join(avg) + " | |")
    if ref:
        for b in bss:
            r = runs[(v, b)]
            if "error" in r: print(f"  bs{b}: {r['error'][:200]}"); continue
            r0 = runs.get((v, 1), ref)
            d = [abs(x - y) for x, y in zip(r["req_logprobs"], r0["req_logprobs"])]
            print(f"  bs{b} vs own bs1: n_req={len(d)} mean|dlogp|={st.mean(d):.4f} max={max(d):.3f} "
                  f"frac>0.1={sum(x > .1 for x in d)/len(d):.3f} seconds={r['seconds']}")

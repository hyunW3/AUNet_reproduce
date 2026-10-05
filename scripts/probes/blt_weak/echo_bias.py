"""echo_all positional bias: share of predictions on the first / last / longest listed option (non-HellaSwag).

  python scripts/probes/blt_weak/echo_bias.py reports/blt_weak
"""
import glob, json, statistics as st
from collections import defaultdict
import sys
R = sys.argv[1] if len(sys.argv) > 1 else "reports/blt_weak"
items = {(json.loads(l)["task"], json.loads(l)["idx"]): json.loads(l) for l in open(f"{R}/data/mc_items.jsonl")}
rows = defaultdict(dict)
for f in glob.glob(f"{R}/results/**/echo_*.jsonl", recursive=True):
    for l in open(f):
        r = json.loads(l)
        rows[r["tag"]][(r["cond"], r["length"], r["pos"])] = r
print("tag cond | pred=first% pred=last% pred=longest% gold=longest% | avg lls margin")
for tag in ["subword_llama", "aunet_static", "byte_greedyroot", "blt_1b"]:
    for cond in ["clean", "echo_all"]:
        first = last = longest = gl = n = 0
        for (c, m, i), r in rows[tag].items():
            if c != cond or m == "hellaswag":
                continue
            ch = items[(m, i)]["choices"]
            L = [len(x) for x in ch]
            n += 1
            first += r["pred"] == 0
            last += r["pred"] == len(ch) - 1
            longest += L[r["pred"]] == max(L)
            gl += L[r["gold"]] == max(L)
        print(f"{tag:16s} {cond:9s} | {100*first/n:5.1f} {100*last/n:5.1f} {100*longest/n:5.1f} {100*gl/n:5.1f}  (n={n}, non-HS)")

#!/usr/bin/env python3
"""Main-table Leet column: |dAcc| of NL-Augmenter leet_letters on the five-task robustness suite.

Same convention as the Noise / Typo columns (scripts/probes/ci_main_table.py): acc_norm on HellaSwag /
ARC-E / ARC-C / PIQA and acc on BoolQ, delta = perturbed - clean per task, five-task mean, absolute
value; paired item bootstrap (B=2000, seed 0) with one set of resampled indices shared by all models
and by the clean / perturbed forms; CI = 2.5/97.5 percentiles, reported as the half-width.

  python leet_main_col.py --raw reports/format_robustness/raw --extra reports/format_robustness/leet_extra
"""
import argparse, glob, json, os

import numpy as np

R5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
ROWS = [("llama", "Transformer"), ("aunet", "AUNet"), ("bpebyte", "BPEByte"),
        ("blt", "BLT (θ=1.34)"), ("blt1609", "BLT (θ=1.61)"), ("hnet", "H-Net")]
SOURCES = {"llama": ["{extra}/llama/results.json"], "aunet": ["{raw}/aunet/results.json"],
           "bpebyte": ["{raw}/bpebyte/results.json"], "blt": ["{raw}/blt_*_B.json"],
           "blt1609": ["{extra}/blt1609_*.json"], "hnet": ["{raw}/hnet.json"]}


def load(model, raw, extra):
    rows = {}
    for pat in SOURCES[model]:
        for f in glob.glob(pat.format(raw=raw, extra=extra)):
            rows.update(json.load(open(f))["results"])
    out = {}
    for t in R5:
        c, p = rows.get(f"fmt_{t}_clean"), rows.get(f"fmt_{t}_nla_leet")
        if c and p:
            m = "acc" if t == "boolq" else "acc_norm"
            out[t] = (np.array(c["bits"][m], float), np.array(p["bits"][m], float))
    return out if len(out) == len(R5) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = {m: d for m, _ in ROWS if (d := load(m, a.raw, a.extra)) is not None}
    n = {t: min(len(D[m][t][0]) for m in D) for t in R5}
    stat = lambda idx: {m: abs(100 * np.mean([D[m][t][1][idx[t]].mean() - D[m][t][0][idx[t]].mean() for t in R5]))
                        for m in D}
    point = stat({t: np.arange(n[t]) for t in R5})
    rng = np.random.default_rng(a.seed)
    boots = {m: np.empty(a.B) for m in D}
    for b in range(a.B):
        idx = {t: rng.integers(0, n[t], n[t]) for t in R5}
        for m, v in stat(idx).items():
            boots[m][b] = v
    res = {}
    for m, name in ROWS:
        if m not in D:
            print(f"{name:14s} (not measured yet)")
            continue
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        signed = 100 * np.mean([D[m][t][1].mean() - D[m][t][0].mean() for t in R5])
        res[m] = {"value": round(float(point[m]), 3), "signed": round(float(signed), 3),
                  "ci95": [round(float(lo), 3), round(float(hi), 3)], "half_width": round(float(hi - lo) / 2, 3),
                  "per_task": {t: round(100 * float(D[m][t][1].mean() - D[m][t][0].mean()), 2) for t in R5}}
        print(f"{name:14s} |Δ| {point[m]:5.2f} ± {res[m]['half_width']:.2f}   (signed {signed:+.2f})  "
              + " ".join(f"{t[:5]}={res[m]['per_task'][t]:+.1f}" for t in R5))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.raw)), "leet_main.json")
    json.dump({"B": a.B, "seed": a.seed, "n_items": n, "results": res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main-table robustness |dAcc| (Noise / Typo / Despace / Leet) under a chosen per-task metric map.

Same machinery as scripts/probes/ci_main_table.py (per-item reruns, items aligned across models, one set of
resampled indices per axis shared by every model and by clean/perturbed, |five-task mean delta|, B=2000),
but the metric per task is a parameter instead of the hardcoded "acc_norm, BoolQ acc" (Noise/Typo) and
"acc" (Despace). Six rows: Transformer, AUNet, BPEByte, BLT theta=1.34, BLT theta=1.61, H-Net.

  python robust_metric_table.py --acc boolq,piqa              # acc on BoolQ+PIQA, acc_norm elsewhere
  python robust_metric_table.py --acc piqa --tasks hellaswag,arc_easy,arc_challenge,piqa   # 4-task mean, no BoolQ
  python robust_metric_table.py --acc hellaswag,arc_easy,arc_challenge,piqa,boolq   # acc everywhere
"""
import argparse, glob, json, os, sys

import numpy as np

A = "/mnt/ssd2/hyun2/AUNet"
HERE = os.path.dirname(os.path.abspath(__file__))
FMT = os.path.join(HERE, "..", "..", "..", "reports", "format_robustness")
sys.path.insert(0, os.path.join(HERE, ".."))
import ci_main_table as C  # noqa: E402  (_ext_json / _vec helpers)

R5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
ROWS = [("llama", "Transformer"), ("aunet", "AUNet"), ("bpebyte", "BPEByte"),
        ("blt", "BLT 1.34"), ("blt1609", "BLT 1.61"), ("hnet", "H-Net")]
LEET = {"llama": ["leet_extra/llama/results.json"], "aunet": ["raw/aunet/results.json"],
        "bpebyte": ["raw/bpebyte/results.json"], "blt": ["raw/blt_*_B.json"],
        "blt1609": ["leet_extra/blt1609_*.json"], "hnet": ["raw/hnet.json"]}


def ext(model, name):
    C.BLT_PREFIX = "t1609_" if model == "blt1609" else ""
    return C._ext_json("blt" if model.startswith("blt") else model, name)


def load(model, metric):
    """{axis: {task: (clean {doc: v}, [variant {doc: v}, ...])}}"""
    out = {"noise": {}, "typo": {}, "despace": {}, "leet": {}}
    if model in ("llama", "aunet", "bpebyte"):
        S = {}
        for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
            S.update(json.load(open(f"{A}/runs/{d}/{model}/results.json"))["samples"])
        pert = {"noise": S, "typo": S}
        D = json.load(open(f"{A}/runs/robustness_despace_bits/{model}/results.json"))["results"]
    else:
        pert = {ax: (ext(model, ax) or {}).get("samples") for ax in ("noise", "typo")}
        D = (ext(model, "despace") or {}).get("results")
    for ax in ("noise", "typo"):
        S = pert[ax]
        for t in R5:
            vs = [k for k in S if k.startswith(f"{t}_{ax}_") and not k.endswith("_avg")]
            if t in S and vs:
                out[ax][t] = (C._vec(S[t], metric[t]), [C._vec(S[k], metric[t]) for k in sorted(vs)])
    for t in R5:
        c, p = D.get(f"despace_mc_{t}_clean"), D.get(f"despace_mc_{t}_despaceall")
        if c and p:
            out["despace"][t] = (dict(enumerate(c["bits"][metric[t]])), [dict(enumerate(p["bits"][metric[t]]))])
    rows = {}
    for pat in LEET[model]:
        for f in glob.glob(os.path.join(FMT, pat)):
            rows.update(json.load(open(f))["results"])
    for t in R5:
        c, p = rows.get(f"fmt_{t}_clean"), rows.get(f"fmt_{t}_nla_leet")
        if c and p:
            out["leet"][t] = (dict(enumerate(c["bits"][metric[t]])), [dict(enumerate(p["bits"][metric[t]]))])
    return out


def main():
    global R5
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="boolq,piqa", help="tasks scored with acc; the rest use acc_norm")
    ap.add_argument("--tasks", default=",".join(R5), help="tasks averaged (e.g. drop boolq for the 4-task mean)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R5 = tuple(a.tasks.split(","))
    acc = set(a.acc.split(",")) if a.acc else set()
    metric = {t: "acc" if t in acc else "acc_norm" for t in R5}
    R = {m: load(m, metric) for m, _ in ROWS}
    rng = np.random.default_rng(a.seed)
    res = {}
    for ax in ("noise", "typo", "despace", "leet"):
        pm = {m: R[m][ax] for m, _ in ROWS if all(t in R[m][ax] for t in R5)}
        Arr, n = {}, {}
        for t in R5:
            common = None
            for m in pm:
                c, vs = pm[m][t]
                ids = set(c).intersection(*[set(v) for v in vs])
                common = ids if common is None else common & ids
            ids = sorted(common)
            n[t] = len(ids)
            Arr[t] = {m: (np.array([pm[m][t][0][i] for i in ids], float),
                          np.array([[v[i] for i in ids] for v in pm[m][t][1]], float)) for m in pm}
        fn = lambda idx: {m: abs(100 * np.mean([Arr[t][m][1][:, idx[t]].mean() - Arr[t][m][0][idx[t]].mean()
                                                for t in R5])) for m in pm}
        point = fn({t: np.arange(n[t]) for t in R5})
        boots = {m: np.empty(a.B) for m in pm}
        for b in range(a.B):
            idx = {t: rng.integers(0, n[t], n[t]) for t in R5}
            for m, v in fn(idx).items():
                boots[m][b] = v
        res[ax] = {"n_items": n}
        for m in pm:
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            res[ax][m] = {"value": round(float(point[m]), 3), "half_width": round(float(hi - lo) / 2, 3)}
    print(f"metric: {metric}")
    print(f"{'row':12s} " + " ".join(f"{ax:>14s}" for ax in res))
    for m, name in ROWS:
        print(f"{name:12s} " + " ".join(f"{res[ax][m]['value']:7.2f} ± {res[ax][m]['half_width']:4.2f}" if m in res[ax]
                                        else f"{'--':>14s}" for ax in res))
    if a.out:
        json.dump({"metric": metric, "B": a.B, "seed": a.seed, "results": res}, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

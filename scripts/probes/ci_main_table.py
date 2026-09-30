#!/usr/bin/env python3
"""Paired item-bootstrap 95% CIs for every main-table accuracy column, all five models.

Columns: Downstream Avg5 (HellaSwag, ARC-E, ARC-C, PIQA, WinoGrande; acc_norm except WinoGrande
acc) at 0/3/5 shots on the full splits, and robustness |dAcc| for Noise / Typo / Despace on the
five-task suite (HellaSwag, ARC-E, ARC-C, PIQA, BoolQ; limit 2000) — the same convention as
robustness_bootstrap_ci.py.

Bootstrap: within each task, items are resampled with replacement; the same resampled indices
are used for every model and, for robustness, for the clean and perturbed forms. The statistic
(five-task mean) is recomputed per resample; the CI is the 2.5/97.5 percentile.

Sources (per-item):
  trio downstream  runs/downstream_ci/<arm>_<k>shot/results.json          (lm-eval samples)
  trio robustness  runs/robustness_paper1p3b{,_ext}/<arm>/results.json    (noise/typo samples)
                   runs/robustness_despace_bits/<arm>/results.json        (despace bits)
  BLT / H-Net      reports/ext_ci/<family>/{ds0,ds3,ds5,noise,typo,despace}.json  (scripts/probes/ext_ci/run_ext.py)

  python ci_main_table.py [--B 2000] -> reports/ci_main_table/ci.json, ci.md
"""
import argparse, json, os
from pathlib import Path

import numpy as np

L = Path("/mnt/ssd2/hyun2/AUNet")
MODELS = ("llama", "aunet", "bpebyte", "blt", "hnet")
NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte", "blt": "BLT", "hnet": "H-Net"}
DS5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande")
R5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc_norm",
          "winogrande": "acc", "boolq": "acc"}
EXT = L / "reports/ext_ci"
# BLT = official bytelatent (xformers: entropy + local windows native), batch size 1. The HF-port runs
# (reports/ext_ci/blt) are not used: the port drops the local sliding windows past 512 bytes.
EXT_DIR = {"blt": EXT / "blt_official", "hnet": EXT / "hnet"}


BLT_PREFIX = ""   # e.g. "t1609_" for the calibrated-threshold runs (--blt_prefix)


def _ext_json(model, name):
    """reports/ext_ci/<model>/<name>.json, or the per-task shards <name>_<task>.json merged."""
    d = EXT_DIR[model]
    if model == "blt":
        name = BLT_PREFIX + name
    p = d / f"{name}.json"
    if p.exists():
        return json.load(open(p))
    shards = sorted(d.glob(f"{name}_*.json"))
    if not shards:
        return None
    out = {"results": {}, "samples": {}}
    for q in shards:
        j = json.load(open(q))
        out["results"].update(j.get("results", {}))
        out["samples"].update(j.get("samples", {}))
    return out


def _vec(rows, metric):
    """lm-eval samples (list of dicts) or compact samples (dict of lists) -> {doc_id: value}."""
    if isinstance(rows, dict):
        m = metric if metric in rows else "acc"
        return dict(zip(rows["doc_id"], rows[m]))
    m = metric if metric in rows[0] else "acc"
    return {int(r["doc_id"]): float(r[m]) for r in rows}


def load_downstream(model, k):
    if model in ("llama", "aunet", "bpebyte"):
        p = L / f"runs/downstream_ci/{model}_{k}shot/results.json"
    else:
        j = _ext_json(model, f"ds{k}")
        if j is None:
            return None
        S = j["samples"]
        return {t: _vec(S[t], METRIC[t]) for t in DS5 if t in S}
    if not p.exists():
        return None
    S = json.load(open(p))["samples"]
    return {t: _vec(S[t], METRIC[t]) for t in DS5 if t in S}


def load_robust(model):
    """{axis: {task: (clean {doc: v}, [variant {doc: v}, ...])}}"""
    out = {"noise": {}, "typo": {}, "despace": {}}
    if model in ("llama", "aunet", "bpebyte"):
        S = {}
        for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
            S.update(json.load(open(L / f"runs/{d}/{model}/results.json"))["samples"])
        D = json.load(open(L / f"runs/robustness_despace_bits/{model}/results.json"))["results"]
        pert = {"noise": S, "typo": S}
    else:
        pert = {}
        for ax in ("noise", "typo"):
            j = _ext_json(model, ax)
            pert[ax] = j["samples"] if j else None
        j = _ext_json(model, "despace")
        D = j["results"] if j else None
    for ax in ("noise", "typo"):
        S = pert[ax]
        if S is None:
            continue
        for t in R5:
            m = "acc" if t == "boolq" else "acc_norm"
            vs = [k for k in S if k.startswith(f"{t}_{ax}_") and not k.endswith("_avg")]
            if t in S and vs:
                out[ax][t] = (_vec(S[t], m), [_vec(S[k], m) for k in sorted(vs)])
    if D is not None:
        for t in R5:
            c, p = D.get(f"despace_mc_{t}_clean"), D.get(f"despace_mc_{t}_despaceall")
            if c and p and "bits" in c:
                out["despace"][t] = (dict(enumerate(c["bits"]["acc"])), [dict(enumerate(p["bits"]["acc"]))])
    return out


def to_arrays(per_model, tasks):
    """Align every model on the doc ids all of them scored -> {task: {model: array}} + ids."""
    aligned = {}
    for t in tasks:
        common = None
        for v in per_model.values():
            common = set(v[t]) if common is None else common & set(v[t])
        ids = sorted(common)
        aligned[t] = {m: np.array([v[t][i] for i in ids]) for m, v in per_model.items()}
    return aligned


def bootstrap(stat_fn, n_items, B, rng):
    """stat_fn(idx: {task: index array}) -> {model: value}; returns point, samples per model."""
    point = stat_fn({t: np.arange(n) for t, n in n_items.items()})
    boots = {m: np.empty(B) for m in point}
    for b in range(B):
        idx = {t: rng.integers(0, n, n) for t, n in n_items.items()}
        for m, v in stat_fn(idx).items():
            boots[m][b] = v
    return point, boots


def summarize(point, boots):
    return {m: {"value": round(float(point[m]), 3),
                "ci95": [round(float(x), 3) for x in np.percentile(boots[m], [2.5, 97.5])],
                "half_width": round(float(np.diff(np.percentile(boots[m], [2.5, 97.5]))[0] / 2), 3)}
            for m in point}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(L / "reports/ci_main_table"))
    ap.add_argument("--blt_prefix", default="", help='BLT result-file prefix, e.g. "t1609_"')
    a = ap.parse_args()
    global BLT_PREFIX
    BLT_PREFIX = a.blt_prefix
    rng = np.random.default_rng(a.seed)
    res = {}

    for k in (0, 3, 5):
        pm = {m: d for m in MODELS if (d := load_downstream(m, k)) is not None and all(t in d for t in DS5)}
        if not pm:
            continue
        A = to_arrays(pm, DS5)
        n = {t: len(next(iter(A[t].values()))) for t in DS5}
        fn = lambda idx: {m: 100 * np.mean([A[t][m][idx[t]].mean() for t in DS5]) for m in pm}
        res[f"downstream_{k}shot"] = {"n_items": n, **summarize(*bootstrap(fn, n, a.B, rng))}
        print(f"downstream {k}-shot:", {NAME[m]: v["value"] for m, v in res[f'downstream_{k}shot'].items() if m in NAME})

    R = {m: load_robust(m) for m in MODELS}
    for ax in ("noise", "typo", "despace"):
        pm = {m: R[m][ax] for m in MODELS if all(t in R[m][ax] for t in R5)}
        if not pm:
            continue
        A = {}
        for t in R5:
            common = None
            for m in pm:
                c, vs = pm[m][t]
                ids = set(c).intersection(*[set(v) for v in vs])
                common = ids if common is None else common & ids
            ids = sorted(common)
            A[t] = {m: (np.array([pm[m][t][0][i] for i in ids]),
                        np.array([[v[i] for i in ids] for v in pm[m][t][1]])) for m in pm}
        n = {t: len(next(iter(A[t].values()))[0]) for t in R5}
        fn = lambda idx: {m: abs(100 * np.mean([A[t][m][1][:, idx[t]].mean() - A[t][m][0][idx[t]].mean()
                                                for t in R5])) for m in pm}
        res[ax] = {"n_items": n, **summarize(*bootstrap(fn, n, a.B, rng))}
        print(f"{ax}:", {NAME[m]: v["value"] for m, v in res[ax].items() if m in NAME})

    os.makedirs(a.out, exist_ok=True)
    json.dump({"B": a.B, "seed": a.seed, "results": res}, open(f"{a.out}/ci.json", "w"), indent=1)
    lines = [f"# Main-table 95% CIs (paired item bootstrap, B={a.B})", "",
             "| column | " + " | ".join(NAME[m] for m in MODELS) + " |", "|---" * 6 + "|"]
    for col, r in res.items():
        cells = [f"{r[m]['value']:.2f} ± {r[m]['half_width']:.2f}" if m in r else "—" for m in MODELS]
        lines.append(f"| {col} | " + " | ".join(cells) + " |")
    open(f"{a.out}/ci.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

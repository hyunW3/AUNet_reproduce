"""Main-table Typo / Leet on the "both" region (context + answer options) over HellaSwag / ARC-E / ARC-C / PIQA
(BoolQ excluded), all six settings. Sources: runs/robustness_typoleet_both (scripts/probes/run_typoleet_both.sh).
Metric as paper_robustness_tables.py: acc_norm, PIQA acc. Typo = mean over its 8 variants per item. Paired item
bootstrap as ci_main_table.py (B=2000, seed 0; same resampled indices for every model and for clean/perturbed).
Writes reports/ci_main_table/ci_typoleet_both.json (value, half_width, per-task clean/perturbed accuracy)."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes")
import ci_main_table as C

R = Path("/mnt/ssd2/hyun2/AUNet/runs/robustness_typoleet_both")
# BPEByte: no-vocab_norm reruns (official AU-Net generator; lingua SCORING_POLICY.md 2026-10-07).
NONORM = Path("/mnt/ssd2/hyun2/AUNet/runs/robustness_paper4task_nonorm")
T4 = ("hellaswag", "arc_easy", "arc_challenge", "piqa")
MET = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc"}
MODELS = ("llama", "aunet", "bpebyte", "blt", "blt161", "hnet")


def typo_samples(m):
    if m == "bpebyte":
        return json.load(open(NONORM / "typo/bpebyte/results.json"))["samples"]
    if m in ("llama", "aunet"):
        return json.load(open(R / f"trio_typo_{m}/results.json"))["samples"]
    S = {}
    pat = {"blt": "blt1335_typoboth_*.json", "blt161": "blt1609_typoboth_*.json", "hnet": "hnet_typoboth.json"}[m]
    for f in sorted(R.glob(pat)):
        S.update(json.load(open(f))["samples"])
    return S


def leet_rows(m):
    files = {"llama": ["trio_leet_llama/results.json"], "aunet": ["trio_leet_aunet/results.json"],
             "bpebyte": [str(NONORM / "leet/bpebyte/results.json")], "blt": sorted(glob.glob(str(R / "blt1335_leetboth_*.json"))),
             "blt161": sorted(glob.glob(str(R / "blt1609_leetboth_*.json"))), "hnet": ["hnet_leetboth.json"]}[m]
    rows = {}
    for f in files:
        d = json.load(open(R / f))
        rows.update(d.get("results", d))
    return rows


def load(m):
    S, L = typo_samples(m), leet_rows(m)
    out = {"typo": {}, "leet": {}}
    for t in T4:
        k = MET[t]
        vs = sorted(x for x in S if x.startswith(f"{t}_typoboth_") and not x.endswith("_avg"))
        assert len(vs) == 8, (m, t, vs)
        out["typo"][t] = (C._vec(S[t], k), [C._vec(S[v], k) for v in vs])
        c, p = L[f"fmt_{t}_clean"]["bits"][k], L[f"fmt_{t}_nla_leet_both"]["bits"][k]
        out["leet"][t] = (dict(enumerate(c)), [dict(enumerate(p))])
    return out


def main():
    rng = np.random.default_rng(0)
    D = {m: load(m) for m in MODELS}
    res = {}
    for ax in ("typo", "leet"):
        A, n = {}, {}
        for t in T4:
            common = None
            for m in MODELS:
                c, vs = D[m][ax][t]
                ids = set(c).intersection(*[set(v) for v in vs])
                common = ids if common is None else common & ids
            ids = sorted(common); n[t] = len(ids)
            A[t] = {m: (np.array([D[m][ax][t][0][i] for i in ids], float),
                        np.array([[v[i] for i in ids] for v in D[m][ax][t][1]], float)) for m in MODELS}
        fn = lambda idx, A=A: {m: abs(100 * np.mean([A[t][m][1][:, idx[t]].mean() - A[t][m][0][idx[t]].mean() for t in T4]))
                               for m in MODELS}
        r = {"n_items": n, **C.summarize(*C.bootstrap(fn, n, 2000, rng))}
        for m in MODELS:
            r[m]["per_task"] = {t: {"clean": 100 * A[t][m][0].mean(), "perturbed": 100 * A[t][m][1].mean()} for t in T4}
        res[ax] = r
    json.dump(res, open("/mnt/ssd2/hyun2/AUNet/reports/ci_main_table/ci_typoleet_both.json", "w"), indent=1)
    for ax, r in res.items():
        print(ax, r["n_items"])
        for m in MODELS:
            pt = " ".join(f"{t[:5]} {v['clean']:.1f}->{v['perturbed']:.1f}" for t, v in r[m]["per_task"].items())
            print(f"  {m:8s} {r[m]['value']:6.2f} ± {r[m]['half_width']:.2f} | {pt}")


if __name__ == "__main__":
    main()

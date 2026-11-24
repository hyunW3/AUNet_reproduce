"""BoolQ 0-shot acc + 95% CI (item bootstrap, B=2000, seed 0) for the 1B main table (tab:main_13b), full validation
split (3270 items), the same runs as table_appendix/downstream_0shot.tex.

BoolQ is not in the downstream_ci reruns. Sources:
  llama          main/main/1.3B/llama_1.8B_paper/evals_5bench/results.json (per-item samples)
  aunet, bpebyte main/main/1.3B/{aunet2_1.3B,bpebyte_br_greedy_root_1.3B}/evals_5bench/results.json: aggregate only
                 (log_samples was off). The +- is each model's own bootstrap CI, which depends only on its number of
                 correct items, so an equivalent 0/1 vector is rebuilt from acc*3270 (pairing across models is unused).
  blt / hnet     reports/ext_ci/{blt_official,hnet}/ds0.json, BLT theta=1.61 t1609_ds0.json (per-item samples)
  python ci_boolq_0shot.py reports/ci_main_table/boolq_0shot.json
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes")
import ci_main_table as C

N = 3270
R = C.L / "main/main/1.3B"
AGG = {"aunet": R / "aunet2_1.3B/evals_5bench/results.json",
       "bpebyte": R / "bpebyte_br_greedy_root_1.3B/evals_5bench/results.json"}


def load_boolq(model):
    if model == "llama":
        S = json.load(open(R / "llama_1.8B_paper/evals_5bench/results.json"))["samples"]
    elif model in AGG:
        k = round(json.load(open(AGG[model]))["results"]["boolq"]["acc,none"] * N)
        return np.r_[np.ones(k), np.zeros(N - k)]
    else:
        S = C._ext_json(model, "ds0")["samples"]
    v = C._vec(S["boolq"], "acc")
    assert len(v) == N, (model, len(v))
    return np.array([v[i] for i in sorted(v)])


out = {}
for prefix, tag in (("", ""), ("t1609_", "_t1609")):
    C.BLT_PREFIX = prefix
    models = C.MODELS if not prefix else ("blt",)
    A = {m: load_boolq(m) for m in models}
    rng = np.random.default_rng(0)
    fn = lambda idx: {m: 100 * A[m][idx["boolq"]].mean() for m in A}
    s = C.summarize(*C.bootstrap(fn, {"boolq": N}, 2000, rng))
    for m, v in s.items():
        out[m + tag] = {"boolq": {"value": round(v["value"], 2), "hw": round(v["half_width"], 1), "n": N}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for m, d in out.items():
    v = d["boolq"]
    print(f"{m:12s} boolq {v['value']:.2f}±{v['hw']:.1f} (n={v['n']})")

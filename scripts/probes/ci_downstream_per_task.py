#!/usr/bin/env python3
"""Per-task 95% CIs for the appendix 0/3/5-shot downstream tables (tab:downstream_{0,3,5}shot), which replace the
former per-benchmark 0-shot table tab:main_13b (tables/1B_table.tex).

Same bootstrap as ci_main_table.py / ci_downstream6.py: item-level resampling, B=2000, seed 0, percentile CI,
one task at a time (a fresh rng per (shot, task)), plus the multi-task averages (every task resampled per draw).
  HS/ARC-E/ARC-C/PIQA/WG : per-item reruns, ci_main_table.load_downstream (runs/downstream_ci, reports/ext_ci/*)
  LAMBADA                : per-item std_bench runs, reports/std_bench/<model>/ds{k}_seed1234_lambada.json
  BoolQ / MMLU           : the matched models' original runs saved aggregates only, so -- as in ci_boolq_0shot.py --
                           a 0/1 vector with the reported number of correct items is rebuilt from acc*n
                           (BoolQ n=3270, MMLU = mmlu_text n=14042, micro acc). A single model's bootstrap CI depends
                           only on that count, so the interval is exact; the cross-model pairing is unused.
BLT = released threshold (blt_1b, reports/ext_ci/blt_official), plus "blt_t1609" = BLT at the compression-matched
threshold theta=1.61 (t1609_ds{k}.json, blt_1b_t1609 LAMBADA); BoolQ/MMLU cells BLT cannot fit are absent.
  python ci_downstream_per_task.py reports/ci_main_table/downstream_per_task_ci.json
"""
import json, sys
import numpy as np
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes")
import ci_main_table as C

REG = {"llama": "llama_1.3b", "aunet": "aunet_1.3b", "bpebyte": "bpebyte_1.3b", "blt": "blt_1b",
       "hnet": "hnet_1stage_XL", "blt_t1609": "blt_1b_t1609"}
MODELS = C.MODELS + ("blt_t1609",)
TASKS6 = C.DS5 + ("lambada_openai",)
N = {"boolq": 3270, "mmlu": 14042}
# BoolQ / MMLU accuracies as reported in table_appendix/downstream_{0,3,5}shot.tex (None = exceeds BLT's context)
AGG = {
    0: {"boolq": dict(llama=63.46, aunet=61.13, bpebyte=62.05, blt=70.43, hnet=58.29, blt_t1609=69.48),
        "mmlu": dict(llama=31.90, aunet=32.15, bpebyte=32.32, blt=35.64, hnet=33.38, blt_t1609=34.87)},
    3: {"boolq": dict(llama=63.46, aunet=59.36, bpebyte=62.72, blt=71.28, hnet=63.73, blt_t1609=69.66),
        "mmlu": dict(llama=33.47, aunet=33.29, bpebyte=33.43, blt=None, hnet=35.18, blt_t1609=None)},
    5: {"boolq": dict(llama=65.11, aunet=58.90, bpebyte=62.87, blt=None, hnet=63.18, blt_t1609=None),
        "mmlu": dict(llama=33.46, aunet=33.27, bpebyte=33.04, blt=None, hnet=35.37, blt_t1609=None)},
}
# extra averages: name -> tasks (each average only for the models that have every task)
AVGS = {0: {"avg6": TASKS6, "avg8": TASKS6 + ("boolq", "mmlu")},
        3: {"avg6": TASKS6, "avg7": TASKS6 + ("boolq",)},
        5: {"avg6": TASKS6}}


def load_lambada(model, k):
    j = json.load(open(C.L / f"reports/std_bench/{REG[model]}/ds{k}_seed1234_lambada.json"))
    assert j["num_fewshot"] == k and j["limit"] is None
    return {int(i.split(":")[1]): b for i, b in zip(j["items"]["lambada_openai"], j["bits"]["lambada_openai"])}


FILL = C.L / "reports/ext_ci/blt_official"


def load_fill(m, k, t):
    """BLT cells beyond its 4096-byte context, re-run left-truncated (run_ext.py, 2026-10-07):
    fill_ds{k}_{task}.json / t1609_fill_ds{k}_{task}.json -> per-item 0/1 vector, or None if not run."""
    if m not in ("blt", "blt_t1609"):
        return None
    name = {"mmlu": "mmlu_text"}.get(t, t)
    p = FILL / f"{'t1609_' if m == 'blt_t1609' else ''}fill_ds{k}_{name}.json"
    if not p.exists():
        return None
    S = json.load(open(p))["samples"]
    if name == "mmlu_text":   # group: every mmlu_text_* subtask, micro-averaged (14042 items)
        v = [b for st_, items in S.items() if st_.startswith("mmlu_text_") for b in C._vec(items, "acc").values()]
    else:
        v = list(C._vec(S[name], "acc").values())
    assert len(v) == N[t], (m, k, t, len(v))
    return np.array(v, float)


def from_count(acc, n):
    c = round(acc * n / 100)
    return np.r_[np.ones(c), np.zeros(n - c)]


def boot(vecs, tasks, seed=0, B=2000):
    """vecs: {model: {task: array}}; statistic = mean over tasks of the task accuracy, per model."""
    n = {t: len(next(iter(vecs.values()))[t]) for t in tasks}
    for v in vecs.values():
        assert all(len(v[t]) == n[t] for t in tasks)
    fn = lambda idx: {m: 100 * np.mean([v[t][idx[t]].mean() for t in tasks]) for m, v in vecs.items()}
    return C.summarize(*C.bootstrap(fn, n, B, np.random.default_rng(seed))), n


out = {}
for k in (0, 3, 5):
    pm = {}
    for m in MODELS:
        C.BLT_PREFIX = "t1609_" if m == "blt_t1609" else ""
        d = C.load_downstream("blt" if m == "blt_t1609" else m, k)
        pm[m] = {**d, "lambada_openai": load_lambada(m, k)}
    C.BLT_PREFIX = ""
    A = C.to_arrays(pm, TASKS6)  # {task: {model: array}} on the doc ids every model scored
    vec = {m: {t: A[t][m] for t in TASKS6} for m in MODELS}
    for t, accs in AGG[k].items():
        for m, a in accs.items():
            f = load_fill(m, k, t) if a is None else None
            if f is not None:
                vec[m][t] = f
            elif a is not None:
                vec[m][t] = from_count(a, N[t])
    res = {}
    for t in TASKS6 + ("boolq", "mmlu"):
        have = {m: {t: v[t]} for m, v in vec.items() if t in v}
        s, n = boot(have, (t,))
        res[t] = {m: {"value": round(x["value"], 2), "hw": round(x["half_width"], 2), "n": n[t]} for m, x in s.items()}
    for name, tasks in AVGS[k].items():
        have = {m: v for m, v in vec.items() if all(t in v for t in tasks)}
        s, _ = boot(have, tasks)
        res[name] = {m: {"value": round(x["value"], 2), "hw": round(x["half_width"], 2)} for m, x in s.items()}
    out[f"{k}shot"] = res
    print(f"== {k}-shot")
    for t, r in res.items():
        print(f"  {t:15s}", "  ".join(f"{m} {x['value']:.2f}±{x['hw']:.2f}" for m, x in r.items()))
json.dump({"B": 2000, "seed": 0, "results": out}, open(sys.argv[1], "w"), indent=1)

#!/usr/bin/env python3
"""Downstream Avg6 = mean of HellaSwag, ARC-E, ARC-C, PIQA, WinoGrande and LAMBADA at 0/3/5 shots,
with the paired item-bootstrap 95% CI of ci_main_table.py (same loaders, same resampling scheme).

The five multiple-choice tasks come from ci_main_table.load_downstream; LAMBADA (lambada_openai acc,
greedy exact match of the last word, all 5,153 items, fewshot seed 1234) from the std_bench runs
  reports/std_bench/<registry name>/ds{k}_seed1234_lambada.json
  (TASKS=lambada_openai SUFFIX=lambada SHOT=k bash scripts/probes/std_bench/run_std_bench.sh <model>).

  python ci_downstream6.py                       # BLT = released threshold (blt_1b)
  python ci_downstream6.py --blt_prefix t1609_   # BLT = theta 1.61 (blt_1b_t1609)
  -> reports/ci_main_table/downstream6{,_t1609}.json, .md
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_main_table as C  # noqa: E402

REG = {"llama": "llama_1.3b", "aunet": "aunet_1.3b", "bpebyte": "bpebyte_1.3b",
       "blt": "blt_1b", "hnet": "hnet_1stage_XL"}
TASKS6 = C.DS5 + ("lambada_openai",)


def load_lambada(model, k, blt_t1609):
    reg = "blt_1b_t1609" if (model == "blt" and blt_t1609) else REG[model]
    p = C.L / f"reports/std_bench/{reg}/ds{k}_seed1234_lambada.json"
    if not p.exists():
        return None
    j = json.load(open(p))
    assert j["num_fewshot"] == k and j["limit"] is None
    return {int(i.split(":")[1]): b for i, b in zip(j["items"]["lambada_openai"], j["bits"]["lambada_openai"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blt_prefix", default="")
    a = ap.parse_args()
    C.BLT_PREFIX = a.blt_prefix
    rng = np.random.default_rng(a.seed)
    res = {}
    for k in (0, 3, 5):
        pm = {}
        for m in C.MODELS:
            d, lam = C.load_downstream(m, k), load_lambada(m, k, bool(a.blt_prefix))
            if d is None or lam is None or not all(t in d for t in C.DS5):
                print(f"skip {m} {k}-shot (missing)")
                continue
            pm[m] = {**d, "lambada_openai": lam}
        A = C.to_arrays(pm, TASKS6)
        n = {t: len(next(iter(A[t].values()))) for t in TASKS6}
        fn = lambda idx: {m: 100 * np.mean([A[t][m][idx[t]].mean() for t in TASKS6]) for m in pm}
        res[f"downstream6_{k}shot"] = {"n_items": n, **C.summarize(*C.bootstrap(fn, n, a.B, rng))}
        res[f"lambada_{k}shot"] = {m: round(100 * float(A["lambada_openai"][m].mean()), 3) for m in pm}
        print(f"{k}-shot Avg6:", {C.NAME[m]: v["value"] for m, v in res[f'downstream6_{k}shot'].items() if m in C.NAME})
    tag = "_" + a.blt_prefix.rstrip("_") if a.blt_prefix else ""
    out = C.L / "reports/ci_main_table"
    os.makedirs(out, exist_ok=True)
    json.dump({"B": a.B, "seed": a.seed, "blt_prefix": a.blt_prefix, "results": res},
              open(out / f"downstream6{tag}.json", "w"), indent=1)
    lines = [f"# Downstream Avg6 (HS/ARC-E/ARC-C/PIQA/WG/LAMBADA) 95% CIs (paired item bootstrap, B={a.B})", "",
             "| column | " + " | ".join(C.NAME[m] for m in C.MODELS) + " |", "|---" * 6 + "|"]
    for col, r in res.items():
        if col.startswith("lambada"):
            cells = [f"{r[m]:.2f}" if m in r else "—" for m in C.MODELS]
        else:
            cells = [f"{r[m]['value']:.2f} ± {r[m]['half_width']:.2f}" if m in r else "—" for m in C.MODELS]
        lines.append(f"| {col} | " + " | ".join(cells) + " |")
    open(out / f"downstream6{tag}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

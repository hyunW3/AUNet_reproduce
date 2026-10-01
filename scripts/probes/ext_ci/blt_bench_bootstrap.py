#!/usr/bin/env python3
"""Item-bootstrap 95% CIs for blt_bench.py results (no GPU, deterministic for a given --seed).

Per task: resample that task's items with replacement B times, percentile CI of the accuracy.
Macro average: every replicate resamples each task independently and averages the task accuracies.
With --ref, the CI is of the paired difference RUN - REF: both runs must cover the same items
(same tasks, num_fewshot, limit) and every replicate uses the same item indices for both.
A two-sided bootstrap p-value (2 * min(P[d<=0], P[d>=0])) is reported for the differences.

  python blt_bench_bootstrap.py out/blt_bench/ds0_seed1234.json
  python blt_bench_bootstrap.py RUN.json --ref REF.json --B 10000 --seed 0 --out ci.json
"""
import argparse, json
import numpy as np


def load(path):
    d = json.load(open(path))
    return d, {t: np.asarray(b, float) for t, b in d["bits"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default=None, help="second blt_bench.py result for a paired difference CI")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d, X = load(a.run)
    tasks = list(X)
    Y = None
    if a.ref:
        dr, Y = load(a.ref)
        for t in tasks:
            if t not in Y or len(Y[t]) != len(X[t]):
                raise SystemExit(f"--ref does not cover the same items for {t}")
            if d["samples"].get(t, {}).get("doc_id") != dr["samples"].get(t, {}).get("doc_id"):
                raise SystemExit(f"--ref doc_ids differ for {t}")
    rng = np.random.default_rng(a.seed)
    lo, hi = 100 * a.alpha / 2, 100 * (1 - a.alpha / 2)

    def stat(v):  # v: [B] in percent
        return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]

    rows, reps, dreps = {}, [], []
    for t in tasks:
        n = len(X[t])
        idx = rng.integers(0, n, size=(a.B, n))  # one draw per task, shared by RUN and REF
        rx = 100 * X[t][idx].mean(1); reps.append(rx)
        row = {"n": n, "acc": 100 * X[t].mean(), "ci": stat(rx)}
        if Y is not None:
            ry = 100 * Y[t][idx].mean(1); dd = rx - ry; dreps.append(dd)
            row.update(ref=100 * Y[t].mean(), diff=100 * (X[t].mean() - Y[t].mean()), diff_ci=stat(dd),
                       p=float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
        rows[t] = row
    avg = np.mean(reps, 0)
    rows["macro_avg"] = {"n": sum(len(X[t]) for t in tasks), "acc": float(np.mean([rows[t]["acc"] for t in tasks])),
                         "ci": stat(avg)}
    if Y is not None:
        dd = np.mean(dreps, 0)
        rows["macro_avg"].update(ref=float(np.mean([rows[t]["ref"] for t in tasks])),
                                 diff=float(np.mean([rows[t]["diff"] for t in tasks])), diff_ci=stat(dd),
                                 p=float(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))

    pct = int(round(100 * (1 - a.alpha)))
    print(f"BLT-1B {d['num_fewshot']}-shot seed={d['seed']} limit={d['limit']}  bootstrap B={a.B} seed={a.seed}")
    hdr = f"| task | metric | n | acc | {pct}% CI |" + (" ref | diff | diff CI | p |" if Y is not None else "")
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for t, r in rows.items():
        m = d["metric"].get(t, "")
        s = f"| {t} | {m} | {r['n']} | {r['acc']:.2f} | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] |"
        if Y is not None:
            s += f" {r['ref']:.2f} | {r['diff']:+.2f} | [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}] | {r['p']:.3f} |"
        print(s)
    if a.out:
        json.dump({"run": a.run, "ref": a.ref, "B": a.B, "seed": a.seed, "alpha": a.alpha, "rows": rows},
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Item-bootstrap 95% CIs for std_bench.py results, one or many models (no GPU, deterministic per --seed).

Per task: resample that task's items with replacement B times, percentile CI of the accuracy.
Macro average: every replicate resamples each task independently and averages the task accuracies.
All runs given together share the same resampled item indices (paired), so with --ref every other
run also gets a CI and a two-sided bootstrap p-value (2 * min(P[d<=0], P[d>=0])) for its paired
difference RUN - REF. Paired runs must cover the same items (checked via the saved item ids).

  python std_bench_bootstrap.py reports/std_bench/aunet_1.3b/ds0_seed1234.json
  python std_bench_bootstrap.py reports/std_bench/*/ds0_seed1234.json --ref reports/std_bench/llama_1.3b/ds0_seed1234.json
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None, help="run to take paired differences against (added to runs if absent)")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="JSON with every number in the printed table")
    a = ap.parse_args()

    paths = list(dict.fromkeys(a.runs + ([a.ref] if a.ref else [])))
    runs = [json.load(open(p)) for p in paths]
    names = [r["model"] for r in runs]
    names = [nm if names.count(nm) == 1 else f"{nm} ({p})" for nm, p in zip(names, paths)]  # repeats of one model
    tasks = [t for t in runs[0]["bits"] if all(t in r["bits"] for r in runs)]
    for t in tasks:
        ids = runs[0]["items"][t]
        for p, r in zip(paths, runs):
            if r["items"][t] != ids:
                raise SystemExit(f"{p}: items of {t} differ from {paths[0]} (same tasks/limit/num_fewshot needed)")
    X = [{t: np.asarray(r["bits"][t], float) for t in tasks} for r in runs]
    ref = paths.index(a.ref) if a.ref else None

    rng = np.random.default_rng(a.seed)
    lo, hi = 100 * a.alpha / 2, 100 * (1 - a.alpha / 2)
    ci = lambda v: [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
    pval = lambda d: float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

    reps = [dict() for _ in runs]  # reps[i][task] = [B] accuracies (percent)
    for t in tasks:
        n = len(X[0][t])
        idx = rng.integers(0, n, size=(a.B, n))  # shared by every run -> paired
        for i in range(len(runs)):
            reps[i][t] = 100 * X[i][t][idx].mean(1)
    rows = {}
    for i, nm in enumerate(names):
        row = {}
        for t in tasks + ["macro_avg"]:
            v = np.mean([reps[i][u] for u in tasks], 0) if t == "macro_avg" else reps[i][t]
            acc = float(np.mean([100 * X[i][u].mean() for u in tasks])) if t == "macro_avg" else 100 * X[i][t].mean()
            cell = {"acc": acc, "ci": ci(v)}
            if ref is not None and i != ref:
                vr = np.mean([reps[ref][u] for u in tasks], 0) if t == "macro_avg" else reps[ref][t]
                accr = (float(np.mean([100 * X[ref][u].mean() for u in tasks])) if t == "macro_avg"
                        else 100 * X[ref][t].mean())
                cell.update(diff=acc - accr, diff_ci=ci(v - vr), p=pval(v - vr))
            row[t] = cell
        rows[nm] = row

    r0 = runs[0]
    print(f"{r0['num_fewshot']}-shot seed={r0['seed']} limit={r0['limit']}  bootstrap B={a.B} seed={a.seed}  "
          f"({int(round(100 * (1 - a.alpha)))}% percentile CI" + (f"; diff vs {names[ref]})" if ref is not None else ")"))
    cols = tasks + ["macro_avg"]
    print("| model | " + " | ".join(f"{t} ({r0['metric'].get(t, '')}, n={len(X[0][t])})" if t in r0['metric']
                                     else t for t in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for nm in names:
        cells = []
        for t in cols:
            c = rows[nm][t]
            s = f"{c['acc']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]"
            if "diff" in c:
                s += f"<br>Δ {c['diff']:+.2f} [{c['diff_ci'][0]:+.2f}, {c['diff_ci'][1]:+.2f}] p={c['p']:.3f}"
            cells.append(s)
        print(f"| {nm} | " + " | ".join(cells) + " |")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"runs": paths, "ref": a.ref, "B": a.B, "seed": a.seed, "alpha": a.alpha, "rows": rows},
                  open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()

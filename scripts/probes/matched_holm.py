#!/usr/bin/env python3
"""Pairwise significance among the matched 1B trio (Transformer / AUNet / BPEByte), Holm-corrected.

Copy of reports_NAACL/sig_holm/scripts/matched_holm.py for the 2026-10-06 main table, whose downstream columns
are six-task means (HellaSwag/ARC-E/ARC-C/PIQA/WinoGrande + LAMBADA at the same shot count) and which has no
separate LAMBADA column. LAMBADA k-shot bits: reports/std_bench/<model>_1.3b/ds{k}_seed1234_lambada.json.

Eight main-table columns with per-item data: downstream Avg6 at 0/3/5 shots, S-NIAH (S-NIAH-1/2/3,
original haystacks), and |dAcc| for Noise / Typo / Despace / Leet over HellaSwag/ARC-E/ARC-C/PIQA (acc_norm on
HellaSwag/ARC, acc on PIQA, as in tab:main_robust; Noise and Despace on the "both" region). BoolQ and the
repeated-haystack S-NIAH variants were dropped from the main table on 2026-10-06. For each column, one paired item bootstrap (B=10,000, seed 0): items are resampled within each
task (S-NIAH: within each task x length cell), the same draw is used for all three models and for the
clean / perturbed forms. Per model pair, the oriented difference (positive = first model better) gets a
two-sided bootstrap p-value 2*min(P(d<=0), P(d>=0)) (floored at 1/B); Holm's step-down correction runs over
all 24 tests (8 columns x 3 pairs). Non-inferiority of BPEByte vs AUNet: one-sided upper bound of AUNet's
advantage, pointwise (95%) and Bonferroni-simultaneous over the eight columns (1 - 0.05/8).

PBP, BPB and latency have no per-item data and are not tested.

Loaders reused from the AUNet repo (main; paths overridable with env):
  scripts/probes/ci_main_table.py                         downstream (lm-eval samples)
  FORMAT_MC  = scripts/probes/format_mc                   robust_metric_table.load
  NIAH_DIR   = scripts/niah                               plot_sniah_heatmap.load
  reports/std_bench/<model>_1.3b/ds{0,3,5}_seed1234_lambada.json  LAMBADA bits

  python scripts/probes/matched_holm.py [--B 10000] \
      [--tex ../paper_overleaf/table_appendix/sig_holm.tex]   -> reports_NAACL/sig_holm/results/summary_avg6.json
"""
import argparse, json, os, sys
from itertools import combinations

import numpy as np

A = "/mnt/ssd2/hyun2/AUNet"
FORMAT_MC = os.environ.get("FORMAT_MC", f"{A}/scripts/probes/format_mc")
NIAH_DIR = os.environ.get("NIAH_DIR", f"{A}/scripts/niah")
sys.path[:0] = [FORMAT_MC, os.path.dirname(FORMAT_MC), NIAH_DIR]
import ci_main_table as C          # noqa: E402
import robust_metric_table as RM   # noqa: E402
from plot_sniah_heatmap import load as sniah_load, LENS, TASKS, REP   # noqa: E402

M3 = ("llama", "aunet", "bpebyte")
NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte"}
PAIRS = (("bpebyte", "aunet"), ("bpebyte", "llama"), ("aunet", "llama"))   # first - second, + = first better
COLS = ("0-shot", "3-shot", "5-shot", "S-NIAH", "Noise", "Typo", "Despace", "Leet")
NOISE = ("antspeak", "drop", "randomcase", "repeat", "uppercase")
T4 = ("hellaswag", "arc_easy", "arc_challenge", "piqa")   # tasks with free-text answers (Noise, Despace)
HIB = {c: c not in ("Noise", "Typo", "Despace", "Leet") for c in COLS}


def run(fn, n, B, rng):
    point = fn({k: np.arange(v) for k, v in n.items()})
    boots = {m: np.empty(B) for m in point}
    for b in range(B):
        idx = {k: rng.integers(0, v, v) for k, v in n.items()}
        for m, v in fn(idx).items():
            boots[m][b] = v
    return point, boots


def columns(B, rng):
    out = {}
    for k in (0, 3, 5):
        pm = {m: C.load_downstream(m, k) for m in M3}
        for m in M3:
            j = json.load(open(f"{A}/reports/std_bench/{m}_1.3b/ds{k}_seed1234_lambada.json"))
            assert j["num_fewshot"] == k and j["limit"] is None
            pm[m]["lambada_openai"] = {int(d.split(":")[1]): b for d, b in
                                       zip(j["items"]["lambada_openai"], j["bits"]["lambada_openai"])}
        T6 = C.DS5 + ("lambada_openai",)
        Ar = C.to_arrays(pm, T6)
        n = {t: len(Ar[t]["llama"]) for t in T6}
        out[f"{k}-shot"] = run(lambda i, Ar=Ar, T6=T6: {m: 100 * np.mean([Ar[t][m][i[t]].mean() for t in T6]) for m in M3},
                               n, B, rng)

    S = sniah_load()
    key = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte (ours)"}
    cells = [f"{t}/{L}" for t in TASKS for L in LENS]   # original haystacks only (REP is appendix-only)
    SA = {c: {m: np.array([ok for _, ok in sorted(S[key[m]][c])], float) for m in M3} for c in cells}
    assert all(len({len(v) for v in SA[c].values()}) == 1 for c in cells)
    out["S-NIAH"] = run(lambda i: {m: 100 * np.mean([SA[c][m][i[c]].mean() for c in cells]) for m in M3},
                        {c: len(SA[c]["llama"]) for c in cells}, B, rng)

    metric = {t: "acc" if t in ("piqa", "boolq") else "acc_norm" for t in RM.R5}
    R = {m: RM.load(m, metric) for m in M3}
    for m in M3:                  # Noise: only the variants that perturb both the context and the answer options
        S = {}
        for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
            S.update(json.load(open(f"{A}/runs/{d}/{m}/results.json"))["samples"])
        R[m]["noise"] = {t: (C._vec(S[t], metric[t]), [C._vec(S[f"{t}_noise_{s}_both"], metric[t]) for s in NOISE])
                         for t in T4}
        # Typo / Leet on the both region (context + options): scripts/probes/run_typoleet_both.sh
        TL = f"{A}/runs/robustness_typoleet_both"
        S = json.load(open(f"{TL}/trio_typo_{m}/results.json"))["samples"]
        R[m]["typo"] = {t: (C._vec(S[t], metric[t]),
                            [C._vec(S[k], metric[t]) for k in sorted(S) if k.startswith(f"{t}_typoboth_")])
                        for t in T4}
        F = json.load(open(f"{TL}/trio_leet_{m}/results.json"))["results"]
        R[m]["leet"] = {t: (dict(enumerate(F[f"fmt_{t}_clean"]["bits"][metric[t]])),
                            [dict(enumerate(F[f"fmt_{t}_nla_leet_both"]["bits"][metric[t]]))]) for t in T4}
        assert all(len(R[m]["typo"][t][1]) == 8 for t in T4)
    for ax in ("noise", "typo", "despace", "leet"):
        tasks = T4
        Arr, n = {}, {}
        for t in tasks:
            common = None
            for m in M3:
                c, vs = R[m][ax][t]
                ids = set(c).intersection(*[set(v) for v in vs])
                common = ids if common is None else common & ids
            ids = sorted(common)
            n[t] = len(ids)
            Arr[t] = {m: (np.array([R[m][ax][t][0][i] for i in ids], float),
                          np.array([[v[i] for i in ids] for v in R[m][ax][t][1]], float)) for m in M3}
        fn = lambda i, Arr=Arr, tasks=tasks: {m: abs(100 * np.mean([Arr[t][m][1][:, i[t]].mean() - Arr[t][m][0][i[t]].mean()
                                                                    for t in tasks])) for m in M3}
        out[ax.capitalize()] = run(fn, n, B, rng)
    return out


def holm(p):
    """{key: p} -> {key: Holm-adjusted p} (step-down, monotone)."""
    order = sorted(p, key=p.get)
    adj, run_max = {}, 0.0
    for i, k in enumerate(order):
        run_max = max(run_max, min(1.0, (len(order) - i) * p[k]))
        adj[k] = run_max
    return adj


def fmt_p(p):
    return r"${<}0.01$" if p < 0.01 else f"{p:.2f}"


def write_tex(res, path, B):
    def cell(c, a, b):
        r = res["pairs"][f"{c}|{a}|{b}"]
        d = f"{r['diff']:+.2f}".replace("-", "$-$")
        if r["p_holm"] < 0.05:
            d = r"\textbf{" + d + "}"
        return f"{d} & {fmt_p(r['p_holm'])}"
    body = []
    for c in COLS:
        ni = res["noninferiority"][c]["simultaneous"]
        ni = "--" if ni < 0 else f"{ni:.2f}"
        arrow = r"$\uparrow$" if HIB[c] else r"$\downarrow$"
        body.append(f"{c} {arrow} & " + " & ".join(cell(c, a, b) for a, b in PAIRS) + f" & {ni} \\\\")
    tex = rf"""% GENERATED by AUNet/scripts/probes/matched_holm.py (B={B}, seed 0) -- edit the script, not this file.
\begin{{table}}[t]
\centering
\caption{{\hyun{{Pairwise comparison of the three matched models on the eight aggregate measures with per-item results: the six-task downstream averages (including LAMBADA) at $0$/$3$/$5$ shots, S-NIAH, and the four robustness axes of Table~\ref{{tab:main_robust}}. $\Delta$ is the difference in percentage points, signed so that a positive value favors the first model (higher accuracy, or a smaller $|\Delta\mathrm{{Acc}}|$ for the robustness columns). $p_{{\mathrm{{Holm}}}}$ is the two-sided paired item-level bootstrap $p$-value (10,000 resamples) after Holm correction over all $24$ tests; bold marks $p_{{\mathrm{{Holm}}}}<0.05$. \textbf{{NI bound}}: the largest margin, in points, by which AUNet could outperform BPEByte on that column, as the one-sided upper confidence bound simultaneous over the eight columns (Bonferroni, $95\%$); -- marks columns where BPEByte is significantly better.}}}}
\label{{tab:sig_holm}}
\small
\setlength{{\tabcolsep}}{{3pt}}
\fitcolumn{{%
\begin{{tabular}}{{l|cc|cc|cc|c}}
\toprule
 & \multicolumn{{2}}{{c|}}{{\textbf{{BPEByte $-$ AUNet}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{BPEByte $-$ Transf.}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{AUNet $-$ Transf.}}}} & \textbf{{NI}} \\
\textbf{{Column}} & $\Delta$ & $p_{{\mathrm{{Holm}}}}$ & $\Delta$ & $p_{{\mathrm{{Holm}}}}$ & $\Delta$ & $p_{{\mathrm{{Holm}}}}$ & bound \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""
    open(path, "w").write(tex)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{A}/reports_NAACL/sig_holm/results/summary_avg6.json")
    ap.add_argument("--tex", default=None)
    a = ap.parse_args()
    X = columns(a.B, np.random.default_rng(a.seed))

    res = {"B": a.B, "seed": a.seed, "points": {}, "pairs": {}, "noninferiority": {}}
    D, praw = {}, {}
    for c in COLS:
        pt, bt = X[c]
        s = 1 if HIB[c] else -1
        res["points"][c] = {NAME[m]: round(float(pt[m]), 3) for m in M3}
        for x, y in combinations(M3, 2):
            for a1, b1 in ((x, y), (y, x)):
                if (a1, b1) in PAIRS:
                    k = f"{c}|{a1}|{b1}"
                    D[k] = s * (bt[a1] - bt[b1])
                    praw[k] = max(2 * min((D[k] <= 0).mean(), (D[k] >= 0).mean()), 1 / a.B)
                    lo, hi = np.percentile(D[k], [2.5, 97.5])
                    res["pairs"][k] = {"diff": round(float(s * (pt[a1] - pt[b1])), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                       "p": float(praw[k])}
        adv = -D[f"{c}|bpebyte|aunet"]          # AUNet advantage over BPEByte
        res["noninferiority"][c] = {"pointwise": round(float(np.percentile(adv, 95)), 3),
                                    "simultaneous": round(float(np.percentile(adv, 100 - 5 / len(COLS))), 3)}
    for k, v in holm(praw).items():
        res["pairs"][k]["p_holm"] = float(v)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    print(f"{'column':9s}" + "".join(f"{NAME[x][:5]}-{NAME[y][:5]:>12s}" for x, y in PAIRS) + "   NI(sim)")
    for c in COLS:
        print(f"{c:9s}" + "".join(f"{res['pairs'][f'{c}|{x}|{y}']['diff']:+8.2f} p={res['pairs'][f'{c}|{x}|{y}']['p_holm']:.3f}"
                                  for x, y in PAIRS) + f"   {res['noninferiority'][c]['simultaneous']:+.2f}")
    if a.tex:
        write_tex(res, a.tex, a.B)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robustness by perturbation REGION (context only / answer options only / both), all models.

Noise (5 strategies, prompt / completion / both) and Despace (despace / despaceans / despaceall) come from the
paper's sources via paper_robustness_tables.load_results(); Typo / Leet "both" likewise. Typo / Leet context-only
and options-only come from scripts/probes/run_typoleet_region.sh (default runs/robustness_typoleet_region):
  Typo : <task>_typoctx_* / <task>_typoopt_* (typoboth seeds; own within-run clean baseline)
  Leet : format_mc nla_leet (context) / nla_leet_opt (options), clean from the same run (paper cache reused)
Metric = paper convention: acc_norm, PIQA acc. Cell = mean over HellaSwag / ARC-E / ARC-C / PIQA of the signed
Delta (perturbed - clean, pp); the paper's main table uses the mean |Delta| instead (also written to the json).

  python robust_region_table.py [--region_dir runs/robustness_typoleet_region] [--out reports_afterNAACL/robust_region]
"""
import argparse, glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_robustness_tables as P  # noqa: E402

T4 = P.T4
MODELS = ("llama", "aunet", "bpebyte", "blt", "blt1609", "hnet")
MNAME = {"llama": "Llama (BPE)", "aunet": "AU-Net", "bpebyte": "BPEByte", "blt": "BLT θ=1.335",
         "blt1609": "BLT θ=1.609", "hnet": "H-Net"}
MTEX = {"llama": "Llama (BPE)", "aunet": "AU-Net", "bpebyte": "BPEByte", "blt": r"BLT$^{\dagger}$ ($\theta{=}1.335$)",
        "blt1609": r"BLT$^{\dagger}$ ($\theta{=}1.609$)", "hnet": r"H-Net$^{\ddagger}$"}
REG = ("ctx", "opt", "both")
RNAME = {"ctx": "Context", "opt": "Options", "both": "Both"}
AXES = ("noise", "typo", "despace", "leet")
ANAME = {"noise": "Noise", "typo": "Typo", "despace": "Despace", "leet": "Leet"}
NOISE_POS = {"ctx": "prompt", "opt": "completion", "both": "both"}
DESPACE_V = {"ctx": "despace", "opt": "despaceans", "both": "despaceall"}
LEET_V = {"ctx": "nla_leet", "opt": "nla_leet_opt", "both": "nla_leet_both"}


def _load(files):
    out = {}
    for f in files:
        out.update(json.load(open(f))["results"])
    return out


def region_runs(D):
    """model -> {'typo': {region: flat results}, 'leet': flat results} for the ctx / opt runs."""
    R = {}
    for m in MODELS:
        typo, leet = {}, {}
        for r in ("ctx", "opt"):
            if m in P.MATCHED:
                fs = [f"{D}/trio_typo{r}_{m}/results.json"]
            elif m in ("blt", "blt1609"):
                th = "1335" if m == "blt" else "1609"
                fs = sorted(glob.glob(f"{D}/blt{th}_typo{r}_*.json"))
            else:
                fs = [f"{D}/hnet_typo{r}.json"]
            typo[r] = _load([f for f in fs if os.path.exists(f)])
        if m in P.MATCHED:
            lf = [f"{D}/trio_leet_{m}/results.json"]
        elif m in ("blt", "blt1609"):
            lf = sorted(glob.glob(f"{D}/blt{'1335' if m == 'blt' else '1609'}_leetregion_*.json"))
        else:
            lf = [f"{D}/hnet_leetregion.json"]
        leet = _load([f for f in lf if os.path.exists(f)])
        R[m] = {"typo": typo, "leet": leet}
    return R


def task_delta(Rp, Rr, m, ax, r, t):
    """(clean, perturbed) in % for one model / axis / region / task, or None if not measured."""
    if ax == "noise":
        res = Rp[m]["noise"]
        vals = [P._norm(res.get(k), t) for k in P.noise_keys(t, None, (NOISE_POS[r],)) if k in res]
        c = P._norm(res.get(t), t)
        return None if c is None or not vals else (c, st.mean(vals))
    if ax == "despace":
        res = Rp[m]["despace"]
        c, p = P._norm(res.get(f"despace_mc_{t}_clean"), t), P._norm(res.get(f"despace_mc_{t}_{DESPACE_V[r]}"), t)
        return None if c is None or p is None else (c, p)
    if ax == "typo":
        res = Rp[m]["typo"] if r == "both" else Rr[m]["typo"][r]
        keys = [f"{t}_typo{r}_{o}_{l}" for o in P.TYPO for l in P.TYPO_LVL]
        vals = [P._norm(res.get(k), t) for k in keys if k in res]
        c = P._norm(res.get(t), t)
        return None if c is None or len(vals) != len(keys) else (c, st.mean(vals))
    if ax == "leet":
        res = Rp[m]["leet"] if r == "both" else Rr[m]["leet"]
        c, p = res.get(f"fmt_{t}_clean"), res.get(f"fmt_{t}_{LEET_V[r]}")
        if not c or not p:
            return None
        k = P._metric(t)
        return 100 * st.mean(c["bits"][k]), 100 * st.mean(p["bits"][k])


def build(region_dir):
    Rp, Rr = P.load_results(), region_runs(region_dir)
    out = {}
    for m in MODELS:
        out[m] = {}
        for ax in AXES:
            out[m][ax] = {}
            for r in REG:
                per = {t: task_delta(Rp, Rr, m, ax, r, t) for t in T4}
                ok = {t: v for t, v in per.items() if v is not None}
                d = {t: p - c for t, (c, p) in ok.items()}
                out[m][ax][r] = {"per_task": {t: {"clean": c, "pert": p, "delta": p - c} for t, (c, p) in ok.items()},
                                 "n_tasks": len(ok),
                                 "delta": st.mean(d.values()) if len(ok) == 4 else None,
                                 "abs_delta": st.mean(abs(x) for x in d.values()) if len(ok) == 4 else None}
    return out


def fmt(x, tex=False):
    if x is None:
        return "–" if not tex else "--"
    s = f"{x:+.1f}"
    return s.replace("-", "$-$") if tex else s


def markdown(T):
    hdr = "| Model | " + " | ".join(f"{ANAME[a]} {RNAME[r]}" for a in AXES for r in REG) + " |"
    sep = "|---|" + "---:|" * (len(AXES) * len(REG))
    rows = [hdr, sep]
    for m in MODELS:
        cells = []
        for a in AXES:
            vals = {r: T[m][a][r]["delta"] for r in REG}
            # bold the region with the largest drop for this model / axis
            worst = min((v, r) for r, v in vals.items() if v is not None)[1] if any(v is not None for v in vals.values()) else None
            cells += [f"**{fmt(vals[r])}**" if r == worst else fmt(vals[r]) for r in REG]
        rows.append(f"| {MNAME[m]} | " + " | ".join(cells) + " |")
    per = ["", "Per-task Δ (pp), HellaSwag / ARC-E / ARC-C / PIQA:", "",
           "| Model | Axis | " + " | ".join(RNAME[r] for r in REG) + " |", "|---|---|---|---|---|"]
    for m in MODELS:
        for a in AXES:
            cs = []
            for r in REG:
                pt = T[m][a][r]["per_task"]
                cs.append(" / ".join(f"{pt[t]['delta']:+.1f}" if t in pt else "–" for t in T4))
            per.append(f"| {MNAME[m]} | {ANAME[a]} | " + " | ".join(cs) + " |")
    return "\n".join(rows + per) + "\n"


def latex(T):
    cols = "l" + "|rrr" * len(AXES)
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Robustness by perturbation region: mean $\Delta$Acc (pp, perturbed $-$ clean) over HellaSwag, "
         r"ARC-Easy, ARC-Challenge and PIQA (acc\_norm; PIQA acc). \textit{Context}: only the context is perturbed; "
         r"\textit{Options}: only the answer options; \textit{Both}: context and options (the setting of "
         r"Table~\ref{tab:main_13b}). $^{\dagger}$BLT and $^{\ddagger}$H-Net are external references.}",
         r"\label{tab:robust_region}", r"\begin{tabular}{" + cols + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{ANAME[a]}}}" for a in AXES) + r" \\",
         "Model & " + " & ".join(RNAME[r][:3] + "." if r != "both" else "Both" for a in AXES for r in REG) + r" \\",
         r"\midrule"]
    for m in MODELS:
        if m == "blt":
            L.append(r"\midrule")
        L.append(MTEX[m] + " & " + " & ".join(fmt(T[m][a][r]["delta"], True) for a in AXES for r in REG) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region_dir", default=f"{P.L}/runs/robustness_typoleet_region")
    ap.add_argument("--out", default=f"{P.L}/reports_afterNAACL/robust_region")
    a = ap.parse_args()
    T = build(a.region_dir)
    os.makedirs(a.out, exist_ok=True)
    json.dump(T, open(f"{a.out}/robust_region.json", "w"), indent=1)
    open(f"{a.out}/robust_region.md", "w").write(markdown(T))
    open(f"{a.out}/robust_region.tex", "w").write(latex(T))
    print(markdown(T))

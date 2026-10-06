#!/usr/bin/env python3
"""Appendix tables + figure for the patch-length statistics of the five main-table models.

The one place in the appendix that reports patch lengths (2026-10-06: replaces the separate
per-condition table (tables/patch_stats_conditions), the BLT-threshold patch table
(table_appendix/blt_threshold_patch) and the theta=1.61 histogram (figure_appendix/patch_len_hist_t161)).
BLT appears at both thresholds: released (theta=1.335) and compression-matched (theta=1.6094).

Reads reports/patch_stats_both4/hist_<model>.json (written by measure_{trio,blt,hnet}.py) and writes
  paper_overleaf/table_appendix/patch_stats.tex           mean +- std patch length per benchmark / condition
  paper_overleaf/table_appendix/patch_dist.tex            distribution summary per evaluation family
  reports/patch_stats_both4/patch_len_hist.{pdf,png}           patch-length distributions (not in the paper since 2026-10-06)
  reports/patch_stats_both4/summary.md                    the same numbers, for the repo

Because patches tile the text, the mean patch length (bytes / patches, pooled over all patches
of a benchmark) IS the compression ratio (bytes per global-model position); it is reported once.
Families that pool several benchmarks pool their histograms (byte-weighted).

  python scripts/probes/patch_stats/make_tables.py [--out_dir DIR]
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/ssd2/hyun2/AUNet")
IN = ROOT / "reports/patch_stats_both4"   # 2026-10-06: Noise/Typo/Despace/Leet = both-region variants, without BoolQ
OVL = ROOT / "paper_overleaf"
MODELS = ("llama", "aunet", "bpebyte", "blt", "blt161", "hnet")
FILE = {"blt": "hist_blt.json", "blt161": "hist_blt_t16094.json"}
EXTERNAL = ("blt", "blt161", "hnet")
NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte (ours)",
        "blt": r"BLT$^{\dagger}$ ($\theta{=}1.34$)", "blt161": r"BLT$^{\dagger}$ ($\theta{=}1.61$)",
        "hnet": r"H-Net$^{\ddagger}$"}
PLAIN = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte (ours)", "blt": "BLT (θ=1.34)",
         "blt161": "BLT (θ=1.61)", "hnet": "H-Net"}
ROB5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
FREE4 = ROB5[:4]                                   # free-text-answer tasks (every robustness row, Clean included)

# (block title, [(row label, [hist keys pooled into this row])])
ROWS = [
    ("Held-out pretraining text (BPB, latency)", [("DCLM windows", ["dclm/dclm"])]),
    ("Downstream (0-shot prompt + gold answer)", [
        ("HellaSwag", ["downstream/hellaswag"]), ("ARC-Easy", ["downstream/arc_easy"]),
        ("ARC-Challenge", ["downstream/arc_challenge"]), ("PIQA", ["downstream/piqa"]),
        ("WinoGrande", ["downstream/winogrande"]), ("BoolQ", ["downstream/boolq"]),
        ("MMLU", ["downstream/mmlu_text"])]),
    ("Retrieval (S-NIAH, 0.5k--4k bytes)", [
        ("S-NIAH-1", ["sniah/sniah1"]), ("S-NIAH-2", ["sniah/sniah2"]), ("S-NIAH-3", ["sniah/sniah3"])]),
    ("Robustness suite (HellaSwag, ARC-E, ARC-C, PIQA)", [
        (r"\textit{Clean} / \textit{PBP}", [f"downstream/{t}" for t in FREE4]),
        (r"\textit{Noise}", [f"noise/{t}" for t in FREE4]),
        (r"\textit{Typo}", [f"typo/{t}" for t in FREE4]),
        (r"\textit{Despace}", [f"despace/{t}" for t in FREE4]),
        (r"\textit{Leet}", [f"leet/{t}" for t in FREE4])]),
]
FAMILIES = [("DCLM", ["dclm/dclm"]),
            ("Downstream", [k for _, ks in ROWS[1][1] for k in ks]),
            ("S-NIAH", [k for _, ks in ROWS[2][1] for k in ks]),
            ("Clean", [f"downstream/{t}" for t in FREE4]),
            ("Noise", [f"noise/{t}" for t in FREE4]),
            ("Typo", [f"typo/{t}" for t in FREE4]),
            ("Despace", [f"despace/{t}" for t in FREE4]),
            ("Leet", [f"leet/{t}" for t in FREE4])]
ROBUST_FAM = ("Clean", "Noise", "Typo", "Despace", "Leet")   # italicised: robustness-suite conditions


def load():
    H = {}
    for m in MODELS:
        p = IN / FILE.get(m, f"hist_{m}.json")
        if p.exists():
            H[m] = json.load(open(p))["benches"]
    return H


def pooled(bench, keys):
    """-> (lengths array, counts array) pooled over keys."""
    c = {}
    for k in keys:
        for L, n in bench[k]["hist"].items():
            c[int(L)] = c.get(int(L), 0) + n
    Ls = np.array(sorted(c)); ns = np.array([c[L] for L in Ls])
    return Ls, ns


def stats(Ls, ns):
    N, B = ns.sum(), (Ls * ns).sum()
    mean = B / N
    std = np.sqrt((ns * (Ls - mean) ** 2).sum() / N)
    cdf = np.cumsum(ns) / N
    q = lambda p: int(Ls[np.searchsorted(cdf, p)])
    return {"mean": mean, "std": std, "median": q(0.5), "p90": q(0.9), "p95": q(0.95), "p99": q(0.99),
            "max": int(Ls.max()), "len1": ns[Ls == 1].sum() / N, "long_bytes": (Ls * ns)[Ls > 16].sum() / B,
            "n_patches": int(N), "bytes": int(B)}


def fam_label(f):
    return r"\textit{" + f + "}" if f in ROBUST_FAM else f


def table_means(H):
    ms = [m for m in MODELS if m in H]
    blt = [m for m in ms if m.startswith("blt")]
    # two-row header: BLT spans its two thresholds
    h1, h2, rule = [r"\textbf{Benchmark}"], [""], ""
    for i, m in enumerate(ms, start=2):
        if m == "blt161":
            continue
        if m == "blt" and len(blt) == 2:
            h1.append(r"\multicolumn{2}{c}{\textbf{BLT$^{\dagger}$}}")
            h2 += [r"$\theta{=}1.34$", r"$\theta{=}1.61$"]
            rule = r"\cmidrule(lr){" + f"{i}-{i + 1}" + "}"
        else:
            h1.append(r"\textbf{" + NAME[m].replace(" (ours)", "") + "}")
            h2.append("")
    out = [r"% GENERATED by AUNet/scripts/probes/patch_stats/make_tables.py — edit the script, not this file.",
           r"\begin{table*}[t]", r"\centering",
           r"\caption{\hyun{Mean patch length in bytes ($\pm$ standard deviation over patches) on the exact byte "
           r"sequences each evaluation scores (context followed by the gold answer; BOS excluded). Because "
           r"patches tile the input, the mean patch length equals the compression ratio $T/M$, i.e.\ the number of "
           r"input bytes per position of the global model; for the subword Transformer the unit is a Llama-3 "
           r"token. Patches come from each model's own parser: Llama-3 BPE tokens, AUNet's whitespace/word "
           r"regex, BPEByte's OnlineBPE, H-Net's learned router (1-stage XL), and BLT's entropy patcher with its "
           r"trained $512$-byte attention window, at the released threshold ($\theta{=}1.34$) and at the threshold "
           r"calibrated to the matched models' patch length ($\theta{=}1.61$; Appendix~\ref{app:blt_threshold}). "
           r"The robustness rows are the robustness suite of Table~\ref{tab:main_robust} (Appendix~\ref{app:protocol}), "
           r"pooled over its four tasks with free-text answers; \textit{Noise}, \textit{Typo}, \textit{Despace}, and "
           r"\textit{Leet} each perturb both the context and every answer option, and \textit{PBP} is byte-identical "
           r"to \textit{Clean}, so it yields identical patches for every model. "
           r"Table~\ref{tab:patch_dist} summarizes the full distributions. "
           r"$^{\dagger}$BLT and $^{\ddagger}$H-Net are external references.}}",
           r"\label{tab:patch_stats}",
           r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\fitcolumn[\textwidth]{%",
           r"\begin{tabular}{l" + "c" * len(ms) + "}", r"\toprule",
           " & ".join(h1) + r" \\"]
    if rule:
        out += [rule, " & ".join(h2) + r" \\"]
    for title, rows in ROWS:
        out += [r"\midrule", r"\multicolumn{" + str(len(ms) + 1) + r"}{l}{\emph{" + title + r"}} \\"]
        for label, keys in rows:
            cells = []
            for m in ms:
                s = stats(*pooled(H[m], keys))
                cells.append(f"{s['mean']:.2f} {{\\scriptsize$\\pm${s['std']:.2f}}}")
            out.append(r"\quad " + label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(out)


# (block title, cell formatter)
DIST_BLOCKS = [
    ("P95 / P99 (bytes)", lambda s: f"{s['p95']} / {s['p99']}"),
    ("Max (bytes)", lambda s: f"{s['max']:,}".replace(",", "{,}")),
    ("1-byte patches (\\%)", lambda s: f"{100 * s['len1']:.2f}"),
    ("Bytes in patches $>$16\\,B (\\%)", lambda s: f"{100 * s['long_bytes']:.2f}"),
]


def table_dist(H):
    ms = [m for m in MODELS if m in H]
    nc = len(FAMILIES) + 1
    out = [r"% GENERATED by AUNet/scripts/probes/patch_stats/make_tables.py — edit the script, not this file.",
           r"\begin{table*}[t]", r"\centering",
           r"\caption{\hyun{Patch-length distribution (bytes) per evaluation family, pooled over the benchmarks of "
           r"Table~\ref{tab:patch_stats} (\textit{Clean} through \textit{Leet}: the four free-text tasks of the "
           r"robustness suite). P95/P99 are percentiles over patches; \emph{1-byte patches} is the share of patches "
           r"that hold a single byte (no compression at that position), and \emph{Bytes in patches $>$16\,B} is the "
           r"share of input bytes that land in patches longer than $16$ bytes (information pooled into one global "
           r"position). "
           r"$^{\dagger}$BLT and $^{\ddagger}$H-Net are external references.}}",
           r"\label{tab:patch_dist}",
           r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\fitcolumn[\textwidth]{%",
           r"\begin{tabular}{l" + "r" * len(FAMILIES) + "}", r"\toprule",
           r"\textbf{Model} & " + " & ".join(r"\textbf{" + fam_label(f) + "}" for f, _ in FAMILIES) + r" \\"]
    S = {m: [stats(*pooled(H[m], keys)) for _, keys in FAMILIES] for m in ms}
    for title, fmt in DIST_BLOCKS:
        out += [r"\midrule", r"\multicolumn{" + str(nc) + r"}{l}{\emph{" + title + r"}} \\"]
        for m in ms:
            cells = [fmt(s) for s in S[m]]
            if m in EXTERNAL:
                out.append(r"\quad \ext{" + NAME[m] + "} & " + " & ".join(r"\ext{" + c + "}" for c in cells) + r" \\")
            else:
                out.append(r"\quad " + NAME[m] + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    return "\n".join(out)


def figure(H, path, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # model colors/styles follow the radar figure (reports/parse/star_draw.py); dashed/dotted = external
    col = {"llama": "#0072B2", "aunet": "#E69F00", "bpebyte": "#009E73", "blt": "#CC79A7", "blt161": "#882255",
           "hnet": "#56B4E9"}
    ls = {"llama": "-", "aunet": "-", "bpebyte": "-", "blt": "--", "blt161": ":", "hnet": "--"}
    mk = {"llama": "o", "aunet": "s", "bpebyte": "D", "blt": "^", "blt161": "<", "hnet": "v"}
    CAP = 24                                       # lengths >= CAP fold into the last bin
    ms = [m for m in MODELS if m in H]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6), sharex=True, sharey=True)
    for ax, (fam, keys) in zip(axes.flat, FAMILIES):
        for m in ms:
            Ls, ns = pooled(H[m], keys)
            y = np.zeros(CAP)
            for L, n in zip(Ls, ns):
                y[min(L, CAP) - 1] += n
            y = y / ns.sum()
            x = np.arange(1, CAP + 1)
            keep = y > 0
            ax.plot(x[keep], y[keep], ls[m], color=col[m], lw=1.6, marker=mk[m], ms=3.2,
                    label=PLAIN[m], zorder=3 if m == "bpebyte" else 2)
        ax.set_title(fam, fontsize=11, style="italic" if fam in ROBUST_FAM else "normal")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.2)
        ax.set_xticks([1, 4, 8, 12, 16, 20, CAP])
        ax.set_xticklabels(["1", "4", "8", "12", "16", "20", f"{CAP}+"])
        ax.grid(color="#e1e0d9", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Patch length (bytes)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of patches")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ms), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    return path


def summary_md(H, path):
    ms = [m for m in MODELS if m in H]
    lines = ["# Patch statistics per benchmark (five main-table models; BLT at two thresholds)", "",
             "Generated by `scripts/probes/patch_stats/make_tables.py` from `hist_<model>.json`.",
             "Mean patch length = compression ratio (bytes per global position); std over patches.", "",
             "| row | " + " | ".join(PLAIN[m] for m in ms) + " |", "|---" * (len(ms) + 1) + "|"]
    for title, rows in ROWS:
        for label, keys in rows:
            cells = []
            for m in ms:
                s = stats(*pooled(H[m], keys))
                cells.append(f"{s['mean']:.2f} ± {s['std']:.2f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| family | model | median | p90 | p95 | p99 | max | 1-byte % | bytes in >16B % |",
              "|---" * 9 + "|"]
    for fam, keys in FAMILIES:
        for m in ms:
            s = stats(*pooled(H[m], keys))
            lines.append(f"| {fam} | {PLAIN[m]} | {s['median']} | {s['p90']} | {s['p95']} | {s['p99']} | {s['max']} | "
                         f"{100 * s['len1']:.2f} | {100 * s['long_bytes']:.2f} |")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=None, help="write all outputs here instead of paper_overleaf / IN")
    a = ap.parse_args()
    H = load()
    missing = [m for m in MODELS if m not in H]
    if missing:
        print("WARNING: no histogram yet for", missing)
    od = Path(a.out_dir) if a.out_dir else None
    if od:
        od.mkdir(parents=True, exist_ok=True)
    (od / "patch_stats.tex" if od else OVL / "table_appendix/patch_stats.tex").write_text(table_means(H))
    (od / "patch_dist.tex" if od else OVL / "table_appendix/patch_dist.tex").write_text(table_dist(H))
    figure(H, (od or IN) / "patch_len_hist.pdf",
           (od or IN) / "patch_len_hist.png")
    md = (od or IN) / "summary.md"
    summary_md(H, md)
    print(md.read_text())


if __name__ == "__main__":
    main()

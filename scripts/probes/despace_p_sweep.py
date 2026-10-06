"""Accuracy vs despace probability p under the current robustness protocol (up to 2,000 items, seed 1234), regions
ctx/ans/all, averaged over the four tasks with free-text answers (BoolQ excluded since 2026-10-06, as for the
main-table Despace axis: its yes/no options have no spaces to remove). Metric = acc_norm on HellaSwag/ARC and acc on PIQA/BoolQ, as for the Despace axis of tab:main_13b /
tab:robustness_category (scripts/probes/paper_robustness_tables.py).

Sources (later entries override earlier ones, so clean and p=1.0 are exactly the Table 3 values)
  matched trio : runs/robustness_paper1p3b{,_ext}/<model>/results.json   p=0.1/0.4/0.7 (+clean, p=1.0)
                 runs/robustness_despace_bits/<model>/results.json      clean, p=1.0  (Table 3 source)
  BLT / H-Net  : reports/despace_p_sweep/graded/*.json                   p=0.1/0.4/0.7 (+clean; snu234)
                 reports/ext_ci/{blt_official,hnet}/[t1609_]despace.json  clean, p=1.0  (Table 3 source)
The two runs of the same checkpoints/items/seed agree within 0.16 pt (macro) on clean and p=1.0.
Writes reports/despace_p_sweep/{sweep.json,sweep.md} and, with --fig, the appendix figure.
"""
import json, glob
from pathlib import Path
L = Path("/mnt/ssd2/hyun2/AUNet"); OUT = L / "reports/despace_p_sweep"; OUT.mkdir(parents=True, exist_ok=True)
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa")
MET = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "piqa": "acc", "boolq": "acc"}
PS = (0.0, 0.1, 0.4, 0.7, 1.0)
VAR = {r: {0.0: "clean", 0.1: f"{r}10", 0.4: f"{r}40", 0.7: f"{r}70",
           1.0: {"ctx": "despace", "ans": "despaceans", "all": "despaceall"}[r]} for r in ("ctx", "ans", "all")}


def cells(paths):
    out = {}
    for p in paths:
        d = json.load(open(p)); res = d.get("results", d)
        for k, v in res.items():
            if k.startswith("despace_mc_") and isinstance(v, dict) and "acc" in v:
                out[k[len("despace_mc_"):]] = v
    return out


def macro(c, variant):
    xs = []
    for t in TASKS:
        k = f"{t}_{variant}"
        if k not in c: return None
        xs.append(c[k][MET[t]])
    return 100 * sum(xs) / len(xs)


SRC = {m: [L / f"runs/{d}/{a}/results.json" for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext", "robustness_despace_bits")]
       for m, a in (("Transformer", "llama"), ("AUNet", "aunet"), ("BPEByte", "bpebyte"))}
G = OUT / "graded"
SRC["BLT (θ=1.34)"] = sorted(G.glob("blt1335_*.json")) + [L / "reports/ext_ci/blt_official/despace.json"]
SRC["BLT (θ=1.61)"] = sorted(G.glob("blt1609_*.json")) + [L / "reports/ext_ci/blt_official/t1609_despace.json"]
SRC["H-Net"] = sorted(G.glob("hnet_*.json")) + [L / "reports/ext_ci/hnet/despace.json"]

res = {}
for m, paths in SRC.items():
    c = cells([p for p in paths if p.exists()])
    res[m] = {r: {str(p): macro(c, VAR[r][p]) for p in PS} for r in ("ctx", "ans", "all")}
json.dump(res, open(OUT / "sweep.json", "w"), indent=1)
lines = ["# Despace accuracy vs p (macro over 4 tasks, BoolQ excluded, %)", ""]
for r in ("ctx", "ans", "all"):
    lines += [f"## {r}", "", "| model | " + " | ".join(f"p={p}" for p in PS) + " | drop at p=1 |", "|---|" + "---:|" * (len(PS) + 1)]
    for m, v in res.items():
        row = v[r]; c0, c1 = row["0.0"], row["1.0"]
        lines.append(f"| {m} | " + " | ".join("—" if row[str(p)] is None else f"{row[str(p)]:.1f}" for p in PS)
                     + f" | {'—' if c0 is None or c1 is None else f'{c0 - c1:.1f}'} |")
    lines.append("")
(OUT / "sweep.md").write_text("\n".join(lines)); print("\n".join(lines))


def figure(res, out_pdf, out_png=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    STY = {"Transformer": ("#52514e", "-", "o"), "AUNet": ("#eb6834", "-", "o"), "BPEByte": ("#2a78d6", "-", "o"),
           "BLT (θ=1.34)": ("#1baf7a", "--", "s"), "BLT (θ=1.61)": ("#4a3aa7", "--", "s"), "H-Net": ("#e87ba4", "--", "^")}
    LAB = {"BPEByte": "BPEByte (ours)", "BLT (θ=1.34)": "BLT† (θ=1.34)", "BLT (θ=1.61)": "BLT† (θ=1.61)", "H-Net": "H-Net‡"}
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    # 2026-10-06: the paper keeps only the whole-input setting (context and answer options both despaced),
    # the Despace axis of tab:main_robust; per-region curves stay in sweep.md.
    fig, ax = plt.subplots(1, 1, figsize=(3.1, 2.35))
    r = "all"
    for m, (c, ls, mk) in STY.items():
        pts = [(p, res[m][r][str(p)]) for p in PS if res[m][r][str(p)] is not None]
        ax.plot([x for x, _ in pts], [y for _, y in pts], ls=ls, color=c, marker=mk, ms=3.2, lw=1.4,
                label=LAB.get(m, m), markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xticks(PS); ax.set_xticklabels(["0", ".1", ".4", ".7", "1"])
    ax.set_xlabel("space-removal probability $p$", fontsize=7.5, labelpad=1)
    ax.grid(axis="y", color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.set_ylabel("accuracy (%, 4-task mean)", fontsize=7.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.01),
               handlelength=2.2, columnspacing=0.9)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.83))
    fig.savefig(out_pdf)
    if out_png: fig.savefig(out_png, dpi=200)


if __name__ == "__main__":
    import sys
    if "--fig" in sys.argv:
        figure(res, sys.argv[sys.argv.index("--fig") + 1], OUT / "despace_sweep.png")

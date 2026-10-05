#!/usr/bin/env python3
"""S-NIAH exact recall as length x needle-depth heatmaps (final set: 0.5k/1k/2k n=250 + fit-4k cell).

Rows: models; columns: S-NIAH-1/2/3. Each panel: x = context length cell, y = needle depth decile
(25 items per cell). Per-item sources:
  matched trio   reports/niah/final/{llama,aunet,bpebyte}_items.jsonl  (score_probe_pairs --items_out)
  H-Net          reports/niah/final/hnet.json (run_ext per_item, pairs-file order)
  BLT            reports/ext_ci/blt_official/{,t1609_}sniah.json for 0.5k-2k  +
                 reports/niah/fit4k/blt_t{1335,1609}.json for 4k (per_item, pairs-file order)
Depth of item i in a cell = (i + 0.5) / 250 (dump order).
Repeated-haystack variants S-NIAH-1/2/3-rep (period 64, scripts/niah/dump_sniah_rep_pairs.py; cells "sniah{k}r/<len>")
come from reports/niah/rep64/scored/<tag>.jsonl (score_items.py output) and are merged in when present.

  python scripts/niah/plot_sniah_heatmap.py [--blt161]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = Path("/mnt/ssd2/hyun2/AUNet")
LENS = (512, 1024, 2048, 4096)
TASKS = ("sniah1", "sniah2", "sniah3")
REP = ("sniah1r", "sniah2r", "sniah3r")
TNAME = {"sniah1": "S-NIAH-1 (repeated noise)", "sniah2": "S-NIAH-2 (essay)", "sniah3": "S-NIAH-3 (essay, UUID)",
         "sniah1r": "S-NIAH-1-rep (noise, 64 B repeated)", "sniah2r": "S-NIAH-2-rep (essay, 64 B repeated)",
         "sniah3r": "S-NIAH-3-rep (UUID, 64 B repeated)"}
REP_TAG = {"Transformer": "llama", "AUNet": "aunet", "BPEByte (ours)": "bpebyte", "BLT ($\\theta$=1.34)": "blt",
           "BLT ($\\theta$=1.61)": "blt161", "H-Net": "hnet"}
NB = 10


def from_per_item(per):
    """{cell: [0/1 in dump order]} -> {cell: [(depth, ok)]}"""
    return {c: [((i + 0.5) / len(v), int(x)) for i, x in enumerate(v)] for c, v in per.items()}


def load():
    M = {}
    for tag, name in (("llama", "Transformer"), ("aunet", "AUNet"), ("bpebyte", "BPEByte (ours)")):
        d = defaultdict(list)
        for l in open(L / f"reports/niah/final/{tag}_items.jsonl"):
            r = json.loads(l)
            d[r["cell"]].append((r["depth"], r["exact"]))
        M[name] = d
    for pre, fit, name in (("", "blt_t1335", "BLT ($\\theta$=1.34)"), ("t1609_", "blt_t1609", "BLT ($\\theta$=1.61)")):
        d = from_per_item(json.load(open(L / f"reports/ext_ci/blt_official/{pre}sniah.json"))["per_item"])
        d.update(from_per_item(json.load(open(L / f"reports/niah/fit4k/{fit}.json"))["per_item"]))
        M[name] = d
    M["H-Net"] = from_per_item(json.load(open(L / "reports/niah/final/hnet.json"))["per_item"])
    for name, tag in REP_TAG.items():
        f = L / f"reports/niah/rep64/scored/{tag}.jsonl"
        if f.exists():
            for l in open(f):
                r = json.loads(l)
                M[name].setdefault(r["cond"], []).append((r["depth"], r["exact"]))
    return M


def grid(d, task):
    G = np.full((NB, len(LENS)), np.nan)
    for j, Ln in enumerate(LENS):
        rows = d.get(f"{task}/{Ln}", [])
        for b in range(NB):
            v = [ok for dep, ok in rows if min(NB - 1, int(dep * NB)) == b]
            if v:
                G[b, j] = np.mean(v)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blt161", action="store_true", help="add the theta=1.61 BLT row")
    ap.add_argument("--rep", action="store_true", help="add the repeated-haystack columns (S-NIAH-k-rep)")
    ap.add_argument("--rep_only", action="store_true", help="only the repeated-haystack columns (same layout as the original)")
    ap.add_argument("--out", default=str(L / "reports/niah/final/sniah_heatmap"))
    a = ap.parse_args()
    M = load()
    tasks = list(REP) if a.rep_only else [t for k in TASKS for t in ((k, k + "r") if a.rep else (k,))]
    order = ["Transformer", "AUNet", "BPEByte (ours)", "BLT ($\\theta$=1.34)"] + \
            (["BLT ($\\theta$=1.61)"] if a.blt161 else []) + ["H-Net"]
    fig, axes = plt.subplots(len(order), len(tasks), figsize=(3.2 * len(tasks), 2.05 * len(order)), sharex=True, sharey=True)
    cmap = plt.get_cmap("Blues")
    for r, name in enumerate(order):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            G = grid(M[name], task)
            ax.imshow(G, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")
            for i in range(NB):
                for j in range(len(LENS)):
                    v = G[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center",
                                fontsize=6.5, color="white" if v > 0.6 else "#1f1f1f")
            ax.set_xticks(range(len(LENS)), ["0.5k", "1k", "2k", "4k"], fontsize=8)
            ax.set_yticks([0, NB - 1], ["0 (start)", "1 (end)"], fontsize=7)
            if r == 0:
                ax.set_title(TNAME[task], fontsize=9)
            if c == 0:
                ax.set_ylabel(name + "\nneedle depth", fontsize=8)
            for s in ax.spines.values():
                s.set_visible(False)
    fig.supxlabel("context length (bytes; the 4k cell counts BOS + prompt + answer)", fontsize=9, y=0.06)
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("exact recall", fontsize=8)
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight")
    print("wrote", a.out + ".{pdf,png}")
    for name in order:
        print(name, {t: round(float(np.nanmean([np.mean([ok for _, ok in M[name][f'{t}/{Ln}']]) for Ln in LENS])), 3) for t in tasks})


if __name__ == "__main__":
    main()

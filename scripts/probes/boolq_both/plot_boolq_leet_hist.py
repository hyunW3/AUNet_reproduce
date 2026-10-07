"""Appendix figure: BLT (released threshold, 512-byte entropy window) vs BPEByte patch-length distributions on the 2,000
BoolQ items, clean (reports/patch_stats downstream/boolq) and under the context-only Leet of the BoolQ appendix table
(reports/patch_stats_conditions/leet leet/boolq: format_mc nla_leet, seed 0, context + gold label). y = share of input
bytes per length bin; each panel lists mean / P95 patch length and patches per item. Style of plot_blt_hist.py.
  python plot_boolq_leet_hist.py --out paper_overleaf/figure_appendix/src/boolq_leet_hist.pdf"""
import argparse, json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

L = Path("/mnt/ssd2/hyun2/AUNet")
EDGES = [1, 2, 4, 8, 16, 64]
LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-64", ">64"]
MODELS = [("blt", "BLT", "#c2185b"), ("bpebyte", "BPEByte", "#2a78d6")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
W_IN = 7.7 / 2.54


def bench(path, key):
    b = json.load(open(path))["benches"][key]
    return {int(k): v for k, v in b["hist"].items()}, b["n_items"]


def byte_share(h):
    T = sum(l * c for l, c in h.items())
    out = np.zeros(len(LABELS))
    for l, c in h.items():
        out[next((k for k, e in enumerate(EDGES) if l <= e), len(EDGES))] += l * c
    return out / T * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panels = [("BoolQ, clean", {m: bench(L / f"reports/patch_stats/hist_{m}.json", "downstream/boolq") for m, *_ in MODELS}),
              ("BoolQ, Leet (context)", {m: bench(L / f"reports/patch_stats_conditions/leet/hist_{m}.json", "leet/boolq")
                                         for m, *_ in MODELS})]
    plt.rcParams.update({"font.size": 5.0, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.edgecolor": INK2, "axes.linewidth": 0.4, "xtick.color": INK2, "ytick.color": INK2,
                         "xtick.major.width": 0.4, "ytick.major.width": 0.4, "axes.labelcolor": INK})
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 1.25), sharey=True)
    x = np.arange(len(LABELS))
    bw = 0.42
    for ax, (pname, H) in zip(axs, panels):
        ax.text(0.98, 0.97, "mean / P95 / patches per item", transform=ax.transAxes, ha="right", va="top",
                fontsize=4.3, color=INK2)
        for k, (m, mname, col) in enumerate(MODELS):
            h, n = H[m]
            ax.bar(x + (k - 0.5) * bw, byte_share(h), width=bw * 0.95, color=col, linewidth=0, label=mname)
            Ls = np.array(sorted(h))
            C = np.array([h[l] for l in Ls])
            cdf = np.cumsum(C) / C.sum()
            mu, p95 = (Ls * C).sum() / C.sum(), int(Ls[np.searchsorted(cdf, .95)])
            ax.text(0.98, 0.97 - 0.12 * (k + 1), f"{mname} {mu:.2f} / {p95} / {C.sum() / n:.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=4.3, color=col, fontweight="bold")
        ax.set_title(pname, fontsize=5.8, fontweight="bold", color=INK, pad=1.5)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=0.3)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=1.2, pad=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=4.3)
        ax.set_xlim(-0.6, len(LABELS) - 0.4)
    axs[0].set_ylabel("% bytes", fontsize=5.0, labelpad=1)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.72), frameon=False, fontsize=4.8, handlelength=0.9,
                  handletextpad=0.3)
    fig.supxlabel("patch length (bytes)", fontsize=5.0, y=0.0)
    fig.tight_layout(pad=0.1, w_pad=0.4, rect=(0, 0.03, 1, 1))
    fig.savefig(a.out)
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=300)
    print(a.out)


if __name__ == "__main__":
    main()

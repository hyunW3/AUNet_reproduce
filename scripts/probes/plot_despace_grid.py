#!/usr/bin/env python3
"""Despace FULL-grid region plot: 1x3 panels (ctx / ans / all), acc vs space-removal prob.

Reads reports/statetrack/despace_grid/{llama,aunet,bpebyte}/results.json (the single
sweep with clean + ctx/ans/all x {10,20,40,70} + despace/despaceans/despaceall = 100%)
and renders one line panel per despaced region, shared y, x = p in {0,10,20,40,70,100}.

Outputs:
  reports/despace/despace_grid_1x3.png       light, with suptitle (report/markdown)
  reports/despace/despace_grid_1x3_dark.png  dark-surface variant (ThingsToMake.html embed)
  reports/despace/despace_grid_1x3.pdf       light vector, no suptitle, paper-sized fonts
  reports/despace/despace_all_1col.pdf       single "all" panel, single-column paper figure
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/ssd2/hyun2/AUNet"
GRID = f"{ROOT}/reports/statetrack/despace_grid"
OUT = f"{ROOT}/reports/despace"
MODELS = [("llama", "Llama"), ("aunet", "AU-Net"), ("bpebyte", "BPEByte"), ("blt", "BLT-1B"), ("hnet1", "H-Net")]
# light = repo model convention (report_despace.py); dark = same hues re-stepped for the
# dark report surface (#171a21) — both orderings validated (CVD + contrast) 2026-07-23.
# blt = external BLT-1B reference (pink, matches the trade-off figure).
COL_LIGHT = {"llama": "#0072B2", "aunet": "#E69F00", "bpebyte": "#009E73", "blt": "#CC79A7", "hnet1": "#56B4E9"}
COL_DARK = {"llama": "#3A8DC9", "aunet": "#C98500", "bpebyte": "#009E73", "blt": "#D98CB3", "hnet1": "#56B4E9"}
MARK = {"llama": "o", "aunet": "s", "bpebyte": "^", "blt": "D", "hnet1": "v"}
PROBS = [0, 10, 20, 40, 70, 100]
REGION_100 = {"ctx": "despace", "ans": "despaceans", "all": "despaceall"}
PANELS = [("ctx", "context only"), ("ans", "answer/choices only"), ("all", "context + answer")]

THEME = {
    "light": dict(col=COL_LIGHT, surface="white", ink="#0b0b0b", muted="#898781", grid="#e1e0d9"),
    "dark": dict(col=COL_DARK, surface="#171a21", ink="#e7eaf0", muted="#9aa4b2", grid="#2a2f3a"),
}


def acc(res, variant):
    r = res.get(f"despace_mc_avg_{variant}")
    return r["acc"] if r else float("nan")


def series(res, region):
    return ([acc(res, "clean")]
            + [acc(res, f"{region}{p:02d}") for p in (10, 20, 40, 70)]
            + [acc(res, REGION_100[region])])


def render(data, path, mode="light", suptitle=True, fs=1.0):
    t = THEME[mode]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    fig.patch.set_facecolor(t["surface"])
    for ax, (region, subtitle) in zip(axes, PANELS):
        ax.set_facecolor(t["surface"])
        for tag, lb in MODELS:
            ax.plot(PROBS, series(data[tag], region), MARK[tag] + "-",
                    color=t["col"][tag], lw=2, ms=6.5, label=lb, zorder=3)
        # direct end labels (contrast relief), dodged where lines converge
        ends = sorted(((series(data[tg], region)[-1], tg, lb) for tg, lb in MODELS), reverse=True)
        min_sep, ypos = 0.021 * fs, []
        for y, _, _ in ends:
            if ypos and ypos[-1] - y < min_sep:
                y = ypos[-1] - min_sep
            ypos.append(y)
        for (yv, tag, lb), y in zip(ends, ypos):
            ax.annotate(lb, (100, yv), xytext=(104, y), fontsize=8.5 * fs,
                        color=t["col"][tag], va="center", annotation_clip=False)
        ax.set_title(f"{region} — {subtitle}", fontsize=10.5 * fs, color=t["ink"])
        ax.set_xticks(PROBS)
        ax.set_xlim(-4, 104)
        ax.set_ylim(0.32, 0.595)
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.tick_params(colors=t["muted"], labelsize=8.5 * fs)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["muted"])
        ax.set_xlabel("space-removal probability p (%)", fontsize=9 * fs, color=t["muted"])
    axes[0].set_ylabel("accuracy (macro-avg, 5 MC tasks)", fontsize=9 * fs, color=t["muted"])
    leg = axes[0].legend(fontsize=8.5 * fs, loc="lower left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    if suptitle:
        fig.suptitle("Graded despace robustness by region — 1.3B, n=500/task "
                     "(HS · ARC-E · ARC-C · PIQA · WinoGrande)", fontsize=11.5 * fs, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 0.985, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print("wrote", path)


def render_single(data, path, region="all", mode="light", fs=1.0, models=None):
    """One panel (default the joint ctx+ans sweep), sized for a single column."""
    t = THEME[mode]
    models = models or MODELS
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for tag, lb in models:
        ax.plot(PROBS, series(data[tag], region), MARK[tag] + "-",
                color=t["col"][tag], lw=2, ms=6.5, label=lb, zorder=3)
    ends = sorted(((series(data[tg], region)[-1], tg, lb) for tg, lb in models), reverse=True)
    min_sep, ypos = 0.014 * fs, []
    for y, _, _ in ends:
        if ypos and ypos[-1] - y < min_sep:
            y = ypos[-1] - min_sep
        ypos.append(y)
    for (yv, tag, lb), y in zip(ends, ypos):
        ax.annotate(lb, (100, yv), xytext=(103, y), fontsize=8.5 * fs,
                    color=t["col"][tag], va="center", annotation_clip=False)
    ax.set_xticks(PROBS)
    ax.set_xlim(-4, 104)
    ax.set_ylim(0.32, 0.595)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.tick_params(colors=t["muted"], labelsize=8.5 * fs)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["muted"])
    ax.set_xlabel("space-removal probability p (%)", fontsize=9 * fs, color=t["muted"])
    ax.set_ylabel("accuracy (macro-avg, 5 MC tasks)", fontsize=9 * fs, color=t["muted"])
    leg = ax.legend(fontsize=8.5 * fs, loc="lower left", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    print("wrote", path)


def main():
    data = {tg: json.load(open(f"{GRID}/{tg}/results.json"))["results"] for tg, _ in MODELS}
    Path(OUT).mkdir(parents=True, exist_ok=True)
    render(data, f"{OUT}/despace_grid_1x3.png", "light", suptitle=True)
    render(data, f"{OUT}/despace_grid_1x3_dark.png", "dark", suptitle=True)
    render(data, f"{OUT}/despace_grid_1x3.pdf", "light", suptitle=False, fs=1.35)
    render_single(data, f"{OUT}/despace_all_1col.pdf", "all", "light", fs=1.15, models=MODELS[:3])


if __name__ == "__main__":
    main()

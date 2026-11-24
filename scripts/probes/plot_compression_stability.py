#!/usr/bin/env python3
"""Figure + tables for the compression-stability sweep.

Reads reports/compression_stability/measurements.jsonl (appended by
compression_stability.py) and emits:

  reports/compression_stability/compression_stability.{png,pdf}
  reports/compression_stability/summary.md
  paper_overleaf/table_appendix/compression_stability.tex

Two panels, because the sweep has two different x-axes and cramming them into one
categorical axis would misread as a single ordered continuum:
  (a) the despace continuum  -- x is a real number p, so it is a line chart
  (b) discrete domains       -- unordered categories, so grouped bars

Values are never hardcoded here (contrast reports/parse/tradeoff_draw.py, flagged in
paper_results/03_consistency_issues.md #16): edit measurements.jsonl and re-run.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports/compression_stability/measurements.jsonl"
OUT = ROOT / "reports/compression_stability"
TEX = ROOT / "paper_overleaf/table_appendix/compression_stability.tex"

# Okabe-Ito, in an order validated by the dataviz palette checker
# (worst adjacent CVD dE 11.0 deutan, worst normal-vision dE 24.2 -- both PASS).
# First three keep the existing repo convention (viz_tradeoff.py, report_despace.py).
COL = {"Llama": "#0072B2", "AU-Net": "#E69F00", "BPEByte": "#009E73",
       "BLT": "#D55E00", "H-Net": "#56B4E9"}
MARK = {"Llama": "o", "AU-Net": "s", "BPEByte": "^", "BLT": "D", "H-Net": "v"}

# measurements.jsonl parser label -> display name. BLT arms are selected by tag.
DISPLAY = {"llama": "Llama", "aunet": "AU-Net", "bpebyte": "BPEByte",
           "blt_design4.5": "BLT", "hnet": "H-Net"}
AUG = "bpebyte_augTe"                       # Telugu vocab-augment arm, drawn as an overlay
ORDER = ["Llama", "AU-Net", "BPEByte", "BLT", "H-Net"]

DESPACE = [("en", 0.0), ("despace_p025", 0.25), ("despace_p050", 0.50),
           ("despace_p075", 0.75), ("despace_p100", 1.0)]
DISCRETE = [("en", "English"), ("code", "Code"), ("zh", "Chinese"),
            ("te", "Telugu"), ("random", "Random")]
BAND = (2.0, 8.0)     # "usable hierarchy" band: below ~2 the hierarchy stops compressing,
                      # above ~8 a single vector must carry a whole clause


# H-Net cannot be re-measured on this host: its harness needs the `hnet` package +
# mamba-ssm + the released weights, and code_appendix/eval_suite/paths.py ships anonymised
# placeholder paths (the real env lives on ece-agpu18 -- paper_results/sources/README.md #3).
# The two published cells are carried over so the arm is visible rather than silently absent;
# every other H-Net domain is REPORTED AS MISSING, never quietly dropped.
PUBLISHED = {
    ("hnet", "en"): {"cr": 4.2, "std": 2.7},
    ("hnet", "zh"): {"cr": 1.1, "std": 0.4},
}
PUBLISHED_SRC = "tab:zh_stats (published; not re-measured on this host)"


def load() -> dict[tuple[str, str], dict]:
    """(parser_label, domain) -> row; later rows win, since the file is append-only."""
    rows: dict[tuple[str, str], dict] = {}
    for line in open(SRC):
        r = json.loads(line)
        r.setdefault("provenance", "measured")
        rows[(r["parser"], r["domain"])] = r
    for (label, dom), v in PUBLISHED.items():
        rows.setdefault((label, dom), dict(
            parser=label, domain=dom, provenance=PUBLISHED_SRC,
            cr=v["cr"], std=v["std"], max=float("nan"), p99=float("nan"),
            frac_len1=float("nan"), frac_bytes_in_long=float("nan"),
            n_windows=None, total_bytes=None, n_patches=None))
    return rows


def missing_report(rows) -> list[str]:
    """Explicitly name every (parser, domain) the sweep did not cover."""
    gaps = []
    for disp in ORDER:
        label = label_of(disp)
        for dom in ("en", "code", "zh", "te", "random", "despace_p025", "despace_p050",
                    "despace_p075", "despace_p100"):
            r = rows.get((label, dom))
            if r is None:
                gaps.append(f"{disp} / {dom}: NOT MEASURED")
            elif r.get("provenance") != "measured":
                gaps.append(f"{disp} / {dom}: {r['provenance']}")
    return gaps


def get(rows, label, domain):
    return rows.get((label, domain))


def label_of(disp: str) -> str | None:
    return next((k for k, v in DISPLAY.items() if v == disp), None)


def present_parsers(rows) -> list[str]:
    """Only parsers that actually have measurements -- an arm that has not run yet must
    not appear in the legend as if it had."""
    have = {l for l, _ in rows}
    return [d for d in ORDER if label_of(d) in have]


def _stagger(ends, min_sep=0.085):
    """Push overlapping right-edge labels apart in log space, keeping their order."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = [math.log10(y) for _, y in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_sep:
            ys[i] = ys[i - 1] + min_sep
    return [(name, 10 ** y, orig) for (name, orig), y in zip(ends, ys)]


def panel_despace(ax, rows):
    ends = []
    for disp in present_parsers(rows):
        label = label_of(disp)
        xs, ys = [], []
        for dom, p in DESPACE:
            r = get(rows, label, dom)
            if r:
                xs.append(p)
                ys.append(r["cr"])
        if not xs:
            continue
        ax.plot(xs, ys, marker=MARK[disp], color=COL[disp], lw=2, ms=6,
                markeredgecolor="white", markeredgewidth=0.9, zorder=3, label=disp)
        ends.append((disp, ys[-1]))
    # direct labels at the right edge (identity never by color alone), de-collided
    for disp, y_label, y_true in _stagger(ends):
        ax.annotate(disp, (1.0, y_true), xytext=(1.055, y_label), fontsize=8.5,
                    va="center", color=COL[disp],
                    arrowprops=dict(arrowstyle="-", color=COL[disp], lw=0.7,
                                    shrinkA=2, shrinkB=0))
    ax.axhspan(*BAND, color="#000000", alpha=0.045, zorder=0, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of intra-line spaces removed  ($p$)")
    ax.set_ylabel("bytes / patch")
    ax.set_title("(a) Whitespace removal", loc="left", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.04, 1.30)
    ax.grid(axis="y", color="#cccccc", lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_discrete(ax, rows):
    """Dot plot, not bars: on a log axis a bar's LENGTH encodes nothing meaningful
    (the baseline is arbitrary), so position is the only honest encoding here."""
    present = present_parsers(rows)
    n = len(present)
    width = 0.72 / max(n, 1)
    xpos = {}
    for i, disp in enumerate(present):
        label = label_of(disp)
        xs, ys = [], []
        for j, (dom, _) in enumerate(DISCRETE):
            r = get(rows, label, dom)
            if r:
                x = j + (i - (n - 1) / 2) * width
                xs.append(x)
                ys.append(r["cr"])
                xpos[(disp, dom)] = x
        # published-but-not-re-measured points are drawn hollow so provenance is visible
        for x, y, dom in zip(xs, ys, [d for d, _ in DISCRETE
                                      if get(rows, label, d)]):
            pub = rows[(label, dom)].get("provenance") != "measured"
            ax.plot([x], [y], MARK[disp], ms=7, zorder=3, linestyle="none",
                    color="white" if pub else COL[disp],
                    markeredgecolor=COL[disp],
                    markeredgewidth=1.5 if pub else 0.9)

    # Telugu vocab-augment overlay: same parser and model, vocabulary repaired
    aug = get(rows, AUG, "te")
    base = get(rows, label_of("BPEByte"), "te")
    if aug and base and ("BPEByte", "te") in xpos:
        x = xpos[("BPEByte", "te")]
        ax.annotate("", (x, aug["cr"]), (x, base["cr"]),
                    arrowprops=dict(arrowstyle="-|>", color=COL["BPEByte"],
                                    lw=1.3, ls=(0, (2, 1.4))), zorder=4)
        ax.plot([x], [aug["cr"]], MARK["BPEByte"], color="white", ms=7, zorder=5,
                markeredgecolor=COL["BPEByte"], markeredgewidth=1.6, linestyle="none")
        ax.annotate("+Telugu\nvocab", (x, aug["cr"]), textcoords="offset points",
                    xytext=(-8, 2), ha="right", va="bottom", fontsize=7.5,
                    linespacing=1.0, color=COL["BPEByte"])

    ax.axhspan(*BAND, color="#000000", alpha=0.045, zorder=0, lw=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(DISCRETE)))
    ax.set_xticklabels([lbl for _, lbl in DISCRETE])
    ax.set_xlim(-0.55, len(DISCRETE) - 0.15)
    ax.set_ylabel("bytes / patch")
    ax.set_title("(b) Domain shift", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#cccccc", lw=0.5, alpha=0.6, zorder=0)
    for j in range(1, len(DISCRETE)):
        ax.axvline(j - 0.5, color="#e5e5e5", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(rows, path, figsize=(7.6, 3.3)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panel_despace(axes[0], rows)
    panel_discrete(axes[1], rows)
    handles = [Line2D([], [], color=COL[d], marker=MARK[d], lw=2, ms=6,
                      markeredgecolor="white", label=d)
               for d in present_parsers(rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, 0.965, "Shaded band = usable hierarchical compression "
                         f"({BAND[0]:.0f}–{BAND[1]:.0f} bytes/patch)",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    plt.close(fig)


def write_summary(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    labels = sorted({k[0] for k in rows})
    domains = [d for d in ("en", "code", "zh", "te", "random", "despace_p025",
                           "despace_p050", "despace_p075", "despace_p100")
               if any((l, d) in rows for l in labels)]
    lines = ["# Compression stability under distribution shift", "",
             f"Source: `{SRC.relative_to(ROOT)}` · "
             "generated by `scripts/probes/plot_compression_stability.py`", "",
             "Bytes per patch (mean over all patches; higher = more pooling). "
             "`I` = |log2(CR_domain / CR_english)|, penalising blow-up and collapse equally.", "",
             "| parser | domain | B/patch | std | max | p99 | % len-1 | % bytes in >16B | I |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for l in labels:
        for d in domains:
            r = rows.get((l, d))
            if not r:
                continue
            inst = r.get("instability")
            lines.append(f"| {l} | {d} | {r['cr']:.2f} | {r['std']:.2f} | {r['max']} | "
                         f"{r['p99']} | {100*r['frac_len1']:.1f}% | "
                         f"{100*r['frac_bytes_in_long']:.1f}% | "
                         f"{'--' if inst is None else f'{inst:.2f}'} |")

    lines += ["", "## Worst-case instability per parser", "",
              "| parser | max I | where |", "|---|---:|---|"]
    worst = defaultdict(lambda: (-1.0, ""))
    for (l, d), r in rows.items():
        i = r.get("instability")
        if i is not None and i > worst[l][0]:
            worst[l] = (i, d)
    for l, (i, d) in sorted(worst.items(), key=lambda kv: kv[1][0]):
        lines.append(f"| {l} | {i:.2f} | {d} |")

    gaps = missing_report(rows)
    lines += ["", "## Coverage gaps", "",
              "Stated explicitly so an absent arm is never read as a covered one.", ""]
    lines += [f"- {g}" for g in gaps] if gaps else ["- none: every arm x domain measured."]

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "summary.md")


def write_tex(rows):
    cols = [(l, DISPLAY[l]) for l in DISPLAY if any((l, d) in rows for d in ("en",))]
    if not cols:
        return
    doms = [("en", "English"), ("code", "Code"), ("zh", "Chinese"),
            ("te", "Telugu"), ("despace_p100", "Despaced EN"), ("random", "Random")]
    out = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\setlength{\tabcolsep}{4pt}", r"\fitcolumn{%",
           r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           " & " + " & ".join(f"\\textbf{{{d}}}" for _, d in cols) + r" \\",
           r"\midrule"]
    for dom, dlabel in doms:
        cells = []
        for l, _ in cols:
            r = rows.get((l, dom))
            cells.append("--" if not r
                         else f"${r['cr']:.1f}\\,{{\\scriptstyle\\pm {r['std']:.1f}}}$")
        out.append(f"{dlabel} & " + " & ".join(cells) + r" \\")
    aug = rows.get((AUG, "te"))
    if aug:
        out += [r"\specialrule{0.01em}{0.3ex}{0.35ex}",
                r"\;\;Telugu, vocab-augmented & "
                + " & ".join(f"${aug['cr']:.1f}\\,{{\\scriptstyle\\pm {aug['std']:.1f}}}$"
                             if DISPLAY[l] == "BPEByte" else "--" for l, _ in cols) + r" \\"]
    out += [r"\bottomrule", r"\end{tabular}", "}",
            r"\caption{Bytes per patch (mean $\pm$ std over patches) on held-out windows "
            r"under the protocol of Table~\ref{tab:zh_stats}. Every parser sees identical "
            r"windows. The entropy patcher's threshold is calibrated once on English and held "
            r"fixed across domains, since per-domain retuning is the very heuristic that "
            r"content-based patchers are meant to avoid.}",
            r"\label{tab:compression_stability}", r"\end{table}"]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(out) + "\n")
    print("wrote", TEX)


def main():
    rows = load()
    # recompute instability so the figure/tables never depend on stale appended values
    ref = {l: rows[(l, "en")]["cr"] for (l, d) in rows if d == "en"}
    for (l, d), r in rows.items():
        r["instability"] = abs(math.log2(r["cr"] / ref[l])) if l in ref else None
    render(rows, OUT / "compression_stability.png")
    render(rows, OUT / "compression_stability.pdf")
    write_summary(rows)
    write_tex(rows)


if __name__ == "__main__":
    main()

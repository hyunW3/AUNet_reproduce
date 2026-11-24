#!/usr/bin/env python3
"""Visualize the computation-boundary (patch) segmentation each parsing method
produces on example text from HellaSwag, ARC-Easy, and DCLM. The parsers are
deterministic and scale-independent, so the boundaries are identical at 100M
and 1B; we render one set per method and say so.

  Llama   : subword BPE (llama3 tiktoken)          -- token boundaries
  AU-Net  : word1 whitespace/word pooling          -- get_levels_mask
  BPEByte : bpe_br online greedy-root trie          -- get_levels_mask

Output: reports/parse/parse_boundaries.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.environ.setdefault("PYTHONPATH", "")
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs  # noqa: E402
from lingua.tokenizer import TikTokenTokenizer                  # noqa: E402

ROOT = "/mnt/ssd2/hyun2/AUNet"
TOK = f"{ROOT}/tokenizer/llama3/tokenizer.model"
OUT = f"{ROOT}/reports/parse"

# paper model colors (report_despace.py convention)
COL = {"Llama": "#0072B2", "AU-Net": "#E69F00", "BPEByte": "#009E73"}


def _pool(strategy, **kw):
    return RegexPool(RegexArgs(strategy=strategy, bpe_tokenizer_path=TOK, **kw))


def build_parsers():
    aunet = _pool({"word1": "1@1"})
    bpe = _pool({"bpe_br": "1@1"}, bpe_online=True,
                bpe_online_mode="greedy", bpe_online_placement="root")
    lt = TikTokenTokenizer(TOK)

    def llama_starts(text):
        b = text.encode()
        toks = lt.tkt_model.decode_tokens_bytes(lt.tkt_model.encode_ordinary(text))
        starts, off = [], 0
        for t in toks:
            starts.append(off); off += len(t)
        return starts

    def mask_starts(pool, text):
        b = list(text.encode())
        m = pool.get_levels_mask(b)
        s = sorted({i for i, v in enumerate(m[:len(b)]) if v >= 1})
        return s if s and s[0] == 0 else [0] + s

    return {
        "Llama": llama_starts,
        "AU-Net": lambda t: mask_starts(aunet, t),
        "BPEByte": lambda t: mask_starts(bpe, t),
    }


def render(examples, parsers, path, dark=False):
    surface = "#171a21" if dark else "white"
    ink = "#e6e6e6" if dark else "#111111"
    muted = "#9aa4b2" if dark else "#555555"
    order = ["Llama", "AU-Net", "BPEByte"]
    names = list(examples)
    maxN = max(len(examples[n].encode()) for n in names)  # ASCII -> bytes == chars

    fig, axes = plt.subplots(len(names), 1, figsize=(min(0.135 * maxN + 1.6, 13),
                                                     2.05 * len(names)),
                             squeeze=False)
    fig.patch.set_facecolor(surface)
    for ax, name in zip(axes[:, 0], names):
        text = examples[name]
        chars = list(text)
        N = len(chars)
        ax.set_facecolor(surface)
        rowH, gap = 1.0, 0.5
        for r, method in enumerate(order):
            starts = parsers[method](text)
            ends = starts[1:] + [N]
            y0 = (len(order) - 1 - r) * (rowH + gap)
            base = COL[method]
            for k, (s, e) in enumerate(zip(starts, ends)):
                ax.add_patch(Rectangle((s, y0), e - s, rowH, facecolor=base,
                                       alpha=0.34 if k % 2 else 0.16,
                                       edgecolor="none", zorder=1))
                # boundary separators
                ax.plot([s, s], [y0, y0 + rowH], color=base, lw=1.4, zorder=3)
            ax.plot([N, N], [y0, y0 + rowH], color=base, lw=1.4, zorder=3)
            # characters
            for i, ch in enumerate(chars):
                ax.text(i + 0.5, y0 + rowH / 2, ch, ha="center", va="center",
                        family="monospace", fontsize=8.5, color=ink, zorder=4)
            # method label + patch count
            ax.text(-0.6, y0 + rowH / 2, method, ha="right", va="center",
                    fontsize=9, color=base, fontweight="bold")
            ax.text(N + 0.6, y0 + rowH / 2, f"{len(starts)}", ha="left",
                    va="center", fontsize=8.5, color=muted)
        ax.set_xlim(-9, maxN + 4)
        ax.set_ylim(-0.3, len(order) * (rowH + gap))
        ax.axis("off")
        ax.set_title(f"{name}   ({N} chars)", loc="left",
                     fontsize=10, color=muted, pad=3, fontweight="bold")
    fig.suptitle("Computation-boundary segmentation per method "
                 "(identical at 100M and 1B — deterministic parsing)",
                 fontsize=11.5, fontweight="bold", color=ink, y=0.998)
    fig.text(0.5, 0.012,
             "Each shaded box is one computation patch; alternating shades mark "
             "consecutive patches, vertical bars are boundaries. Right-hand number "
             "= patch count (fewer = more pooling).",
             ha="center", fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=surface)
    print("wrote", path)


def main():
    from pathlib import Path
    Path(OUT).mkdir(parents=True, exist_ok=True)
    ex = json.load(open("/tmp/parse_examples.json"))
    # trim to one readable ASCII line each
    dclm = ex["dclm"]
    i = dclm.find("Here what happened")
    examples = {
        "HellaSwag": ex["hellaswag"],
        "ARC-Easy": ex["arc_easy"].replace("\n", "  "),
        "DCLM": dclm[i:i + 88].strip(),
    }
    parsers = build_parsers()
    render(examples, parsers, f"{OUT}/parse_boundaries.png")
    render(examples, parsers, f"{OUT}/parse_boundaries.pdf")
    render(examples, parsers, f"{OUT}/parse_boundaries_dark.png", dark=True)


if __name__ == "__main__":
    main()

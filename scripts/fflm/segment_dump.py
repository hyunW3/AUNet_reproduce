#!/usr/bin/env python3
"""Show how each model chunks an FFLM string into the units it actually pools /
tokenizes at inference. This is the direct evidence for the "atomicity" claim:
does the governing bit stay its own retrievable unit, or get blended in?

  subword (Llama)   : tiktoken BPE tokens
  AU-Net (word1)    : whitespace-word pooling boundaries (RegexPool.str_offset)
  byte  (greedy-rt) : online greedy byte-trie boundaries (RegexPool.online_byte_boundaries)

CPU only. Run with PYTHONPATH=<lingua> from anywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from omegaconf import OmegaConf
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs
from lingua.args import dataclass_from_dict
from lingua.tokenizer import build_tokenizer

L = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet"
TOK = f"{L}/tokenizer/llama3/tokenizer.model"
CKPT = {
    "subword_llama":   f"{L}/runs/main/1.3B/llama_1.8B_paper/checkpoints/*/consolidated",
    "aunet_static":    f"{L}/runs/main/1.3B/aunet2_1.3B/checkpoints/*/consolidated",
    "byte_greedyroot": f"{L}/runs/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/*/consolidated",
}


def _params(tag):
    return sorted(glob.glob(CKPT[tag] + "/params.json"))[-1]


def show_segments(segs):
    # make spaces visible with a middle dot
    return " | ".join("'" + s.replace(" ", "·") + "'" for s in segs)


def _spans(pieces):
    """[(text, start, end)] from a list of contiguous unit strings."""
    out, pos = [], 0
    for p in pieces:
        out.append((p, pos, pos + len(p)))
        pos += len(p)
    return out


def seg_subword(text):
    tok = build_tokenizer("tiktoken", TOK)
    ids = tok.encode(text, add_bos=False, add_eos=False)
    return [tok.decode([i]) for i in ids]


def _pool_from(tag):
    cfg = OmegaConf.load(_params(tag))
    rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
    if getattr(rargs, "bpe_tokenizer_path", None):
        rargs.bpe_tokenizer_path = TOK   # portability
    return RegexPool(rargs)


def seg_word(text):
    """AU-Net whitespace pooling: cut after each end-offset from str_offset."""
    pool = _pool_from("aunet_static")
    offsets, _ = pool.str_offset(text)
    ends = sorted(set(int(o) for o in offsets))
    segs, prev = [], 0
    for e in ends:
        segs.append(text[prev:e + 1])          # end-offset is inclusive last char
        prev = e + 1
    if prev < len(text):
        segs.append(text[prev:])
    return segs


def seg_byte_greedy(text):
    """byte greedy-root: online greedy byte-trie unit starts."""
    pool = _pool_from("byte_greedyroot")
    b = list(text.encode("utf-8"))
    starts = sorted(set(pool.online_byte_boundaries(b, mode="greedy")))
    cut = sorted(set([0] + starts + [len(b)]))
    segs = []
    for i in range(len(cut) - 1):
        seg = bytes(b[cut[i]:cut[i + 1]]).decode("utf-8", errors="replace")
        if seg:
            segs.append(seg)
    return segs


MODELS = [("subword (Llama)", seg_subword),
          ("AUNet", seg_word),
          ("BPEByte rg", seg_byte_greedy)]


def _readbit_offsets(text):
    """Char offsets of bits that follow an 'r' instruction (the predicted targets)."""
    syms, pos = [], 0
    for part in text.split(" "):
        syms.append((part, pos))
        pos += len(part) + 1
    return {off for k, (s, off) in enumerate(syms)
            if k % 2 == 1 and syms[k - 1][0] == "r"}


def render_png(text, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    C_WRITEIGN = "#4C78A8"   # write/ignore bit (state-carrying or distractor)
    C_READ = "#E45756"       # read bit = prediction target
    C_INSTR = "#D8DEE9"      # instruction-containing unit
    C_SPACE = "#F2F2F2"      # space-only unit
    reads = _readbit_offsets(text)

    fig, ax = plt.subplots(figsize=(max(7, len(text) * 0.42), 3.3))
    n = len(MODELS)
    for row, (label, fn) in enumerate(MODELS):
        y = n - 1 - row
        for seg, s, e in _spans(fn(text)):
            core = seg.strip()
            is_bit = core in ("0", "1")
            is_read = is_bit and any(s <= o < e for o in reads)
            if is_read:
                fc, tc = C_READ, "white"
            elif is_bit:
                fc, tc = C_WRITEIGN, "white"
            elif core == "":
                fc, tc = C_SPACE, "#999999"
            else:
                fc, tc = C_INSTR, "#2B2B2B"
            ax.add_patch(FancyBboxPatch(
                (s + 0.06, y + 0.12), (e - s) - 0.12, 0.76,
                boxstyle="round,pad=0,rounding_size=0.12",
                linewidth=1, edgecolor="white", facecolor=fc, mutation_aspect=0.5))
            ax.text((s + e) / 2, y + 0.5, seg.replace(" ", "·"),
                    ha="center", va="center", fontsize=11, color=tc,
                    family="monospace", fontweight="bold")
        ax.text(-0.4, y + 0.5, label, ha="right", va="center", fontsize=10)
        ax.text(len(text) + 0.3, y + 0.5, f"{len(fn(text))} units",
                ha="left", va="center", fontsize=8, color="#888")

    ax.set_xlim(-0.5, len(text) + 3.5)
    ax.set_ylim(-0.15, n + 0.15)
    ax.axis("off")
    ax.set_title(f"FFLM segmentation per model    string: \"{text}\"",
                 fontsize=11, fontweight="bold", loc="left")
    # legend
    handles = [plt.matplotlib.patches.Patch(facecolor=c, label=l) for c, l in
               [(C_READ, "read bit (target)"), (C_WRITEIGN, "write/ignore bit"),
                (C_INSTR, "instruction"), (C_SPACE, "space")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print("wrote", outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="w 0 i 1 i 0 r 0 w 1 i 0 r 1")
    ap.add_argument("--png", default=None, help="also render a figure here")
    args = ap.parse_args()
    text = args.text
    print(f"FFLM string : '{text}'  ({len(text)} chars)\n")
    for tag, fn in MODELS:
        try:
            segs = fn(text)
            print(f"{tag:24s} [{len(segs):2d} units]  {show_segments(segs)}")
        except Exception as e:
            print(f"{tag:24s} ERROR: {e}")
    if args.png:
        render_png(text, args.png)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MK-NIAH pressure-test GRID (needle depth x context size), per model.

Classic Needle-in-a-Haystack visualization (cf. mm-niah.github.io / gkamradt):
place the QUERIED needle at a controlled depth, K-1 distractor needles elsewhere,
sweep (context size x target depth), and render an exact-match heatmap per model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = os.environ.get("AUNET_ROOT", "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet")
sys.path.insert(0, f"{L}/scripts/probes")
sys.path.insert(0, f"{L}/scripts/niah")
from common import build_generator, ckpt_for, fmt_pair, score  # noqa: E402
from niah_data import _essay_pool, KEYS, INSTRUCTION  # noqa: E402

MODELS = [("subword_llama", "subword (Llama)"), ("aunet_static", "AUNet"),
          ("byte_greedyroot", "BPEByte rg")]


def make_grid_sample(target_bytes, target_depth, K, rng):
    keys = rng.sample(KEYS, K)
    vals = ["".join(rng.choice("0123456789") for _ in range(7)) for _ in keys]
    needles = [f"One of the special magic numbers for {k} is {v}. "
               for k, v in zip(keys, vals)]
    pool = _essay_pool()
    start = rng.randint(0, max(0, len(pool) - target_bytes - 1))
    body = pool[start:start + target_bytes]
    # target (index 0) at target_depth; distractors at random depths
    offs = [target_depth] + [rng.random() for _ in range(K - 1)]
    order = sorted(range(K), key=lambda i: offs[i])
    parts, prev = [], 0
    for i in order:
        c = min(len(body), max(prev, int(offs[i] * len(body))))
        while 0 < c < len(body) and body[c] != " ":
            c += 1
        parts.append(body[prev:c]); parts.append(needles[i]); prev = c
    parts.append(body[prev:])
    haystack = "".join(parts)
    qk, ans = keys[0], vals[0]
    query = (f"What is the special magic number for {qk} mentioned in the "
             f"provided text? The special magic number for {qk} is")
    return f"{INSTRUCTION}\n\n{haystack}\n\n{query}", ans


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[m for m, _ in MODELS])
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 6144])
    ap.add_argument("--n_depths", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_per_cell", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=f"{L}/reports/niah/mk_grid.jsonl")
    args = ap.parse_args()
    depths = [round(i / (args.n_depths - 1), 3) for i in range(args.n_depths)]
    lbl = dict(MODELS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"MK-NIAH grid | K={args.K} contexts={args.contexts} depths={depths} "
          f"n/cell={args.n_per_cell}")

    import math
    # hard context window per family (len(encode(prompt)) units): bytes for byte
    # models, tokens for subword. A prompt over this gets truncated -> cell is N/A,
    # not a (vacuous) score.
    WINDOW = {"aunet": 8192, "subword": 2048}   # byte: 8192 bytes; Llama: 2048 tokens
    grids = {}
    for tag in args.models:
        fam, ckpt = ckpt_for(tag)
        gen, tok = build_generator(fam, ckpt, None, args.max_tokens)
        rng = random.Random(args.seed)
        grid = [[float("nan")] * len(args.contexts) for _ in range(len(depths))]
        for di, dp in enumerate(depths):
            for ci, cb in enumerate(args.contexts):
                pairs = []
                for _ in range(args.n_per_cell):
                    p, a = make_grid_sample(cb, dp, args.K, rng)
                    pairs.append(fmt_pair(p, a, fam))
                # a cell is scored only if EVERY sample fits the window (else the
                # cell would mix real scores with silently-truncated ones)
                plen = max(len(tok.encode(c, add_bos=False, add_eos=False))
                           for c, _ in pairs)
                if plen > WINDOW[fam]:            # any sample exceeds the model window
                    with open(args.out, "a") as f:
                        f.write(json.dumps({"tag": tag, "depth": dp, "context": cb,
                                            "exact_match": None, "n": 0,
                                            "over_window": plen}) + "\n")
                    continue
                res = score(gen, tok, pairs, args.batch_size)
                em = sum(r[0] for r in res) / len(res)
                grid[di][ci] = em
                with open(args.out, "a") as f:
                    f.write(json.dumps({"tag": tag, "depth": dp, "context": cb,
                                        "exact_match": em, "n": len(res)}) + "\n")
        grids[tag] = grid
        vals = [v for row in grid for v in row if not math.isnan(v)]
        print(f"  {tag}: mean={sum(vals)/len(vals):.3f} over {len(vals)} in-window cells")
        del gen

    # heatmap grid, one panel per model
    import numpy as np
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d0d0d0")               # N/A cells (prompt over the model window)
    tags = [t for t in args.models]
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.6), squeeze=False)
    for ax, tag in zip(axes[0], tags):
        g = np.array(grids[tag], dtype=float)
        im = ax.imshow(g, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(args.contexts)))
        ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in args.contexts], fontsize=8)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{int(d*100)}%" for d in depths], fontsize=7)
        ax.set_xlabel("context size (bytes)")
        if ax is axes[0][0]:
            ax.set_ylabel("needle depth (position in haystack)")
        ax.set_title(lbl.get(tag, tag), fontsize=10)
        for di in range(len(depths)):
            for ci in range(len(args.contexts)):
                v = g[di][ci]
                txt = "n/a" if np.isnan(v) else f"{v:.2f}"
                ax.text(ci, di, txt, ha="center", va="center", fontsize=6,
                        color="#777" if np.isnan(v) else "black")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, label="exact-match")
    fig.text(0.5, -0.02, "n/a = prompt exceeds the model's context window "
             "(byte models: 8192 bytes; Llama: 2048 tokens). At 8k body every "
             "family is out-of-window — the largest testable body is ~6k.",
             ha="center", fontsize=8, color="#777")
    fig.suptitle(f"MK-NIAH pressure test (K={args.K} needles) — depth x context, frozen 1.3B",
                 fontweight="bold")
    out = Path(args.out).with_name("mk_grid.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Trunk-context (sink-preserving) sweep: BPB per patch-position vs how many patches back the
global trunk may attend to.  Modes trunk_s{K}w{N} keep K sink patches + the last N patches."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
ROOT = Path(__file__).resolve().parents[2]
POS_BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 10**6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "reports/global_ablation"))
    ap.add_argument("--tags", default="aunet_static,byte_greedyroot")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    P = lines.append
    for tag in args.tags.split(","):
        d = np.load(Path(args.dir) / f"{tag}.npz")
        pos, bnd = d["pos_in_patch"], d["is_boundary"]
        bpp = 1 / bnd.mean()
        sweep = sorted(((int(re.fullmatch(r"nll_trunk_s\d+w(\d+)", k).group(1)), k)
                        for k in d.files if re.fullmatch(r"nll_trunk_s\d+w\d+", k)))
        rows = [("local_only (no global at all)", d["nll_local_only"])] if "nll_local_only" in d else []
        rows += [(f"{n} patches (~{n * bpp:.0f} B)", d[k]) for n, k in sweep]
        rows += [("full (~790 patches)", d["nll_full"])]
        P(f"\n### {tag} — BPB vs trunk context (sink kept), by patch position\n")
        P("| trunk context | all | pos 0 | pos 1 | pos 2 | pos 3–4 | pos 5–9 | pos 10+ |")
        P("|---|---|---|---|---|---|---|---|")
        for name, nll in rows:
            cells = [f"{nll.mean() / LN2:.3f}"]
            for lo, hi in POS_BINS:
                m = (pos >= lo) & (pos <= hi)
                cells.append(f"{nll[m].mean() / LN2:.3f}" if m.any() else "-")
            P(f"| {name} | " + " | ".join(cells) + " |")

        base = d["nll_full"]
        P(f"\n**Δ BPB vs full** (how much the byte at that position still needs the missing context)\n")
        P("| trunk context | all | pos 0 | pos 1 | pos 2 | pos 3–4 | pos 5–9 | pos 10+ |")
        P("|---|---|---|---|---|---|---|---|")
        for name, nll in rows[:-1]:
            cells = [f"{(nll - base).mean() / LN2:+.3f}"]
            for lo, hi in POS_BINS:
                m = (pos >= lo) & (pos <= hi)
                cells.append(f"{(nll[m] - base[m]).mean() / LN2:+.3f}" if m.any() else "-")
            P(f"| {name} | " + " | ".join(cells) + " |")

    txt = "\n".join(lines)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()

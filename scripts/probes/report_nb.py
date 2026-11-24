#!/usr/bin/env python3
"""Patch-offset / patch-length conditioned BPB for the T_N (non-boundary transformer) arms.

`eval_sp_models.py` already writes one per-byte NLL array per arm (`<tag>_bpb.npy`).  This script
adds the axis the experiment actually turns on: WHERE inside its patch each scored byte sits.

    python scripts/probes/report_nb.py --tags nb_base nb_2l nb_2l_noanchor \
        --params runs/nb/nb_base/checkpoints/0000003344/params.json

The metadata is recomputed from the same windows and the same `RegexPool`, in the same order
`eval_sp_models.py` scores them, so element i of `<tag>_bpb.npy` and element i of the metadata are
the same byte.  A mismatch in length is a hard error rather than a silent truncation.

Two offsets exist and confusing them inverts the story:

  `off_in`   the offset of the INPUT byte t inside its patch.  This is what T_N conditions on:
             at off_in = 0 the byte reads a vector pooled at itself and T_N is a structural no-op.
  `off_pred` the offset of the byte being PREDICTED (t+1).  off_pred = 0 means "predict the first
             byte of the next patch", which is where most of the loss mass lives.

`off_in` is the causal axis for this experiment; both are reported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))

from lingua.args import dataclass_from_dict  # noqa: E402
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs  # noqa: E402

LN2 = math.log(2.0)
BOS = 256


def metadata(windows, pool, max_bytes, seqlen):
    """Per scored byte: input-byte offset, predicted-byte offset, patch length, window id.

    Mirrors `eval_sp_models.prep` exactly -- same clip, same skip rule, same scored range
    (`ids[1:n+1]` predicted from `ids[:n]`).
    """
    off_in, off_pred, plen, win = [], [], [], []
    for wi, w in enumerate(windows):
        clip = w["text"].encode()[:max_bytes].decode("utf-8", errors="ignore")
        bs = list(clip.encode())
        n = len(bs)
        if n == 0 or n + 2 >= seqlen:
            continue
        ids = [BOS] + bs
        lm = np.asarray(pool.get_levels_mask(ids)[: n + 1])
        starts = np.flatnonzero(lm)                      # patch-start positions in `ids`
        if starts.size == 0 or starts[0] != 0:
            starts = np.concatenate([[0], starts[starts > 0]])
        pid = np.searchsorted(starts, np.arange(n + 1), side="right") - 1
        off = np.arange(n + 1) - starts[pid]
        lens = np.diff(np.concatenate([starts, [n + 1]]))
        off_in.append(off[:n])                           # input bytes 0..n-1
        off_pred.append(off[1 : n + 1])                  # predicted bytes 1..n
        plen.append(lens[pid[:n]])                       # length of the INPUT byte's patch
        win.append(np.full(n, wi))
    return {k: np.concatenate(v) for k, v in
            dict(off_in=off_in, off_pred=off_pred, plen=plen, win=win).items()}


def boot(a, b, win, n=2000, seed=0):
    """Paired bootstrap over WINDOWS (bytes inside a window are not independent)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(win)
    idx_of = [np.flatnonzero(win == w) for w in ids]
    d = []
    for _ in range(n):
        pick = np.concatenate([idx_of[j] for j in rng.integers(0, len(ids), len(ids))])
        d.append((b[pick].mean() - a[pick].mean()) / LN2)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def table(title, key, bins, meta, arms, base):
    """One row per bin, with BOTH absolute and RELATIVE delta.

    Relative matters here and absolute alone misleads: base BPB ranges from 2.67 at patch offset 0
    to 0.36 at offset 5, so the loss available to recover differs ~7x across the rows.  A flat
    absolute delta across offsets is therefore a strongly INCREASING relative one, which is the
    quantity the patch-lag hypothesis actually predicts.
    """
    print(f"\n## {title}\n")
    others = [a for a in arms if a != base]
    hdr = (f"| {key} | bytes | share | " + " | ".join(arms) + " | "
           + " | ".join(f"D {a}" for a in others) + " | "
           + " | ".join(f"%D {a}" for a in others) + " |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    v, tot = meta[key], meta[key].size
    for lo, hi, lab in bins:
        m = (v >= lo) & (v <= hi)
        if not m.any():
            continue
        b = arms[base][m].mean()
        cells = [f"{arms[a][m].mean() / LN2:.4f}" for a in arms]
        deltas = [f"{(arms[a][m].mean() - b) / LN2:+.4f}" for a in others]
        rel = [f"{100 * (arms[a][m].mean() - b) / b:+.2f}%" for a in others]
        print(f"| {lab} | {m.sum():,} | {100 * m.sum() / tot:.1f}% | "
              + " | ".join(cells) + " | " + " | ".join(deltas) + " | " + " | ".join(rel) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="first tag is the baseline")
    ap.add_argument("--params", required=True, help="params.json of any arm (all share the data cfg)")
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--indir", default=str(ROOT / "reports/patch_lag/sp_eval"))
    ap.add_argument("--max_bytes", type=int, default=3840)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--tokenizer_path", default=None)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.params)
    rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
    if a.tokenizer_path:
        rargs.bpe_tokenizer_path = a.tokenizer_path
    pool = RegexPool(rargs)

    wins = [json.loads(l) for l in open(a.windows)]
    meta = metadata(wins, pool, a.max_bytes, a.seqlen)

    arms, indir = {}, Path(a.indir)
    for t in a.tags:
        nll = np.load(indir / f"{t}_bpb.npy").astype(np.float64)
        if nll.size != meta["off_in"].size:
            raise SystemExit(
                f"{t}: {nll.size} scored bytes but metadata has {meta['off_in'].size}. "
                "The windows / max_bytes / tokenizer must match the eval run exactly.")
        arms[t] = nll
    base = a.tags[0]

    print(f"# T_N arms -- {meta['off_in'].size:,} bytes, {len(np.unique(meta['win']))} windows")
    print(f"\n## Overall\n\n| arm | BPB | D vs {base} | 95% CI |")
    print("|---|---|---|---|")
    for t in a.tags:
        d = (arms[t].mean() - arms[base].mean()) / LN2
        ci = "" if t == base else "[{:+.4f}, {:+.4f}]".format(*boot(arms[base], arms[t], meta["win"]))
        print(f"| {t} | {arms[t].mean() / LN2:.4f} | {d:+.4f} | {ci} |")

    off_bins = [(r, r, str(r)) for r in range(6)] + [(6, 7, "6-7"), (8, 15, "8-15"), (16, 10**6, "16+")]
    table("BPB by INPUT-byte patch offset (the axis T_N acts on; 0 = T_N is a no-op)",
          "off_in", off_bins, meta, arms, base)
    table("BPB by PREDICTED-byte patch offset", "off_pred", off_bins, meta, arms, base)
    table("BPB by patch length (P=1 has no non-boundary position -- arms must agree)",
          "plen", [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 4, "4"),
                   (5, 7, "5-7"), (8, 10**6, "8+")], meta, arms, base)


if __name__ == "__main__":
    main()

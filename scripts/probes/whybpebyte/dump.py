#!/usr/bin/env python3
"""Stages 1-2: dump per-arm boundary masks and per-byte probe surprisal for a window corpus.

Both are cached to .npz because every downstream experiment reuses them, and because the design
requires that all arms be scored against IDENTICAL underlying uncertainty values (section 1.1 /
stage 2): the probe runs once per corpus, never once per arm.

    python dump.py --domain en --probe blt_entropy
    python dump.py --domain en --arms llama aunet bpebyte --no-probe

Layout (reports_afterAAAIsub/whybpebyte/cache/):
    seg_<domain>.npz     starts_<arm> (ragged, stored flat + offs_<arm>), n_bytes
    probe_<probe>_<domain>.npz   s (flat float32), h (flat float32), offs, n_bytes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / "reports_afterAAAIsub/whybpebyte/cache"

DOMAINS = {
    "en": ROOT / "reports/bpb_windows.jsonl",
    "zh": ROOT / "reports/bpb_windows_zh.jsonl",
    "code": ROOT / "reports/windows/code.jsonl",
    "te": ROOT / "reports/windows/te.jsonl",
    "random": ROOT / "reports/windows/random.jsonl",
}


def load_windows(domain: str, limit=None):
    wins = [json.loads(l) for l in open(DOMAINS[domain])]
    return wins[:limit] if limit else wins


def _pack(list_of_arrays):
    flat = np.concatenate([np.asarray(a, dtype=np.int64) for a in list_of_arrays]) \
        if list_of_arrays else np.zeros(0, np.int64)
    offs = np.cumsum([0] + [len(a) for a in list_of_arrays]).astype(np.int64)
    return flat, offs


def unpack(flat, offs):
    return [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]


def dump_segmentations(domain: str, arms, limit=None, **kw):
    from segmenters import build
    wins = load_windows(domain, limit)
    out = {"n_bytes": np.array([len(w["text"].encode()) for w in wins], dtype=np.int64)}
    for arm in arms:
        seg = build(arm, **kw)
        starts = []
        for w in wins:
            s = np.asarray(seg(w["text"]), dtype=np.int64)
            n = len(w["text"].encode())
            s = np.unique(s[(s >= 0) & (s < n)])
            if s.size == 0 or s[0] != 0:
                s = np.concatenate(([0], s))
            starts.append(s)
        flat, offs = _pack(starts)
        out[f"starts_{arm}"], out[f"offs_{arm}"] = flat, offs
        bpp = out["n_bytes"].sum() / sum(len(s) for s in starts)
        print(f"  {arm:10s} units={sum(len(s) for s in starts):>8d}  bytes/unit={bpp:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"seg_{domain}.npz"
    if p.exists():                      # merge, so arms can be added incrementally
        old = dict(np.load(p))
        old.update(out)
        out = old
    np.savez_compressed(p, **out)
    print(f"  -> {p}")


def dump_probe(domain: str, probe: str, device="cuda", limit=None):
    from probe_lm import ByteLMProbe, PROBES
    wins = load_windows(domain, limit)
    lm = ByteLMProbe(str(ROOT / PROBES[probe]), device=device)
    print(f"  probe {probe}: {lm.info()}")
    ss, hh = [], []
    for i, w in enumerate(wins):
        s, h = lm.score(list(w["text"].encode()))
        ss.append(s)
        hh.append(h)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(wins)}  running bpb={np.concatenate(ss).mean():.4f}", flush=True)
    offs = np.cumsum([0] + [len(s) for s in ss]).astype(np.int64)
    s_all, h_all = np.concatenate(ss), np.concatenate(hh)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"probe_{probe}_{domain}.npz"
    np.savez_compressed(p, s=s_all, h=h_all, offs=offs,
                        n_bytes=np.array([len(s) for s in ss], dtype=np.int64),
                        info=json.dumps(lm.info() | {"bpb": float(s_all.mean())}))
    print(f"  probe bpb={s_all.mean():.4f} bits/byte over {len(s_all)} bytes -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="en", choices=sorted(DOMAINS))
    ap.add_argument("--arms", nargs="*", default=["llama", "aunet", "bpebyte", "whitespace"])
    ap.add_argument("--probe", default="blt_entropy")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--entropy_model", default=None, help="blt arm: consolidated dir")
    ap.add_argument("--theta", type=float, default=None)
    a = ap.parse_args()

    if not a.no_seg:
        print(f"[seg] {a.domain}")
        kw = {}
        if a.entropy_model:
            kw = {"entropy_model": a.entropy_model, "theta": a.theta, "device": a.device}
        dump_segmentations(a.domain, a.arms, a.limit, **kw)
    if not a.no_probe:
        print(f"[probe] {a.domain} / {a.probe}")
        dump_probe(a.domain, a.probe, a.device, a.limit)


if __name__ == "__main__":
    main()

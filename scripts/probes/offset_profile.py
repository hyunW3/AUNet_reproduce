#!/usr/bin/env python3
"""Per-byte held-out NLL tagged with patch_offset, so the offset curve can be compared across arms.

Every patch-lag statistic we have (79.5% of bytes at offset>=1 carrying 45.9% of the loss) was
measured on a FROZEN checkpoint.  The question this answers is whether that gap survives joint
training, and -- decisively -- whether an arm that never touches patch lag (base_150m: the same
~50 M spent on trunk depth) flattens the offset curve just as much as scratchpad1 does.  If it
does, high offset is simply "harder position" and patch lag is not a separable phenomenon.

Offsets are recomputed from the levels mask rather than read out of the model, so the same
definition applies to every arm regardless of which forward path it takes:
    patch_offset[t] = t - (position of the last boundary at or before t)
The loss at index i is produced by the byte at position i (it predicts byte i+1), so it is
attributed to offset[i] -- the staleness of the vector the PREDICTING byte read.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))

from lingua.args import dataclass_from_dict  # noqa: E402
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints  # noqa: E402
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs  # noqa: E402
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs  # noqa: E402

LN2 = math.log(2.0)
BOS = 256


def load(ckpt_dir, tok_path=None):
    p = Path(ckpt_dir)
    if not (p / CONSOLIDATE_NAME).exists():
        p = Path(consolidate_checkpoints(str(p)))
    cfg = OmegaConf.load(p / "params.json")
    margs = dataclass_from_dict(HierarchicalArgs, cfg.model, strict=False)
    model = HierarchicalTransformer(margs)
    st = torch.load(p / CONSOLIDATE_NAME, weights_only=True, map_location="cpu")
    model.load_state_dict({k.replace("parametrizations", "_orig_mod.parametrizations")
                           if "transitions." in k and "parametrizations" in k else k: v
                           for k, v in st["model"].items()})
    del st
    model = model.to(dtype=torch.bfloat16).cuda().eval()
    rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
    if tok_path and getattr(rargs, "bpe_tokenizer_path", None):
        rargs.bpe_tokenizer_path = tok_path
    return model, RegexPool(rargs), cfg


def offsets_from_mask(lm_real: np.ndarray):
    """levels mask over [BOS]+bytes -> (patch_offset per position, patch length per position)."""
    lm = (np.asarray(lm_real) > 0).astype(np.int64)
    lm[0] = 1                                   # BOS always opens a patch
    b = np.flatnonzero(lm)                      # boundary positions
    pidx = np.cumsum(lm) - 1                    # which patch each position belongs to
    starts = b[pidx]
    off = np.arange(len(lm)) - starts
    ends = np.append(b[1:], len(lm))            # exclusive end of each patch
    plen = (ends - b)[pidx]
    return off, plen


def prep(text, pool, max_bytes, seqlen):
    clip = text.encode()[:max_bytes].decode("utf-8", errors="ignore")
    bs = list(clip.encode())
    n = len(bs)
    if n == 0 or n + 2 >= seqlen:
        return None
    ids_real = [BOS] + bs
    lm_real = pool.get_levels_mask(ids_real)
    off, plen = offsets_from_mask(np.asarray(lm_real[: n + 1]))
    ids = torch.zeros(seqlen, dtype=torch.long)
    lm = torch.zeros(seqlen, dtype=torch.long)
    ids[: n + 1] = torch.tensor(ids_real)
    lm[: n + 1] = torch.tensor(lm_real[: n + 1])
    lm[n + 1] = 1
    return ids, lm, n, off[:n], plen[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--max_bytes", type=int, default=3840)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tokenizer_path", default=str(ROOT / "tokenizer/llama3/tokenizer.model"))
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag/offset"))
    ap.add_argument("--trunk_loops", type=int, default=None,
                    help="override the looped trunk's depth at inference (looped models only)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, pool, cfg = load(args.ckpt, args.tokenizer_path)
    if args.trunk_loops is not None:
        assert hasattr(model, "trunk_loops"), "checkpoint has no looped trunk"
        model.trunk_loops = int(args.trunk_loops)
    seqlen = int(model.transitions[0].seqlen_in)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.tag}] loops={int(getattr(model, 'trunk_loops', 1))} "
          f"scratchpad_k={int(getattr(model, 'scratchpad_k', 0))} "
          f"updater={str(getattr(model, 'updater_kind', '')) or '-'} "
          f"layers={cfg.model.layers} seqlen={seqlen}", flush=True)

    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[: args.limit]
    prepped = [p for p in (prep(w["text"], pool, args.max_bytes, seqlen) for w in wins) if p]

    nll, off, plen = [], [], []
    t0 = time.time()
    for i in range(0, len(prepped), 2):
        ch = prepped[i : i + 2]
        if len(ch) < 2:
            ch = ch + [ch[-1]]
        ids = torch.stack([c[0] for c in ch]).cuda()
        lm = torch.stack([c[1] for c in ch]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True)
        for b in range(min(2, len(prepped) - i)):
            n = ch[b][2]
            nll.append(F.cross_entropy(logits[b, :n].float(), ids[b, 1 : n + 1],
                                       reduction="none").cpu().numpy())
            off.append(ch[b][3]); plen.append(ch[b][4])
    nll = np.concatenate(nll).astype(np.float32)
    off = np.concatenate(off).astype(np.int32)
    plen = np.concatenate(plen).astype(np.int32)
    assert nll.shape == off.shape == plen.shape, (nll.shape, off.shape, plen.shape)

    np.savez_compressed(outdir / f"{args.tag}.npz", nll=nll, offset=off, plen=plen)
    print(f"[{args.tag}] BPB = {nll.mean() / LN2:.4f}  ({nll.size} bytes, "
          f"{time.time() - t0:.0f}s)", flush=True)
    print(f"[{args.tag}] offset>=1 share = {(off >= 1).mean():.3f}  "
          f"loss share = {nll[off >= 1].sum() / nll.sum():.3f}", flush=True)
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 1 << 30)]:
        m = (off >= lo) & (off <= hi)
        if m.any():
            lab = f"{lo}" if lo == hi else (f"{lo}+" if hi > 1 << 20 else f"{lo}-{hi}")
            print(f"    offset {lab:>4s}  n={m.sum():8d} ({m.mean():5.1%})  "
                  f"BPB={nll[m].mean() / LN2:.4f}", flush=True)


if __name__ == "__main__":
    main()

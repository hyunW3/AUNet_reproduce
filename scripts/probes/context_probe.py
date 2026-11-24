#!/usr/bin/env python3
"""Does the boundary byte actually USE the preceding context to choose the next word?

The offset split says patch_offset == 0 -- the byte that is a space 75% of the time and must
predict the first letter of the next word -- carries 49.3% of the held-out loss while responding
least to extra capacity.  Two readings of that are possible and they call for opposite work:

  (a) irreducible entropy   -- the next word genuinely is not determined, nothing to fix;
  (b) context not delivered -- the information needed to choose the word exists in the sequence
                               but does not reach that position, which IS fixable.

This separates them WITHOUT touching the model: score the SAME target bytes while varying how
much preceding context the model is given.  If offset-0 loss keeps falling as context grows, the
boundary byte is extracting context and (a) dominates.  If it flattens while offset>=1 keeps
improving, the context is not reaching the boundary and (b) is real headroom.

Cutting the prefix can only change patching near the cut; the scored region's offsets are
recomputed each time and asserted identical, so the comparison is byte-for-byte the same targets.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offset_profile import load, offsets_from_mask  # noqa: E402

LN2 = math.log(2.0)
BOS = 256


@torch.no_grad()
def run(model, pool, wins, prefix, score_bytes, seqlen, bs=2):
    """Score the last `score_bytes` of each window given `prefix` bytes of context before it."""
    nll, off = [], []
    batch = []
    for w in wins:
        b = w.encode()[: 3840].decode("utf-8", "ignore").encode()
        if len(b) < score_bytes + 16:
            continue
        # Cut only ON a patch boundary.  BPE root cutting is not position-invariant -- slicing
        # mid-root re-tokenises the remainder and the scored bytes stop being the same targets
        # (the first attempt asserted out on exactly this).  Starting the string at a root
        # boundary makes the tokenisation of everything after it identical.
        full = pool.get_levels_mask([BOS] + list(b))
        bnd = np.flatnonzero(np.asarray(full[: 1 + len(b)]) > 0)      # in [BOS]+b coordinates
        bnd = bnd[bnd > 0] - 1                                        # -> byte coordinates
        if bnd.size < 8:
            continue
        j_s = int(np.searchsorted(bnd, len(b) - score_bytes))         # scored region starts here
        j_c = int(np.searchsorted(bnd, max(0, s_i_prefix := bnd[j_s] - prefix)))
        if bnd[j_c] > s_i_prefix and j_c > 0:
            j_c -= 1
        j_c = min(j_c, j_s - 1)                                       # >= 1 patch of context:
        # the mask value at a truncated string's very FIRST byte differs from its value in the
        # full string (that position always opens a patch), and only there.  Keeping the cut at
        # least one patch before the scored region puts that single byte outside the targets.
        if j_c < 0:
            continue
        s_i, c_i = int(bnd[j_s]), int(bnd[j_c])
        seg = list(b[c_i:])
        n_ctx = s_i - c_i
        ids_real = [BOS] + seg
        lm_real = pool.get_levels_mask(ids_real)
        o, _ = offsets_from_mask(np.asarray(lm_real[: len(ids_real)]))
        T = len(ids_real)
        if T + 2 >= seqlen:
            continue
        batch.append((ids_real, lm_real, n_ctx, o))
    for i in range(0, len(batch), bs):
        ch = batch[i : i + bs]
        while len(ch) < 2:
            ch = ch + [ch[-1]]
        T = max(len(c[0]) for c in ch) + 1
        ids = torch.zeros(len(ch), T, dtype=torch.long)
        lm = torch.zeros(len(ch), T, dtype=torch.long)
        for b_, c in enumerate(ch):
            t = len(c[0])
            ids[b_, :t] = torch.tensor(c[0]); lm[b_, :t] = torch.tensor(c[1][:t]); lm[b_, t] = 1
        ids, lm = ids.cuda(), lm.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True)
        for b_ in range(min(bs, len(batch) - i)):
            ids_real, _, n_ctx, o = ch[b_]
            lo, hi = 1 + n_ctx, len(ids_real) - 1        # score positions [lo, hi)
            e = F.cross_entropy(logits[b_, lo:hi].float(), ids[b_, lo + 1 : hi + 1],
                                reduction="none").cpu().numpy()
            nll.append(e); off.append(o[lo:hi])
    return np.concatenate(nll), np.concatenate(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--score_bytes", type=int, default=384)
    ap.add_argument("--prefixes", default="0,128,384,896,1920,3328")
    ap.add_argument("--tokenizer_path", default=str(ROOT / "tokenizer/llama3/tokenizer.model"))
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag/offset"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, pool, cfg = load(args.ckpt, args.tokenizer_path)
    seqlen = int(model.transitions[0].seqlen_in)
    wins = [json.loads(l)["text"] for l in open(args.windows)]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.tag}] 같은 {args.score_bytes} 바이트를, 앞 문맥 길이만 바꿔가며 채점", flush=True)
    print(f"{'prefix':>8s}{'BPB 전체':>11s}{'offset 0':>11s}{'offset>=1':>11s}"
          f"{'   Δoff0 vs prefix=0':>22s}{'  Δoff>=1':>11s}", flush=True)
    rows, ref = {}, None
    base0 = base1 = None
    for P in [int(x) for x in args.prefixes.split(",")]:
        t0 = time.time()
        nll, off = run(model, pool, wins, P, args.score_bytes, seqlen)
        if ref is None:
            ref = off
        else:
            assert off.shape == ref.shape and (off == ref).all(), \
                "prefix 절단이 채점 구간의 패치 경계를 바꿨다 -- 같은 타깃이 아니다"
        m0 = off == 0
        b, b0, b1 = nll.mean() / LN2, nll[m0].mean() / LN2, nll[~m0].mean() / LN2
        if base0 is None:
            base0, base1 = b0, b1
        rows[P] = (b, b0, b1)
        print(f"{P:>8d}{b:>11.4f}{b0:>11.4f}{b1:>11.4f}"
              f"{b0 - base0:>+22.4f}{b1 - base1:>+11.4f}   ({time.time() - t0:.0f}s)", flush=True)
    json.dump({str(k): [float(x) for x in v] for k, v in rows.items()},
              open(outdir / f"{args.tag}_ctx.json", "w"), indent=1)


if __name__ == "__main__":
    main()

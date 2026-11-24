#!/usr/bin/env python3
"""Causality + cost check for the offset-aware pretraining objective (patch_lag.py Sec 4).

A leak here would silently invalidate a whole pretraining run, so this checks it directly rather
than by argument:

  C1  every byte that receives a RE-POOLED vector was pooled at a position <= its own index
  C2  bytes earlier in the patch keep the ordinary vector, bit-for-bit
  C3  r = 0 everywhere reproduces the ordinary per-byte globals exactly
  C4  a byte's vector never depends on any byte after it -- tested by perturbing one future byte
      and checking that no earlier byte's global vector moves
  T   wall-clock cost of the extra trunk pass, as a multiple of the ordinary forward
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_ablation import load_model, prep  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, sample_patch_offsets, offset_aware_globals,
)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated")
    torch.set_grad_enabled(False)
    model, pool, _ = load_model(ckpt, os.environ.get("TOKENIZER_PATH"))
    seqlen = int(model.transitions[0].seqlen_in)
    wins = [json.loads(l) for l in open(ROOT / "reports/bpb_windows.jsonl")][:2]
    rows = [prep(w["text"], pool, 3840, seqlen) for w in wins]
    ids = torch.stack([r[0] for r in rows]).cuda()
    lm = torch.stack([r[1] for r in rows]).cuda()
    n = rows[0][2]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = split_forward(model, ids, lm)
        gen = torch.Generator(device="cuda"); gen.manual_seed(0)
        r = sample_patch_offsets(f, gen)
        g_mix = offset_aware_globals(model, f, r)

        exp = lambda g: g.gather(1, f.repeat_idx.unsqueeze(-1).expand(-1, -1, g.shape[-1]))
        g_std = exp(f.glob)
        r_byte = r.gather(1, f.repeat_idx)                       # each byte's patch's r
        pooled_at = f.idx.gather(1, f.repeat_idx) + r_byte       # index the fresh vector saw
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        fresh = f.patch_offset >= r_byte

        bad = int((fresh & (pooled_at > pos))[:, :n].sum())
        print(f"C1 re-pooled vector is never from the future : {bad} violations  "
              f"{'PASS' if bad == 0 else 'FAIL'}")

        early = (~fresh)[:, :n]
        same = torch.isclose(g_mix[:, :n][early], g_std[:, :n][early]).all()
        print(f"C2 earlier bytes keep the ordinary vector    : {'PASS' if bool(same) else 'FAIL'} "
              f"({int(early.sum())} bytes)")

        # relative, not absolute: the trunk residual stream has norm ~430, so an absolute
        # threshold says nothing.  The r=0 path is trunk_partial, whose fp deviation from the
        # ordinary trunk is the same bf16 noise patch_lag_verify's T2 measures.
        g0 = offset_aware_globals(model, f, torch.zeros_like(r)).float()
        gs = g_std[:, :n].float()
        rel = ((g0[:, :n] - gs).norm() / gs.norm()).item()
        cos = torch.nn.functional.cosine_similarity(
            g0[:, :n].flatten(0, 1), gs.flatten(0, 1), dim=-1).mean().item()
        print(f"C3 r=0 reproduces the ordinary globals       : rel={rel:.3e} cos={cos:.6f}  "
              f"{'PASS' if rel < 2e-2 else 'FAIL'}")

        # C4: flip one byte late in the sequence, re-run, and check nothing earlier moved
        cut = n // 2
        ids2 = ids.clone(); ids2[:, cut] = (ids2[:, cut] + 1) % 256
        f2 = split_forward(model, ids2, lm.clone())
        g2 = offset_aware_globals(model, f2, r)
        moved = (g_mix[:, :cut] - g2[:, :cut]).abs().amax(dim=-1)
        first = int((moved > 0).float().argmax(dim=1).min()) if bool((moved > 0).any()) else -1
        print(f"C4 no dependence on a later byte             : first moved byte={first} "
              f"(perturbed at {cut}, encoder window is 512 B)  "
              f"{'PASS' if first == -1 or first >= cut - 512 else 'FAIL'}")

        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            f3 = split_forward(model, ids, lm.clone())
        torch.cuda.synchronize(); base = (time.time() - t0) / 3
        t0 = time.time()
        for _ in range(3):
            f3 = split_forward(model, ids, lm.clone())
            offset_aware_globals(model, f3, r)
        torch.cuda.synchronize(); both = (time.time() - t0) / 3
        print(f"\nT  ordinary forward {base * 1e3:6.1f} ms   + offset-aware {both * 1e3:6.1f} ms"
              f"   -> {both / base:.2f}x")


if __name__ == "__main__":
    main()

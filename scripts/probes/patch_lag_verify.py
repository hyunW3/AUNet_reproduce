#!/usr/bin/env python3
"""Correctness tests for apps/aunet/patch_lag.py (the Stage -1 teacher).

  T1  pool_at_offset(0) == the trunk input the model itself pooled            (exact)
  T2  teacher_globals(0) == the standard trunk output                         (fp noise only)
  T3  partial slots do not see each other: perturbing slot j leaves every other slot untouched
  T4  causality: slot j's output does not depend on standard patches > j
  T5  magnitudes -- ||p||, ||g||, ||p~(r) - p||, ||g_teacher(r) - g|| -- so the oracle numbers
      can be read against the scale of the representations they perturb.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_ablation import load_model, prep  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, pool_at_offset, teacher_globals, trunk_partial,
)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-9)).item()


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt/aunet2_1.3B_180k/consolidated"
    torch.set_grad_enabled(False)
    model, pool, cfg = load_model(ckpt, os.environ.get("TOKENIZER_PATH"))
    seqlen = int(model.transitions[0].seqlen_in)
    wins = [json.loads(l) for l in open(ROOT / "reports/bpb_windows.jsonl")][:2]
    prepped = [prep(w["text"], pool, 3840, seqlen) for w in wins]
    ids = torch.stack([p[0] for p in prepped]).cuda()
    lm = torch.stack([p[1] for p in prepped]).cuda()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = split_forward(model, ids, lm)
        P = f.p_eff
        print(f"patches (p_eff)={P}  n_patches={f.n_patches.tolist()}")

        p0 = pool_at_offset(model, f, 0)
        d = (p0 - f.patch_in).abs().max().item()
        print(f"T1 pool_at_offset(0) vs trunk input : max|d|={d:.3e}  "
              f"{'PASS' if d == 0 else 'FAIL'}")

        g0 = teacher_globals(model, f, 0)[:, :P]
        gs = f.glob[:, :P]
        r = rel(g0.float(), gs.float())
        cos = torch.nn.functional.cosine_similarity(
            g0.float().flatten(0, 1), gs.float().flatten(0, 1), dim=-1).mean().item()
        print(f"T2 teacher(0) vs standard trunk     : rel={r:.3e} cos={cos:.6f} "
              f"max|d|={(g0 - gs).abs().max().item():.3e}  "
              f"{'PASS' if r < 2e-2 else 'FAIL'}")

        # T3: perturb one partial slot, check the others are bit-identical
        alt = pool_at_offset(model, f, 2)[:, :P]
        cache = [(k[:, :P], v[:, :P]) for k, v in f.cache]
        a = trunk_partial(model.trunk, alt, cache, f.freq_cis)
        alt2 = alt.clone()
        alt2[:, P // 2] += 10.0
        b = trunk_partial(model.trunk, alt2, cache, f.freq_cis)
        diff = (a - b).abs().amax(dim=-1)
        moved = (diff > 0).nonzero()
        ok = bool((moved[:, 1] == P // 2).all())
        print(f"T3 partial slots independent        : slots moved={sorted(set(moved[:, 1].tolist()))[:5]}"
              f" (expect [{P // 2}])  {'PASS' if ok else 'FAIL'}")

        # T4: zero a LATER standard patch, check earlier partial slots are untouched
        j = P // 2
        cache2 = [(k.clone(), v.clone()) for k, v in cache]
        for k, v in cache2:
            k[:, j + 1:] = 0
            v[:, j + 1:] = 0
        c = trunk_partial(model.trunk, alt, cache2, f.freq_cis)
        d4 = (a[:, : j + 1] - c[:, : j + 1]).abs().max().item()
        print(f"T4 no dependence on later patches   : max|d| over slots<=j: {d4:.3e}  "
              f"{'PASS' if d4 == 0 else 'FAIL'}")

        print("\nT5 magnitudes (RMS over slots):")
        print(f"   ||trunk input  p||   = {f.patch_in[:, :P].float().norm(dim=-1).mean():.3f}")
        print(f"   ||trunk output g||   = {f.glob[:, :P].float().norm(dim=-1).mean():.3f}")
        print(f"   ||up_norm(g)||       = "
              f"{model.transitions[0].up_norm(f.glob)[:, :P].float().norm(dim=-1).mean():.3f}")
        for r_ in (1, 2, 4, 8):
            pa = pool_at_offset(model, f, r_)[:, :P].float()
            gt = teacher_globals(model, f, r_)[:, :P].float()
            cs = torch.nn.functional.cosine_similarity(
                gt.flatten(0, 1), gs.float().flatten(0, 1), dim=-1).mean().item()
            print(f"   r={r_:2d}  ||p~-p||={((pa - f.patch_in[:, :P].float()).norm(dim=-1).mean()):8.3f}"
                  f"  ||g_t-g||={((gt - gs.float()).norm(dim=-1).mean()):8.3f}"
                  f"  cos(g_t,g)={cs:.4f}")


if __name__ == "__main__":
    main()

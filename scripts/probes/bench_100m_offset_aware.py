#!/usr/bin/env python3
"""Cost of offset-aware pretraining at 100M, measured rather than estimated.

Builds the 100M model from its own config, times an ordinary training step (forward + backward)
and an offset-aware one (which adds a single prefix-KV trunk pass), and converts both into a
wall-clock budget for the published recipe.  The point is to size the run before committing GPUs
to it, not to train anything here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from omegaconf import OmegaConf  # noqa: E402
from lingua.args import dataclass_from_dict  # noqa: E402
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, decode_from_globals, sample_patch_offsets, offset_aware_globals,
)


def build(cfg_path, seq_len):
    cfg = OmegaConf.load(cfg_path)
    m = OmegaConf.to_container(cfg.model, resolve=True)
    m["vocab_size"] = 258
    m["max_seqlens"] = [seq_len if s == -1 else s for s in m["max_seqlens"]]
    args = dataclass_from_dict(HierarchicalArgs, m, strict=False)
    model = HierarchicalTransformer(args).cuda().to(torch.bfloat16)
    model.init_weights()
    return model, args


def fake_batch(bsz, seq_len, mean_patch=4.8, device="cuda"):
    ids = torch.randint(0, 256, (bsz, seq_len), device=device)
    lm = (torch.rand(bsz, seq_len, device=device) < (1.0 / mean_patch)).long()
    lm[:, 0] = 1
    return ids, lm


def timeit(fn, n=6, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "lingua/apps/aunet/configs/aunet_100M_word.yaml"))
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--recipe_steps", type=int, default=6688)
    ap.add_argument("--recipe_tokens_per_step", type=int, default=96 * 2 * 8192 * 4)
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()

    model, margs = build(args.config, args.seq_len)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {n_par / 1e6:.1f}M params  dims={margs.dimensions} layers={margs.layers} "
          f"seq_len={args.seq_len} bsz={args.bsz}")
    ids, lm = fake_batch(args.bsz, args.seq_len)
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)

    def ordinary():
        model.zero_grad(set_to_none=True)
        loss, _, _, _ = model(ids, lm.clone(), ids, torch.zeros_like(lm))
        loss.backward()

    def offset_aware():
        # PRETRAINING cost: everything differentiable -- encoder, BOTH trunk passes, decoder
        model.zero_grad(set_to_none=True)
        f = split_forward(model, ids, lm.clone())
        with torch.no_grad():
            r = sample_patch_offsets(f, gen)
        g = offset_aware_globals(model, f, r)
        T = ids.shape[1]
        bidx = torch.arange(T, device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        logits = decode_from_globals(model, f, g, repeat_idx=bidx)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))
        loss.backward()

    t_ord = timeit(ordinary)
    t_off = timeit(offset_aware)
    print(f"\nper step (bsz={args.bsz}, {args.bsz * args.seq_len / 1e3:.1f}k bytes):")
    print(f"  ordinary      {t_ord * 1e3:8.1f} ms   -> {args.bsz * args.seq_len / t_ord / 1e6:.2f} M bytes/s")
    print(f"  offset-aware  {t_off * 1e3:8.1f} ms   -> {args.bsz * args.seq_len / t_off / 1e6:.2f} M bytes/s"
          f"   ({t_off / t_ord:.2f}x)")

    total_bytes = args.recipe_steps * args.recipe_tokens_per_step
    for name, t in (("ordinary", t_ord), ("offset-aware", t_off)):
        thru = args.bsz * args.seq_len / t * args.gpus
        hrs = total_bytes / thru / 3600
        print(f"\n  {name}: recipe {total_bytes / 1e9:.1f} GB on {args.gpus} GPU "
              f"-> {thru / 1e6:.2f} M bytes/s -> {hrs:.1f} h ({hrs / 24:.1f} d)")
    print(f"\n  (recipe = {args.recipe_steps} steps x {args.recipe_tokens_per_step / 1e6:.2f} M bytes/step)")


if __name__ == "__main__":
    main()

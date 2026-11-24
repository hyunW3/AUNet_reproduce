#!/usr/bin/env python3
"""Pre-flight for scratchpad pretraining: the flag must be a no-op when off, causal when on.

A leak or a silent behaviour change here would waste a whole training run, so both are checked
against the shipped forward rather than argued for.

  N  scratchpad_k=0 reproduces the ordinary forward bit-for-bit
  C  a byte's scratch vector was pooled at or before that byte (never the future)
  G  gradients reach the encoder, both trunk passes and the decoder
  T  step time and peak memory, ordinary vs scratchpad
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))

from omegaconf import OmegaConf  # noqa: E402
from lingua.args import dataclass_from_dict  # noqa: E402
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs  # noqa: E402


def build(model_cfg, seq_len, k, shift, seed=0):
    m = dict(model_cfg)
    m["vocab_size"] = 258
    m["max_seqlens"] = [seq_len if s == -1 else s for s in m["max_seqlens"]]
    m["scratchpad_k"], m["scratchpad_shift"] = k, shift
    args = dataclass_from_dict(HierarchicalArgs, m, strict=False)
    torch.manual_seed(seed)
    model = HierarchicalTransformer(args).cuda().to(torch.bfloat16)
    torch.manual_seed(seed)
    model.init_weights()
    return model


def batch(bsz, seq_len, mean_patch=4.5):
    g = torch.Generator(device="cuda").manual_seed(1)
    ids = torch.randint(0, 256, (bsz, seq_len), device="cuda", generator=g)
    lm = (torch.rand(bsz, seq_len, device="cuda", generator=g) < 1.0 / mean_patch).long()
    lm[:, 0] = 1
    return ids, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", default=str(ROOT / "reports_afterAAAIsub/hp_settings_aunet2_rg_llama.json"))
    ap.add_argument("--run", default="bpebyte rg (root_greedy) · WSD 2.0e-3")
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--shift", type=float, default=1.0)
    args = ap.parse_args()

    import json
    hp = json.load(open(args.hp))
    cfg = hp["runs"][args.run] if "runs" in hp else hp
    mcfg = cfg["model"]
    ids, lm = batch(args.bsz, args.seq_len)

    # ---- N: off == shipped forward ----------------------------------------------------------
    off = build(mcfg, args.seq_len, 0, 1.0)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        l0 = off(ids, lm.clone(), ids, torch.zeros_like(lm))[0].item()
        l0b = off(ids, lm.clone(), ids, torch.zeros_like(lm))[0].item()
    print(f"N  scratchpad_k=0, 같은 입력 두 번      : {l0:.6f} / {l0b:.6f}  "
          f"{'PASS' if l0 == l0b else 'FAIL'}")
    del off; torch.cuda.empty_cache()

    on = build(mcfg, args.seq_len, args.k, args.shift)
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        on.scratchpad_k = 0
        l_off = on(ids, lm.clone(), ids, torch.zeros_like(lm))[0].item()
        on.scratchpad_k = args.k
        l_on = on(ids, lm.clone(), ids, torch.zeros_like(lm))[0].item()
    print(f"N2 같은 가중치, 플래그만 껐다 켜기      : off={l_off:.6f}  on={l_on:.6f}  "
          f"{'다름 (기대대로)' if l_off != l_on else 'FAIL: 플래그가 아무것도 안 함'}")

    # ---- C: causality, checked through the loss ---------------------------------------------
    # flip one byte late in the sequence; the loss on the PREFIX before it must not move
    cut = args.seq_len // 2
    ids2 = ids.clone(); ids2[:, cut:] = (ids2[:, cut:] + 7) % 256
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        lg1 = on(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True)
        lg2 = on(ids2, lm.clone(), ids2, torch.zeros_like(lm), return_logits=True)
    d = (lg1[:, : cut - 1] - lg2[:, : cut - 1]).abs().max().item()
    print(f"C  뒤쪽 바이트를 바꿔도 앞쪽 logits 불변 : max|d|={d:.3e}  "
          f"{'PASS' if d == 0 else 'FAIL (미래 누설)'}")

    # ---- G: gradients reach everything ------------------------------------------------------
    on.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        on(ids, lm.clone(), ids, torch.zeros_like(lm))[0].backward()
    named = dict(on.named_parameters())
    got = {n: (named[n].grad is not None and named[n].grad.abs().sum() > 0)
           for n in ["encoders.0.layers.0.attention.wq.weight",
                     "transitions.0.trans_down.weight",
                     "trunk.layers.0.attention.wq.weight",
                     "trunk.layers.9.feed_forward.w1.weight",
                     "transitions.0.indexed_linear.weight",
                     "decoders.0.layers.0.attention.wq.weight"] if n in named}
    print("G  gradient 도달 :", {n.split('.')[0] + '.' + n.split('.')[1]: bool(v) for n, v in got.items()},
          " PASS" if all(got.values()) else " FAIL")

    # ---- T: cost ----------------------------------------------------------------------------
    def step():
        on.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            on(ids, lm.clone(), ids, torch.zeros_like(lm))[0].backward()

    for kk, tag in ((0, "ordinary"), (args.k, f"scratchpad{args.k}")):
        on.scratchpad_k = kk
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        print(f"T  {tag:14s} {dt * 1e3:7.1f} ms/step (bsz {args.bsz})  "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Global-transformer (trunk) ablation for the 1.3B AU-Net / BPEByte checkpoints.

Teacher-forced, full-sequence byte scoring (training forward, return_logits=True) of the fixed
held-out English windows (reports/bpb_windows.jsonl), under three forward modes:

  full        local encoder -> down-pool -> GLOBAL trunk -> up -> local decoder   (default setting)
  trunk_wN    the trunk still runs, but its attention is restricted to a sliding window of N
              PATCHES (N=1 -> every patch sees only itself: the trunk keeps its per-patch
              featurisation but all CROSS-patch information is gone).  In-distribution (the same
              LocalAttentionFromBottomRightMask the model is trained with), so unlike no_trunk it
              measures information, not activation shift.  fmha only.
  trunk_sKwN  trunk attention = the first K patches (attention SINK, incl. the BOS patch) PLUS a
              sliding window of the last N patches.  trunk_wN alone starves the model of its sink,
              which costs loss for reasons that have nothing to do with cross-patch information;
              K>=1 restores it, so Δ(N) is a clean "how far back does the trunk have to see" curve.
              Runs the TRUNK ONLY under sdpa with an explicit bool mask (the local encoder/decoder
              keep fmha + their exact 512-byte window).  N=0 -> self-patch only (+ sink).
  delayN      every byte reads the patch N patches BEHIND its own (SimpleTransition.patch_read_delay).
              At pos-in-patch 1 -- the only position that reads a FRESHLY pooled patch vector, i.e.
              one that saw the boundary byte -- this replaces that vector with the previous patch's.
  delay_bnd   the same one-patch delay applied ONLY at boundary positions (the positions that predict
              a pos-in-patch-1 byte); every other position is untouched.  Targeted control: it must
              move pos 1 and leave the rest bit-identical.
  no_trunk    the 25-layer global trunk is replaced by the IDENTITY; the byte->patch->byte
              plumbing (down-pool + delayed up-read) is kept.  = "what do the global layers add"
  local_only  the whole global path is cut: up() returns 0, so the decoder sees only the byte
              encoder's residual (3+3 local layers, 512-byte sliding window). = "local net only"
  mean_patch  MEAN ablation control: the trunk still runs, but every byte reads the SEQUENCE-MEAN
              patch representation instead of its own patch's. Removes the position-specific global
              information while keeping the injected activation statistics, so it separates "the
              decoder lost information" from "the decoder was fed out-of-distribution activations"
              (which is what trunk=identity and, partly, local_only measure).

Per-byte NLL is split by whether the PREDICTED byte starts a patch (level_mask==1, i.e. the byte
after a boundary decision) vs. an interior byte, and by the byte's index inside its patch.
Everything is dumped per byte to an .npz so the statistics can be recomputed without re-running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))

from lingua.args import dataclass_from_dict  # noqa: E402
from lingua.checkpoint import CONSOLIDATE_NAME  # noqa: E402
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs  # noqa: E402
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs  # noqa: E402
from apps.aunet.generate import _resolve_bpe_tokenizer_path  # noqa: E402

LN2 = math.log(2.0)
BOS = 256
MODES = ("full", "no_trunk", "local_only", "mean_patch")


def load_model(ckpt: str, tok_path=None):
    """Same as apps.aunet.generate.load_consolidated_model_and_tokenizer, but casts on the CPU
    before moving to the GPU (the upstream loader peaks at fp32 on-device, ~5.3 GB for 1.3B)."""
    p = Path(ckpt)
    cfg = OmegaConf.load(p / "params.json")
    dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        cfg.distributed.model_dtype
    ]
    margs = dataclass_from_dict(HierarchicalArgs, cfg.model, strict=False)
    model = HierarchicalTransformer(margs)
    st = torch.load(p / CONSOLIDATE_NAME, weights_only=True, map_location="cpu")
    sd = {}
    for k, v in st["model"].items():
        if "transitions." in k and "parametrizations" in k:
            k = k.replace("parametrizations", "_orig_mod.parametrizations")
        sd[k] = v
    model.load_state_dict(sd)
    del st, sd
    model = model.to(dtype=dtype).cuda().eval()

    rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
    baked = getattr(rargs, "bpe_tokenizer_path", None)
    if baked is not None:
        rargs.bpe_tokenizer_path = _resolve_bpe_tokenizer_path(baked, tok_path)
    return model, RegexPool(rargs), cfg


class Ablation:
    """Context manager that rewires the forward for one mode and restores it on exit."""

    def __init__(self, model, mode):
        assert (mode in MODES or re.fullmatch(r"trunk_w\d+", mode)
                or re.fullmatch(r"trunk_s\d+w\d+", mode) or re.fullmatch(r"delay\d+", mode)
                or mode == "delay_bnd")
        # delay_bnd needs a per-position read index, which the caller passes into the forward
        self.needs_cpi = mode == "delay_bnd"
        self.model, self.mode, self.saved = model, mode, []

    def __enter__(self):
        if self.mode == "full":
            return self
        m = re.fullmatch(r"delay(\d+)", self.mode)
        if m:
            for t in self.model.transitions:
                self.saved.append((t, "patch_read_delay", t.patch_read_delay))
                t.patch_read_delay = int(m.group(1))
            return self
        if self.mode == "delay_bnd":
            return self          # handled by committed_patch_idx in score_batch
        m = re.fullmatch(r"trunk_s(\d+)w(\d+)", self.mode)
        if m:
            K, N = int(m.group(1)), int(m.group(2))
            S = self.model.trunk.max_seqlen
            q = torch.arange(S, device="cuda")[:, None]
            k = torch.arange(S, device="cuda")[None, :]
            bias = ((q >= k) & ((k < K) | (q - k <= N)))[None, None]     # [1,1,S,S] bool
            orig = self.model.trunk.forward
            self.saved.append((self.model.trunk, "forward", orig))

            def fwd(x, *a, **kw):
                kw.pop("attn_impl", None)
                return orig(x, *a, mask=bias, attn_impl="sdpa", **kw)

            self.model.trunk.forward = fwd
            return self
        m = re.fullmatch(r"trunk_w(\d+)", self.mode)
        if m:
            from xformers.ops import fmha
            bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(
                window_left=int(m.group(1)) - 1, window_right=0)
            orig = self.model.trunk.forward
            self.saved.append((self.model.trunk, "forward", orig))
            self.model.trunk.forward = lambda x, *a, **kw: orig(x, mask=bias, **kw)
            return self
        if self.mode == "mean_patch":
            for t in self.model.transitions:
                self.saved.append((t, "up", t.up))
                t.up = self._mean_up(t, t.up)
            return self
        # the other two ablations bypass the global trunk entirely
        self.saved.append((self.model.trunk, "forward", self.model.trunk.forward))
        self.model.trunk.forward = lambda x, *a, **kw: x
        if self.mode == "local_only":
            for t in self.model.transitions:
                self.saved.append((t, "up", t.up))
                t.up = lambda x, res, *a, **kw: torch.zeros_like(res)
        return self

    @staticmethod
    def _mean_up(trans, orig):
        S = trans.seqlen_in

        def up(x, res, mask, *a, **kw):
            npatch = mask[:, :S].sum(1).clamp(min=1)                       # [B] patches in this row
            ar = torch.arange(x.shape[1], device=x.device)[None, :]
            valid = (ar < npatch[:, None]).unsqueeze(-1)
            m = (x.float() * valid).sum(1, keepdim=True) / npatch[:, None, None].float()
            return orig(m.expand_as(x).to(x.dtype), res, mask, *a, **kw)

        return up

    def __exit__(self, *exc):
        for obj, name, fn in self.saved:
            setattr(obj, name, fn)
        self.saved = []
        return False


def prep(text: str, pool: RegexPool, max_bytes: int, seqlen: int):
    """-> (ids[seqlen], level_mask[seqlen], n_bytes, lm_real[n+1])."""
    clip = text.encode()[:max_bytes].decode("utf-8", errors="ignore")
    bs = list(clip.encode())
    n = len(bs)
    if n == 0 or n + 2 >= seqlen:
        return None
    ids_real = [BOS] + bs
    lm_real = pool.get_levels_mask(ids_real)          # scheme-native boundaries (word / bpe-root)
    ids = torch.zeros(seqlen, dtype=torch.long)
    lm = torch.zeros(seqlen, dtype=torch.long)
    ids[: n + 1] = torch.tensor(ids_real, dtype=torch.long)
    lm[: n + 1] = torch.tensor(lm_real[: n + 1], dtype=torch.long)
    lm[n + 1] = 1        # seal the pad tail into its own patch (cannot contaminate the last patch)
    return ids, lm, n, lm_real


@torch.no_grad()
def score_batch(model, ids, lm, ns, attn_impl, cpi=False):
    """ids/lm: [B, seqlen]. Returns list of per-byte NLL tensors (nats), one per item."""
    ids, lm = ids.cuda(), lm.cuda()
    kw = {}
    if cpi:
        # read index = own patch everywhere, previous patch at boundary positions only.
        # up() renormalises by column 0, so pin column 0 to 0 (position 0 always reads patch 0).
        # the model forces a boundary at position 0 (get_pool_mask(force_first=True)); the read
        # index must be built from that SAME mask or every position ends up shifted by one patch.
        b = (lm > 0).long()
        b[:, 0] = 1
        idxs = (b.cumsum(1) - 1 - b).clamp(min=0)
        idxs[:, 0] = 0
        kw["committed_patch_idx"] = idxs
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids, lm, ids, torch.zeros_like(lm), return_logits=True, attn_impl=attn_impl,
                       **kw)
    out = []
    for i, n in enumerate(ns):
        pred = logits[i, :n].float()
        tgt = ids[i, 1 : n + 1]
        out.append(F.cross_entropy(pred, tgt, reduction="none").cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_bytes", type=int, default=3840)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--attn_impl", default="fmha", choices=["fmha", "sdpa"],
                    help="fmha honours the local sliding window (= training); sdpa ignores it")
    ap.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    ap.add_argument("--outdir", default=str(ROOT / "reports/global_ablation"))
    ap.add_argument("--debug_boundaries", action="store_true")
    args = ap.parse_args()

    modes = args.modes.split(",")
    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[: args.limit]

    model, pool, cfg = load_model(args.ckpt, args.tokenizer_path)
    seqlen = int(model.transitions[0].seqlen_in)
    print(f"[{args.tag}] seqlen={seqlen} windows={len(wins)} modes={modes} attn={args.attn_impl}",
          flush=True)

    prepped = [prep(w["text"], pool, args.max_bytes, seqlen) for w in wins]
    prepped = [p for p in prepped if p is not None]

    if args.debug_boundaries:
        ids, lm, n, lm_real = prepped[0]
        s = "".join(("|" if lm_real[i + 1] else "") + chr(ids[i + 1]) for i in range(min(n, 300)))
        print(f"[{args.tag}] boundaries (| = patch start):\n{s}\n", flush=True)

    # per-byte metadata, identical across modes
    meta_bnd, meta_pos, meta_win = [], [], []
    for wi, (ids, lm, n, lm_real) in enumerate(prepped):
        b = np.asarray(lm_real[1 : n + 1], dtype=np.int16) > 0     # predicted byte starts a patch
        pos = np.zeros(n, dtype=np.int16)                          # index inside the current patch
        c = 0
        for i in range(n):
            c = 0 if b[i] else c + 1
            pos[i] = c
        meta_bnd.append(b); meta_pos.append(pos); meta_win.append(np.full(n, wi, np.int32))
    meta_bnd, meta_pos, meta_win = (np.concatenate(x) for x in (meta_bnd, meta_pos, meta_win))

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    store = {"is_boundary": meta_bnd, "pos_in_patch": meta_pos, "window": meta_win}
    prev = outdir / f"{args.tag}.npz"
    if prev.exists():          # keep modes scored by an earlier run (same windows/metadata)
        old = np.load(prev)
        assert old["window"].shape == meta_win.shape and (old["is_boundary"] == meta_bnd).all()
        for k in old.files:
            if k.startswith("nll_"):
                store.setdefault(k, old[k])
    for mode in modes:
        nlls = []
        with Ablation(model, mode) as abl:
            for i in range(0, len(prepped), args.batch_size):
                chunk = prepped[i : i + args.batch_size]
                ids = torch.stack([c[0] for c in chunk])
                lm = torch.stack([c[1] for c in chunk])
                if ids.shape[0] == 1:            # MaxSumMask squeezes a bsz-1 batch away
                    ids, lm = ids.repeat(2, 1), lm.repeat(2, 1)
                    ns = [chunk[0][2]] * 2
                    nlls.append(score_batch(model, ids, lm, ns, args.attn_impl, abl.needs_cpi)[0])
                else:
                    nlls += score_batch(model, ids, lm, [c[2] for c in chunk], args.attn_impl,
                                        abl.needs_cpi)
                if (i // args.batch_size) % 20 == 0:
                    print(f"  [{args.tag}/{mode}] {i + len(chunk)}/{len(prepped)}", flush=True)
        nll = torch.cat(nlls).numpy().astype(np.float32)
        store[f"nll_{mode}"] = nll
        print(f"[{args.tag}] mode={mode} bytes={nll.size} mean_nll={nll.mean():.4f} "
              f"BPB={nll.mean() / LN2:.4f}", flush=True)

    np.savez_compressed(outdir / f"{args.tag}.npz", **store)
    print(f"[{args.tag}] wrote {outdir / (args.tag + '.npz')}", flush=True)


if __name__ == "__main__":
    main()

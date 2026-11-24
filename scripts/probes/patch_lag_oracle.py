#!/usr/bin/env python3
"""Stage -1 of plans/patch_lag_residual_updater_implementation_plan.md: how much is patch lag
actually worth?

A byte at offset r inside its patch reads a global vector that was pooled at the patch's FIRST
byte, so it has never seen the r bytes of the word being written.  This script measures what the
model would do if that vector were refreshed -- the Scratchpad / oracle upper bound -- and two
controls, on the same held-out windows and with the same per-byte bookkeeping as
`scripts/probes/global_ablation.py`.

Modes (all scored teacher-forced, per byte):

  full        the unmodified model, but with the trunk run through this file's own sdpa path.
              Identical maths to the shipped fmha trunk (its sliding window, 4096, exceeds the
              3200 patch slots), so `--check` reports the fp difference as a sanity number.
  oracle      byte at staleness r reads the trunk output patch j WOULD have had if it had been
              pooled at byte start_j + r.  = Scratchpad with an exact prefix, i.e. the ceiling for
              any updater.  The IndexedMatMul offset index stays r (what the model was trained on).
  oracle_p0   the same vectors, but with the offset index forced to 0 -- the index that pairs with
              a fresh vector during training.  Separates "the vector was patch_offset" from "the vector
              and the offset index no longer agree".
  m0          zero-parameter control: inject the RAW pooled difference into the post-trunk vector,
              g + (pool(start+r) - pool(start)), i.e. the current patch's content WITHOUT the
              trunk's 25 layers reprocessing it.  Splits "content injection" from "global compute".
  look1       LEAKY upper bound: a patch_offset byte reads the NEXT patch's vector.  That vector is a
              genuine word-boundary pooled vector (in-distribution for the frozen trunk) and it
              has seen the whole current word -- but it has also seen the bytes we are predicting,
              so this OVER-estimates.  Useful only in the negative direction: if even look1 buys
              little, no updater can buy more.  `look1_p0` is the same with offset index 0.
  oracle_full same re-pooling, but the trunk runs as one ordinary forward with EVERY slot replaced
              by its offset-r vector, so the partial vectors also see each other.  No custom
              attention anywhere -- the control for "is the prefix-KV shortcut the problem?"
  acc<k>      ADDITIVE, not substitutive: k extra trunk slots per patch, pooled at the k bytes
              after the boundary, so the global path refreshes k+1 times per patch.  acc0 must
              reproduce `full` exactly.  `accdup<k>` lengthens the sequence identically but fills
              the extra slots with DUPLICATES of the boundary vector -- same length, same RoPE
              shift, no new information, so it separates structure from content.
  scratchpad<k>
              The transient scratch entry: patch j keeps its ordinary entry AND gets a second one
              pooled at start_j+k appended after it, which its own bytes read from offset k on.
              Nothing else attends to it -- the moment the next patch starts the scratch entry is
              dropped and only the ordinary entry stays in the KV.  A `_5` or `frac` suffix puts
              it at position j+0.5 instead of j+1.
  refresh<k>  the REPLACE variant: same scratch vector, but it stands in for the patch's ordinary
              entry rather than following it (the partial does not attend to start_j).  Kept as
              the control that shows what the ordinary entry is worth as context.
  (old) refresh<k>app == scratchpad<k>.  TRANSIENT version of acc<k>: patch j is re-pooled at start_j+k and read by its own
              bytes from offset k on, but the re-pooled vector never joins the trunk sequence --
              no RoPE position moves and no later patch attends to it.  Isolates the CONTENT of a
              mid-patch vector from every structural consequence of adding one.
  prev1       the mirror control: a patch_offset byte reads the PREVIOUS patch's vector.  Together with
              look1 it puts "one patch back / own / one patch forward" on one axis, separating
              "the vector is patch_offset" from "the vector is the wrong patch's".

Positions whose read vector is already fresh (staleness 0) are bit-identical to `full` in every
mode by construction; positions with staleness > --max_offset fall back to `full` and are counted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_ablation import load_model, prep, LN2  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, decode_from_globals, teacher_globals, teacher_globals_full, pool_at_offset,
    accumulate_globals, refresh_globals,
)

MODES = ("oracle", "oracle_full", "oracle_p0", "m0", "look1", "look1_p0", "prev1")


def nll_of(logits, ids, n):
    """[B,T,V] logits -> per-item per-byte NLL (nats) for the first n bytes."""
    out = []
    for i, ni in enumerate(n):
        out.append(F.cross_entropy(logits[i, :ni].float(), ids[i, 1 : ni + 1],
                                   reduction="none"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="aunet_static")
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_bytes", type=int, default=3840)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_offset", type=int, default=15)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--attn_impl", default="fmha", choices=["fmha", "sdpa"])
    ap.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag"))
    ap.add_argument("--check", action="store_true",
                    help="also score the stock model.forward and report the BPB difference")
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[: args.limit]

    torch.set_grad_enabled(False)          # Stage -1 is inference only
    model, pool, cfg = load_model(args.ckpt, args.tokenizer_path)
    seqlen = int(model.transitions[0].seqlen_in)
    print(f"[{args.tag}] seqlen={seqlen} windows={len(wins)} modes={modes} "
          f"max_offset={args.max_offset}", flush=True)

    prepped = [p for p in (prep(w["text"], pool, args.max_bytes, seqlen) for w in wins)
               if p is not None]

    store = {m: [] for m in modes}
    store["full"] = []
    if args.check:
        store["full_fmha"] = []
    meta_stale, meta_pos, meta_bnd, meta_win = [], [], [], []
    beyond = 0
    t0 = time.time()

    bs = max(2, args.batch_size)          # MaxSumMask squeezes a bsz-1 batch away
    for bi in range(0, len(prepped), bs):
        chunk = prepped[bi : bi + bs]
        pad = 2 - len(chunk) if len(chunk) < 2 else 0
        rows = chunk + [chunk[-1]] * pad
        ids = torch.stack([c[0] for c in rows]).cuda()
        lm = torch.stack([c[1] for c in rows]).cuda()
        ns = [c[2] for c in rows]
        keep = len(chunk)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = split_forward(model, ids, lm, attn_impl=args.attn_impl)
            T = f.res.shape[1]
            byte_idx = torch.arange(T, device=ids.device).unsqueeze(0).expand(len(rows), -1)
            exp = lambda g: g.gather(1, f.repeat_idx.unsqueeze(-1).expand(-1, -1, g.shape[-1]))

            # every mode is scored as ONE self-consistent model: a per-BYTE global vector, so a
            # byte's neighbours (which the 512-byte decoder window sees) carry the same rule.
            g_base = exp(f.glob)                                        # [B,T,d]
            base = decode_from_globals(model, f, g_base, repeat_idx=byte_idx,
                                       attn_impl=args.attn_impl)
            base_nll = nll_of(base, ids, ns)

            g_or, g_orf, g_m0 = g_base.clone(), g_base.clone(), g_base.clone()
            need_or = any(m.startswith("oracle") or m.startswith("interp") for m in modes)
            for r in range(1, args.max_offset + 1):
                hit = (f.patch_offset == r)
                if not bool(hit.any()):
                    continue
                if need_or:
                    g_or = torch.where(hit.unsqueeze(-1), exp(teacher_globals(model, f, r)), g_or)
                if "oracle_full" in modes:
                    g_orf = torch.where(hit.unsqueeze(-1),
                                        exp(teacher_globals_full(model, f, r)), g_orf)
                if "m0" in modes:
                    d = exp(pool_at_offset(model, f, r) - f.patch_in)
                    g_m0 = torch.where(hit.unsqueeze(-1), g_base + d, g_m0)

            acc = {}
            zero = torch.zeros_like(f.patch_offset)
            for m in modes:                      # interp<beta>: g + beta*(g_oracle - g)
                mm = re.fullmatch(r"interp([0-9.]+)", m)
                if mm:
                    beta = float(mm.group(1))
                    acc[m] = nll_of(decode_from_globals(
                        model, f, g_base + beta * (g_or - g_base), repeat_idx=byte_idx,
                        attn_impl=args.attn_impl), ids, ns)
            if "oracle" in modes:
                acc["oracle"] = nll_of(decode_from_globals(
                    model, f, g_or, repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)
            if "oracle_p0" in modes:
                acc["oracle_p0"] = nll_of(decode_from_globals(
                    model, f, g_or, repeat_idx=byte_idx, position=zero,
                    attn_impl=args.attn_impl), ids, ns)
            if "oracle_full" in modes:
                acc["oracle_full"] = nll_of(decode_from_globals(
                    model, f, g_orf, repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)
            if "m0" in modes:
                acc["m0"] = nll_of(decode_from_globals(
                    model, f, g_m0, repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)
            if "look1" in modes or "look1_p0" in modes:
                nxt = (f.repeat_idx + 1).minimum(
                    (f.n_patches - 1).unsqueeze(1).to(f.repeat_idx.dtype))
                g_next = f.glob.gather(1, nxt.unsqueeze(-1).expand(-1, -1, f.glob.shape[-1]))
                g_lk = torch.where((f.patch_offset >= 1).unsqueeze(-1), g_next, g_base)
                if "look1" in modes:
                    acc["look1"] = nll_of(decode_from_globals(
                        model, f, g_lk, repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)
                if "look1_p0" in modes:
                    acc["look1_p0"] = nll_of(decode_from_globals(
                        model, f, g_lk, repeat_idx=byte_idx, position=zero,
                        attn_impl=args.attn_impl), ids, ns)

            for m in modes:                      # acc<k>: k extra global slots per patch
                mm = re.fullmatch(r"acc(dup)?(\d+)(frac|int|same)?", m)
                if mm:
                    gb, off = accumulate_globals(model, f, int(mm.group(2)),
                                                 dup=mm.group(1) is not None,
                                                 pos=mm.group(3) or "table")
                    acc[m] = nll_of(decode_from_globals(
                        model, f, gb, position=off % model.transitions[0].max_pos,
                        repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)

            for m in modes:                      # refresh<k>: transient re-pool, sequence unchanged
                mm = re.fullmatch(r"(?:refresh(\d+)(app)?|scratchpad(\d+))(?:_(\d+)|(frac))?", m)
                if mm:
                    sp = mm.group(3) is not None            # scratchpad<k> == refresh<k>app
                    app = sp or mm.group(2) is not None
                    # `frac` == `_5`: put the scratch entry at j+0.5 rather than j+1.  Note this
                    # means something different from acc's `frac`, where the sequence lengthens and
                    # fractional positions are what KEEP the original entries where they were; a
                    # scratchpad never lengthens the sequence, so both variants already preserve them.
                    sh = 0.5 if mm.group(5) else \
                        ((int(mm.group(4)) / 10) if mm.group(4) else (1.0 if app else 0.0))
                    acc[m] = nll_of(decode_from_globals(
                        model, f, refresh_globals(model, f, int(mm.group(1) or mm.group(3)),
                                                  append=app, shift=sh),
                        repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)

            if "prev1" in modes:
                prv = (f.repeat_idx - 1).clamp(min=0)
                g_prev = f.glob.gather(1, prv.unsqueeze(-1).expand(-1, -1, f.glob.shape[-1]))
                acc["prev1"] = nll_of(decode_from_globals(
                    model, f, torch.where((f.patch_offset >= 1).unsqueeze(-1), g_prev, g_base),
                    repeat_idx=byte_idx, attn_impl=args.attn_impl), ids, ns)

            if args.check:
                chk = model(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True,
                            attn_impl=args.attn_impl)
                chk_nll = nll_of(chk, ids, ns)

        for i in range(keep):
            n = ns[i]
            store["full"].append(base_nll[i].float().cpu().numpy())
            for m in modes:
                store[m].append(acc[m][i].float().cpu().numpy())
            if args.check:
                store["full_fmha"].append(chk_nll[i].float().cpu().numpy())
            st = f.patch_offset[i, :n].cpu().numpy().astype(np.int16)
            meta_stale.append(st)
            beyond += int((st > args.max_offset).sum())
            p = chunk[i][3][1 : n + 1]                       # scheme-native level mask
            b = np.asarray(p, dtype=np.int16) > 0
            posp = np.zeros(n, dtype=np.int16)
            c = 0
            for j in range(n):
                c = 0 if b[j] else c + 1
                posp[j] = c
            meta_bnd.append(b); meta_pos.append(posp)
            meta_win.append(np.full(n, bi + i, np.int32))

        if (bi // bs) % 10 == 0:
            el = time.time() - t0
            print(f"  [{args.tag}] {bi + keep}/{len(prepped)}  {el:.0f}s "
                  f"({el / max(1, bi + keep):.2f}s/win)", flush=True)

    out = {"patch_offset": np.concatenate(meta_stale), "pos_in_patch": np.concatenate(meta_pos),
           "is_boundary": np.concatenate(meta_bnd), "window": np.concatenate(meta_win)}
    for k, v in store.items():
        out[f"nll_{k}"] = np.concatenate(v).astype(np.float32)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"{args.tag}_oracle.npz", **out)

    nb = out["nll_full"].size
    print(f"\n[{args.tag}] bytes={nb}  patch_offset>{args.max_offset} (fell back to full): "
          f"{beyond} ({100 * beyond / nb:.2f}%)")
    base = out["nll_full"].mean() / LN2
    for k in out:
        if not k.startswith("nll_"):
            continue
        v = out[k].mean() / LN2
        print(f"  {k:16s} BPB={v:.4f}  delta={v - base:+.4f}")
    st = out["patch_offset"]
    print("\n  by patch_offset (BPB):")
    hdr = "  %-9s %8s" % ("patch_off", "n")
    for k in out:
        if k.startswith("nll_"):
            hdr += " %10s" % k[4:]
    print(hdr)
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 10 ** 6)]:
        m = (st >= lo) & (st <= hi)
        if not m.any():
            continue
        line = "  %-9s %8d" % (f"{lo}" if lo == hi else f"{lo}-{hi}", m.sum())
        for k in out:
            if k.startswith("nll_"):
                line += " %10.4f" % (out[k][m].mean() / LN2)
        print(line)
    print(f"\n[{args.tag}] wrote {outdir / (args.tag + '_oracle.npz')}", flush=True)


if __name__ == "__main__":
    main()

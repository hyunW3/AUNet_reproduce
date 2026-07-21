#!/usr/bin/env python3
"""Diagnose the residual PBP for byte models: is the nonzero delta a benign
boundary-byte effect or a real patch/alignment shift?

For each NIAH sample we build the two boundary cuts (identical total text):
  canonical: ctx = "...is",  cont = " <value>"
  space    : ctx = "...is ", cont = "<value>"
Then we compare the model's per-byte greedy predictions over the *value* bytes
between the two cuts. If they are identical, the parse is prefix-stable over the
value and the PBP delta is purely the extra boundary-space byte that canonical
must also emit. If they differ, moving the space re-parses the tail (a real
prefix-instability of the boundary).
"""
from __future__ import annotations
import os, sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fflm_probe import build_generator          # noqa: E402
import niah_data as N                            # noqa: E402


def probe(family, ckpt, label, n=30):
    gen, tok = build_generator(family, ckpt, os.environ.get("AUNET_TOK"), 8192)
    bl = len(tok.encode("hello world", add_bos=False, add_eos=False))
    print(f"\n### {label} | encode('hello world') -> {bl} tokens "
          f"({'byte-level' if bl == 11 else 'NOT byte-level'})")
    rng = random.Random(0)
    val_diff = space_flip = enc_unstable = 0
    for _ in range(n):
        s = N.make_sample(512, rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]), 7, rng, "noise", "number")
        base, v = s["prompt"].rstrip(" "), s["value"]
        vb = len(v.encode())                      # value byte length (7)
        pairs = [(base, " " + v), (base + " ", v)]   # canonical, space
        # encode prefix-stability (the assumption score() makes)
        for ctx, cont in pairs:
            e_ctx = tok.encode(ctx, add_bos=False, add_eos=False)
            e_full = tok.encode(ctx + cont, add_bos=False, add_eos=False)
            if list(e_full[:len(e_ctx)]) != list(e_ctx):
                enc_unstable += 1
        _, _lls, greedy = gen.generate([c + k for c, k in pairs])
        g_can, g_spc = greedy[0], greedy[1]
        gv_can = [bool(x) for x in g_can[-vb:].tolist()]   # value-byte greedy, canonical
        gv_spc = [bool(x) for x in g_spc[-vb:].tolist()]   # value-byte greedy, space
        if gv_can != gv_spc:
            val_diff += 1
        space_greedy = bool(g_can[-(vb + 1)].item())       # the boundary space byte (canonical only)
        if all(gv_spc) and not (space_greedy and all(gv_can)):
            space_flip += 1
    print(f"  encode(ctx) NOT a prefix of encode(ctx+cont): {enc_unstable}/{2*n} pairs")
    print(f"  VALUE-byte greedy differs (canonical vs space): {val_diff}/{n} "
          f"-> {'REAL patch shift' if val_diff else 'value parse is prefix-stable'}")
    print(f"  exact-match flips driven by the boundary-space byte only: {space_flip}/{n}")
    del gen


if __name__ == "__main__":
    R = os.environ["AUNET_ROOT"]
    probe("aunet", f"{R}/runs/small/cmp_100M/v4_root_greedy_ot/checkpoints/0000006688/consolidated",
          "BPEByte (root-greedy)")
    probe("aunet", f"{R}/runs/small/cmp_100M/aunet_orig_100M/checkpoints/0000001672/consolidated",
          "AU-Net (word-pool)")

#!/usr/bin/env python3
"""Is the byte models' large canonical->space PBP gap at 1.3B a real tokenization
effect, or the query-format confound (the needle says "is: <val>" with a colon, so
after "...is" the model greedily emits ':' rather than the scored ' ')?

For each sample we score the canonical cut and decompose the failure:
  - boundary-only failure : the FIRST answer byte (the space) is wrong but EVERY
    value byte is still the greedy argmax -> the model retrieves the value fine and
    only 'disagrees' about the boundary byte (the confound).
  - value failure         : a value byte itself is not greedy (a real retrieval/parse effect).
We also check whether the value-byte greedy is identical across the two cuts.
"""
from __future__ import annotations
import os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fflm_probe import build_generator          # noqa: E402
import niah_data as N                            # noqa: E402


def probe(family, ckpt, label, task="3", n=40):
    ht, vt = N.TASKS[task]
    gen, tok = build_generator(family, ckpt, os.environ.get("AUNET_TOK"), 8192)
    rng = random.Random(0)
    canon_fail = boundary_only = val_diff = 0
    old = gen.max_gen_len; gen.max_gen_len = 1
    try:
        for _ in range(n):
            s = N.make_sample(2048, rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]), 7, rng, ht, vt)
            base, v = s["prompt"].rstrip(" "), s["value"]
            _, _lls, greedy = gen.generate([base + " " + v, (base + " ") + v])
            pc = len(tok.encode(base, add_bos=False, add_eos=False))
            ps = len(tok.encode(base + " ", add_bos=False, add_eos=False))
            gc = [bool(x) for x in greedy[0][pc:].tolist()]   # [boundary-space, val...]
            gs = [bool(x) for x in greedy[1][ps:].tolist()]   # [val...]
            if not all(gc):
                canon_fail += 1
                if len(gc) > 1 and (not gc[0]) and all(gc[1:]):
                    boundary_only += 1
            if gc[1:] != gs:
                val_diff += 1
    finally:
        gen.max_gen_len = old
    den = canon_fail or 1
    print(f"### {label} [S-NIAH-{task}] n={n}")
    print(f"  canonical failures ......................... {canon_fail}/{n}")
    print(f"  ...purely the BOUNDARY byte (value all OK) . {boundary_only}/{canon_fail}  "
          f"({100*boundary_only/den:.0f}% of failures = the colon confound)")
    print(f"  value-byte greedy differs canon vs space ... {val_diff}/{n}  "
          f"({'REAL parse shift' if val_diff else 'value parse prefix-stable'})")
    del gen


if __name__ == "__main__":
    R = os.environ["AUNET_ROOT"]
    bpe = os.environ.get("BPE_CKPT", f"{R}/main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated")
    aun = os.environ.get("AUNET_CKPT", f"{R}/main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated")
    for T in ("1", "3"):
        probe("aunet", bpe, "BPEByte (root-greedy) 1.3B", task=T)
        probe("aunet", aun, "AU-Net (word-pool) 1.3B", task=T)

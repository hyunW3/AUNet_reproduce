#!/usr/bin/env python3
"""Per-prompt patch counts for a hierarchical byte model (CPU-side parser only).

The patch transformer has a hard per-sequence limit (trunk.max_seqlen, 3200 for the
1.3B models); a prompt whose patches + generation exceed it cannot be scored. Writes
{id: n_patches} so the suite can be capped identically for every family and the
patch budget per byte can be reported (e.g. hex/UUID text costs ~2x more patches).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_longctx import build_generator, DEFAULT_TOK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g, tok = build_generator("aunet", a.ckpt, DEFAULT_TOK, 8192)
    limit = int(g.model.trunk.max_seqlen)
    res = {"_patch_limit": limit}
    for f in a.data:
        for line in open(f):
            r = json.loads(line)
            pb = tok.encode(r["prompt"], add_bos=False, add_eos=False)
            res[r["id"]] = int(sum(1 for x in g.regex_pool.get_levels_mask(pb) if x > 0))
    json.dump(res, open(a.out, "w"))
    print("patch_limit", limit, "max", max(v for k, v in res.items() if k[0] != "_"))


if __name__ == "__main__":
    main()

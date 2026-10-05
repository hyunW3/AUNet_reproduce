#!/usr/bin/env python3
"""Score dumped NoLiMa/MK-NIAH pairs (dump_probe_pairs.py) with a local model
family (Llama/AU-Net/BPEByte), so all four models are compared on IDENTICAL
prompts. Reuses the exact teacher-forcing scorer from niah_ext_probe (sep=" ",
greedy exact-match over the value bytes). Aggregates exact-match per cell.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK   # noqa: E402
from niah_ext_probe import score                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items_out", default=None, help="also write per-item exact-match (jsonl, pairs order)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pairs)]
    gen, tok = build_generator(args.family, args.ckpt, args.tok_path, args.max_tokens)
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)

    rows, items = [], []
    for cell in sorted(by_cell):
        sm = [{"prompt": r["prompt"], "values": r["values"]} for r in by_cell[cell]]
        res = score(gen, tok, sm, " ", args.batch_size)
        em = sum(x["exact"] for x in res) / len(res)
        for r, x in zip(by_cell[cell], res):
            items.append({"tag": args.tag, "cell": cell, "depth": r.get("depth"), "target_bytes": r.get("target_bytes"),
                          "exact": int(x["exact"])})
        rows.append({"tag": args.tag, "cell": cell, "n": len(res), "exact_match": em})
        print(f"[{args.tag}] {cell:22} n={len(res):4} exact={em:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if args.items_out:
        with open(args.items_out, "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

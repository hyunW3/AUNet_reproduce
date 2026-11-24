#!/usr/bin/env python3
"""Does repairing OnlineBPE's Telugu vocabulary coverage also repair modelling quality?

The compression sweep (plans/plan_compression_stability.md) showed that grafting BLOOM's
Telugu range into the trie lifts BPEByte from 1.57 to 7.40 bytes/patch on Telugu while
leaving English and Chinese untouched. That is a COMPUTE result. This probe asks the
separate question of whether held-out BPB moves with it.

Expect it not to, or to get worse: the 1.3B checkpoint was TRAINED with the un-augmented
parser, so scoring it under augmented boundaries is a train/test mismatch. Measuring it is
the only way to bound the claim honestly -- "one vocabulary knob repairs the parser" must
not be read as "one vocabulary knob repairs the model".

    python scripts/probes/bpb_augment_te.py --domain te --domain en

Reports base vs augmented BPB on identical windows, so the delta isolates the boundaries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(ROOT / "scripts/fflm"))

# The segmentation cache key (RegexPool._seg_sig) does NOT include bpe_augment_*, so an
# enabled cache would serve un-augmented boundaries to the augmented run and silently
# invalidate this comparison. Refuse to run with it on.
if os.environ.get("REGEX_SEG_CACHE"):
    sys.exit("REGEX_SEG_CACHE is set; unset it -- _seg_sig omits bpe_augment_* "
             "so cached boundaries would be stale for the augmented arm.")

DOMAINS = {
    "en": ROOT / "reports/bpb_windows.jsonl",
    "zh": ROOT / "reports/bpb_windows_zh.jsonl",
    "te": ROOT / "reports/windows/te.jsonl",
}
CKPT = ROOT / "main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"
AUG_TOK = ROOT / "tokenizer/bloom/tokenizer.json"
AUG_RANGE = "0C00-0C7F"


def score(generator, tokenizer, windows, batch_size):
    from fflm_probe import score_pairs
    res = score_pairs(generator, tokenizer, [("", w["text"]) for w in windows], batch_size)
    nll = -sum(lp for lp, _ in res)
    nbytes = sum(w["n_bytes"] for w in windows)
    return nll / (math.log(2) * nbytes), nbytes


def build(augment: bool, tok_path: str, max_tokens: int):
    from fflm_probe import build_generator
    gen, tok = build_generator("aunet", str(CKPT), tok_path, max_tokens)
    pool = gen.regex_pool
    if augment:
        pool.bpe_augment_tokenizer_path = str(AUG_TOK)
        pool.bpe_augment_unicode_range = AUG_RANGE
        pool._byte_trie = None          # force a rebuild so the graft is applied
        _ = pool.byte_trie              # build now, so the log line lands before scoring
    return gen, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", action="append", default=None)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--tok_path", default=str(ROOT / "tokenizer/llama3/tokenizer.model"))
    ap.add_argument("--arms", default="both", choices=["both", "base", "augmented"],
                    help="skip an arm when it has already been measured (one model on GPU at a time)")
    ap.add_argument("--out", default=str(ROOT / "reports/compression_stability/bpb_augment_te.jsonl"))
    args = ap.parse_args()
    doms = args.domain or ["te", "en"]
    arms = {"both": (False, True), "base": (False,), "augmented": (True,)}[args.arms]

    rows = []
    for augment in arms:
        gen, tok = build(augment, args.tok_path, args.max_tokens)
        label = "BPEByte+teVocab" if augment else "BPEByte"
        for d in doms:
            wins = [json.loads(l) for l in open(DOMAINS[d])]
            if args.limit:
                wins = wins[:args.limit]
            bpb, nbytes = score(gen, tok, wins, args.batch_size)
            rows.append({"tag": f"{label}|{d}", "model": label, "domain": d,
                         "augmented": augment, "n_windows": len(wins),
                         "total_bytes": nbytes, "bpb": bpb})
            print(f"[{label}|{d}] windows={len(wins)} bytes={nbytes} BPB={bpb:.4f}", flush=True)
        del gen
        import torch
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"{'domain':<8}{'base':>10}{'augmented':>12}{'delta':>10}")
    for d in doms:
        b = next((r["bpb"] for r in rows if r["domain"] == d and not r["augmented"]), None)
        a = next((r["bpb"] for r in rows if r["domain"] == d and r["augmented"]), None)
        if b is not None and a is not None:
            print(f"{d:<8}{b:>10.4f}{a:>12.4f}{a-b:>+10.4f}")
    print(f"\nappended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

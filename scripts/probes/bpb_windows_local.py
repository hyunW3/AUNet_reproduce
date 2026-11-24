#!/usr/bin/env python3
"""Self-consistent byte-normalized BPB for Llama/AU-Net/BPEByte on fixed held-out
windows (reports/bpb_windows.jsonl). Each window is scored teacher-forced (ctx="",
cont=window); BPB = -sum(logprob_nats) / (ln2 * total_utf8_bytes). Byte models sum
over byte tokens, Llama over subword tokens -> both divided by the SAME byte count,
so the numbers are directly comparable (and comparable to the BLT-1B scorer).
"""
import argparse, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK, score_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default="reports/bpb_windows.jsonl")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N windows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[:args.limit]
    gen, tok = build_generator(args.family, args.ckpt, args.tok_path, args.max_tokens)
    pairs = [("", w["text"]) for w in wins]
    res = score_pairs(gen, tok, pairs, args.batch_size)   # [(sum_logprob_nats, greedy_all)]
    total_nll = -sum(lp for lp, _ in res)
    total_bytes = sum(w["n_bytes"] for w in wins)
    bpb = total_nll / (math.log(2) * total_bytes)
    row = {"tag": args.tag, "family": args.family, "n_windows": len(wins),
           "total_bytes": total_bytes, "total_nll_nats": total_nll, "bpb": bpb}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{args.tag}] windows={len(wins)} bytes={total_bytes} BPB={bpb:.4f}", flush=True)


if __name__ == "__main__":
    main()

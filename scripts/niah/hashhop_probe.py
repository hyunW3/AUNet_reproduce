#!/usr/bin/env python3
"""HashHop (hops=1) associative-recall probe for frozen 1.3B checkpoints.

HashHop (Magic; https://github.com/magicproduct/hash-hop) fills the context with
random hash pairs ``KEY = 'VAL'`` (incompressible 16-char letter strings),
shuffled, and asks the model to recall ``VAL`` for a queried ``KEY``. With
``hops=1`` the multi-hop benchmark reduces to single-hop key->value recall over
strings with no word/semantic structure -- an official, harder analog of
S-NIAH-3 (UUID). We reproduce the exact hops=1 prompt format (a shuffled list of
``KEY = 'VAL'`` lines followed by ``COMPLETION:\\nHops=1\\n\\nCoT=False``), then
score with the same teacher-forced greedy exact-match used for S-NIAH: the model
passes an item only if its argmax next-token prediction equals the gold token at
*every* answer position (i.e. free-running greedy reproduces all 16 VAL bytes).

Fit: HashHop hashes are random letters, so under the llama3 BPE they cost
~0.64 tok/byte; the byte models bind at ~bytes/token. A 6144-char haystack is
~6.7 KB / ~4.3 K llama tokens -- inside the 8192 window for all three models.

Run from repo root with the same env as run_ext_1p3b.sh (AUNET_ROOT, AUNET_TOK,
PYTHONPATH=<repo>/lingua). One model per GPU via CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

# reuse the family loaders + the teacher-forced exact-match scorer from S-NIAH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_probe import score  # noqa: E402  (returns [(exact, n_correct_tok, n_tok, ll_sum)])

# HashHop alphabet: upper+lower ASCII letters only (no digits) -> incompressible.
_ALPHA = string.ascii_lowercase + string.ascii_uppercase
# HashHop hops=1 completion header (verbatim from hashhop/generate.py).
_HEADER = "\n\nCOMPLETION:\nHops=1\n\nCoT=False\n\n"


def _rand_hash(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(_ALPHA) for _ in range(n))


def make_hashhop(target_bytes, n_samples, hstr, seed):
    """n independent hops=1 haystacks sized to ~target_bytes of KEY = 'VAL' lines.

    Each sample queries one random key (random depth, since HashHop shuffles).
    Returns dicts with the teacher-forcing context and the gold value.
    """
    samples = []
    line_len = hstr * 2 + 6  # "KEY = 'VAL'\n" == 16 + " = '" (4) + "'" (1) + "\n" (1) + 16
    for i in range(n_samples):
        rng = random.Random(seed * 100003 + i)
        n_pairs = max(2, round(target_bytes / line_len))
        pairs = {}
        while len(pairs) < n_pairs:            # keys unique (dict) -> retry on rare collision
            pairs[_rand_hash(hstr, rng)] = _rand_hash(hstr, rng)
        keys = list(pairs)
        rng.shuffle(keys)
        lines = [f"{k} = '{pairs[k]}'" for k in keys]
        prompt = "\n".join(lines)
        qk = rng.choice(keys)
        ctx = prompt + _HEADER + qk + " = '"    # model must now emit VAL (16 bytes)
        samples.append({"ctx": ctx, "value": pairs[qk],
                        "prompt_bytes": len(ctx.encode()),
                        "n_pairs": len(keys),
                        "depth": keys.index(qk) / len(keys)})
    return samples


def run(family, ckpt, tag, lengths, n, hstr, seed, batch, tok_path, max_tokens, out, per_out):
    generator, tokenizer = build_generator(family, ckpt, tok_path, max_tokens)
    rows, per_rows = [], []
    for tb in lengths:
        samples = make_hashhop(tb, n, hstr, seed)
        pairs = [(s["ctx"], s["value"]) for s in samples]
        avg_b = sum(s["prompt_bytes"] for s in samples) / len(samples)
        avg_np = sum(s["n_pairs"] for s in samples) / len(samples)
        print(f"[{tag}] HashHop hops=1 len~{tb} ({avg_b:.0f}B prompt, {avg_np:.0f} pairs) "
              f"x {len(samples)} ...", flush=True)
        res = score(generator, tokenizer, pairs, batch)
        em = [r[0] for r in res]
        frac = [r[1] / r[2] if r[2] else 0.0 for r in res]
        for s, r in zip(samples, res):
            per_rows.append({"tag": tag, "family": family, "target_bytes": tb,
                             "prompt_bytes": s["prompt_bytes"], "n_pairs": s["n_pairs"],
                             "depth": s["depth"], "exact": int(r[0]),
                             "tok_frac": r[1] / r[2] if r[2] else 0.0})
        row = {"tag": tag, "family": family, "task": "hashhop_h1", "target_bytes": tb,
               "avg_prompt_bytes": round(avg_b), "avg_pairs": round(avg_np),
               "hash_len": hstr, "n": len(samples),
               "exact_match": sum(em) / len(em), "tok_frac": sum(frac) / len(frac)}
        rows.append(row)
        print(f"   -> exact_match={row['exact_match']:.3f} tok_frac={row['tok_frac']:.3f}",
              flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if per_out:
        with open(per_out, "a") as f:
            for r in per_rows:
                f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 6144])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--hash_len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_out", default=None)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.lengths, args.n, args.hash_len,
        args.seed, args.batch_size, args.tok_path, args.max_tokens, args.out, args.per_out)


if __name__ == "__main__":
    main()

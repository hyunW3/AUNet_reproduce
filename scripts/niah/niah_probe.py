#!/usr/bin/env python3
"""S-NIAH-1 retrieval probe for frozen 1.3B checkpoints (llama / aunet / bpebyte).

Teacher-forces the correct magic number after the query and checks whether the
model would greedily generate it:
  exact_match : ALL answer tokens are the argmax (== free-running greedy emits the
                number verbatim) — the standard NIAH retrieval metric.
  tok_frac    : fraction of answer tokens that are argmax (partial credit;
                per-byte for byte models, per-subword for llama — not strictly
                comparable, shown as a diagnostic).

Reuses fflm_probe.build_generator (same loaders). Run from lingua/ with
PYTHONPATH=<lingua>.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Despace: strip ALL intra-line spaces/tabs from the prompt ("I have a boy" -> "Ihaveaboy"),
# preserving newlines. The magic number / UUID value has no internal spaces, so it survives and
# must still be retrieved. Mirrors the MCQ despace probe (eval_despace_mc.py).
_WS = re.compile(r"[ \t]+")


def _despace(text: str) -> str:
    return "\n".join(_WS.sub("", line) for line in text.split("\n"))

# reuse the family loaders from the FFLM probe
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_data import make_dataset, TASKS  # noqa: E402


def score(generator, tokenizer, pairs, batch):
    """Return list of (exact_match, n_correct_tok, n_tok, ll_sum)."""
    out = []
    old = generator.max_gen_len
    generator.max_gen_len = 1
    try:
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            inputs = [c + k for c, k in chunk]
            _, lls, greedy = generator.generate(inputs)
            for (ctx, _cont), ll, gr in zip(chunk, lls, greedy):
                p_len = len(tokenizer.encode(ctx, add_bos=False, add_eos=False))
                g = gr[p_len:]
                n = int(g.numel())
                # empty slice = answer truncated (prompt over the model window); never a pass
                out.append((bool(n > 0 and g.all().item()), int(g.sum().item()),
                            n, ll[p_len:].sum().item()))
    finally:
        generator.max_gen_len = old
    return out


def build_pairs(samples, family):
    pairs = []
    for s in samples:
        if family == "subword":
            ctx, cont = s["prompt"], " " + s["value"]
        else:  # bytes: fold separating space into context
            ctx, cont = s["prompt"] + " ", s["value"]
        pairs.append((ctx, cont))
    return pairs


def run(family, ckpt, tag, lengths, depths, n_per_cell, value_digits, seed,
        batch, tok_path, max_tokens, out, per_out, task="1", control=False, despace=False):
    ht, vt = TASKS[task]
    generator, tokenizer = build_generator(family, ckpt, tok_path, max_tokens)
    rows, per_rows = [], []
    for tb in lengths:
        samples = make_dataset(tb, depths, n_per_cell, value_digits, seed, ht, vt, control)
        if despace:
            # Strip ALL prompt spaces (haystack + needle + query). The magic number / UUID value has
            # no internal spaces, so it survives and must still be retrieved.
            for s in samples:
                s["prompt"] = _despace(s["prompt"])
                s["prompt_bytes"] = len(s["prompt"].encode())
        pairs = build_pairs(samples, family)
        avg_bytes = sum(s["prompt_bytes"] for s in samples) / len(samples)
        print(f"[{tag}] S-NIAH-{task} len~{tb} ({avg_bytes:.0f}B prompt) x {len(samples)} ...",
              flush=True)
        res = score(generator, tokenizer, pairs, batch)
        em = [r[0] for r in res]
        frac = [r[1] / r[2] if r[2] else 0.0 for r in res]
        for s, r in zip(samples, res):
            per_rows.append({"tag": tag, "family": family, "task": task,
                             "target_bytes": tb,
                             "prompt_bytes": s["prompt_bytes"], "depth": s["depth"],
                             "exact": int(r[0]), "tok_frac": r[1] / r[2] if r[2] else 0.0})
        row = {"tag": tag, "family": family, "task": task, "target_bytes": tb,
               "avg_prompt_bytes": round(avg_bytes), "n": len(samples),
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
    ap.add_argument("--task", choices=list(TASKS), default="1",
                    help="1=noise+number, 2=essay+number, 3=essay+uuid")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--n_per_cell", type=int, default=4)
    ap.add_argument("--value_digits", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--control", action="store_true",
                    help="remove the needle (absent-value control for leak checks)")
    ap.add_argument("--despace", action="store_true",
                    help="strip ALL spaces from the prompt (haystack+needle+query); value survives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_out", default=None)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.lengths, args.depths,
        args.n_per_cell, args.value_digits, args.seed, args.batch_size,
        args.tok_path, args.max_tokens, args.out, args.per_out, args.task, args.control,
        args.despace)


if __name__ == "__main__":
    main()

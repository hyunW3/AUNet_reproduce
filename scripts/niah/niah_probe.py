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


def build_pairs(samples, family, variant=None, sep="", answer_sep=None):
    """(context, continuation) per sample.

    variant=None keeps the family-default cut (unchanged legacy behavior).
    The two PBP variants place exactly ONE space at the prompt/answer cut, on
    opposite sides; the total text is identical, only the cut position moves
    (mirror of apps/aunet/eval_pbp_mc.py):
      canonical : context ends before the space, continuation = " " + value
      space     : trailing space committed into the context, continuation = value
    A byte model sees identical bytes either way (delta == 0 exactly); a subword
    model re-tokenizes the prompt tail, so exact-match can drop -- that drop is
    the prompt-boundary problem (PBP).
    """
    pairs = []
    for s in samples:
        if variant is None:
            if answer_sep is not None:
                # put the needle separator on the ANSWER side: the model must emit
                # e.g. ": <value>" (its natural continuation of "...is"), which is the
                # arguably-more-natural cut than scoring only " <value>".
                ctx, cont = s["prompt"].rstrip(" "), answer_sep + s["value"]
            elif family == "subword":
                ctx, cont = s["prompt"], " " + s["value"]
            else:  # bytes: fold separating space into context
                ctx, cont = s["prompt"] + " ", s["value"]
        else:
            # `sep` (e.g. ":") aligns the query end to the needle's value separator
            # ("...is: <value>"), so the boundary shift is a pure trailing space and
            # not confounded by the model expecting the memorized colon after "is".
            base = s["prompt"].rstrip(" ") + sep
            if variant == "canonical":
                ctx, cont = base, " " + s["value"]
            elif variant == "space":
                ctx, cont = base + " ", s["value"]
            else:
                raise ValueError(f"unknown PBP variant {variant!r}")
        pairs.append((ctx, cont))
    return pairs


def run(family, ckpt, tag, lengths, depths, n_per_cell, value_digits, seed,
        batch, tok_path, max_tokens, out, per_out, task="1", control=False, despace=False,
        pbp=False, pbp_sep="", query_sep="", answer_sep=None):
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
        if query_sep and not pbp:
            # Align the query end to the needle's value separator (e.g. ":") so the
            # scored continuation follows the same format as "...is: <value>" in the
            # needle. Without this the model greedily emits the memorized separator
            # after "is" and exact-match penalizes the format mismatch, not retrieval.
            for s in samples:
                s["prompt"] = s["prompt"].rstrip(" ") + query_sep
                s["prompt_bytes"] = len(s["prompt"].encode())
        avg_bytes = sum(s["prompt_bytes"] for s in samples) / len(samples)

        if pbp:
            # Score BOTH boundary variants (identical total text; the single cut
            # space sits on opposite sides) and report delta = space - canonical.
            # Byte models: identical bytes -> delta == 0 exactly. Subword: the
            # re-tokenized tail can drop exact-match -> that drop is the PBP.
            stats = {}
            for var in ("canonical", "space"):
                r = score(generator, tokenizer, build_pairs(samples, family, var, pbp_sep), batch)
                stats[var] = (sum(x[0] for x in r) / len(r),
                              sum((x[1] / x[2] if x[2] else 0.0) for x in r) / len(r))
            d_exact = stats["space"][0] - stats["canonical"][0]
            d_frac = stats["space"][1] - stats["canonical"][1]
            row = {"tag": tag, "family": family, "task": task, "target_bytes": tb,
                   "avg_prompt_bytes": round(avg_bytes), "n": len(samples),
                   "exact_canonical": stats["canonical"][0], "exact_space": stats["space"][0],
                   "pbp_delta_exact": round(d_exact, 4),
                   "tok_frac_canonical": round(stats["canonical"][1], 4),
                   "tok_frac_space": round(stats["space"][1], 4),
                   "pbp_delta_tok_frac": round(d_frac, 4)}
            rows.append(row)
            print(f"[{tag}] S-NIAH-{task} len~{tb} ({avg_bytes:.0f}B) PBP | exact "
                  f"canon={stats['canonical'][0]:.3f} space={stats['space'][0]:.3f} "
                  f"Δ={d_exact:+.3f}  (tok_frac Δ={d_frac:+.3f})", flush=True)
            continue

        pairs = build_pairs(samples, family, answer_sep=answer_sep)
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
    ap.add_argument("--pbp", action="store_true",
                    help="prompt-boundary problem: score canonical vs trailing-space cut, "
                         "report delta (byte models ~0, subword drops)")
    ap.add_argument("--pbp_sep", default="",
                    help="align the query end to the needle value separator (e.g. ':') so the "
                         "PBP shift is a pure trailing space, not the memorized-colon confound")
    ap.add_argument("--query_sep", default="",
                    help="append this to the query end (e.g. ':') in the STANDARD eval to match "
                         "the needle format '...is: <value>' and remove the boundary-format confound")
    ap.add_argument("--answer_sep", default=None,
                    help="put the needle separator on the ANSWER side (e.g. ': '): score the "
                         "continuation as '<sep><value>' so the model emits its natural '...is: value'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_out", default=None)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.lengths, args.depths,
        args.n_per_cell, args.value_digits, args.seed, args.batch_size,
        args.tok_path, args.max_tokens, args.out, args.per_out, args.task, args.control,
        args.despace, args.pbp, args.pbp_sep, args.query_sep, args.answer_sep)


if __name__ == "__main__":
    main()

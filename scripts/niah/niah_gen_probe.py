#!/usr/bin/env python3
"""Generation-style S-NIAH probe: free-running greedy decode + substring scoring.

Unlike niah_probe.py (teacher-forced argmax over the gold continuation), this
lets the model GENERATE max_gen_len tokens after the query and scores RULER-style:
  gen_match : gold value appears verbatim in the generated text.

Because the model chooses its own separator (" 123", ": 123", ...) there is no
prompt-boundary/format confound, so this cleanly measures whether the colon in
the NEEDLE ("...is: {value}" vs "...is {value}") changes actual retrieval.

Diagnostics per sample: the head of the generation and whether it opens with a
colon (does the model reproduce the needle's separator?).

Run from lingua/ with PYTHONPATH=<lingua>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niah_data import make_dataset, TASKS  # noqa: E402


def generate_batch(generator, prompts, batch):
    texts = []
    for i in range(0, len(prompts), batch):
        gen, _, _ = generator.generate(prompts[i:i + batch])
        texts.extend(gen)
    return texts


def run(family, ckpt, tag, tasks, needle_seps, lengths, depths, n_per_cell,
        value_digits, seed, batch, tok_path, max_tokens, max_gen_len, control,
        out, per_out):
    generator, tokenizer = build_generator(family, ckpt, tok_path, max_tokens)
    generator.max_gen_len = max_gen_len
    rows, per_rows = [], []
    for task in tasks:
        ht, vt = TASKS[task]
        for sep in needle_seps:
            cond = "colon" if sep == ":" else ("nosep" if sep == "" else repr(sep))
            for tb in lengths:
                samples = make_dataset(tb, depths, n_per_cell, value_digits,
                                       seed, ht, vt, control, needle_sep=sep)
                avg_bytes = sum(s["prompt_bytes"] for s in samples) / len(samples)
                print(f"[{tag}] S-NIAH-{task} {cond} len~{tb} "
                      f"({avg_bytes:.0f}B) x {len(samples)} ...", flush=True)
                gens = generate_batch(generator, [s["prompt"] for s in samples], batch)
                n_match = n_colon = 0
                for s, g in zip(samples, gens):
                    hit = s["value"] in g
                    n_match += hit
                    starts_colon = g.lstrip().startswith(":")
                    n_colon += starts_colon
                    per_rows.append({
                        "tag": tag, "family": family, "task": task, "cond": cond,
                        "target_bytes": tb, "prompt_bytes": s["prompt_bytes"],
                        "depth": s["depth"], "value": s["value"],
                        "gen_match": int(hit), "starts_colon": int(starts_colon),
                        "gen_head": g[:80]})
                row = {"tag": tag, "family": family, "task": task, "cond": cond,
                       "control": bool(control), "target_bytes": tb,
                       "avg_prompt_bytes": round(avg_bytes), "n": len(samples),
                       "max_gen_len": max_gen_len,
                       "gen_match": n_match / len(samples),
                       "starts_colon": n_colon / len(samples)}
                rows.append(row)
                print(f"   -> gen_match={row['gen_match']:.3f} "
                      f"starts_colon={row['starts_colon']:.3f}", flush=True)

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
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["1"])
    ap.add_argument("--needle_seps", nargs="+", default=["", ":"],
                    help='needle separators to compare; "" = "...is {v}", ":" = RULER "...is: {v}"')
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 6144])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--n_per_cell", type=int, default=4)
    ap.add_argument("--value_digits", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--max_gen_len", type=int, default=48,
                    help="tokens to generate (48 covers the 36-char UUID + preamble)")
    ap.add_argument("--control", action="store_true",
                    help="omit the needle: value absent from context (leak check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_out", default=None)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.tasks, args.needle_seps,
        args.lengths, args.depths, args.n_per_cell, args.value_digits,
        args.seed, args.batch_size, args.tok_path, args.max_tokens,
        args.max_gen_len, args.control, args.out, args.per_out)


if __name__ == "__main__":
    main()

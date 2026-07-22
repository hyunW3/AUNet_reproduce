#!/usr/bin/env python3
"""Probe for the extended retrieval tasks (niah_ext_data): RULER multi-value /
multi-query, Sequential-NIAH, and NoLiMa-lite. Same teacher-forced, generation-
free protocol as niah_probe:

  exact_match : ALL answer tokens are the argmax (the whole value list is emitted
                verbatim, in order) -- the strict RULER retrieval metric;
  recall      : fraction of the K values whose OWN tokens are all argmax (partial
                credit per value). Value spans are found by longest-common-token-
                prefix so a boundary merge is attributed to the right value.

NoLiMa reports acc (single value) for the literal and paraphrase queries; the
NoLiMa effect is literal-acc - paraphrase-acc.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import niah_ext_data as D  # noqa: E402


def _cpl(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def score(generator, tokenizer, samples, sep, batch):
    """Teacher-force ctx + ''.join(sep+v). Return per-sample {exact, recall,
    n_val, n_correct}."""
    enc = lambda s: tokenizer.encode(s, add_bos=False, add_eos=False)
    out = []
    old = generator.max_gen_len
    generator.max_gen_len = 1
    try:
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            inputs = [s["prompt"] + "".join(sep + v for v in s["values"]) for s in chunk]
            _, _lls, greedy = generator.generate(inputs)
            for s, gr in zip(chunk, greedy):
                ctx, vals = s["prompt"], s["values"]
                full = enc(ctx + "".join(sep + v for v in vals))
                # token index where each cumulative prefix diverges from `full`
                bounds = [_cpl(enc(ctx + "".join(sep + v for v in vals[:j])), full)
                          for j in range(len(vals) + 1)]
                per = []
                for j in range(len(vals)):
                    seg = gr[bounds[j]:bounds[j + 1]]
                    per.append(bool(seg.numel() > 0 and seg.all().item()))
                ans = gr[bounds[0]:]
                out.append({"exact": int(bool(ans.numel() > 0 and ans.all().item())),
                            "recall": sum(per) / len(per) if per else 0.0,
                            "n_val": len(vals), "n_correct": sum(per)})
    finally:
        generator.max_gen_len = old
    return out


def run(family, ckpt, tag, task, lengths, K, n, seed, batch, tok_path,
        max_tokens, out, haystack, n_distract):
    gen, tok = build_generator(family, ckpt, tok_path, max_tokens)
    rows = []
    for tb in lengths:
        if task == "nolima":
            for mode, lit in (("literal", True), ("paraphrase", False)):
                sm = D.make_dataset("nolima", tb, K, n, seed, haystack, lit, n_distract)
                res = score(gen, tok, sm, " ", batch)
                ab = round(sum(s["prompt_bytes"] for s in sm) / len(sm))
                acc = sum(r["exact"] for r in res) / len(res)
                rows.append({"tag": tag, "family": family, "task": "nolima", "mode": mode,
                             "target_bytes": tb, "avg_prompt_bytes": ab, "n": len(res),
                             "n_distract": n_distract, "acc": acc})
                print(f"[{tag}] nolima/{mode} len~{tb} ({ab}B, 1+{n_distract}) -> acc={acc:.3f}",
                      flush=True)
        else:
            sm = D.make_dataset(task, tb, K, n, seed, haystack)
            res = score(gen, tok, sm, " ", batch)
            ab = round(sum(s["prompt_bytes"] for s in sm) / len(sm))
            em = sum(r["exact"] for r in res) / len(res)
            rc = sum(r["recall"] for r in res) / len(res)
            rows.append({"tag": tag, "family": family, "task": task, "K": K,
                         "target_bytes": tb, "avg_prompt_bytes": ab, "n": len(res),
                         "exact_match": em, "recall": rc})
            print(f"[{tag}] {task} K={K} len~{tb} ({ab}B) -> exact={em:.3f} recall={rc:.3f}",
                  flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["mv", "mq", "seq", "nolima"], required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_distract", type=int, default=3, help="nolima decoy needle count")
    ap.add_argument("--haystack", choices=["essay", "noise"], default="essay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.family, args.ckpt, args.tag, args.task, args.lengths, args.K, args.n,
        args.seed, args.batch_size, args.tok_path, args.max_tokens, args.out,
        args.haystack, args.n_distract)


if __name__ == "__main__":
    main()

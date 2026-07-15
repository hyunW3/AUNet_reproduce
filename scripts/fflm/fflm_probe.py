#!/usr/bin/env python3
"""In-context FFLM state-tracking probe for frozen 1.3B checkpoints.

Loads a checkpoint from one of two lingua codebases and scores next-token
prediction at every READ bit of flip-flop sequences:

  family=subword -> apps.main   (flat BPE Llama transformer)
  family=aunet   -> apps.aunet  (byte-input hierarchical: AU-Net static pooling
                                 OR byte greedy/bt boundary pooling; same code)

For each read we form (context, continuation) and reuse the exact
loglikelihood machinery the downstream evals use (`generator.generate`):

  * greedy_acc  : model's argmax next token(s) == the correct bit  (PAPER metric;
                  <100% = a "reasoning error" per Liu et al. 2023)
  * binary_acc  : logprob(correct bit) > logprob(wrong bit)        (discrimination)
  * margin      : mean [ logprob(correct) - logprob(wrong) ]

Continuation is formatted per family so the graded token isolates the bit:
  subword:  ctx = "... r",   cont = " 0"      (leading-space bit is one BPE token)
  aunet  :  ctx = "... r ",  cont = "0"       (bytes; space folded into context)

Run from the lingua/ dir so `apps.*` imports resolve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# fflm_data lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fflm_data import generate_dataset, REGIMES, FFLMSequence  # noqa: E402

DEFAULT_TOK = "/NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/tokenizer/llama3/tokenizer.model"


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def build_generator(family: str, ckpt: str, tok_path: str, max_tokens: int):
    """Return (generator, tokenizer). `ckpt` is a consolidated dir (params.json +
    consolidated.pth)."""
    if family == "subword":
        from apps.main.generate import (
            load_consolidated_model_and_tokenizer,
            PackedCausalTransformerGenerator,
            PackedCausalTransformerGeneratorArgs,
        )
        model, tokenizer, _ = load_consolidated_model_and_tokenizer(ckpt)
        gen_args = PackedCausalTransformerGeneratorArgs(
            temperature=0.0, max_gen_len=1, max_tokens=max_tokens,
        )
        generator = PackedCausalTransformerGenerator(gen_args, model, tokenizer)
        return generator, tokenizer

    elif family == "aunet":
        from apps.aunet.generate import (
            load_consolidated_model_and_tokenizer,
            PackedHierarchicalCausalTransformerGenerator,
            PackedHierarchicalCausalTransformerGeneratorArgs,
        )
        from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs
        model, tokenizer, regex_pool, _ = load_consolidated_model_and_tokenizer(
            ckpt,
            model_cls=HierarchicalTransformer,
            model_args_cls=HierarchicalArgs,
            regex_bpe_tokenizer_path=tok_path,
            regex_bpe_tokenizer_kind=None,
        )
        gen_args = PackedHierarchicalCausalTransformerGeneratorArgs(
            temperature=0.0, max_gen_len=1, max_tokens=max_tokens,
        )
        generator = PackedHierarchicalCausalTransformerGenerator(
            gen_args, model, tokenizer, regex_pool
        )
        return generator, tokenizer

    raise ValueError(f"unknown family {family}")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_pairs(generator, tokenizer, pairs: List[Tuple[str, str]],
                batch_size: int) -> List[Tuple[float, bool]]:
    """For each (context, continuation) return (sum_logprob_of_cont, greedy_all).

    Mirrors EvalHarnessLM.loglikelihood in both codebases."""
    out: List[Tuple[float, bool]] = []
    old = generator.max_gen_len
    generator.max_gen_len = 1
    try:
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            inputs = [c + k for c, k in chunk]
            _, lls, greedy = generator.generate(inputs)
            for (ctx, _cont), ll, gr in zip(chunk, lls, greedy):
                p_len = len(tokenizer.encode(ctx, add_bos=False, add_eos=False))
                out.append((ll[p_len:].sum().item(), bool(gr[p_len:].all().item())))
    finally:
        generator.max_gen_len = old
    return out


def make_fewshot_prefix(shots: int, num_instr: int, p_i: float, seed: int) -> str:
    if shots <= 0:
        return ""
    demos = generate_dataset(shots, num_instr, p_i, seed + 987654)
    return "".join(d.text + "\n" for d in demos)


def build_requests(seqs: List[FFLMSequence], family: str, prefix: str,
                   max_reads: int, rng_seed: int):
    """Return (correct_pairs, wrong_pairs, meta) aligned index-for-index.

    meta[i] = dict(seq, read_index, dist, prev_bit, state) for diagnostics:
      dist     = #instructions back to the governing write (dependency distance)
      prev_bit = data bit of the immediately preceding instruction (recency
                 distractor); recency-glitch if the model copies this instead
      state    = correct bit (== governing write's bit)."""
    import random
    rng = random.Random(rng_seed)
    correct_pairs, wrong_pairs, meta = [], [], []
    for si, s in enumerate(seqs):
        reads = s.reads
        if max_reads and len(reads) > max_reads:
            reads = rng.sample(reads, max_reads)
        for instr_index, correct in reads:
            prefix_str, corr, wrong = s.context_and_bit(instr_index)
            if family == "subword":
                ctx = prefix + prefix_str            # ends at "r"
                cont_c, cont_w = " " + corr, " " + wrong
            else:  # aunet / bytes: fold the separating space into context
                ctx = prefix + prefix_str + " "      # ends at "r "
                cont_c, cont_w = corr, wrong
            correct_pairs.append((ctx, cont_c))
            wrong_pairs.append((ctx, cont_w))
            w_idx = max(k for k in range(instr_index) if s.tokens[2 * k] == "w")
            prev_bit = s.tokens[2 * (instr_index - 1) + 1] if instr_index > 0 else corr
            meta.append({"seq": si, "read_index": instr_index,
                         "dist": instr_index - w_idx, "prev_bit": prev_bit,
                         "state": corr})
    return correct_pairs, wrong_pairs, meta


def run(family, ckpt, regime, num_instr, n, seed, shots, max_reads,
        batch_size, tok_path, max_tokens):
    p_i = REGIMES[regime]
    seqs = generate_dataset(n, num_instr, p_i, seed)
    prefix = make_fewshot_prefix(shots, num_instr, p_i, seed)
    correct_pairs, wrong_pairs, meta = build_requests(
        seqs, family, prefix, max_reads, rng_seed=seed
    )
    if not meta:
        return {"family": family, "regime": regime, "n_reads": 0,
                "note": "no reads sampled (increase n for sparse)"}

    print(f"[{family}/{regime}] scoring {len(meta)} reads "
          f"({n} seqs, shots={shots}) ...", flush=True)
    generator, tokenizer = getattr(run, "_gen_cache", (None, None))
    if generator is None:
        generator, tokenizer = build_generator(family, ckpt, tok_path, max_tokens)
        run._gen_cache = (generator, tokenizer)

    corr = score_pairs(generator, tokenizer, correct_pairs, batch_size)
    wrong = score_pairs(generator, tokenizer, wrong_pairs, batch_size)

    greedy_ok = [c[1] for c in corr]
    binary_ok = [c[0] > w[0] for c, w in zip(corr, wrong)]
    margins = [c[0] - w[0] for c, w in zip(corr, wrong)]
    N = len(greedy_ok)
    # per-read diagnostic records (accuracy-vs-distance, recency-glitch test)
    records = []
    for m, g, b, mg in zip(meta, greedy_ok, binary_ok, margins):
        records.append({"family": family, "regime": regime, "dist": m["dist"],
                        "prev_bit": m["prev_bit"], "state": m["state"],
                        # recency-glitch: did the nearest distractor bit disagree
                        # with the true state? (hard case for a copy-nearest model)
                        "prev_disagrees": int(m["prev_bit"] != m["state"]),
                        "greedy_ok": int(g), "binary_ok": int(b), "margin": mg})
    return {
        "family": family, "regime": regime, "p_i": p_i, "ckpt": ckpt,
        "num_instr": num_instr, "n_seqs": n, "shots": shots,
        "max_reads": max_reads, "n_reads": N,
        "greedy_acc": sum(greedy_ok) / N,
        "binary_acc": sum(binary_ok) / N,
        "mean_margin": sum(margins) / N,
        "_records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True, help="consolidated checkpoint dir")
    ap.add_argument("--regimes", nargs="+", default=["dense", "indist", "sparse"])
    ap.add_argument("--num_instr", type=int, default=64)
    ap.add_argument("--n", type=int, default=100, help="seqs per regime (dense/indist)")
    ap.add_argument("--n_sparse", type=int, default=1000, help="seqs for sparse regime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--max_reads", type=int, default=16,
                    help="cap reads scored per sequence (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tag", default=None, help="label for this checkpoint")
    ap.add_argument("--out", required=True, help="append summary JSONL here")
    ap.add_argument("--per_read_out", default=None,
                    help="append per-read diagnostic JSONL here (dist, prev_bit, ...)")
    args = ap.parse_args()

    results, all_records = [], []
    for regime in args.regimes:
        n = args.n_sparse if regime == "sparse" else args.n
        res = run(args.family, args.ckpt, regime, args.num_instr, n, args.seed,
                  args.shots, args.max_reads, args.batch_size, args.tok_path,
                  args.max_tokens)
        res["tag"] = args.tag or Path(args.ckpt).parts[-3]
        recs = res.pop("_records", [])
        for r in recs:
            r["tag"] = res["tag"]
        all_records.extend(recs)
        print("  ->", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in res.items()
                                  if k in ("regime", "greedy_acc", "binary_acc",
                                           "mean_margin", "n_reads")}), flush=True)
        results.append(res)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(results)} rows -> {args.out}")
    if args.per_read_out:
        Path(args.per_read_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_read_out, "a") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(all_records)} per-read rows -> {args.per_read_out}")


if __name__ == "__main__":
    main()

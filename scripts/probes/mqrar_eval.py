#!/usr/bin/env python3
"""In-context evaluation of the OFFICIAL MQRAR benchmark (sundai-research/mqrar-bench,
the Zoology multi-query associative recall + update generator) on our frozen 1.3B LMs.

The official generator emits integer token-ID sequences with a label (the recalled
value) at every query position -- a repeated key whose CURRENT value must be
recalled; the token after a key updates that key's value. We use the generator
UNCHANGED (scripts/probes/mqrar_bench/data_gen.py) and only render its sequences to
text for teacher-forced in-context scoring: at each query the model must place the
gold value token as its top-1 prediction (exact-match). Swept over N = num_kv_pairs.
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

# Load the vendored OFFICIAL generator functions unchanged, but exec only the
# function-definition portion (before the repo's module-level demo/torch.save,
# marked by "# ========================") so importing has no side effects.
_DG = Path(__file__).resolve().parent / "mqrar_bench" / "data_gen.py"
_src = _DG.read_text().split("# ========================")[0]
_dg = types.ModuleType("mqrar_dg")
exec(compile(_src, str(_DG), "exec"), _dg.__dict__)


def _render(inputs, labels):
    """One sequence -> (reads, full_inline_text). reads[i]=(context_ending_at_query,
    gold_value). Each query's answer is inlined into the running text so later
    queries (and few-shot demos) see the resolved recall pattern."""
    parts, reads = [], []
    for t, lab in zip(inputs, labels):
        parts.append(str(int(t)))
        if int(lab) != -100:
            reads.append((" ".join(parts), str(int(lab))))
            parts.append(str(int(lab)))          # reveal the recalled value inline
    return reads, " ".join(parts)


def run_model(tag, n_vars_list, vocab, seq_len, n_seq, shots, seed, batch):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)
    rows = []
    for N in n_vars_list:
        data = _dg.multiquery_ar(vocab_size=vocab, num_train_examples=max(shots, 1),
                                 num_test_examples=n_seq, input_seq_len=seq_len,
                                 num_kv_pairs=N, random_non_queries=True, seed=seed)
        demos = [_render(data.train_inputs[i].tolist(), data.train_labels[i].tolist())[1]
                 for i in range(shots)]
        prefix = ("\n\n".join(demos) + "\n\n") if shots > 0 else ""
        pairs = []
        for i in range(n_seq):
            reads, _ = _render(data.test_inputs[i].tolist(), data.test_labels[i].tolist())
            for ctx, gold in reads:
                pairs.append(fmt_pair(prefix + ctx, gold, fam))
        res = score(gen, tok, pairs, batch)
        acc = sum(r[0] for r in res) / len(res)
        tf = sum(r[1] / r[2] if r[2] else 0.0 for r in res) / len(res)
        rows.append({"tag": tag, "n_kv": N, "vocab": vocab, "seq_len": seq_len,
                     "n_queries": len(res), "top1_acc": acc, "tok_frac": tf})
        print(f"  {tag:16s} N={N:3d} top1={acc:.3f} tok_frac={tf:.3f} (q={len(res)})", flush=True)
    del gen
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--n_kv", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--vocab", type=int, default=256)  # must exceed seq_len
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--n_seq", type=int, default=30)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=f"{L}/reports/statetrack/mqrar_results.jsonl")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    open(args.out, "w").close()
    print(f"MQRAR (official gen) | N={args.n_kv} vocab={args.vocab} seq_len={args.seq_len} "
          f"n_seq={args.n_seq} shots={args.shots}")
    for tag in args.models:
        for r in run_model(tag, args.n_kv, args.vocab, args.seq_len, args.n_seq,
                           args.shots, args.seed, args.batch_size):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

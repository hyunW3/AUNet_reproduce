#!/usr/bin/env python3
"""What is the model's EFFECTIVE context window, and does it match the nominal one?

reports/statetrack/REPORT.md records the Llama 1.3B checkpoint's effective window
as 2048 tokens "(RoPE theta=10000)" even though its params.json and its training
config both say 4096 (`seq_len: 4096`, `max_seqlen: 4096`). That number is load
bearing: it is why the MK-NIAH pressure grid marks its 8 KB cells N/A, and it
decides which long-context cells are legitimate cross-family comparisons at all.
This probe measures it instead of asserting it.

Two measurements, because they can disagree and the disagreement is the point:

  A. LANGUAGE-MODELLING loss vs position. Feed held-out text and bucket the
     per-token NLL by absolute position. A model pushed past its usable window
     shows loss rising with position. This says whether the weights still work
     that far out.

  B. RETRIEVAL vs distance (run separately). A model can keep low LM loss at a
     position and still be unable to USE information there. Nominal window,
     LM-usable window and retrieval-usable window are three different numbers;
     only the first is in params.json.

Positions are bucketed in the model's own units: tokens for subword, bytes for
the byte families (whose generator caps at 8192 bytes).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from common import build_generator, ckpt_for, CKPT, L


def load_docs(n_chars, n_docs, seed=0):
    """Concatenated held-out DCLM text, one long string per doc slot."""
    import random
    rng = random.Random(seed)
    # the shard directory is named differently on different hosts
    files = []
    for sub in ("dclm_baseline_1.0_2shards_shuffled", "dclm_r10lr_2chunk"):
        files = sorted(glob.glob(f"{L}/data/{sub}/*.jsonl"))
        if files:
            break
    if not files:
        raise SystemExit(f"no DCLM shards found under {L}/data")
    texts, buf = [], ""
    # take from the LAST shard to reduce overlap with what the runs trained on
    with open(files[-1]) as fh:
        for line in fh:
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if not t:
                continue
            buf += t + "\n\n"
            while len(buf) >= n_chars and len(texts) < n_docs:
                texts.append(buf[:n_chars]); buf = buf[n_chars:]
            if len(texts) >= n_docs:
                break
    return texts


def run(tag, docs, bucket, max_units, batch, out):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, max_units + 64)
    unit = "tok" if fam == "subword" else "byte"

    sums, cnts = {}, {}
    old = gen.max_gen_len
    gen.max_gen_len = 1
    try:
        for i in range(0, len(docs), batch):
            chunk = docs[i:i + batch]
            _, lls, _ = gen.generate(chunk)
            for ll in lls:
                ll = ll.float().cpu()
                for p in range(ll.numel()):
                    b = (p // bucket) * bucket
                    if b >= max_units:
                        break
                    sums[b] = sums.get(b, 0.0) + float(ll[p])
                    cnts[b] = cnts.get(b, 0) + 1
    finally:
        gen.max_gen_len = old
    del gen

    rows = []
    for b in sorted(sums):
        n = cnts[b]
        nll = -sums[b] / n                      # nats/unit
        rows.append({"pos": b, "n": n, "nll": nll})
    res = {"tag": tag, "family": fam, "unit": unit, "bucket": bucket,
           "n_docs": len(docs), "rows": rows}
    Path(out).write_text(json.dumps(res, indent=2) + "\n")
    base = rows[len(rows) // 4]["nll"] if rows else float("nan")
    print(f"  {tag:16s} unit={unit} buckets={len(rows)}", flush=True)
    for r in rows:
        flag = ""
        if r["nll"] > base * 1.10:
            flag = "  <-- +10% vs mid-window"
        print(f"      pos {r['pos']:6d}  nll/{unit}={r['nll']:.4f}  n={r['n']}{flag}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--bucket", type=int, default=256)
    ap.add_argument("--max_units", type=int, default=4096)
    ap.add_argument("--n_docs", type=int, default=24)
    ap.add_argument("--chars", type=int, default=None,
                    help="chars per doc; default = max_units * 5 (subword) to overfill")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    chars = a.chars or a.max_units * 6
    docs = load_docs(chars, a.n_docs)
    print(f"ctx_window | {len(docs)} docs x {chars} chars | bucket={a.bucket} "
          f"max_units={a.max_units}", flush=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for t in a.models:
        run(t, docs, a.bucket, a.max_units, a.batch_size,
            os.path.join(a.outdir, f"{t}.json"))


if __name__ == "__main__":
    main()

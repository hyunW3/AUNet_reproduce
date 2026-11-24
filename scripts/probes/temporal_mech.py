#!/usr/bin/env python3
"""Is the below-chance Temporal-sequences result a DISTRACTOR bias?

BIG-bench temporal_sequences gives a passage of observed intervals and asks for
the one interval the person was free. The dataset has a perfect structural
asymmetry (verified over all 1000 items):

    the GOLD interval appears verbatim in the passage    0/1000  =   0.0%
    a DISTRACTOR interval appears verbatim in the passage 3000/3000 = 100.0%

The three distractors are exactly the observation intervals; the answer is the
gap, which is never stated. So "prefer a span you have already read" is not just
a wrong strategy, it is an *anti*-correct one: following it scores 0%, far under
the 25% floor. That is the shape of AUNet's result (0.207-0.220 under every
scoring rule), and this probe tests it directly.

For each item we score all four options and record the pick under each of the
three decision rules the harness offers (raw sum / length-normalised / PMI), then
report:

    seen_rate  P(picked option appears verbatim in the passage)
               0.75 if picking uniformly at random, 1.00 for a pure
               already-seen preference, 0.00 for a perfect solver.
    pos_dist   distribution over the choice-list index, to separate a
               positional/primacy bias from a content bias.

seen_rate is the statistic: it is what accuracy cannot distinguish, because every
wrong answer is a seen span and every right answer is an unseen one.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from common import build_generator, ckpt_for, fmt_pair, score, CKPT, L

NEUTRAL = "Possible times:"          # unconditional context for the PMI term


def load_items(limit=None):
    from datasets import load_dataset
    d = load_dataset("hails/bigbench", "temporal_sequences_zero_shot", split="default")
    out = []
    for x in d:
        ch = list(x["multiple_choice_targets"])
        if not ch:
            continue
        body = x["inputs"].split("  choice:")[0]
        passage = body.split("We know that:")[1] if "We know that:" in body else body
        out.append({"ctx": x["inputs"], "choices": ch,
                    "gold": ch.index(x["targets"][0]),
                    "seen": [c in passage for c in ch]})
        if limit and len(out) >= limit:
            break
    return out


def run_model(tag, items, batch, out_path):
    fam, ckpt = ckpt_for(tag)
    gen, tok = build_generator(fam, ckpt, None, 8192)

    cond, uncond = [], []
    for it in items:
        for c in it["choices"]:
            cond.append(fmt_pair(it["ctx"], " " + c, fam))
            uncond.append(fmt_pair(NEUTRAL, " " + c, fam))
    rc = score(gen, tok, cond, batch)
    ru = score(gen, tok, uncond, batch)
    del gen

    rules = {"acc": [], "acc_norm": [], "pmi": []}
    rows = []
    for i, it in enumerate(items):
        k = len(it["choices"])
        ll = [rc[i * k + j][3] for j in range(k)]
        lu = [ru[i * k + j][3] for j in range(k)]
        clen = [float(len(c)) for c in it["choices"]]
        picks = {
            "acc": max(range(k), key=lambda j: ll[j]),
            "acc_norm": max(range(k), key=lambda j: ll[j] / clen[j]),
            "pmi": max(range(k), key=lambda j: ll[j] - lu[j]),
        }
        for r, p in picks.items():
            rules[r].append(p)
        rows.append({"gold": it["gold"], "seen": it["seen"], **{f"pick_{r}": p for r, p in picks.items()}})

    n = len(items)
    summary = {"tag": tag, "n": n}
    for r, picks in rules.items():
        acc = sum(p == it["gold"] for p, it in zip(picks, items)) / n
        seen = sum(it["seen"][p] for p, it in zip(picks, items)) / n
        pos = Counter(picks)
        summary[r] = {"acc": acc, "seen_rate": seen,
                      "pos_dist": {str(j): pos.get(j, 0) / n for j in range(4)}}
    # reference points that do not depend on the model
    summary["seen_rate_if_uniform"] = sum(sum(it["seen"]) / len(it["seen"]) for it in items) / n
    summary["gold_seen_rate"] = sum(it["seen"][it["gold"]] for it in items) / n
    Path(out_path).write_text(json.dumps(summary, indent=2) + "\n")
    Path(str(out_path) + ".rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    for r in ("acc", "acc_norm", "pmi"):
        d = summary[r]
        print(f"  {tag:16s} {r:9s} acc={d['acc']:.3f} seen_rate={d['seen_rate']:.3f} "
              f"(uniform {summary['seen_rate_if_uniform']:.3f}) pos={ {k: round(v,2) for k,v in d['pos_dist'].items()} }",
              flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(CKPT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    items = load_items(a.limit)
    print(f"temporal_mech | n={len(items)} | gold-seen={sum(i['seen'][i['gold']] for i in items)}"
          f" | mean options-seen={sum(sum(i['seen']) for i in items)/len(items):.2f}/4", flush=True)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for t in a.models:
        run_model(t, items, a.batch_size, os.path.join(a.outdir, f"{t}.json"))


if __name__ == "__main__":
    main()

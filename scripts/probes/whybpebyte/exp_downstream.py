#!/usr/bin/env python3
"""Experiment F: does segmentation disruption predict loss degradation? (design section 9)

Paired design. Each example is a held-out span with exactly ONE character edit placed in its
first 40%. The model is scored teacher-forced twice -- on the clean span and on the edited span --
and the loss is read ONLY over a downstream region that starts `--margin` bytes after the edit and
at a space byte. Two properties make the pair exactly comparable:

  * the region text is byte-identical in both runs (the edit is entirely upstream), so the BPB
    denominators match;
  * the region starts at a space, and Llama-3's pre-tokenizer splits there, so the subword arm
    tokenizes the region identically in both runs too (verified per example, reported as
    `region_token_mismatch`).

Whatever Delta BPB remains is therefore the cost of conditioning on a corrupted CONTEXT -- for the
subword arm, a context that was re-tokenized; for the byte arms, one that was re-pooled.

    python exp_downstream.py --arm bpebyte --ckpt <consolidated> --out results/downstream_bpebyte.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/fflm"))
sys.path.insert(0, str(ROOT / "lingua"))

LN2 = math.log(2.0)
# arm -> (model family, segmenter name used by the model's data pipeline)
ARM_FAMILY = {"llama": "subword", "aunet": "aunet", "bpebyte": "aunet"}


def token_byte_spans(tokenizer, ids):
    """Byte [start, end) of every token id, in the concatenated decoding."""
    parts = tokenizer.tkt_model.decode_tokens_bytes(ids)
    lens = np.fromiter((len(p) for p in parts), dtype=np.int64, count=len(parts))
    ends = np.cumsum(lens)
    return np.concatenate(([0], ends[:-1])), ends


def region_bpb(family, tokenizer, ids, ll, lo, hi):
    """Bits/byte over clean-byte region [lo, hi) from per-token logprobs `ll` (nats)."""
    if family == "aunet":                      # one token == one byte
        seg = ll[lo:hi]
        return float(-seg.sum().item() / LN2 / max(hi - lo, 1)), int(hi - lo), None
    st, en = token_byte_spans(tokenizer, ids)
    sel = (st >= lo) & (en <= hi)
    nb = int((en[sel] - st[sel]).sum())
    tok = [int(t) for t, s in zip(ids, sel) if s]
    return float(-ll[np.flatnonzero(sel)].sum().item() / LN2 / max(nb, 1)), nb, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_FAMILY))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="en")
    ap.add_argument("--span", type=int, default=640, help="bytes per example")
    ap.add_argument("--sites", type=int, default=5, help="edit sites per window")
    ap.add_argument("--margin", type=int, default=8, help="protected zone after the edit")
    ap.add_argument("--families", nargs="*", default=["sub_key", "insert", "delete", "swap"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    import torch
    import perturb as P
    from segmenters import build, starts_to_mask
    from dump import load_windows
    from fflm_probe import build_generator, DEFAULT_TOK

    rng = random.Random(a.seed)
    wins = load_windows(a.domain, a.limit)
    seg = build(a.arm)

    # ---- build the example set once, so every arm scores identical (span, edit) pairs
    examples = []
    for wi, w in enumerate(wins):
        text = w["text"]
        for _ in range(a.sites):
            fam = rng.choice(a.families)
            # a span that starts and ends on character boundaries
            cs = P._char_starts(text)
            if int(cs[-1]) < a.span + 16:
                continue
            c0 = rng.randrange(0, len(text) - 1)
            b0 = int(cs[c0])
            if b0 + a.span > int(cs[-1]):
                continue
            c1 = int(np.searchsorted(cs, b0 + a.span))
            span = text[c0:c1]
            scs = P._char_starts(span)
            sites = [s for s in P.sample_edits(span, fam, 16, rng, margin=16)
                     if scs[s] < 0.4 * len(span.encode())]
            if not sites:
                continue
            r = P.apply_edit(span, fam, rng.choice(sites), rng)
            if r is None:
                continue
            pert, align, touched = r
            nb = len(span.encode())
            lo = int(touched[-1]) + 1 + a.margin
            raw = span.encode()
            while lo < nb and raw[lo] != 0x20:      # start the region at a space
                lo += 1
            if lo >= nb - 32:
                continue
            examples.append({"window": wi, "family": fam, "span": span, "pert": pert,
                             "align": align, "touched": touched, "lo": lo, "nb": nb})
    print(f"{len(examples)} examples", flush=True)

    # ---- per-example segmentation disruption for THIS arm (same quantity as Experiment D)
    for ex in examples:
        n, align, touched = ex["nb"], ex["align"], ex["touched"]
        cm = starts_to_mask(seg(ex["span"]), n)
        pm = starts_to_mask(seg(ex["pert"]), len(ex["pert"].encode()))
        d = np.zeros(n, np.int64)
        lo_t, hi_t = int(touched[0]), int(touched[-1])
        d[:lo_t] = lo_t - np.arange(lo_t)
        d[hi_t + 1:] = np.arange(hi_t + 1, n) - hi_t
        keep = (align >= 0) & (d >= 1)
        A = cm[keep] == pm[align[keep]]
        bad = ~A
        dk = d[keep]
        down = np.flatnonzero(keep) > hi_t
        ex["radius"] = int(dk[bad].max()) if bad.any() else 0
        ex["n_changed"] = int(bad.sum())
        ex["n_changed_downstream"] = int((bad & down).sum())
        ex["n_changed_gt_margin"] = int((bad & (dk > a.margin)).sum())
        ex["n_changed_in_region"] = int((bad & (np.flatnonzero(keep) >= ex["lo"])).sum())
        ex["n_units"] = int(cm.sum())

    # ---- score
    fam_model = ARM_FAMILY[a.arm]
    gen, tok = build_generator(fam_model, a.ckpt, os.environ.get("AUNET_TOK", DEFAULT_TOK), 8192)
    gen.max_gen_len = 1

    def score(texts):
        _, lls, _ = gen.generate(list(texts))
        return lls

    rows = []
    for i in range(0, len(examples), a.batch):
        chunk = examples[i:i + a.batch]
        ll_c = score([e["span"] for e in chunk])
        ll_p = score([e["pert"] for e in chunk])
        for e, lc, lp in zip(chunk, ll_c, ll_p):
            ids_c = tok.encode(e["span"], add_bos=False, add_eos=False)
            ids_p = tok.encode(e["pert"], add_bos=False, add_eos=False)
            lo_c, hi_c = e["lo"], e["nb"]
            lo_p, hi_p = int(e["align"][lo_c]), len(e["pert"].encode())
            bc, nbc, tc = region_bpb(fam_model, tok, ids_c, lc, lo_c, hi_c)
            bp, nbp, tp = region_bpb(fam_model, tok, ids_p, lp, lo_p, hi_p)
            rows.append({
                "arm": a.arm, "window": e["window"], "family": e["family"],
                "n_bytes": e["nb"], "region_bytes_clean": nbc, "region_bytes_pert": nbp,
                "region_token_mismatch": int(tc != tp) if tc is not None else 0,
                "clean_bpb": bc, "pert_bpb": bp, "delta_bpb": bp - bc,
                "radius": e["radius"], "n_changed": e["n_changed"],
                "n_changed_downstream": e["n_changed_downstream"],
                "n_changed_gt_margin": e["n_changed_gt_margin"],
                "n_changed_in_region": e["n_changed_in_region"],
                "n_units": e["n_units"],
            })
        if (i // a.batch) % 25 == 0:
            print(f"  {i + len(chunk)}/{len(examples)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dl = np.array([r["delta_bpb"] for r in rows])
    mm = np.mean([r["region_token_mismatch"] for r in rows])
    print(f"[{a.arm}] n={len(rows)}  mean dBPB={dl.mean():+.4f}  median={np.median(dl):+.4f}  "
          f"region_token_mismatch={mm:.3f}  -> {a.out}")


if __name__ == "__main__":
    main()

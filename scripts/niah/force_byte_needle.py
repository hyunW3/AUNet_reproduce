#!/usr/bin/env python3
"""Intervention probe for the AU-Net S-NIAH-3 gap: rescore the final S-NIAH pairs with the regex level mask
overridden so that the needle VALUE bytes are byte-level patches (every byte a boundary at every level).

  --force ctx   only the value's occurrence inside the haystack (what the latent stage has to retrieve)
  --force ans   only the teacher-forced answer bytes (what the decoder has to emit)
  --force both  both occurrences
  --force none  unchanged mask (baseline; must reproduce reports/niah/final/aunet_items.jsonl)
  --force ctrl  control: a same-length span of ordinary haystack text (ending 64 bytes before the needle value, or
                starting 64 bytes after it when there is no room before) -- does byte-forcing ANY span break the model?
  --force coarse_ctx / coarse_ans / coarse_both: the opposite direction -- one patch per hyphen group of the UUID
                (8/4/4/4/12 bytes; each '-' its own patch), i.e. MORE bytes per patch than the regex parse (~2.4)
--per_byte also writes the greedy correctness of every answer byte (where inside the value the first error lands).

The override is keyed by the exact token tuple (as the eval atomic_tail hook does), so batching order does not
matter. Scoring is the same teacher-forced greedy exact match as score_probe_pairs.py.
  cd lingua && PYTHONPATH=$PWD .venv/bin/python <this> --force ctx --ckpt ... --out ...
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, f"{L}/scripts/niah")
sys.path.insert(0, f"{L}/scripts/fflm")
from fflm_probe import build_generator, DEFAULT_TOK   # noqa: E402
from niah_ext_probe import score                      # noqa: E402

OFF = 0  # byte-tokenizer id offset (token = byte + OFF), set in main()


def value_spans(prompt, value, tok, force):
    """Token-index spans [s, e) of the value bytes in enc(prompt + ' ' + value) with BOS (generator layout)."""
    full = prompt + " " + value
    ids = tok.encode(full, add_bos=True, add_eos=False)
    assert len(ids) == 1 + len(full.encode()), "byte tokenizer expected"
    b, v = full.encode(), value.encode()
    ctx, ans = b.find(v), len(b) - len(v)
    assert 0 <= ctx < len(prompt.encode()), "needle value not found in the haystack"
    spans = {"ctx": [(1 + ctx, 1 + ctx + len(v))], "ans": [(1 + ans, 1 + ans + len(v))]}
    spans["both"] = spans["ctx"] + spans["ans"]
    s0 = ctx - 64 - len(v) if ctx - 64 - len(v) >= 0 else ctx + len(v) + 64
    assert s0 + len(v) <= len(prompt.encode())
    spans["ctrl"] = [(1 + s0, 1 + s0 + len(v))]
    for k in ("ctx", "ans", "both"):
        spans["coarse_" + k] = spans[k]
    return tuple(ids), spans.get(force, [])


def install(regex_pool, override, coarse=False):
    orig = regex_pool.get_levels_mask_prefill

    def wrapped(byte, size=None, force_first=False):
        flat = orig(byte, size=size, force_first=force_first)
        out, o = [], 0
        for b in byte:
            m = list(flat[o:o + len(b)]); o += len(b)
            top = max(max(m), 1)
            for s, e in override.get(tuple(b), ()):  # m[j] = top: byte j STARTS a patch ("x 12345" -> 0110010)
                for j in range(max(s - 1, 0), min(e, len(m))):  # the preceding space (its own patch natively), then 1 byte/patch
                    m[j] = top
                if coarse:  # keep only the starts of each hyphen group and of each '-'
                    for j in range(s + 1, e):
                        if b[j] != ord("-") + OFF and b[j - 1] != ord("-") + OFF:
                            m[j] = 0
                    if e < len(m):
                        m[e] = top  # the byte after the value starts a new patch
            out.extend(m)
        assert o == len(flat)
        return out

    regex_pool.get_levels_mask_prefill = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--force", choices=["none", "ctx", "ans", "both", "ctrl", "coarse_ctx", "coarse_ans", "coarse_both"], required=True)
    ap.add_argument("--pairs", default=f"{L}/reports/niah/sniah123_n250_final_pairs.jsonl")
    ap.add_argument("--probe", default="sniah3")
    ap.add_argument("--tag", default="aunet")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--cells", default=None, help="comma-separated subset of cells")
    ap.add_argument("--per_byte", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [r for r in map(json.loads, open(args.pairs)) if r["probe"] == args.probe]
    gen, tok = build_generator("aunet", args.ckpt, DEFAULT_TOK, args.max_tokens)
    override = {}
    for r in recs:
        key, spans = value_spans(r["prompt"], r["values"][0], tok, args.force)
        override[key] = spans
    global OFF
    OFF = tok.encode("-", add_bos=False, add_eos=False)[0] - ord("-")
    install(gen.regex_pool, override, coarse=args.force.startswith("coarse"))

    by_cell = defaultdict(list)
    for r in recs:
        by_cell[r["cell"]].append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for cell in sorted(by_cell, key=lambda c: int(c.split("/")[1])):
            if args.cells and cell not in args.cells.split(","):
                continue
            if args.per_byte:
                old = gen.max_gen_len; gen.max_gen_len = 1
                for i in range(0, len(by_cell[cell]), args.batch_size):
                    ch = by_cell[cell][i:i + args.batch_size]
                    _, _, greedy = gen.generate([r["prompt"] + " " + r["values"][0] for r in ch])
                    for r, gr in zip(ch, greedy):  # gr[t]: byte t+1 predicted correctly; value = last len(v) bytes
                        pb = [int(x) for x in gr[-len(r["values"][0].encode()):].tolist()]
                        f.write(json.dumps({"tag": args.tag, "force": args.force, "cell": cell, "depth": r["depth"],
                                            "exact": int(all(pb)), "per_byte": pb}) + "\n")
                gen.max_gen_len = old
                print(f"[{args.tag}/{args.force}] {cell} per_byte done", flush=True)
                continue
            sm = [{"prompt": r["prompt"], "values": r["values"]} for r in by_cell[cell]]
            res = score(gen, tok, sm, " ", args.batch_size)
            for r, x in zip(by_cell[cell], res):
                f.write(json.dumps({"tag": args.tag, "force": args.force, "cell": cell, "depth": r["depth"],
                                    "exact": int(x["exact"])}) + "\n")
            print(f"[{args.tag}/{args.force}] {cell:14} n={len(res)} exact={sum(x['exact'] for x in res) / len(res):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()

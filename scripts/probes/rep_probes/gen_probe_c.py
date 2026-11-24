#!/usr/bin/env python3
"""Probe C — duplicated-context QA (BoolQ, multiple choice).

BoolQ prompts are "<passage>\nQuestion: <q>?\nAnswer:". For k in {2, 3, 4} we prepend k-1 blocks before the
item's own passage, separated by blank lines, so the evidence stays adjacent to the question in every
condition (evidence-question distance fixed; only what precedes it changes):
  dup{k}   k-1 exact copies of the same passage  (repetition)
  ctrl{k}  k-1 other BoolQ passages of similar length, drawn by a seeded RNG (no repetition)
plus clean. Items are the first --n BoolQ items (robustness-run samples, byte-identical prompts) whose
4-copy version fits BLT's 4096-byte context.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python gen_probe_c.py --out reports/rep_probes/C/items.jsonl
"""
import argparse, json, os, random, zlib
from pathlib import Path

L = Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet"))
KS = (2, 3, 4)


def boolq_items():
    S = json.load(open(L / "runs/robustness_paper1p3b_ext/llama/results.json"))["samples"]["boolq"]
    for s in sorted(S, key=lambda s: int(s["doc_id"])):
        args = s["arguments"]
        if isinstance(args, dict):
            args = [(x["arg_0"], x["arg_1"]) for x in args.values()]
        ctx = args[0][0]
        i = ctx.rfind("\nQuestion:")
        yield int(s["doc_id"]), ctx[:i], ctx[i:], [c for _, c in args], int(s["target"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allp = list(boolq_items())
    passages = [p for _, p, _, _, _ in allp]
    out, k_used = [], 0
    for doc, passage, qpart, choices, gold in allp:
        if len(("\n\n".join([passage] * 4) + qpart + max(choices, key=len)).encode()) + 1 > 4096:
            continue
        base = {"type": "mc", "probe": "C", "task": "boolq", "choices": choices, "gold": gold}
        out.append({**base, "key": f"{doc}/clean", "cond": "clean", "length": 1, "ctx": passage + qpart})
        rng = random.Random(zlib.crc32(f"C|{doc}".encode()))
        n0 = len(passage.encode())
        similar = [p for p in passages if p != passage and 0.7 * n0 <= len(p.encode()) <= 1.3 * n0]
        for k in KS:
            out.append({**base, "key": f"{doc}/dup{k}", "cond": f"dup{k}", "length": k,
                        "ctx": "\n\n".join([passage] * k) + qpart})
            others = rng.sample(similar, k - 1) if len(similar) >= k - 1 else rng.sample(passages, k - 1)
            ctx = "\n\n".join(others + [passage]) + qpart
            if len((ctx + max(choices, key=len)).encode()) + 1 > 4096:
                others = [o.encode()[: n0].decode("utf-8", "ignore") for o in others]
                ctx = "\n\n".join(others + [passage]) + qpart
            out.append({**base, "key": f"{doc}/ctrl{k}", "cond": f"ctrl{k}", "length": k, "ctx": ctx})
        k_used += 1
        if k_used == a.n:
            break
    worst = max(len((it["ctx"] + max(it["choices"], key=len)).encode()) + 1 for it in out)
    assert worst <= 4096, worst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items ({k_used} BoolQ docs) -> {a.out}; worst {worst} B")


if __name__ == "__main__":
    main()

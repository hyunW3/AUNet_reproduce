#!/usr/bin/env python3
"""Probe A — repeated-filler robustness axis (multiple choice).

Base items: the clean lm-eval prompts of the five-task robustness suite (HellaSwag, ARC-E, ARC-C, PIQA,
BoolQ), the first --n items each that still fit 4096 B with the longest filler, taken byte-for-byte from the matched-model robustness run's samples.
A filler of L bytes is inserted:
  BoolQ                  between the passage and "\nQuestion:" (evidence -> question distance grows)
  HS / ARC-E / ARC-C / PIQA   before the prompt (these items carry no separate evidence passage)
Filler kinds (same byte length, so the contrast isolates repetition from length/distance):
  rep  the S-NIAH-1 noise passage ("The grass is green. ...") repeated
  nat  natural DCLM text (the S-NIAH-2/3 essay pool), a window chosen per item by a seeded RNG
Conditions: clean, rep{256,1024,2048}, nat{256,1024,2048}. Every scored sequence stays < 4096 B.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python gen_probe_a.py --out reports/rep_probes/A/items.jsonl
"""
import argparse, json, os, random, sys, zlib
from pathlib import Path

L = Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet"))
sys.path.insert(0, str(L / "scripts/niah"))
from niah_data import NOISE, _essay_pool  # noqa: E402

TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
LENGTHS = (256, 1024, 2048)


def clean_items(n):
    S = {}
    for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
        S.update(json.load(open(L / f"runs/{d}/llama/results.json"))["samples"])
    for t in TASKS:
        k = 0
        for s in sorted(S[t], key=lambda s: int(s["doc_id"])):
            args = s["arguments"]
            if isinstance(args, dict):
                args = [(x["arg_0"], x["arg_1"]) for x in args.values()]
            ctx, choices = args[0][0], [c for _, c in args]
            # keep only items that still fit BLT's 4096-byte context with the longest filler (+ separators, BOS)
            if len((ctx + max(choices, key=len)).encode()) + max(LENGTHS) + 8 > 4096:
                continue
            yield t, int(s["doc_id"]), ctx, choices, int(s["target"])
            k += 1
            if k == n:
                break


def cut_bytes(text, n):
    b = text.encode()[:n]
    return b.decode("utf-8", "ignore")


def filler(kind, n, rng, pool):
    if kind == "rep":
        return cut_bytes(NOISE * (n // len(NOISE) + 2), n).strip()
    start = rng.randint(0, len(pool) - 4 * n)
    return cut_bytes(pool[start:start + 2 * n], n).strip()


def insert(task, ctx, fill):
    if task == "boolq":
        i = ctx.rfind("\nQuestion:")
        assert i > 0, ctx[:80]
        return ctx[:i] + "\n\n" + fill + ctx[i:]
    return fill + "\n\n" + ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = _essay_pool()
    out = []
    for task, doc, ctx, choices, gold in clean_items(a.n):
        base = {"type": "mc", "probe": "A", "task": task, "choices": choices, "gold": gold}
        out.append({**base, "key": f"{task}/{doc}/clean", "cond": "clean", "length": 0, "ctx": ctx})
        for kind in ("rep", "nat"):
            for n in LENGTHS:
                rng = random.Random(zlib.crc32(f"{task}|{doc}|{n}".encode()))
                c = insert(task, ctx, filler(kind, n, rng, pool))
                out.append({**base, "key": f"{task}/{doc}/{kind}{n}", "cond": f"{kind}{n}", "length": n, "ctx": c})
    worst = max(len((it["ctx"] + max(it["choices"], key=len)).encode()) + 1 for it in out)
    assert worst <= 4096, worst
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items -> {a.out}; worst scored sequence {worst} B (+BOS)")


if __name__ == "__main__":
    main()

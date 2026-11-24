#!/usr/bin/env python3
"""Probe B — needle retrieval with a haystack of controlled repetitiveness.

The haystack is a P-byte window of natural DCLM text (the S-NIAH-2/3 essay pool) repeated to the target
length; P = "nat" uses a single long window (no repetition, = S-NIAH-2). Content type is the same at every
P, so only the repetition period changes. Needle, query, instruction and scoring are those of S-NIAH
(7-digit number, colon-free template, greedy exact match of " " + value).

Grid: P in {64, 128, 256, 512, 1024, nat} x length {2048, 4096} x 10 needle depths x --per_depth items.
The length is the WHOLE scored sequence (BOS + prompt + " " + value <= length bytes), as in the fit-4k
S-NIAH cell, so BLT never truncates.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python gen_probe_b.py --out reports/rep_probes/B/items.jsonl
"""
import argparse, json, os, random, sys, zlib
from pathlib import Path

L = Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet"))
sys.path.insert(0, str(L / "scripts/niah"))
from niah_data import INSTRUCTION, KEYS, _essay_pool  # noqa: E402

PERIODS = (64, 128, 256, 512, 1024, "nat")
LENGTHS = (2048, 4096)
DEPTHS = [(i + 0.5) / 10 for i in range(10)]


def snap(text, i):
    while 0 < i < len(text) and text[i] != " ":
        i += 1
    return i


def build(period, length, depth, rng, pool):
    key = rng.choice(KEYS)
    value = "".join(rng.choice("0123456789") for _ in range(7))
    start = rng.randint(0, len(pool) - 20000)
    if period == "nat":
        unit = pool[start:start + 12000]
    else:
        a = snap(pool, start) + 1
        unit = pool[a:a + period]
        unit = unit[:unit.rfind(" ") + 1] if " " in unit else unit + " "     # end on a word boundary
    body_full = unit * (12000 // max(1, len(unit)) + 2) if period != "nat" else unit
    needle = f"One of the special magic numbers for {key} is {value}. "
    query = (f"What is the special magic number for {key} mentioned in the provided text? "
             f"The special magic number for {key} is")

    def make(nchar):
        body = body_full[:nchar]
        cut = snap(body, int(len(body) * depth))
        hay = body[:cut] + needle + body[cut:]
        return f"{INSTRUCTION}\n\n{hay}\n\n{query}"
    lo, hi = 200, 12000                                    # largest body (chars) whose scored bytes fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if 1 + len((make(mid) + " " + value).encode()) <= length:
            lo = mid
        else:
            hi = mid - 1
    prompt = make(lo)
    return prompt, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_depth", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool = _essay_pool()
    out = []
    for period in PERIODS:
        for length in LENGTHS:
            for di, depth in enumerate(DEPTHS):
                for j in range(a.per_depth):
                    rng = random.Random(zlib.crc32(f"B|{length}|{di}|{j}".encode()))   # same key/value/window across P
                    prompt, value = build(period, length, depth, rng, pool)
                    n = 1 + len((prompt + " " + value).encode())
                    assert n <= length and value in prompt
                    out.append({"type": "gen", "probe": "B", "task": f"P{period}", "cond": f"P{period}/{length}",
                                "length": length, "depth": depth, "key": f"P{period}/{length}/{di}/{j}",
                                "prompt": prompt, "value": value})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items -> {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S-NIAH-1/2/3 with a period-P repeated haystack ("-rep" variants, Probe B at P=64 applied to S-NIAH).

Each item is the S-NIAH item of the final set (sniah123_n250_final_pairs.jsonl) with the same task, cell,
depth (i+0.5)/n, key and value (same seed-0 RNG stream and draw order as niah_data.make_sample). Only the
haystack body changes: it is a P-byte window of the task's own haystack source, cut at word boundaries and
repeated to the body length.
  S-NIAH-1 (noise): the window starts at the noise passage ("The grass is green. ... Here we ") -> period 64
                    instead of the original 90-byte passage.
  S-NIAH-2/3 (essay): the window starts where the original essay slice starts (snapped to a word), as in
                    Probe B.
Length: every item is sized to at most the scored bytes (BOS + prompt + " " + value) of its original
counterpart (binary search over the haystack size), so the two variants differ only in repetition. (Sizing by
characters would not do: a window holding multi-byte characters inflates when repeated, and 2k items would
exceed BLT's 4096-byte context.) The 4096 cell is therefore also capped at 4096 bytes, as fit-4k.

Output is the probe item format of scripts/probes/rep_probes/score_items.py (type "gen": greedy
teacher-forced exact match of " " + value = S-NIAH scoring), so all five models run through one scorer.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python scripts/niah/dump_sniah_rep_pairs.py \
      --out reports/niah/rep64/sniah_rep64_items.jsonl
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from niah_data import INSTRUCTION, KEYS, NOISE, TASKS, _essay_pool, _uuid  # noqa: E402


def snap(text, i):
    while 0 < i < len(text) and text[i] != " ":
        i += 1
    return i


def make_rep(target_bytes, depth, rng, ht, vt, period, digits=7):
    """make_sample with the same RNG draws (key, essay start, value), haystack = repeated period window."""
    key = rng.choice(KEYS)
    noun = "UUID" if vt == "uuid" else "number"
    body_target = max(target_bytes, 80 + 200)
    if ht == "essay":
        pool = _essay_pool()
        start = rng.randint(0, max(0, len(pool) - body_target - 1))
        a = snap(pool, start) + (1 if start else 0)
        src = pool[a:a + period]
    else:
        src = NOISE[:period]
    unit = src[:src.rfind(" ") + 1] if " " in src else src + " "
    body = (unit * (body_target // len(unit) + 2))[:body_target]
    value = _uuid(rng) if vt == "uuid" else "".join(rng.choice("0123456789") for _ in range(digits))
    needle = f"One of the special magic {noun}s for {key} is {value}. "
    cut = snap(body, int(len(body) * depth))
    hay = body[:cut] + needle + body[cut:]
    query = (f"What is the special magic {noun} for {key} mentioned in the "
             f"provided text? The special magic {noun} for {key} is")
    return f"{INSTRUCTION}\n\n{hay}\n\n{query}", value, len(unit)


def scored(prompt, value):
    return 1 + len((prompt + " " + value).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=64)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--orig", default="reports/niah/sniah123_n250_final_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    orig = {}
    for l in open(a.orig):
        d = json.loads(l)
        orig.setdefault(d["cell"], []).append(scored(d["prompt"], d["values"][0]))
    recs, units = [], Counter()
    for task in ("1", "2", "3"):
        ht, vt = TASKS[task]
        for tb in (512, 1024, 2048, 4096):
            rng = random.Random(a.seed)
            for i in range(a.n):
                depth = (i + 0.5) / a.n
                budget = orig[f"sniah{task}/{tb}"][i]
                state = rng.getstate()

                def build(t):
                    rng.setstate(state)
                    return make_rep(t, depth, rng, ht, vt, a.period)
                lo, hi = 64, 8192
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    p, v, _ = build(mid)
                    lo, hi = (mid, hi) if scored(p, v) <= budget else (lo, mid - 1)
                prompt, value, u = build(lo)
                assert scored(prompt, value) <= budget and value in prompt
                units[(task, u)] += 1
                cell = f"sniah{task}r/{tb}"
                recs.append({"type": "gen", "probe": "sniah_rep", "task": f"sniah{task}r", "cond": cell,
                             "cell": cell, "length": tb, "depth": depth, "target_bytes": lo,
                             "budget_bytes": budget, "period": a.period, "key": f"{cell}/{i}",
                             "prompt": prompt, "value": value})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {out}")
    for k, v in sorted(Counter(r["cell"] for r in recs).items()):
        print(f"  {k:16s} {v}")
    print("unit lengths (task, chars):", sorted(Counter({(t, u // 8 * 8): c for (t, u), c in units.items()}).items())[:12])


if __name__ == "__main__":
    main()

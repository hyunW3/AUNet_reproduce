#!/usr/bin/env python3
"""S-NIAH needle-value variants on the S-NIAH-3 items (tab:niah_aunet, tab:niah_variants).

Every item is an S-NIAH-3 item of the final set (essay haystack, 0.5k/1k/2k haystack cells + fit-4k cell, n=250 per
cell, depth (i+0.5)/250, seed-0 stream): same key, haystack slice and depth, because the main RNG makes exactly the
draws of niah_data.make_sample (key, essay start, UUID) and the variant value comes from a separate per-item RNG.
Only the value (and the noun naming it in needle and query) changes:
  letters32    32 random lowercase letters                          noun "code"
  digits32     32 random digits                                     noun "number"
  uuid_space   a UUID with its hyphens replaced by spaces           noun "UUID"
  hex32_sp4    32 hex characters with a space after every 4         noun "code"
  tok4         7 random 4-letter lowercase Llama-3 BPE vocab tokens joined by '-' (vocab4.txt, 4117 tokens,
               ~12 bits/group, 84 bits)                              noun "code"
  rand4        7 groups of 4 random lowercase letters joined by '-' (~18.8 bits/group, 132 bits)   noun "code"
  rand4x4/x5   rand4 with 4 / 5 groups (75 / 94 bits: brackets tok4's total)                      noun "code"
  tok4x11      tok4 with 11 groups (132 bits: rand4's total)                                     noun "code"
The 7-digit number (S-NIAH-2) and the UUID (S-NIAH-3) rows come from the final S-NIAH runs. The 4096 cell caps
BOS + prompt + " " + value at 4096 bytes (binary search over the haystack size, as dump_sniah_fit_pairs.py).
Output: score_items.py "gen" items.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python scripts/niah/dump_needle_variants.py --out reports/niah/needle_variants/items.jsonl
"""
import argparse, json, os, random, sys, zlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from niah_data import INSTRUCTION, KEYS, _essay_pool, _uuid  # noqa: E402

HEX = "0123456789abcdef"
_V4 = []


def vocab4():
    if not _V4:
        _V4.extend(open(Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet")) / "reports/niah/needle_variants/vocab4.txt").read().split())
    return _V4


def groups(r, n, tok):
    abc = "abcdefghijklmnopqrstuvwxyz"
    return "-".join(r.choice(vocab4()) if tok else "".join(r.choice(abc) for _ in range(4)) for _ in range(n))
VARIANTS = {
    "letters32": ("code", lambda r: "".join(r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(32))),
    "digits32": ("number", lambda r: "".join(r.choice("0123456789") for _ in range(32))),
    "uuid_space": ("UUID", lambda r: _uuid(r).replace("-", " ")),
    "hex32_sp4": ("code", lambda r: " ".join("".join(r.choice(HEX) for _ in range(4)) for _ in range(8))),
    "tok4": ("code", lambda r: groups(r, 7, True)),
    "rand4": ("code", lambda r: groups(r, 7, False)),
    "rand4x4": ("code", lambda r: groups(r, 4, False)),
    "rand4x5": ("code", lambda r: groups(r, 5, False)),
    "tok4x11": ("code", lambda r: groups(r, 11, True)),
}


def make(target_bytes, depth, rng, vrng, variant):
    """niah_data.make_sample(essay, uuid) with the value swapped; identical main-RNG draws."""
    key = rng.choice(KEYS)
    body_target = max(target_bytes, 80 + 200)
    pool = _essay_pool()
    start = rng.randint(0, max(0, len(pool) - body_target - 1))
    body = pool[start:start + body_target]
    _uuid(rng)                                             # the S-NIAH-3 value draw, discarded
    noun, gen = VARIANTS[variant]
    value = gen(vrng)
    needle = f"One of the special magic {noun}s for {key} is {value}. "
    cut = int(len(body) * depth)
    while 0 < cut < len(body) and body[cut] != " ":
        cut += 1
    hay = body[:cut] + needle + body[cut:]
    query = (f"What is the special magic {noun} for {key} mentioned in the "
             f"provided text? The special magic {noun} for {key} is")
    return f"{INSTRUCTION}\n\n{hay}\n\n{query}", value, key


def scored(p, v):
    return 1 + len((p + " " + v).encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset")
    a = ap.parse_args()
    recs = []
    for variant in a.variants.split(","):
        for tb in (512, 1024, 2048, 4096):
            rng = random.Random(0)
            for i in range(a.n):
                depth = (i + 0.5) / a.n
                vseed = zlib.crc32(f"{variant}|{tb}|{i}".encode())
                if tb == 4096:
                    state = rng.getstate()

                    def build(t):
                        rng.setstate(state)
                        return make(t, depth, rng, random.Random(vseed), variant)
                    lo, hi = 64, 4096
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        p, v, _ = build(mid)
                        lo, hi = (mid, hi) if scored(p, v) <= 4096 else (lo, mid - 1)
                    p, v, key = build(lo)
                else:
                    p, v, key = make(tb, depth, rng, random.Random(vseed), variant)
                assert scored(p, v) <= 4096 and v in p
                cell = f"{variant}/{tb}"
                recs.append({"type": "gen", "probe": "needle_variants", "task": variant, "cond": cell, "cell": cell,
                             "length": tb, "depth": depth, "key": f"{cell}/{i}", "niah_key": key,
                             "prompt": p, "value": v})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} -> {a.out}")
    for c, n in sorted(Counter(r["cell"] for r in recs).items()):
        print(f"  {c:18s} {n}")


if __name__ == "__main__":
    main()

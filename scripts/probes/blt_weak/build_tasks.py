#!/usr/bin/env python3
"""Probe suite for BLT's structural weak spots vs. BPEByte (reports/blt_weak/CANDIDATES.md).

Writes run_longctx.py-format JSONL (gen rows: greedy decode cut to `window` bytes, scored
`substr`; rank rows: argmax log P(option | prompt)), so Llama / AU-Net / BPEByte / BLT-1B
(official bytelatent) are scored on byte-identical prompts by one runner. Every prompt +
generation stays <= 4096 bytes (official BLT's RoPE table).

  hashhop      HashHop hops=1 (16-letter KEY = 'VAL'), BLT arm for reports/niah/hashhop_1p3b
  mkniah_word  RULER MK-NIAH (word keys, 7-digit values), K needles -- BLT arm
  mkniah_nc    MK-NIAH with 8-char code keys; distractor keys are random (rand) or the target
               key with ONE char substituted at the last / middle / first position (nc_*)
  sniah{1,2,3}_nc  S-NIAH-1/2/3 + 3 distractor needles: clean (none) | distract (unrelated
               keys) | key_nc (keys 1 letter off the target key) | keyval_nc (key_nc + values
               sharing all but the tail with the gold value)
  count        few-shot repetition counting (rank over " 1\\n".." 20\\n")
  copy         few-shot verbatim copy of random / periodic / periodic-with-1-mutation strings
  dist512      fixed ~3.5 KB essay, needle placed d bytes before the query (d sweeps 512)

  python scripts/probes/blt_weak/build_tasks.py --out_dir reports/blt_weak/data
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import zlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "niah"))
from niah_data import _essay_pool, _uuid, KEYS, NOISE, INSTRUCTION  # noqa: E402

MAX_BYTES = 4096
LETTERS = string.ascii_letters
LETTERS_LU = string.ascii_lowercase + string.ascii_uppercase   # HashHop alphabet order
ALNUM = string.ascii_lowercase + string.digits


def _seed(*parts) -> int:
    """Process-stable seed (built-in hash() of str is salted per process)."""
    return zlib.crc32(repr(parts).encode())


def _nb(s: str) -> int:
    return len(s.encode())


def gen_row(rid, task, cond, length, pos, prompt, answer, slack=2):
    return {"id": rid, "task": task, "cond": cond, "length": length, "pos": pos, "mode": "gen",
            "score": "substr", "prompt": prompt, "answers": [answer],
            "window": _nb(answer) + slack, "prompt_bytes": _nb(prompt)}


def rank_row(rid, task, cond, length, pos, prompt, options, gold):
    return {"id": rid, "task": task, "cond": cond, "length": length, "pos": pos, "mode": "rank",
            "prompt": prompt, "options": options, "gold": gold, "prompt_bytes": _nb(prompt)}


def essay(rng, n):
    pool = _essay_pool()
    s = rng.randint(0, len(pool) - n - 1)
    return pool[s:s + n].encode()[:n].decode(errors="ignore")   # n BYTES, not chars


def snap(body, cut):
    while 0 < cut < len(body) and body[cut] != " ":
        cut += 1
    return cut


def insert_needles(body, needles_at):
    """needles_at: [(depth, text)] -> body with each text inserted at a word boundary."""
    cuts = sorted(((snap(body, int(len(body) * d)), t) for d, t in needles_at), key=lambda x: x[0])
    out, prev = [], 0
    for c, t in cuts:
        out.append(body[prev:c])
        out.append(t)
        prev = c
    out.append(body[prev:])
    return "".join(out)


def sub1(s, i, rng, alphabet):
    return s[:i] + rng.choice([c for c in alphabet if c != s[i]]) + s[i + 1:]


# ------------------------------------------------------------------------------ hashhop
_HH_HEADER = "\n\nCOMPLETION:\nHops=1\n\nCoT=False\n\n"   # verbatim hashhop/generate.py


def make_hashhop(target_bytes, n_samples, hstr, seed):
    """Same generator as scripts/niah/hashhop_probe.py (byte-identical samples for a seed)."""
    samples = []
    line_len = hstr * 2 + 6
    for i in range(n_samples):
        rng = random.Random(seed * 100003 + i)
        n_pairs = max(2, round(target_bytes / line_len))
        pairs = {}
        while len(pairs) < n_pairs:
            pairs["".join(rng.choice(LETTERS_LU) for _ in range(hstr))] = \
                "".join(rng.choice(LETTERS_LU) for _ in range(hstr))
        keys = list(pairs)
        rng.shuffle(keys)
        prompt = "\n".join(f"{k} = '{pairs[k]}'" for k in keys)
        qk = rng.choice(keys)
        samples.append({"ctx": prompt + _HH_HEADER + qk + " = '", "value": pairs[qk],
                        "depth": keys.index(qk) / len(keys)})
    return samples


def build_hashhop(n, lengths, seed=0):
    rows = []
    for tb in lengths:
        for i, s in enumerate(make_hashhop(tb, n, 16, seed)):
            rows.append(gen_row(f"hh-{tb}-{i}", "hashhop", "h1", tb, round(s["depth"], 3),
                                s["ctx"], s["value"]))
    return rows


# ------------------------------------------------------------------------------ MK-NIAH
def _mk_prompt(rng, target_bytes, keys, vals, ti):
    noun = "number"
    needles = [f"One of the special magic {noun}s for {k} is {v}. " for k, v in zip(keys, vals)]
    body = essay(rng, target_bytes)
    order = list(range(len(keys)))
    rng.shuffle(order)
    depths = [(j + 0.5) / len(keys) + rng.uniform(-0.3, 0.3) / len(keys) for j in range(len(keys))]
    hay = insert_needles(body, [(depths[j], needles[order[j]]) for j in range(len(keys))])
    qk = keys[ti]
    query = (f"What is the special magic {noun} for {qk} mentioned in the provided text? "
             f"The special magic {noun} for {qk} is")
    return f"{INSTRUCTION}\n\n{hay}\n\n{query}", depths[order.index(ti)]


def build_mkniah_word(n, Ks, target_bytes=2048, seed=0):
    rows = []
    for K in Ks:
        rng = random.Random(seed * 7919 + K)
        for i in range(n):
            keys = rng.sample(KEYS, K)
            vals = ["".join(rng.choice("0123456789") for _ in range(7)) for _ in keys]
            ti = rng.randrange(K)
            p, d = _mk_prompt(rng, target_bytes, keys, vals, ti)
            rows.append(gen_row(f"mkw-K{K}-{i}", "mkniah_word", f"K{K}", target_bytes, round(d, 3),
                                p, vals[ti]))
    return rows


def build_mkniah_nc(n, Ks, target_bytes=2048, seed=0, klen=8):
    rows = []
    for mode in ("rand", "nc_last", "nc_mid", "nc_first"):
        for K in Ks:
            rng = random.Random(_seed(seed, mode, K))
            for i in range(n):
                tk = "".join(rng.choice(ALNUM) for _ in range(klen))
                keys = {tk}
                while len(keys) < K:
                    if mode == "rand":
                        keys.add("".join(rng.choice(ALNUM) for _ in range(klen)))
                    else:
                        pos = {"nc_last": klen - 1, "nc_mid": klen // 2, "nc_first": 0}[mode]
                        keys.add(sub1(tk, pos, rng, ALNUM))
                keys = [tk] + sorted(keys - {tk})
                vals = set()
                while len(vals) < K:
                    vals.add("".join(rng.choice("0123456789") for _ in range(7)))
                vals = list(vals)
                p, d = _mk_prompt(rng, target_bytes, keys, vals, 0)
                rows.append(gen_row(f"mknc-{mode}-K{K}-{i}", "mkniah_nc", f"{mode}_K{K}",
                                    target_bytes, round(d, 3), p, vals[0]))
    return rows


# ------------------------------------------------------------------------------ S-NIAH-1/2/3 + near-collision
SNIAH = {"1": ("noise", "number"), "2": ("essay", "number"), "3": ("essay", "uuid")}


def _val(rng, vt):
    return _uuid(rng) if vt == "uuid" else "".join(rng.choice("0123456789") for _ in range(7))


def _near_val(rng, gold, vt):
    """Shares everything with gold except the tail (last 2 digits / last 4 hex chars)."""
    if vt == "uuid":
        tail = 4
        alpha = "0123456789abcdef"
    else:
        tail = 2
        alpha = "0123456789"
    while True:
        v = gold[:-tail] + "".join(rng.choice(alpha) for _ in range(tail))
        if v != gold:
            return v


def _near_key(rng, key, avoid):
    while True:
        i = rng.randrange(1, len(key))            # keep the first letter: same "word" onset
        k = sub1(key, i, rng, string.ascii_lowercase)
        if k not in avoid:
            return k


def build_sniah_nc(n, lengths, D=3, seed=0):
    rows = []
    for t, (ht, vt) in SNIAH.items():
        noun = "UUID" if vt == "uuid" else "number"
        for cond in ("clean", "distract", "key_nc", "keyval_nc"):
            for tb in lengths:
                rng = random.Random(_seed(seed, t, cond, tb))
                for i in range(n):
                    depth = (i + 0.5) / n
                    key = rng.choice(KEYS)
                    gold = _val(rng, vt)
                    if ht == "essay":
                        body = essay(rng, tb)
                    else:
                        body = (NOISE * (tb // len(NOISE) + 3))[:tb]
                    nd = [(depth, f"One of the special magic {noun}s for {key} is {gold}. ")]
                    if cond != "clean":
                        used = {key}
                        for _ in range(D):
                            if cond == "distract":
                                k = rng.choice([x for x in KEYS if x not in used])
                            else:
                                k = _near_key(rng, key, used | set(KEYS))
                            used.add(k)
                            v = _near_val(rng, gold, vt) if cond == "keyval_nc" else _val(rng, vt)
                            dd = rng.uniform(0.02, 0.98)
                            nd.append((dd, f"One of the special magic {noun}s for {k} is {v}. "))
                    hay = insert_needles(body, nd)
                    query = (f"What is the special magic {noun} for {key} mentioned in the provided "
                             f"text? The special magic {noun} for {key} is")
                    p = f"{INSTRUCTION}\n\n{hay}\n\n{query}"
                    rows.append(gen_row(f"s{t}nc-{cond}-{tb}-{i}", f"sniah{t}_nc", cond, tb,
                                        round(depth, 3), p, gold))
    return rows


# ------------------------------------------------------------------------------ counting
_WORDS = ["cat", "apple", "river", "stone", "blue", "house", "green", "tree", "music", "light",
          "paper", "chair", "cloud", "bread", "glass", "horse", "money", "table", "water", "smile"]
_CHUNKS = ["abc", "xyz", "kpm", "the", "ing", "tion", "ab", "qrs"]


def _count_item(rng, cond, c):
    if cond == "word":
        u = rng.choice(_WORDS)
        return " ".join([u] * c), u
    if cond == "char":
        u = rng.choice(string.ascii_lowercase)
        return u * c, u
    if cond == "digit":
        u = rng.choice("123456789")
        return u * c, u
    if cond == "chunk_sep":
        u = rng.choice(_CHUNKS)
        return " ".join([u] * c), u
    if cond == "chunk_nosep":
        u = rng.choice(_CHUNKS)
        return u * c, u
    raise ValueError(cond)


def build_count(n, lo=2, hi=15, shots=4, seed=0, n_opt=20):
    rows = []
    opts = [f" {k}\n" for k in range(1, n_opt + 1)]
    for cond in ("word", "char", "digit", "chunk_sep", "chunk_nosep"):
        rng = random.Random(_seed(seed, "count", cond))
        for i in range(n):
            demo = []
            for _ in range(shots):
                c = rng.randint(lo, hi)
                s, u = _count_item(rng, cond, c)
                demo.append(f'Text: {s}\nNumber of times "{u}" appears: {c}\n')
            c = lo + i % (hi - lo + 1)                      # balanced gold counts
            s, u = _count_item(rng, cond, c)
            p = "\n".join(demo) + f'\nText: {s}\nNumber of times "{u}" appears:'
            rows.append(rank_row(f"cnt-{cond}-{i}", "count", cond, c, None, p, opts, c - 1))
    return rows


# ------------------------------------------------------------------------------ copy
def _copy_str(rng, cond, L):
    if cond == "rand":
        return "".join(rng.choice(LETTERS) for _ in range(L)), None
    per = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.choice([3, 4, 5])))
    s = (per * (L // len(per) + 1))[:L]
    if cond == "periodic":
        return s, None
    if cond == "periodic_mut":                      # one substitution after the first 2 periods
        i = rng.randrange(2 * len(per), L)
        return sub1(s, i, rng, string.ascii_lowercase), round(i / L, 3)
    if cond == "words_mut":                         # repeated word list with one swapped word
        w = rng.sample(_WORDS, 3)
        toks = []
        while len(" ".join(toks + [w[len(toks) % 3]])) <= L:
            toks.append(w[len(toks) % 3])
        j = rng.randrange(3, len(toks)) if len(toks) > 3 else len(toks) - 1
        toks[j] = rng.choice([x for x in _WORDS if x != toks[j]])
        return " ".join(toks), round(j / len(toks), 3)
    raise ValueError(cond)


def build_copy(n, Ls, shots=3, seed=0):
    rows = []
    for cond in ("rand", "periodic", "periodic_mut", "words_mut"):
        for L in Ls:
            rng = random.Random(_seed(seed, "copy", cond, L))
            for i in range(n):
                demo = []
                for _ in range(shots):
                    d, _ = _copy_str(rng, cond, rng.choice([16, 24, 32]))
                    demo.append(f"Input: {d}\nOutput: {d}\n")
                s, pos = _copy_str(rng, cond, L)
                p = "\n".join(demo) + f"\nInput: {s}\nOutput:"
                rows.append(gen_row(f"cp-{cond}-{L}-{i}", "copy", cond, L, pos, p, s, slack=2))
    return rows


# ------------------------------------------------------------------------------ 512-byte distance sweep
def build_dist512(n, dists, total=3500, seed=0):
    rows = []
    for cond in ("hash", "num"):
        for d in dists:
            rng = random.Random(_seed(seed, "dist", cond, d))
            for i in range(n):
                if cond == "hash":
                    key = "".join(rng.choice(LETTERS) for _ in range(12))
                    val = "".join(rng.choice(LETTERS) for _ in range(12))
                else:
                    key = rng.choice(KEYS)
                    val = "".join(rng.choice("0123456789") for _ in range(7))
                needle = f" The access code for {key} is {val}. "
                query = f"\n\nWhat is the access code for {key}? The access code for {key} is"
                head = f"{INSTRUCTION.replace('magic numbers', 'access codes').replace('the numbers', 'the codes')}\n\n"
                post = essay(rng, d)                             # needle end -> query start = d bytes
                pre_n = total - _nb(head) - _nb(needle) - d
                pre = essay(rng, max(0, pre_n))
                p = head + pre + needle + post + query
                rows.append(gen_row(f"d512-{cond}-{d}-{i}", "dist512", cond, d, None, p, val))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sets = {
        "hashhop": build_hashhop(a.n, [512, 1024, 2048, 3584]),
        "mkniah": build_mkniah_word(a.n, [1, 2, 4, 8]) + build_mkniah_nc(a.n, [2, 4, 8]),
        "sniah_nc": build_sniah_nc(a.n, [1024, 2048, 3200]),
        "count": build_count(2 * a.n),
        "copy": build_copy(a.n, [32, 64, 128]),
        "dist512": build_dist512(a.n, [64, 128, 256, 384, 448, 512, 576, 640, 768, 1024, 2048, 3072]),
    }
    for name, rows in sets.items():
        def need(r):
            return 1 + r["prompt_bytes"] + (r["window"] if r["mode"] == "gen"
                                            else max(_nb(o) for o in r["options"]))
        big = [r["id"] for r in rows if need(r) > MAX_BYTES]
        assert not big, f"{name}: {len(big)} rows exceed {MAX_BYTES} B, e.g. {big[:3]}"
        assert len({r["id"] for r in rows}) == len(rows), name
        with open(od / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mx = max(r["prompt_bytes"] for r in rows)
        print(f"{name:10s} {len(rows):5d} rows  max prompt {mx} B  "
              + " ".join(f"{k}={v}" for k, v in sorted(Counter((r['task'], r['cond']) for r in rows).items())[:6]))


if __name__ == "__main__":
    main()

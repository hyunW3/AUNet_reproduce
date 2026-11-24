#!/usr/bin/env python3
"""Probe D — key-value retrieval from structured, template-repetitive text vs prose.

A list of user records (id, name, city, 7-digit code) is rendered in one of four formats:
  json   one JSON object per record (repeated field names / punctuation)
  csv    header + comma-separated rows
  log    timestamped log lines with a fixed template
  prose  the same fields in sentences drawn from 8 templates (control: far less template repetition)
The question asks for the code of one target record; scoring = greedy exact match of " " + code
(teacher forced, as S-NIAH). Grid: format x length {1024, 2048, 4096} x 10 target positions x --per_depth.
The length is the WHOLE scored sequence (BOS + prompt + " " + code <= length bytes); records are added
until the next one would not fit. Record contents depend only on (length, position, item), so all
formats carry identical information.

  AUNET_ROOT=/mnt/ssd2/hyun2/AUNet python gen_probe_d.py --out reports/rep_probes/D/items.jsonl
"""
import argparse, json, random, zlib
from pathlib import Path

FORMATS = ("json", "csv", "log", "prose")
LENGTHS = (1024, 2048, 4096)
DEPTHS = [(i + 0.5) / 10 for i in range(10)]
NAMES = ["Alice", "Bruno", "Chen", "Dana", "Elif", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kofi", "Lena",
         "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wei", "Yara"]
CITIES = ["Lisbon", "Osaka", "Quito", "Tunis", "Oslo", "Lima", "Hanoi", "Perth", "Dakar", "Riga", "Cork",
          "Porto", "Accra", "Bern", "Cusco", "Delft", "Gent", "Izmir", "Kobe", "Lyon", "Malmo", "Nice"]
PROSE = ["{name} from {city} holds account {id}, and its code is {code}.",
         "Account {id} belongs to {name}, who lives in {city}; the code on file is {code}.",
         "The code for {id} is {code}. That account is registered to {name} in {city}.",
         "{name} ({city}) uses the account {id}, whose code reads {code}.",
         "Living in {city}, {name} opened account {id} with code {code}.",
         "Records list {id} under {name} of {city}, with code {code}.",
         "For account {id}, owned by {name} in {city}, the code is {code}.",
         "In {city}, {name} manages account {id}; its code is {code}."]
INSTR = {"json": "Below is a JSON list of user records.", "csv": "Below is a CSV table of user records.",
         "log": "Below is a log of user records.", "prose": "Below is a description of user records."}


def records(rng, n):
    ids = rng.sample(range(1000, 9999), n)
    return [{"id": f"user_{i}", "name": rng.choice(NAMES), "city": rng.choice(CITIES),
             "code": "".join(rng.choice("0123456789") for _ in range(7))} for i in ids]


def render(fmt, recs, rng):
    if fmt == "json":
        return "[\n" + ",\n".join(json.dumps(r) for r in recs) + "\n]"
    if fmt == "csv":
        return "id,name,city,code\n" + "\n".join(f"{r['id']},{r['name']},{r['city']},{r['code']}" for r in recs)
    if fmt == "log":
        return "\n".join(f"2026-03-{1 + i % 28:02d} 10:{i % 60:02d}:00 INFO record id={r['id']} name={r['name']} "
                         f"city={r['city']} code={r['code']}" for i, r in enumerate(recs))
    return " ".join(rng.choice(PROSE).format(**r) for r in recs)


def prompt_for(fmt, recs, target, rng_seed):
    body = render(fmt, recs, random.Random(rng_seed))
    q = f"What is the code for {target['id']}? The code for {target['id']} is"
    return f"{INSTR[fmt]}\n\n{body}\n\n{q}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_depth", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for length in LENGTHS:
        for di, depth in enumerate(DEPTHS):
            for j in range(a.per_depth):
                seed = zlib.crc32(f"D|{length}|{di}|{j}".encode())
                pool = records(random.Random(seed), 400)
                for fmt in FORMATS:
                    def fits(k):
                        recs = pool[:k]
                        t = recs[min(k - 1, int(k * depth))]
                        p = prompt_for(fmt, recs, t, seed)
                        return 1 + len((p + " " + t["code"]).encode()) <= length
                    lo, hi = 2, 400                         # most records that fit the budget
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
                    recs = pool[:lo]
                    t = recs[min(lo - 1, int(lo * depth))]
                    p = prompt_for(fmt, recs, t, seed)
                    assert 1 + len((p + " " + t["code"]).encode()) <= length and p.count(t["code"]) == 1
                    out.append({"type": "gen", "probe": "D", "task": fmt, "fmt": fmt, "cond": f"{fmt}/{length}",
                                "length": length, "depth": depth, "n_records": lo,
                                "key": f"{fmt}/{length}/{di}/{j}", "prompt": p, "value": t["code"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in out:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(out)} items -> {a.out}")


if __name__ == "__main__":
    main()

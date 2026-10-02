#!/usr/bin/env python3
"""Convert dumped teacher-forcing pairs ({cell, prompt, values}) into run_longctx.py gen rows, so the
existing NoLiMa-lite / MK-NIAH prompts (reports/niah/probe_pairs.jsonl, the ones the matched trio was
scored on) can be re-scored by all four models -- including official BLT -- with one runner.

  python convert_pairs.py --pairs reports/niah/probe_pairs.jsonl --only nolima --task nolima --out nolima.jsonl
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--only", default=None, help="keep cells starting with this prefix")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen = Counter()
    with open(a.out, "w") as f:
        for line in open(a.pairs):
            r = json.loads(line)
            if a.only and not r["cell"].startswith(a.only):
                continue
            parts = r["cell"].split("/")            # e.g. nolima/literal/1024
            cond, length = parts[1], int(parts[-1]) if parts[-1].isdigit() else parts[-1]
            i = seen[r["cell"]]
            seen[r["cell"]] += 1
            v = r["values"][0]
            f.write(json.dumps({"id": f"{a.task}-{cond}-{length}-{i}", "task": a.task, "cond": cond,
                                "length": length, "pos": None, "mode": "gen", "score": "substr",
                                "prompt": r["prompt"], "answers": [v], "window": len(v.encode()) + 2,
                                "prompt_bytes": len(r["prompt"].encode())}, ensure_ascii=False) + "\n")
    print(dict(seen))


if __name__ == "__main__":
    main()

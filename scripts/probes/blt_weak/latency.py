#!/usr/bin/env python3
"""End-to-end teacher-forced scoring latency of one 3000-byte window at batch size 1, per domain
(ood_windows.jsonl), so OOD patch blow-up shows up as wall-clock rather than only as a patch count.

  BLT   : ext_models.build_blt harness .score_pairs([("", text)]) -- entropy model + patching + model
  trio  : run_longctx.build_generator(...).generate([text]) with max_gen_len=1 -- parser + prefill

Each window: 2 warm-up calls (first domain only), then `--reps` timed calls; median per window,
then median over windows. Run alone on an otherwise idle GPU.

  python latency.py --family blt --ckpt <lc>/ext/blt_weights --tag blt_1b --windows ood_windows.jsonl --out o.jsonl
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt", "aunet", "subword"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--per_domain", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.environ["LC_SCRIPTS"])
    if a.family == "blt":
        import ext_models
        h = ext_models.build_blt(a.ckpt)
        call = lambda t: h.score_pairs([("", t)])
    else:
        from run_longctx import build_generator
        g, _ = build_generator(a.family, a.ckpt, os.environ.get("AUNET_TOK", ""), 8192)
        g.max_gen_len = 1
        call = lambda t: g.generate([t])
    wins = [json.loads(l) for l in open(a.windows)]
    doms = list(dict.fromkeys(w["domain"] for w in wins))
    for _ in range(2):
        call(wins[0]["text"])
    with open(a.out, "w") as f:
        for d in doms:
            ws = [w for w in wins if w["domain"] == d][: a.per_domain]
            per = []
            for w in ws:
                ts = []
                for _ in range(a.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    call(w["text"])
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                per.append(st.median(ts))
            rec = {"tag": a.tag, "domain": d, "n": len(ws), "bytes": st.mean(w["n_bytes"] for w in ws),
                   "ms_median": 1000 * st.median(per), "ms_min": 1000 * min(per), "ms_max": 1000 * max(per)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec, flush=True)


if __name__ == "__main__":
    main()

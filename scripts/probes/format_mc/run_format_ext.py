#!/usr/bin/env python3
"""format_mc on the external references, built exactly as scripts/probes/ext_ci/run_ext.py builds them
for the paper robustness axes: BLT-1B through official bytelatent (xformers, native entropy/local
windows, batch size 1, released threshold) and H-Net 1-stage XL (eval_suite HNetHarness, batch 8).

  python run_format_ext.py --family blt_official --tasks piqa --items_dir I --cache C --out X.json
Env: EVAL_SUITE, HNET_REPO (H-Net), PYTHONPATH with the bytelatent checkout (BLT).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ext_ci"))
import format_mc as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt_official", "hnet"], required=True)
    ap.add_argument("--tasks", nargs="*", default=list(F.TASKS))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--group", choices=["A", "B"], default=None, help="variant half (format_mc.variant_group)")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--items_dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--reverse", action="store_true", help="score the to-do list from the end (helper worker)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/AUNet_lc/ext/blt_weights"))
    ap.add_argument("--hnet_model", default="hnet_1stage_XL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import run_ext as R
    R._lm_eval_compat()
    t0 = time.time()
    if a.family == "blt_official":
        lm = R.build_blt_official(a.blt_weights, a.max_len, a.threshold)
        meta = {"model": "facebook/blt-1b (official bytelatent, xformers)", "threshold": a.threshold,
                "batch_size": 1}
    else:
        lm = R.build_hnet_lm(batch_size=8, max_len=a.max_len, name=a.hnet_model)
        meta = {"model": f"cartesia-ai/{a.hnet_model}", "batch_size": 8}
    variants = F.variant_group(a.group) if a.group else a.variants
    res = F.run_format_mc(lm.loglikelihood, tasks=a.tasks, variants=variants, limit=a.limit,
                          items_dir=a.items_dir, cache=a.cache, chunk=a.chunk, reverse=a.reverse,
                          log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    out = {"family": a.family, "tasks": a.tasks, "limit": a.limit, "max_len": a.max_len, **meta,
           "results": res, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

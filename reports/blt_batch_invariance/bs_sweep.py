#!/usr/bin/env python3
"""Does BLT-1B's lm-eval score depend on the eval batch size?

Loads the official BLT-1B once (run_ext.build_blt_official: xformers, threshold 1.3354, native windows)
and scores the same 0-shot requests at several batch sizes, saving per-request logprobs and per-doc
correctness so runs can be compared item by item.

  --variant fixed    eval_suite harness as is (per-row entropies, _row_entropies)
  --variant fixed_entfp32  as fixed, entropy model in fp32
  --variant unfixed  BLT's calculate_entropies path (flattens the batch into one 8192-byte stream;
                     the pre-Aug-29 harness, = code_appendix/eval_suite/blt_eval/harness.py)
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes/ext_ci")
import run_ext  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["fixed", "unfixed", "fixed_entfp32"], required=True)
ap.add_argument("--bs", type=int, nargs="+", default=[1, 4, 8, 16])
ap.add_argument("--tasks", nargs="+", default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq"])
ap.add_argument("--limit", type=int, default=500)
ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)) + "/out")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

if os.environ.get("STATIC_SHAPES"):  # avoid the dynamic-shape create_block_mask kernel that faults at bs>1
    import torch._dynamo
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.recompile_limit = 4096
    torch._dynamo.config.cache_size_limit = 4096
run_ext._lm_eval_compat()
lm = run_ext.build_blt_official("/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/blt_weights", 4096)
if a.variant == "fixed_entfp32":  # entropy model in fp32 (main model stays bf16)
    lm.patcher.entropy_model.float()
if a.variant == "unfixed":
    orig = lm.patcher.patch
    lm.patcher.patch = lambda toks, include_next_token=False, entropies=None, **kw: orig(
        toks, include_next_token=include_next_token, **kw)

log = []
orig_ll = lm.loglikelihood
def ll(requests):
    out = orig_ll(requests)
    log.extend([lp for lp, _ in out])
    return out
lm.loglikelihood = ll

for bs in a.bs:
    path = f"{a.out_dir}/{a.variant}_bs{bs}.json"
    if os.path.exists(path):
        print("skip", path); continue
    lm.batch_size = bs; log.clear(); t0 = time.time()
    try:
        res, samples = run_ext.run_lm_eval(lm, a.tasks, 0, a.limit)
        rec = {"variant": a.variant, "bs": bs, "limit": a.limit, "seconds": round(time.time() - t0),
               "results": res, "samples": samples, "req_logprobs": list(log)}
    except Exception as e:  # e.g. the bs=4 CUDA error seen in reports/verify_blt_1335
        rec = {"variant": a.variant, "bs": bs, "error": repr(e)[:2000], "seconds": round(time.time() - t0)}
        print("ERROR", bs, repr(e)[:500], flush=True)
    json.dump(rec, open(path, "w"), default=str)
    print("wrote", path, rec["seconds"], "s", flush=True)
    if "error" in rec:
        break  # CUDA context is likely poisoned

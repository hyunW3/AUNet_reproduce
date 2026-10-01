#!/usr/bin/env python3
"""Isolate why the (fixed) BLT harness is batch-size dependent.

For the first N lm-eval requests (hellaswag + arc_easy, 0-shot) compare each request's logprob and
real-region patch boundaries under:
  A  bs=1 run twice                        (determinism)
  B  [x]*8  (same Np padding, larger batch) (batch-dimension kernel numerics only)
  C  [x, longest request]                  (extra right padding from a longer batch-mate)
  D  [x] alone but entropies from the unfixed flattened path with a batch-mate before it
"""
import json, sys, torch
sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/scripts/probes/ext_ci")
import run_ext  # noqa: E402

run_ext._lm_eval_compat()
lm = run_ext.build_blt_official("/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/blt_weights", 4096)
pairs = []
orig_ll = lm.loglikelihood
lm.loglikelihood = lambda reqs: (pairs.extend(r.args for r in reqs), orig_ll(reqs))[1]
run_ext.run_lm_eval(lm, ["hellaswag", "arc_easy"], 0, 16)
import os
if os.environ.get("ENT_FP32"): lm.patcher.entropy_model.float()
lm.loglikelihood = orig_ll
print("requests", len(pairs))

tok, boe = lm.tokenizer, lm.tokenizer.boe_id
def real_bounds(rows):
    """patch start ids within each row's real bytes, using the harness's padding + per-row entropies."""
    enc = [tok.encode(c + x, add_bos=True, add_eos=False) for c, x in rows]
    Np = ((max(map(len, enc)) + 127) // 128) * 128
    t = torch.full((len(enc), Np), boe, dtype=torch.long, device="cuda")
    for i, f in enumerate(enc):
        t[i, :len(f)] = torch.tensor(f, device="cuda")
    with torch.inference_mode():
        pl, _ = lm.patcher.patch(t, include_next_token=False, entropies=lm._row_entropies(t))
    out = []
    for i, f in enumerate(enc):
        s = torch.cumsum(pl[i], 0).tolist(); starts = [0] + s[:-1]
        out.append([p for p, l in zip(starts, pl[i].tolist()) if l > 0 and p < len(f)])
    return out

longest = max(pairs, key=lambda p: len(p[0]) + len(p[1]))
stats = {k: [] for k in "ABCD"}
bflip = {k: 0 for k in "BC"}
for x in pairs:
    lp1 = lm._score_batch([x])[0][0]
    stats["A"].append(abs(lm._score_batch([x])[0][0] - lp1))
    stats["B"].append(abs(lm._score_batch([x] * 8)[0][0] - lp1))
    stats["C"].append(abs(lm._score_batch([x, longest])[0][0] - lp1))
    b1 = real_bounds([x])[0]
    bflip["B"] += real_bounds([x] * 8)[0] != b1
    bflip["C"] += real_bounds([x, longest])[0] != b1
# D: unfixed path, x placed after a batch-mate
orig = lm.patcher.patch
lm.patcher.patch = lambda t, include_next_token=False, entropies=None, **kw: orig(t, include_next_token=include_next_token, **kw)
for x in pairs:
    lp1 = stats_ref = None
lm.patcher.patch = orig
res = {}
fixed_lp1 = [lm._score_batch([x])[0][0] for x in pairs]
lm.patcher.patch = lambda t, include_next_token=False, entropies=None, **kw: orig(t, include_next_token=include_next_token, **kw)
for x, lp1 in zip(pairs, fixed_lp1):
    stats["D"].append(abs(lm._score_batch([longest, x])[1][0] - lp1))
lm.patcher.patch = orig
for k, v in stats.items():
    v = torch.tensor(v)
    res[k] = {"n": len(v), "mean_abs": v.mean().item(), "max_abs": v.max().item(), "frac_gt_0.01": (v > 0.01).float().mean().item()}
res["boundary_changed_B"] = bflip["B"]; res["boundary_changed_C"] = bflip["C"]
print(json.dumps(res, indent=1))
json.dump(res, open("/mnt/ssd2/hyun2/AUNet/reports/blt_batch_invariance/diag%s.json" % ("_entfp32" if os.environ.get("ENT_FP32") else "") + "", "w"), indent=1)

#!/usr/bin/env python3
"""H-Net (1-stage / 2-stage XL) robustness on the unified HellaSwag + ARC-Easy suite.

Mirrors scripts/probes/blt_robustness_native.py so the two external references are scored on
byte-identical perturbations with the same aggregation, using the AU-Net repo's own
apps.aunet.eval_{noise,typo,despace_mc,pbp_mc} code.

Two deviations from eval_suite/hnet_eval/harness.py, both required here:
  * build_hnet() fetches the model config over HTTPS from raw.githubusercontent.com. This box has
    the configs locally (hnet repo `configs/`), so we read them from disk -- no network at eval time.
  * batch_size is exposed and defaults to 8. Unlike BLT, H-Net's harness passes an explicit
    attention `mask`, so padded rows should not shift the routing -- but "should not" is what was
    also assumed of BLT before bs16 vs bs8 moved ARC-Easy by 3.8 points. --check_batch runs the
    same items at two batch sizes and reports the gap, so the assumption is tested, not trusted.

  $HNET_PY hnet_robustness.py --variant hnet_1stage_XL --limit 2000 --out out.json
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
import torch

ROOT = "/mnt/ssd2/hyun2/AUNet"
HNET_REPO = os.environ.get("HNET_REPO", "/mnt/ssd2/hyun2/hnet")
EVAL_SUITE = os.environ.get("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite")

TASKS = ("hellaswag", "arc_easy")
_MK = "acc_norm,none"
AU = {}


def _preload_aunet():
    """AU-Net perturbation code must bind `apps` before any repo that ships its own."""
    from apps.aunet.eval_noise import expand_noise_tasks, summarize_noise
    from apps.aunet.eval_typo import expand_typo_tasks, summarize_typo
    from apps.aunet.eval_typo_ds import expand_typo_ds_tasks, summarize_typo_ds
    from apps.aunet.eval_despace_mc import run_despace_mc, DESPACE_TASKS
    from apps.aunet.eval_pbp_mc import run_pbp_mc, MC_TASKS
    AU.update(locals())
    print(f"[aunet] despace={DESPACE_TASKS} pbp_mc={MC_TASKS}", flush=True)


def build_local(variant):
    """build_hnet() but with the config read from the local checkout instead of GitHub."""
    from huggingface_hub import hf_hub_download
    # Upstream moved these under hnet.models/ after the eval_suite harness was written
    # (harness.py still imports the flat paths and now ImportErrors). Try new layout first.
    try:
        from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
        from hnet.models.mixer_seq import HNetForCausalLM
    except ModuleNotFoundError:
        from hnet.config_hnet import AttnConfig, HNetConfig, SSMConfig
        from hnet.mixer_seq import HNetForCausalLM
    wp = hf_hub_download(f"cartesia-ai/{variant}", filename=f"{variant}.pt")
    cfg = json.load(open(os.path.join(HNET_REPO, "configs", f"{variant}.json")))
    cfg["ssm_cfg"] = SSMConfig(**cfg.get("ssm_cfg", {}))
    cfg["attn_cfg"] = AttnConfig(**cfg.get("attn_cfg", {}))
    model = HNetForCausalLM(HNetConfig(**cfg), device="cuda", dtype=torch.bfloat16)
    model.load_state_dict(torch.load(wp, map_location="cuda", weights_only=True), strict=False)
    model.eval()
    print(f"[hnet] loaded {variant} from {wp}", flush=True)
    return model


def _acc(res, k):
    r = res.get(k) or {}
    return r.get(_MK, r.get("acc_norm"))


def axis_noise(lm, limit):
    from lm_eval import simple_evaluate
    from common.compat import safe_task_manager
    tl = AU["expand_noise_tasks"](["hellaswag", "hellaswag_noise", "arc_easy", "arc_easy_noise"])
    res = simple_evaluate(lm, tasks=tl, num_fewshot=0, task_manager=safe_task_manager(),
                          limit=limit, log_samples=False, bootstrap_iters=0)["results"]
    res.update(AU["summarize_noise"](res))
    out = {}
    for t in TASKS:
        c, n = _acc(res, t), _acc(res, f"{t}_noise_avg")
        out[t] = {"clean": c, "pert": n,
                  "delta_pp": round(100 * (n - c), 2) if (c is not None and n is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_typo(lm, limit):
    from lm_eval import simple_evaluate
    from common.compat import safe_task_manager
    tl = AU["expand_typo_tasks"](["hellaswag", "hellaswag_typo", "arc_easy", "arc_easy_typo"])
    tl = AU["expand_typo_ds_tasks"](tl)
    res = simple_evaluate(lm, tasks=tl, num_fewshot=0, task_manager=safe_task_manager(),
                          limit=limit, log_samples=False, bootstrap_iters=0)["results"]
    res.update(AU["summarize_typo"](res)); res.update(AU["summarize_typo_ds"](res))
    out = {}
    for t in TASKS:
        c, n = _acc(res, t), _acc(res, f"{t}_typo_avg")
        out[t] = {"clean": c, "pert": n,
                  "delta_pp": round(100 * (n - c), 2) if (c is not None and n is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_despace(lm, limit):
    res = AU["run_despace_mc"](lm.loglikelihood, limit=limit)
    out = {}
    for t in TASKS:
        c = (res.get(f"despace_mc_{t}_clean") or {}).get("acc")
        a = (res.get(f"despace_mc_{t}_despaceall") or res.get(f"despace_mc_{t}_all100") or {}).get("acc")
        out[t] = {"clean": c, "pert": a,
                  "delta_pp": round(100 * (a - c), 2) if (c is not None and a is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_pbp(lm, limit):
    res = AU["run_pbp_mc"](lm.loglikelihood, limit=limit)
    out = {}
    for t in list(TASKS) + ["curated"]:
        c = (res.get(f"pbp_mc_{t}_canonical") or {}).get("acc")
        s = (res.get(f"pbp_mc_{t}_space") or {}).get("acc")
        out[t] = {"clean": c, "pert": s,
                  "delta_pp": round(100 * (s - c), 2) if (c is not None and s is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


AXES = {"noise": axis_noise, "typo": axis_typo, "despace": axis_despace, "pbp": axis_pbp}


def check_batch(lm_cls, model, n=120):
    """Same items at bs=1 and bs=8 -> does batch composition move the score? (BLT's failure mode)"""
    from lm_eval import simple_evaluate
    from common.compat import safe_task_manager
    got = {}
    for bs in (1, 8):
        lm = lm_cls(model, batch_size=bs, max_len=8192)
        r = simple_evaluate(lm, tasks=["arc_easy"], num_fewshot=0,
                            task_manager=safe_task_manager(), limit=n,
                            log_samples=False, bootstrap_iters=0)["results"]["arc_easy"]
        got[bs] = round(100 * r.get("acc_norm,none", r.get("acc_norm", 0)), 4)
        print(f"  bs={bs}: arc_easy acc_norm={got[bs]}", flush=True)
    got["gap_pp"] = round(abs(got[1] - got[8]), 4)
    got["n"] = n
    print(f"  batch-composition gap = {got['gap_pp']} pp", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="hnet_1stage_XL",
                    choices=["hnet_1stage_XL", "hnet_2stage_XL", "hnet_1stage_L", "hnet_2stage_L"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--axes", default="noise,typo,despace,pbp")
    ap.add_argument("--check_batch", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["DESPACE_TASKS"] = ",".join(TASKS)
    os.environ["PBP_MC_TASKS"] = ",".join(TASKS)
    os.environ["DESPACE_FULL"] = "1"
    os.environ["DESPACE_ANSWER"] = "1"

    sys.path.insert(0, HNET_REPO)
    sys.path.insert(0, EVAL_SUITE)
    sys.path.insert(0, f"{ROOT}/lingua")
    _preload_aunet()
    # eval_suite/hnet_eval/harness.py does `from hnet.mixer_seq import ...` at import time.
    # Register the new module paths under the old names so that import still resolves.
    import importlib
    for old_name, new_name in (("hnet.config_hnet", "hnet.models.config_hnet"),
                               ("hnet.mixer_seq",   "hnet.models.mixer_seq")):
        if old_name not in sys.modules:
            try:
                sys.modules[old_name] = importlib.import_module(new_name)
            except ModuleNotFoundError:
                pass
    from hnet_eval.harness import HNetHarness

    model = build_local(args.variant)
    row = {"model": args.variant, "batch_size": args.batch_size, "limit": args.limit,
           "tasks": list(TASKS), "axes": {}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.check_batch:
        row["batch_check"] = check_batch(HNetHarness, model)
        json.dump(row, open(args.out, "w"), indent=2)

    lm = HNetHarness(model, batch_size=args.batch_size, max_len=args.max_len)
    for name in [a.strip() for a in args.axes.split(",") if a.strip()]:
        t0 = time.time()
        print(f"\n===== {args.variant} · axis {name} =====", flush=True)
        try:
            row["axes"][name] = AXES[name](lm, args.limit)
        except Exception as e:
            import traceback; traceback.print_exc()
            row["axes"][name] = {"error": f"{type(e).__name__}: {e}"}
        row["axes"][name]["_minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(row, open(args.out, "w"), indent=2)
        print(f"[{name}] {row['axes'][name]['_minutes']} min "
              f"Δ={row['axes'][name].get('delta_mean_pp')}", flush=True)

    json.dump(row, open(args.out, "w"), indent=2)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

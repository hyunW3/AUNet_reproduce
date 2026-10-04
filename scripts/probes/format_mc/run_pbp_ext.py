#!/usr/bin/env python3
"""PBP-MC (apps.aunet.eval_pbp_mc, the matched models' scorer) on BLT / H-Net, keeping acc AND acc_norm.

The paper's BLT/H-Net PBP files (reports/robustness_ext/*_pbp_*.json) stored acc only; this reruns the
same five tasks (limit 2000) so the PBP column can be compared under acc_norm as well. Note that
eval_pbp_mc deliberately reports PBP in acc: under acc_norm the trailing-space cut changes each option's
normalising length, so even a cut-invariant byte model can move.

  python run_pbp_ext.py --family blt_official [--threshold T] --tasks hellaswag ... --out X.json
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext_ci"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt_official", "hnet"], required=True)
    ap.add_argument("--tasks", nargs="*", default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq"])
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/AUNet_lc/ext/blt_weights"))
    ap.add_argument("--hnet_model", default="hnet_1stage_XL")
    ap.add_argument("--items_dir", default=None,
                    help="frozen format_mc items (<task>_2000.json); bypasses lm-eval/datasets doc loading")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["PBP_MC_TASKS"] = ",".join(a.tasks)          # read by eval_pbp_mc at import
    import run_ext as R
    R._lm_eval_compat()
    sys.path.insert(0, R.LINGUA)
    sys.modules.pop("apps", None)
    import apps.aunet.eval_pbp_mc as P
    if a.items_dir:   # same {context, choices, gold} records _items_from_task builds, frozen to json
        P._items_from_task = lambda task, limit=2000: json.load(open(os.path.join(a.items_dir, f"{task}_2000.json")))[:limit]
    P.MC_TASKS = list(a.tasks)   # older lingua checkouts (ece) hardcode the task list instead of PBP_MC_TASKS
    run_pbp_mc = P.run_pbp_mc
    t0 = time.time()
    if a.family == "blt_official":
        lm = R.build_blt_official(a.blt_weights, a.max_len, a.threshold)
        meta = {"model": "facebook/blt-1b (official bytelatent, xformers)", "threshold": a.threshold, "batch_size": 1}
    else:
        lm = R.build_hnet_lm(batch_size=8, max_len=a.max_len, name=a.hnet_model)
        meta = {"model": f"cartesia-ai/{a.hnet_model}", "batch_size": 8}
    res = run_pbp_mc(lm.loglikelihood, limit=a.limit)
    out = {"family": a.family, "axis": "pbp", "tasks": a.tasks, "limit": a.limit, **meta, "results": res,
           "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    for t in a.tasks:
        c, s = res.get(f"pbp_mc_{t}_canonical"), res.get(f"pbp_mc_{t}_space")
        if c and s:
            print(f"{t}: acc {c['acc']:.4f}->{s['acc']:.4f}  acc_norm {c['acc_norm']:.4f}->{s['acc_norm']:.4f}", flush=True)
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

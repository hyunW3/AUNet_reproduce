#!/usr/bin/env python3
"""Visual-attack robustness (VIPER-ICES / Bad Characters homoglyphs / ANTHRO at p in {0.1, 0.5},
plus the official Bad Characters MNLI homoglyph release) for the external references, through
the same model builders and lm-eval path as ext_ci/run_ext.py.

  python run_visual.py --family blt_official --tasks piqa --limit 1000 --out X.json
  python run_visual.py --family hnet --tasks hellaswag arc_easy ... mnli_badchars --out X.json

A task name in --tasks expands to "<task>" (clean) + "<task>_visual"; "mnli_badchars" is passed
through as the MNLI sentinel (budget 0 = clean). Perturbation code: lingua/apps/aunet/eval_visual.py.
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ext_ci"))
import run_ext as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt", "blt_official", "hnet"], required=True)
    ap.add_argument("--tasks", nargs="*", default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq",
                                                   "mnli_badchars"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--hnet_model", default="hnet_1stage_XL")
    ap.add_argument("--hnet_bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    R._lm_eval_compat()
    sys.path.insert(0, R.LINGUA)
    sys.modules.pop("apps", None)
    from apps.aunet.eval_noise import expand_noise_tasks, summarize_noise
    from apps.aunet import eval_visual as ev

    t0 = time.time()
    if a.family == "blt_official":
        lm = R.build_blt_official(a.blt_weights, a.max_len, a.threshold)
        meta = {"model": "facebook/blt-1b (official bytelatent, xformers)", "threshold": a.threshold, "batch_size": 1}
    elif a.family == "blt":
        lm = R.BLTHF(max_len=a.max_len)
        meta = {"model": "itazap/blt-1b-hf"}
    else:
        lm = R.build_hnet_lm(batch_size=a.hnet_bs, max_len=a.max_len, name=a.hnet_model)
        meta = {"model": f"cartesia-ai/{a.hnet_model}"}
    names = [x for t in a.tasks for x in ((t,) if t == ev.MNLI_SENTINEL else (t, f"{t}_visual"))]
    tl = expand_noise_tasks(names, base_seed=R.PERTURB_SEED)
    res, samples = R.run_lm_eval(lm, tl, 0, a.limit)
    res.update(summarize_noise(res))
    out = {"family": a.family, "axis": "visual", "tasks": a.tasks, "limit": a.limit, "max_len": a.max_len,
           "perturb_seed": R.PERTURB_SEED, "attacks": ev.ATTACKS, "probs": ev.PROBS, "targets": ev.TARGETS,
           **meta, "results": res, "samples": samples, "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

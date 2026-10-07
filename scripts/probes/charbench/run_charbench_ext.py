#!/usr/bin/env python3
"""CharBench + Strawberry (gen + cloze) on the external byte models, same YAMLs as the trio.

BLT = official bytelatent + xformers at batch size 1 (run_ext.build_blt_official); H-Net through
eval_suite's HNetHarness (run_ext.build_hnet_lm). Environments as for run_ext.py
(see scripts/probes/ext_ci/README.md); prompts are < 2 KB, well inside BLT's 4096-byte RoPE table.

  python run_charbench_ext.py --family blt_official --blt_weights <dir> --threshold 1.61 --out X.json
  python run_charbench_ext.py --family hnet --hnet_model hnet_1stage_XL --out X.json
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ext_ci"))
import run_ext  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["blt_official", "hnet"], required=True)
    ap.add_argument("--items", default="/mnt/ssd2/hyun2/AUNet/reports/charbench/items")
    ap.add_argument("--groups", nargs="*", default=["charbench_gen", "charbench_cloze",
                                                    "strawberry_gen", "strawberry_cloze"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--hnet_model", default="hnet_1stage_XL")
    ap.add_argument("--batch_size", type=int, default=8, help="H-Net only; BLT is always bs1")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run_ext._lm_eval_compat()
    if a.family == "blt_official":
        lm = run_ext.build_blt_official(a.blt_weights, a.max_len, a.threshold)
        meta = {"model": "facebook/blt-1b (official bytelatent, xformers)", "threshold": a.threshold,
                "batch_size": 1}
    else:
        lm = run_ext.build_hnet_lm(batch_size=a.batch_size, max_len=a.max_len, name=a.hnet_model)
        meta = {"model": f"cartesia-ai/{a.hnet_model}", "batch_size": a.batch_size}
    names = json.load(open(os.path.join(a.items, "task_lists.json")))
    tasks = [t for g in a.groups for t in names[g]]

    from lm_eval import simple_evaluate
    t0 = time.time()
    r = simple_evaluate(lm, tasks=tasks, num_fewshot=0,
                        task_manager=run_ext.safe_task_manager(os.path.join(a.items, "tasks")),
                        limit=a.limit, log_samples=True, bootstrap_iters=0, random_seed=0,
                        numpy_random_seed=run_ext.PERTURB_SEED, torch_random_seed=run_ext.PERTURB_SEED)
    out = {"family": a.family, **meta, "items": a.items, "limit": a.limit,
           "elapsed_s": round(time.time() - t0, 1), "results": r["results"], "samples": r["samples"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)
    print("WROTE", a.out, out["elapsed_s"], "s")


if __name__ == "__main__":
    main()

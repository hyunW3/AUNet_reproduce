#!/usr/bin/env python3
"""BLT-1B standard-benchmark accuracy, single seed, reproducible.

Official bytelatent BLT-1B (run_ext.build_blt_official: xformers, native 512 windows, threshold
1.335442066192627, batch size 1) scored with lm-eval on the standard multiple-choice tasks. One run
= one seed. Batch size is fixed at 1 because larger batches change scores (BLT_execution_guide.md
§2.5); at bs=1 the forward pass is bit-for-bit deterministic, so re-running with the same seed,
weights and environment reproduces every per-item result.

The seed sets lm-eval's fewshot / numpy / torch seeds (random_seed stays 0, as in run_ext.py).
At 0-shot nothing is sampled, so the seed does not change the score; with --num_fewshot > 0 it
selects the few-shot examples (mmlu_text always takes the first n dev items).

Output JSON: per-task headline metric (run_ext.METRIC), macro average, per-item correctness bits
(input of blt_bench_bootstrap.py) and a manifest (code commits, package versions, GPU, weight
hashes, command line) for reproduction.

  python blt_bench.py --out out/blt_bench/ds0_seed1234.json
  python blt_bench.py --num_fewshot 5 --seed 1234 --out out/blt_bench/ds5_seed1234.json
Environment: see run_blt_bench.sh (snu55 paths) and BLT_execution_guide.md §4.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ext  # noqa: E402

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq", "mmlu_text"]


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def manifest(a):
    import torch, lm_eval
    try:
        import xformers; xf = xformers.__version__
    except Exception:
        xf = None
    import bytelatent
    blt_repo = os.path.dirname(os.path.dirname(os.path.abspath(bytelatent.__file__)))
    return {
        "argv": sys.argv, "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "xformers": xf, "lm_eval": lm_eval.__version__, "gpu": torch.cuda.get_device_name(0),
        "aunet_commit": git_head(os.path.dirname(os.path.abspath(__file__))),
        "bytelatent_commit": git_head(blt_repo), "eval_suite": os.environ.get("EVAL_SUITE"),
        "tasks_dir": run_ext.TASKS_DIR,
        "weights_sha256": {f"{d}/model.safetensors": sha256(f"{a.blt_weights}/{d}/model.safetensors")
                           for d in ("blt_1b", "entropy")},
    }


def task_bits(samples, task):
    """Per-item 0/1 for the task's headline metric; mmlu_text pools its 57 subjects (size-weighted, as lm-eval)."""
    m = run_ext.METRIC[task]
    keys = [k for k in samples if k.startswith("mmlu_text_")] if task == "mmlu_text" else [task]
    return [x for k in sorted(keys) for x in samples[k][m]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (default: full test set)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--blt_weights", default="/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/blt_weights")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    run_ext._lm_eval_compat()
    lm = run_ext.build_blt_official(a.blt_weights, a.max_len, a.threshold)
    assert lm.batch_size == 1
    from lm_eval import simple_evaluate
    r = simple_evaluate(lm, tasks=a.tasks, num_fewshot=a.num_fewshot,
                        task_manager=run_ext.safe_task_manager(run_ext.TASKS_DIR), limit=a.limit,
                        log_samples=True, bootstrap_iters=0, fewshot_random_seed=a.seed, random_seed=0,
                        numpy_random_seed=a.seed, torch_random_seed=a.seed)
    samples = run_ext.compact_samples(r["samples"])
    bits = {t: task_bits(samples, t) for t in a.tasks}
    score = {t: 100 * sum(b) / len(b) for t, b in bits.items()}
    out = {"model": "facebook/blt-1b (official bytelatent, xformers)", "batch_size": 1, "threshold": a.threshold,
           "max_len": a.max_len, "num_fewshot": a.num_fewshot, "seed": a.seed, "limit": a.limit,
           "metric": {t: run_ext.METRIC[t] for t in a.tasks},
           "score": score, "n_items": {t: len(b) for t, b in bits.items()},
           "macro_avg": sum(score.values()) / len(score),
           "bits": bits, "lm_eval_results": r["results"], "samples": samples,
           "manifest": manifest(a), "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)

    print(f"\nBLT-1B  {a.num_fewshot}-shot  seed={a.seed}  limit={a.limit}")
    for t in a.tasks:
        print(f"  {t:14s} {run_ext.METRIC[t]:8s} {score[t]:6.2f}  (n={len(bits[t])})")
    print(f"  {'macro avg':14s} {'':8s} {out['macro_avg']:6.2f}")
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

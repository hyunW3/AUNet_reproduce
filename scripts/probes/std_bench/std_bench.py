#!/usr/bin/env python3
"""Standard-benchmark accuracy for any registered model, single seed, reproducible.

Every model (BLT-1B, H-Net, the matched Transformer / AU-Net / BPEByte, or any lingua checkpoint via
--ckpt) is scored by the same lm-eval call: same task YAMLs ($AUNET_LINGUA/eval_tasks for mmlu_text),
same seeds, same headline metric per task (run_ext.METRIC), single process / single GPU. Models and
their harnesses are in models.py; run it through run_std_bench.sh, which picks each family's
interpreter and environment.

The seed sets lm-eval's fewshot / numpy / torch seeds (random_seed stays 0, as in run_ext.py).
At 0-shot nothing is sampled, so the seed does not change the score; with --num_fewshot > 0 it
selects the few-shot examples (mmlu_text always takes the first n dev items).

Output JSON: per-task score, macro average, per-item correctness bits (input of
std_bench_bootstrap.py) and a manifest (code commits + dirty flags, package versions, GPU, weight
hashes, harness settings, command line) for reproduction.

  python std_bench.py --model aunet_1.3b --out reports/std_bench/aunet_1.3b/ds0_seed1234.json
  python std_bench.py --model my_run --family lingua_aunet --ckpt <consolidated dir> --out X.json
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
sys.path.insert(0, f"{models.REPO}/scripts/probes/ext_ci")
import run_ext  # noqa: E402

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq", "mmlu_text"]


def git_state(path):
    def g(*a):
        return subprocess.check_output(["git", "-C", path, *a], text=True, stderr=subprocess.DEVNULL)
    try:
        diff = g("diff", "HEAD")
        return {"path": path, "commit": g("rev-parse", "HEAD").strip(), "dirty": bool(diff),
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None}
    except Exception:
        return {"path": path, "commit": None}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def manifest(spec, weight_files, lm):
    import torch, lm_eval
    repos = {"aunet": models.REPO, "lingua": os.environ.get("AUNET_LINGUA", f"{models.AUNET}/lingua")}
    if spec["family"] == "blt":
        import bytelatent
        repos["bytelatent"] = os.path.dirname(os.path.dirname(os.path.abspath(bytelatent.__file__)))
    if spec["family"] == "hnet":
        repos["hnet"] = os.environ.get("HNET_REPO", "/mnt/ssd2/hyun2/hnet")
    vers = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "lm_eval": lm_eval.__version__}
    for mod in ("xformers", "transformers", "mamba_ssm", "flash_attn"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            pass
    return {"argv": sys.argv, "versions": vers, "gpu": torch.cuda.get_device_name(0),
            "repos": {k: git_state(v) for k, v in repos.items()},
            "eval_suite": os.environ.get("EVAL_SUITE"), "tasks_dir": run_ext.TASKS_DIR,
            "spec": spec, "harness": type(lm).__module__ + "." + type(lm).__name__,
            "harness_batch_size": getattr(lm, "batch_size", None),
            "weights_sha256": {p: sha256(p) for p in weight_files}}


def task_bits(samples, task):
    """Per-item 0/1 for the task's headline metric; mmlu_text pools its 57 subjects (size-weighted, as lm-eval)."""
    m = run_ext.METRIC[task]
    keys = sorted(k for k in samples if k.startswith("mmlu_text_")) if task == "mmlu_text" else [task]
    return [x for k in keys for x in samples[k][m]], [f"{k}:{d}" for k in keys for d in samples[k]["doc_id"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"registry name ({', '.join(models.MODELS)}) or a label for --ckpt")
    ap.add_argument("--ckpt", default=None, help="ad-hoc lingua consolidated checkpoint dir")
    ap.add_argument("--family", default=None, choices=models.FAMILIES)
    ap.add_argument("--max_tokens", type=int, default=None, help="lingua generator max_tokens override")
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (default: full test set)")
    ap.add_argument("--max_len", type=int, default=4096, help="BLT / H-Net context cap in bytes")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    spec = models.resolve(a.model, a.ckpt, a.family, a.max_tokens)
    run_ext._lm_eval_compat()
    lm, weight_files = models.build(spec, a.max_len)
    from lm_eval import simple_evaluate
    r = simple_evaluate(lm, tasks=a.tasks, num_fewshot=a.num_fewshot,
                        task_manager=run_ext.safe_task_manager(run_ext.TASKS_DIR), limit=a.limit,
                        log_samples=True, bootstrap_iters=0, fewshot_random_seed=a.seed, random_seed=0,
                        numpy_random_seed=a.seed, torch_random_seed=a.seed)
    samples = run_ext.compact_samples(r["samples"])
    bits, items = {}, {}
    for t in a.tasks:
        bits[t], items[t] = task_bits(samples, t)
    score = {t: 100 * sum(b) / len(b) for t, b in bits.items()}
    out = {"model": a.model, "family": spec["family"], "num_fewshot": a.num_fewshot, "seed": a.seed,
           "limit": a.limit, "metric": {t: run_ext.METRIC[t] for t in a.tasks},
           "score": score, "n_items": {t: len(b) for t, b in bits.items()},
           "macro_avg": sum(score.values()) / len(score),
           "bits": bits, "items": items, "lm_eval_results": r["results"],
           "manifest": manifest(spec, weight_files, lm), "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)

    print(f"\n{a.model}  {a.num_fewshot}-shot  seed={a.seed}  limit={a.limit}")
    for t in a.tasks:
        print(f"  {t:14s} {run_ext.METRIC[t]:8s} {score[t]:6.2f}  (n={len(bits[t])})")
    print(f"  {'macro avg':14s} {'':8s} {out['macro_avg']:6.2f}")
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

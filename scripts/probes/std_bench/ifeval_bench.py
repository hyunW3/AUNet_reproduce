#!/usr/bin/env python3
"""IFEval (google/IFEval, 541 prompts) for any registered model, with lm-eval's stock task config
(0-shot, greedy, until=[], max_gen_toks=1280; byte models scale it by GEN_BYTES_PER_TOKEN in their
EvalHarnessLM). Base models get the raw prompt, no chat template.

Writes per-prompt strict/loose bits, the generations, lm-eval's aggregate metrics, and 95% item-bootstrap
CIs of prompt-level accuracy. Run through run_ifeval.sh (environment + extra deps).

  python ifeval_bench.py --model bpebyte_1.3b --out reports/ifeval/bpebyte_1.3b/ifeval.json [--limit 20]
"""
import argparse, json, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models  # noqa: E402
sys.path.insert(0, f"{models.REPO}/scripts/probes/ext_ci")
import run_ext  # noqa: E402

KEYS = ["prompt_level_strict_acc", "prompt_level_loose_acc", "inst_level_strict_acc", "inst_level_loose_acc"]


def boot_ci(bits, B=10000, seed=0):
    rng = random.Random(seed); n = len(bits)
    means = sorted(sum(bits[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * means[int(0.025 * B)], 100 * means[int(0.975 * B) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--family", default=None, choices=models.FAMILIES)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--max_gen_toks", type=int, default=None, help="override the task's 1280 (byte models: x6 bytes)")
    ap.add_argument("--cut_bytes", type=int, default=None,
                    help="also score every response truncated to this many UTF-8 bytes (common budget across models)")
    ap.add_argument("--shard", default=None, help="i/n: score items i, i+n, ... of the 541 (merge with ifeval_merge.py)")
    ap.add_argument("--items", default=None, help="comma-separated dataset indices (debugging)")
    ap.add_argument("--resume_jsonl", default=None, help="one prompt per call, appended here; skip prompts already in it")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    spec = models.resolve(a.model, a.ckpt, a.family, a.max_tokens)
    run_ext._lm_eval_compat()
    if os.environ.get("IFEVAL_EAGER_MASK") and spec["family"] == "lingua_aunet":
        # The compiled create_block_mask (setup_prefilling, _compile=True) hits a CUDA illegal memory
        # access when the first trunk prefill of a process is short (e.g. IFEval #43: 37 patches,
        # KV 16384); warming up on a longer prompt avoids it. The eager path builds the same mask
        # densely (Q <= 3200 patches x 16384 here: fits) and does not crash.
        import apps.aunet.generate as _g
        _cbm = _g.create_block_mask
        _g.create_block_mask = lambda *x, **kw: _cbm(*x, **{**kw, "_compile": False})
    lm, _ = models.build(spec, a.max_len)
    if os.environ.get("IFEVAL_TRACE") or os.environ.get("IFEVAL_RESET"):
        # TRACE: log every generate() chunk, to locate a crashing prompt. RESET: drop torch.compile
        # caches before every chunk (BPEByte cached path hits an illegal memory access in the
        # compiled block-mask kernel when shapes vary across consecutive chunks).
        import hashlib, torch
        _gen = lm.generator.generate
        def _traced(prompts, _n=[0]):
            _n[0] += 1
            if os.environ.get("IFEVAL_RESET"):
                torch._dynamo.reset()
            print(f"[trace] chunk {_n[0]} rows={len(prompts)} bytes={[len(p.encode()) for p in prompts]} "
                  f"md5={[hashlib.md5(p.encode()).hexdigest()[:8] for p in prompts]}", file=sys.stderr, flush=True)
            return _gen(prompts)
        lm.generator.generate = _traced
    from lm_eval import simple_evaluate
    idx = ([int(i) for i in a.items.split(",")] if a.items else
           list(range(int(a.shard.split("/")[0]), 541, int(a.shard.split("/")[1]))) if a.shard else None)

    def run(samples, limit=None):
        r = simple_evaluate(lm, tasks=["ifeval"], num_fewshot=0, task_manager=run_ext.safe_task_manager(),
                            limit=limit, samples=samples, log_samples=True, bootstrap_iters=0,
                            gen_kwargs=f"max_gen_toks={a.max_gen_toks}" if a.max_gen_toks else None,
                            random_seed=0, numpy_random_seed=1234, torch_random_seed=1234)
        return r, [{"doc": s["doc"],
                    "response": s["filtered_resps"][0] if s.get("filtered_resps") else s["resps"][0][0],
                    **{k: s[k] for k in KEYS}} for s in r["samples"]["ifeval"]]

    if a.resume_jsonl:
        # One prompt per lm-eval call, each appended to the jsonl as it finishes. The BPEByte cached
        # path can hit a CUDA illegal memory access after a few consecutive generate() calls, which
        # kills the process; run_ifeval.sh restarts it and the finished prompts are skipped.
        done = set()
        if os.path.exists(a.resume_jsonl):
            done = {json.loads(l)["idx"] for l in open(a.resume_jsonl) if l.strip()}
        todo = [i for i in (idx if idx is not None else range(a.limit or 541)) if i not in done]
        print(f"[resume] {len(done)} done, {len(todo)} to go", file=sys.stderr, flush=True)
        for i in todo:
            _, its = run({"ifeval": [i]})
            with open(a.resume_jsonl, "a") as f:
                f.write(json.dumps({"idx": i, **its[0]}, default=str) + "\n")
        rows = [json.loads(l) for l in open(a.resume_jsonl) if l.strip()]
        want = set(idx if idx is not None else range(a.limit or 541))
        items = [{k: v for k, v in rw.items() if k != "idx"} for rw in sorted(rows, key=lambda x: x["idx"])
                 if rw["idx"] in want]
        r = {"results": None}
    else:
        r, items = run({"ifeval": idx} if idx is not None else None, None if idx is not None else a.limit)
    if a.cut_bytes:  # rescore with lm-eval's own checker on the truncated responses
        from lm_eval.tasks.ifeval.utils import process_results
        for it in items:
            it["response_full"] = it["response"]
            it["response"] = it["response"].encode()[:a.cut_bytes].decode("utf-8", errors="ignore")
            it.update(process_results(it["doc"], [it["response"]]))
    agg = {}
    for k in KEYS:
        v = [b for it in items for b in (it[k] if isinstance(it[k], list) else [it[k]])]
        agg[k] = 100 * sum(v) / len(v)
    ci = {}
    for k in ("prompt_level_strict_acc", "prompt_level_loose_acc"):
        ci[k] = boot_ci([int(it[k]) for it in items], a.B)
    for k in ("inst_level_strict_acc", "inst_level_loose_acc"):
        flat = [int(b) for it in items for b in it[k]]
        ci[k] = boot_ci(flat, a.B)  # instruction-level (ignores within-prompt clustering)
    resp_bytes = [len(it["response"].encode()) for it in items]
    out = {"model": a.model, "family": spec["family"], "limit": a.limit, "shard": a.shard, "n_prompts": len(items),
           "max_gen_toks": a.max_gen_toks, "cut_bytes": a.cut_bytes,
           "score": agg, "ci95": ci, "mean_response_bytes": sum(resp_bytes) / len(resp_bytes),
           "items": items, "lm_eval_results": r["results"], "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), default=str)
    print(f"\n{a.model}  IFEval  n={len(items)}  mean response {out['mean_response_bytes']:.0f} B")
    for k in KEYS:
        print(f"  {k:24s} {agg[k]:6.2f}  [{ci[k][0]:.2f}, {ci[k][1]:.2f}]")
    print("wrote", a.out, f"({out['seconds']}s)")


if __name__ == "__main__":
    main()

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


def _early_stop_generate(lm, family):
    """Swap in eval_suite's greedy loop with an early exit once every row has emitted a stop byte.

    The stock loops always run max_gen_toks steps (128 for Strawberry) and cut at the first `until`
    afterwards. Decoding is causal (BLT's entropy patcher too), so stopping when each row has produced
    its stop byte returns the same strings; it only skips the steps whose bytes are thrown away.
    Only single-byte `until` strings take the fast path; anything else runs the original loop."""
    import types
    import torch
    orig = lm._generate_batch

    def stop_ids(until):
        if family == "hnet":
            ids = [list(u.encode()) for u in until]
        else:
            ids = [lm.tokenizer.encode(u, add_bos=False, add_eos=False) for u in until]
        return [i[0] for i in ids] if all(len(i) == 1 for i in ids) else None

    @torch.inference_mode()
    def gen(self, prompts, until, max_gen_toks):
        sid = stop_ids(until)
        if sid is None:
            return orig(prompts, until, max_gen_toks)
        cap = self.max_len - max_gen_toks - 1
        if family == "hnet":
            from hnet_eval.harness import BOS, DEVICE
            ptoks = [([BOS] + self._enc(p))[-cap:] for p in prompts]
            fill = 0
        else:
            DEVICE = "cuda"
            ptoks = [self.tokenizer.encode(p, add_bos=True, add_eos=False)[-cap:] for p in prompts]
            fill = self.tokenizer.boe_id
        a128 = lambda n: ((n + 127) // 128) * 128
        a64 = lambda n: ((n + 63) // 64) * 64
        plen = torch.tensor([len(t) for t in ptoks], device=DEVICE)
        start, end, bs = int(plen.min()), int(plen.max()) + max_gen_toks, len(ptoks)
        toks = torch.full((bs, end), fill, dtype=torch.long, device=DEVICE)
        real = torch.zeros((bs, end), dtype=torch.bool, device=DEVICE)
        for i, t in enumerate(ptoks):
            toks[i, :len(t)] = torch.tensor(t, device=DEVICE); real[i, :len(t)] = True
        full = torch.ones((bs, end), dtype=torch.bool, device=DEVICE)
        stop = torch.tensor(sid, device=DEVICE)
        done = torch.zeros(bs, dtype=torch.bool, device=DEVICE)
        for curr in range(start, end):
            if family == "hnet":
                nxt = self.model(toks[:, :curr], mask=full[:, :curr]).logits[:, curr - 1].argmax(-1)
            else:   # verbatim blt_eval.harness.BLTHarness._generate_batch step
                Np = a128(curr); cur = toks[:, :curr]
                if Np > curr:
                    cur = torch.cat([cur, torch.full((bs, Np - curr), fill, dtype=torch.long, device=DEVICE)], dim=1)
                pl, _ = self.patcher.patch(cur, include_next_token=False, entropies=self._row_entropies(cur))
                P, Pp = pl.shape[1], a64(pl.shape[1])
                if Pp > P:
                    pl = torch.cat([pl, torch.zeros((bs, Pp - P), dtype=pl.dtype, device=DEVICE)], dim=1)
                nxt = self.model(cur, patch_lengths=pl)[:, curr - 1].argmax(-1)
            toks[:, curr] = torch.where(real[:, curr], toks[:, curr], nxt)
            done |= (curr >= plen) & torch.isin(toks[:, curr], stop)
            if bool(done.all()):
                break
        outs = []
        for i, t in enumerate(ptoks):
            row = toks[i, len(t):len(t) + max_gen_toks].tolist()
            s = (bytes(b for b in row if b < 254).decode("utf-8", "ignore") if family == "hnet"
                 else self.tokenizer.decode(row))
            for u in until:
                j = s.find(u)
                if j >= 0:
                    s = s[:j]
            outs.append(s)
        return outs

    lm._generate_batch = types.MethodType(gen, lm)


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
    ap.add_argument("--no_early_stop", action="store_true", help="keep eval_suite's fixed-length greedy loop")
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
    if not a.no_early_stop:
        _early_stop_generate(lm, "hnet" if a.family == "hnet" else "blt")
    meta["early_stop"] = not a.no_early_stop
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

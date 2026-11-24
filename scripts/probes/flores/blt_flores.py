#!/usr/bin/env python3
"""FLORES-200 X->en for BLT-1B, on the same prompts/protocol as the AU-Net trio.

Uses the transformers conversion `itazap/blt-1b-hf` (the bytelatent repo's own venv is not
readable from this account). BLT patches internally from input_ids on every forward, so generation re-runs the full forward
per step rather than using a KV cache.

Two decoding facts established by probing this checkpoint (see reports_afterAAAIsub/flores):
  * `model.generate()` (the KV-cache path) is BROKEN for BLT in transformers 5.14.1 -- it emits
    "thandendendend..." where the uncached loop emits a correct translation. Do not use it.
  * Batching rows of DIFFERENT tokenized length corrupts the output: `patch_lengths_from_entropies`
    sizes its patch axis from the row with the most boundaries and pads the others with degenerate
    patches, which misaligns cross-attention for the short rows. A batch of IDENTICAL-length rows
    reproduces batch-1 exactly; a mixed batch degenerates into repeated characters.
    Therefore generation runs at batch 1, and languages are sharded across GPUs instead.

Protocol is byte-identical to lingua/apps/{main,aunet}/configs/eval_flores_*_local.yaml:
5-shot, greedy, limit=200, stop at "\\n". Generation budget is given in BYTES: the AU-Net byte
models run max_gen_len = ceil(96 * 6 / 2) = 288 B, so BLT gets 288 B too.

  python blt_flores.py --split dev --theta 2.9219 --limit 200 --out out/blt_flores_dev.json
"""
import argparse, json, math, os, sys, time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blt_hf_sliding_window as _sw
from lm_eval import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.tasks import TaskManager

MODEL_ID = "itazap/blt-1b-hf"
LANGS = ["de", "fr", "es", "it", "nl", "ko"]


def build(theta, max_len):
    from transformers import AutoTokenizer, BltConfig, BltForCausalLM
    _sw.enable()   # restore the 512-byte sliding window the conversion drops
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BltConfig.from_pretrained(MODEL_ID)
    if not hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = cfg.global_config.num_hidden_layers
    released = float(cfg.patching_threshold)
    if theta is not None:
        cfg.patching_threshold = float(theta)
    cfg.max_position_embeddings = max(cfg.max_position_embeddings, max_len)
    model = BltForCausalLM.from_pretrained(MODEL_ID, config=cfg, dtype=torch.bfloat16).cuda().eval()
    print(f"[blt] patching_threshold: released={released:.6f} -> in use={float(cfg.patching_threshold):.6f}")
    return model, tok, released


class BltHF(LM):
    def __init__(self, model, tok, batch_size=16, max_len=4096, gen_bytes=288):
        super().__init__()
        self.model, self.tok = model, tok
        self.batch_size, self.max_len, self.gen_bytes = batch_size, max_len, gen_bytes
        self._device, self._rank, self._world_size = "cuda", 0, 1
        self.n_trunc = 0
        self.n_gen = 0

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def _generate_one(self, prompt, until, max_new):
        """Greedy, batch 1, no KV cache -- the only decoding path this checkpoint gets right."""
        tok, model = self.tok, self.model
        cap = self.max_len - max_new - 2
        ids = tok(prompt, add_special_tokens=True)["input_ids"][-cap:]
        n0 = len(ids)
        t = torch.tensor([ids], device="cuda")
        hit = False
        for step in range(max_new):
            nxt = model(input_ids=t).logits[:, -1].argmax(-1, keepdim=True)
            t = torch.cat([t, nxt], dim=1)
            if step % 8 == 7 or step == max_new - 1:
                s = tok.decode(t[0, n0:].tolist())
                if any(u in s for u in until):
                    hit = True
                    break
        s = tok.decode(t[0, n0:].tolist())
        for u in until:
            j = s.find(u)
            if j >= 0:
                s, hit = s[:j], True
        self.n_gen += 1
        if not hit:
            self.n_trunc += 1
        return s

    def generate_until(self, requests):
        groups = defaultdict(list)
        for i, req in enumerate(requests):
            gk = req.args[1] or {}
            u = gk.get("until", ["\n"]) or ["\n"]
            u = [u] if isinstance(u, str) else list(u)
            groups[tuple(u)].append(i)
        res = [None] * len(requests)
        t0 = time.time()
        n_done = 0
        for u, idxs in groups.items():
            for i in idxs:
                res[i] = self._generate_one(requests[i].args[0], list(u), self.gen_bytes)
                n_done += 1
                if n_done % 20 == 0 or n_done == len(requests):
                    el = time.time() - t0
                    print(f"  [gen] {n_done}/{len(requests)}  {el/60:.1f} min "
                          f"({el/n_done:.2f} s/sample)", flush=True)
        return res

    # ------------------------------------------------------------- loglikelihood
    @torch.inference_mode()
    def loglikelihood(self, requests):
        out = []
        for r in requests:
            ctx, cont = r.args
            c = self.tok(ctx, add_special_tokens=True)["input_ids"]
            f = self.tok(ctx + cont, add_special_tokens=True)["input_ids"][-self.max_len:]
            x = torch.tensor([f], device="cuda")
            lp = torch.log_softmax(self.model(input_ids=x).logits[0].float(), -1)
            tgt = torch.tensor(f[len(c):], device="cuda")
            sl = lp[len(c) - 1:len(f) - 1]
            out.append((float(sl.gather(-1, tgt.unsqueeze(-1)).sum()),
                        bool((sl.argmax(-1) == tgt).all())))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "devtest"], required=True)
    ap.add_argument("--theta", type=float, default=None, help="patching_threshold; default = released")
    ap.add_argument("--tasks_dir", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--gen_bytes", type=int, default=288)
    ap.add_argument("--langs", default=",".join(LANGS),
                    help="comma-separated subset, for sharding languages across GPUs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    pre = "flores_dev_" if args.split == "dev" else "flores_"
    tasks = [f"{pre}{l}-en" for l in langs]
    model, tok, released = build(args.theta, args.max_len)
    lm = BltHF(model, tok, args.batch_size, args.max_len, args.gen_bytes)

    tm = TaskManager(include_path=args.tasks_dir)
    res = simple_evaluate(lm, tasks=tasks, num_fewshot=args.num_fewshot, task_manager=tm,
                          limit=args.limit, log_samples=True, bootstrap_iters=1000,
                          gen_kwargs="max_gen_toks=%d" % args.gen_bytes)
    r = res["results"]
    per = {l: {"bleu": r[f"{pre}{l}-en"]["bleu,none"],
               "bleu_stderr": r[f"{pre}{l}-en"]["bleu_stderr,none"],
               "chrf": r[f"{pre}{l}-en"]["chrf,none"]} for l in langs}
    b = [per[l]["bleu"] for l in langs]
    c = [per[l]["chrf"] for l in langs]
    bl = [per[l]["bleu"] for l in langs if l != "ko"]
    cl = [per[l]["chrf"] for l in langs if l != "ko"]
    row = {"model": "BLT-1B", "family": "blt-entropy", "source": MODEL_ID, "split": args.split,
           "n": args.limit, "num_fewshot": args.num_fewshot, "decoding": "greedy",
           "gen_bytes": args.gen_bytes, "released_threshold": released,
           "patching_threshold": args.theta if args.theta is not None else released,
           "truncated_generations": f"{lm.n_trunc}/{lm.n_gen}",
           "langs": langs, "per_lang": per,
           "avg_bleu": round(sum(b) / len(b), 3), "avg_chrf": round(sum(c) / len(c), 3),
           "avg_latin_bleu": round(sum(bl) / len(bl), 3) if bl else None,
           "avg_latin_chrf": round(sum(cl) / len(cl), 3) if cl else None}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    sp = args.out.replace(".json", "_samples.json")
    json.dump({k: v for k, v in res.get("samples", {}).items()}, open(sp, "w"), default=str)
    print(json.dumps(row, indent=2))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

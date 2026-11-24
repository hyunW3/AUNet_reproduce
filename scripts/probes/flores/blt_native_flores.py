#!/usr/bin/env python3
"""FLORES-200 X->en for BLT-1B through Meta's NATIVE bytelatent code path.

Cross-check for scripts/probes/flores/blt_flores.py, which drives the `itazap/blt-1b-hf`
transformers conversion. Same tasks, same 5-shot prompts, same greedy decoding, same 288-byte
budget, so the two numbers are directly comparable and any gap is attributable to the conversion.

Three things this path needs that the conversion does not:
  * bytelatent's `setup_torch_distributed` dies on `mp.Manager()` here, so a 1-rank process
    group is initialised directly and `load_consolidated_model_and_tokenizer` called by hand.
  * byte length must be padded to a multiple of 128 and the patch axis to a multiple of 64,
    or BLT's cross-attention over-indexes (same constraint eval_suite/blt_eval documents).
  * the entropy model must run its ORIGINAL xformers attention. This checkout ships a local
    workaround -- `bytelatent/entropy_model.py` hardcodes `attn_impl="sdpa"` with the comment
    "was xformers; sdpa+BLT_SUPPRESS -> full-causal (exact for seqlen<=512)" -- because the
    venv's xformers had no CUDA kernels. With `attn_bias_type="local_block_causal"` that path
    faults (CUDA illegal memory access) a couple of decode steps into a ~1.6 kB prompt, and
    would be numerically wrong above 512 bytes anyway, which every 5-shot FLORES prompt is.
    xformers 0.0.29.post2 (matching torch 2.6) restores the kernels, and `--attn_impl xformers`
    monkeypatches `load_entropy_model` back to the upstream setting WITHOUT editing the shared
    checkout. `--attn_impl sdpa` reproduces the old broken behaviour for comparison.

  BLT_SUPPRESS_ATTN_ERROR=1 <blt-venv>/bin/python blt_native_flores.py --split dev \
      --tasks_dir .../lingua/eval_tasks --langs de,fr --limit 200 --out out/native.json
"""
import argparse, json, os, sys, time
from collections import defaultdict

import torch

BLT_REPO = "/mnt/ssd2/hyun2/lzspacebyte/blt"
sys.path.insert(0, BLT_REPO)
LANGS = ["de", "fr", "es", "it", "nl", "ko"]

a128 = lambda n: ((n + 127) // 128) * 128
a64 = lambda n: ((n + 63) // 64) * 64


def _patch_entropy_attn(attn_impl):
    """Rebind load_entropy_model with the upstream attn_impl, leaving the checkout untouched.

    patcher.py does `from bytelatent.entropy_model import load_entropy_model` at import time,
    so the name has to be replaced on the patcher module, not on entropy_model.
    """
    import json as _json
    import bytelatent.data.patcher as _patcher
    from bytelatent.transformer import LMTransformer, LMTransformerArgs  # noqa: F401

    def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
        with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
            reloaded = _json.loads(fr.read())
        torch.set_default_dtype(torch.bfloat16)
        mp = reloaded["entropy_model"]
        args = LMTransformerArgs(
            dim=mp["dim"], n_layers=mp["n_layers"], n_heads=mp["n_heads"],
            max_seqlen=mp["max_seqlen"], ffn_dim_multiplier=mp["ffn_dim_multiplier"],
            vocab_size=mp["vocab_size"],
            attn_bias_type="local_block_causal", attn_impl=attn_impl, sliding_window=512,
        )
        m = LMTransformer(args)
        m.load_state_dict(torch.load(state_dict_path, map_location=device)["model"], strict=False)
        m.to(device).eval()
        for prm in m.parameters():
            prm.requires_grad = False
        return m, args

    _patcher.load_entropy_model = load_entropy_model
    print(f"[native] entropy model attn_impl -> {attn_impl}")


def build(ckpt_dir, entropy_dir, theta, max_patch_length, port, attn_impl="xformers"):
    for k, v in [("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(port)),
                 ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
        os.environ.setdefault(k, v)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from bytelatent.generate import load_consolidated_model_and_tokenizer
    if attn_impl:
        _patch_entropy_attn(attn_impl)
    model, tok, cfg = load_consolidated_model_and_tokenizer(ckpt_dir)
    pa = cfg.data.patcher_args.model_copy(deep=True)
    pa.realtime_patching = True
    pa.entropy_model_checkpoint_dir = entropy_dir
    released = float(pa.threshold)
    if theta is not None:
        pa.threshold = float(theta)
    pa.max_patch_length = max_patch_length
    patcher = pa.build()
    print(f"[native] threshold: released={released:.6f} -> in use={float(pa.threshold):.6f} "
          f"| max_patch_length={max_patch_length}")
    return model, tok, patcher, released


from lm_eval import simple_evaluate                       # noqa: E402
from lm_eval.api.model import LM                          # noqa: E402
from lm_eval.tasks import TaskManager                     # noqa: E402


class BltNative(LM):
    def __init__(self, model, tok, patcher, max_len=4096, gen_bytes=288):
        super().__init__()
        self.model, self.tok, self.patcher = model, tok, patcher
        self.max_len, self.gen_bytes, self.batch_size = max_len, gen_bytes, 1
        self._device, self._rank, self._world_size = "cuda", 0, 1
        self.n_trunc = self.n_gen = 0

    @torch.inference_mode()
    def _generate_one(self, prompt, until, max_new):
        tok, model, patcher = self.tok, self.model, self.patcher
        cap = self.max_len - max_new - 2
        ids = tok.encode(prompt, add_bos=True, add_eos=False)[-cap:]
        n0, cur = len(ids), list(ids)
        boe = tok.boe_id
        hit = False
        for step in range(max_new):
            L = len(cur)
            Np = a128(L)
            x = torch.tensor([cur + [boe] * (Np - L)], device="cuda")
            pl, _ = patcher.patch(x, include_next_token=False)
            P = pl.shape[1]
            Pp = a64(P)
            if Pp > P:
                pl = torch.cat([pl, torch.zeros((1, Pp - P), dtype=pl.dtype, device="cuda")], 1)
            cur.append(int(model(x, patch_lengths=pl)[0, L - 1].argmax(-1).item()))
            if step % 8 == 7 or step == max_new - 1:
                s = tok.decode(cur[n0:])
                if any(u in s for u in until):
                    hit = True
                    break
        s = tok.decode(cur[n0:])
        for u in until:
            j = s.find(u)
            if j >= 0:
                s, hit = s[:j], True
        self.n_gen += 1
        self.n_trunc += (not hit)
        return s

    def generate_until(self, requests):
        groups = defaultdict(list)
        for i, req in enumerate(requests):
            gk = req.args[1] or {}
            u = gk.get("until", ["\n"]) or ["\n"]
            u = [u] if isinstance(u, str) else list(u)
            groups[tuple(u)].append(i)
        res = [None] * len(requests)
        t0, n = time.time(), 0
        for u, idxs in groups.items():
            for i in idxs:
                res[i] = self._generate_one(requests[i].args[0], list(u), self.gen_bytes)
                n += 1
                if n % 20 == 0 or n == len(requests):
                    el = time.time() - t0
                    print(f"  [gen] {n}/{len(requests)}  {el/60:.1f} min ({el/n:.2f} s/sample)", flush=True)
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=f"{BLT_REPO}/hf-weights/blt_1b")
    ap.add_argument("--entropy_ckpt_dir", default=f"{BLT_REPO}/hf-weights/entropy_model")
    ap.add_argument("--split", choices=["dev", "devtest"], default="dev")
    ap.add_argument("--tasks_dir", required=True)
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--max_patch_length", type=int, default=None)
    ap.add_argument("--attn_impl", default="xformers", choices=["xformers", "sdpa"],
                    help="entropy-model attention; xformers is upstream, sdpa is this "
                         "checkout's <=512-byte-only workaround")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--gen_bytes", type=int, default=288)
    ap.add_argument("--port", type=int, default=29600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    pre = "flores_dev_" if args.split == "dev" else "flores_"
    tasks = [f"{pre}{l}-en" for l in langs]

    model, tok, patcher, released = build(args.ckpt_dir, args.entropy_ckpt_dir, args.theta,
                                          args.max_patch_length, args.port, args.attn_impl)
    lm = BltNative(model, tok, patcher, args.max_len, args.gen_bytes)
    tm = TaskManager(include_path=args.tasks_dir)
    res = simple_evaluate(lm, tasks=tasks, num_fewshot=args.num_fewshot, task_manager=tm,
                          limit=args.limit, log_samples=True, bootstrap_iters=1000,
                          gen_kwargs="max_gen_toks=%d" % args.gen_bytes)
    r = res["results"]
    per = {l: {"bleu": r[f"{pre}{l}-en"]["bleu,none"],
               "bleu_stderr": r[f"{pre}{l}-en"]["bleu_stderr,none"],
               "chrf": r[f"{pre}{l}-en"]["chrf,none"]} for l in langs}
    b = [per[l]["bleu"] for l in langs]; c = [per[l]["chrf"] for l in langs]
    row = {"model": "BLT-1B (native bytelatent)", "family": "blt-entropy", "source": args.ckpt_dir,
           "split": args.split, "n": args.limit, "num_fewshot": args.num_fewshot,
           "decoding": "greedy", "gen_bytes": args.gen_bytes,
           "released_threshold": released,
           "patching_threshold": args.theta if args.theta is not None else released,
           "max_patch_length": args.max_patch_length, "entropy_attn_impl": args.attn_impl,
           "truncated_generations": f"{lm.n_trunc}/{lm.n_gen}", "langs": langs, "per_lang": per,
           "avg_bleu": round(sum(b)/len(b), 3), "avg_chrf": round(sum(c)/len(c), 3)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    json.dump(res.get("samples", {}), open(args.out.replace(".json", "_samples.json"), "w"), default=str)
    print(json.dumps(row, indent=2)); print("WROTE", args.out)


if __name__ == "__main__":
    main()

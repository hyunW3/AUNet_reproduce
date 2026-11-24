#!/usr/bin/env python3
"""Re-measure BLT-1B's SCORING axes with the entropy model's upstream xformers attention.

Every BLT number in this project was produced after bytelatent/entropy_model.py was edited to
`attn_impl="sdpa"` (see reports_afterAAAIsub/blt_threshold_ablation.md). That fallback does not
implement the specified `local_block_causal` + `sliding_window=512` bias, so BLT ran at
~1.15 bytes/patch instead of its ~4.07. `realtime_patching=True` in this harness, so patching is
on the critical path for every scored sequence -- the downstream / robustness / retrieval / BPB
numbers were all taken at the wrong operating point.

This reruns exactly those axes through eval_suite's own BLTHarness and common.axes, changing one
thing: the entropy model's attention. Same checkpoints, same tasks, same prompts, same batch size.

  --attn_impl xformers   the upstream setting (default)
  --attn_impl sdpa       reproduces the numbers currently in the paper

bytelatent's setup_torch_distributed dies on mp.Manager() here, so a 1-rank group is initialised
directly -- the same bypass the other scripts in this directory use.
"""
import argparse, json, os, sys, time

import torch

BLT_REPO = "/mnt/ssd2/hyun2/lzspacebyte/blt"
SUITE = "/mnt/ssd2/hyun2/eval_suite"
sys.path.insert(0, SUITE)
sys.path.insert(0, BLT_REPO)


def patch_entropy_attn(attn_impl):
    import bytelatent.data.patcher as _patcher
    from bytelatent.transformer import LMTransformer, LMTransformerArgs

    def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
        with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
            mp = json.load(fr)["entropy_model"]
        torch.set_default_dtype(torch.bfloat16)
        args = LMTransformerArgs(
            dim=mp["dim"], n_layers=mp["n_layers"], n_heads=mp["n_heads"],
            max_seqlen=mp["max_seqlen"], ffn_dim_multiplier=mp["ffn_dim_multiplier"],
            vocab_size=mp["vocab_size"],
            attn_bias_type="local_block_causal", attn_impl=attn_impl, sliding_window=512,
        )
        m = LMTransformer(args)
        m.load_state_dict(torch.load(state_dict_path, map_location=device)["model"], strict=False)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m, args

    _patcher.load_entropy_model = load_entropy_model
    print(f"[blt] entropy attn_impl -> {attn_impl}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=f"{BLT_REPO}/hf-weights/blt_1b")
    ap.add_argument("--entropy_ckpt_dir", default=f"{BLT_REPO}/hf-weights/entropy_model")
    ap.add_argument("--attn_impl", default="xformers", choices=["xformers", "sdpa"])
    ap.add_argument("--axis", required=True,
                    help="downstream | noise | typo | despace | formatted_string | retrieval | bpb")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--port", type=int, default=29620)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for k, v in [("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(args.port)),
                 ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
        os.environ.setdefault(k, v)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)

    patch_entropy_attn(args.attn_impl)

    import paths
    from common import axes
    from blt_eval.harness import BLTHarness
    from bytelatent.generate import load_consolidated_model_and_tokenizer

    model, tok, cfg = load_consolidated_model_and_tokenizer(args.ckpt_dir)
    pa = cfg.data.patcher_args.model_copy(deep=True)
    pa.realtime_patching = True
    pa.entropy_model_checkpoint_dir = args.entropy_ckpt_dir
    patcher = pa.build()
    lm = BLTHarness(model, tok, patcher, args.batch_size, args.max_len)

    # BLT ships its own top-level `apps/` package and `blt_eval.harness` re-inserts BLT_REPO at
    # sys.path[0] when imported, so lingua's `apps.aunet.eval_*` (which the noise/typo/despace
    # axes import) gets shadowed. Undo it here, after every bytelatent import is done: drop the
    # cached name, then put lingua ahead of the BLT repo.
    for _m in [m for m in sys.modules if m == "apps" or m.startswith("apps.")]:
        del sys.modules[_m]
    while paths.BLT_REPO in sys.path:
        sys.path.remove(paths.BLT_REPO)
    sys.path.insert(0, paths.LINGUA)

    tasks = args.tasks or paths.TASKS_6
    a, t0 = args.axis, time.time()
    lim = int(args.limit) if args.limit else None
    if a == "downstream":
        ip = paths.MMLU_TASKS if "mmlu_text" in tasks else None
        res = axes.downstream(lm, tasks, args.num_fewshot, lim, ip)
    elif a == "noise":
        res = axes.noise(lm, lim or 500)
    elif a == "typo":
        res = axes.typo(lm, lim or 500)
    elif a == "despace":
        res = axes.despace(lm, lim or 500)
    elif a == "formatted_string":
        res = axes.formatted_string(lm)
    elif a == "retrieval":
        res = axes.retrieval(lm)
    elif a == "bpb":
        res = axes.bpb(lm)
    else:
        raise SystemExit(f"unknown axis {a!r}")

    row = {"model": "blt-1b (native bytelatent)", "axis": a, "entropy_attn_impl": args.attn_impl,
           "threshold": float(pa.threshold), "num_fewshot": args.num_fewshot, "limit": lim,
           "elapsed_s": round(time.time() - t0, 1), **res}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in row.items() if k != "raw"}, indent=2, default=str))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

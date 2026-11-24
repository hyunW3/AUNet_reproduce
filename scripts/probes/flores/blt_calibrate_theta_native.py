#!/usr/bin/env python3
"""Calibrate BLT-1B's entropy patching threshold on the NATIVE bytelatent path.

Companion to blt_calibrate_theta.py, which fits the same target through the
`itazap/blt-1b-hf` conversion. The two disagree by ~4x in realized bytes/patch because the
conversion's patcher builds an ordinary causal mask while the real entropy model runs
`attn_bias_type="local_block_causal"` with `sliding_window=512`. This script is the
authoritative one; see reports_afterAAAIsub/flores/summary.md ("CORRECTION").

Rule (verified against find_entropy_patch_start_ids, monotonicity=False):
    starts = {0, 1} u {t >= 1 : H[t] > theta}
with the entropy vector's first element (BOS) dropped before thresholding, and -- matching
how eval_suite's harness actually calls patch(include_next_token=False) -- the last element
of the mask dropped too. n_patches is monotone decreasing in theta, so bisection is exact.

Must run in the BLT venv, and with attn_impl=xformers: this checkout hardcodes "sdpa" in
bytelatent/entropy_model.py, which mis-patches above 512 bytes (the windows here are 3840 B).

  $BLT_PY blt_calibrate_theta_native.py --target 4.53 --out .../blt_theta_native.json
"""
import argparse, json, math, os, sys

import torch

ROOT = "/mnt/ssd2/hyun2/AUNet"
BLT_REPO = os.environ.get("BLT_REPO", "/mnt/ssd2/hyun2/lzspacebyte/blt")
sys.path.insert(0, BLT_REPO)
CKPT = f"{BLT_REPO}/hf-weights/blt_1b"
ENTROPY_CKPT = f"{BLT_REPO}/hf-weights/entropy_model"


def patch_entropy_attn(attn_impl):
    import json as _json
    import bytelatent.data.patcher as _patcher
    from bytelatent.transformer import LMTransformer, LMTransformerArgs

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
    print(f"[native] entropy attn_impl -> {attn_impl}", flush=True)


@torch.inference_mode()
def collect(tok, entropy_model, windows, max_len):
    """Per-window entropy vectors, already trimmed to what the rule thresholds."""
    from bytelatent.data.patcher import calculate_entropies
    ents, total = [], 0
    for i, w in enumerate(windows):
        ids = tok.encode(w["text"], add_bos=True, add_eos=False)[:max_len]
        x = torch.tensor([ids], device="cuda")
        e = calculate_entropies(x, entropy_model, patching_batch_size=1, device="cuda")
        if isinstance(e, tuple):
            e = e[0]
        ents.append(e[0, 1:-1].float().cpu())   # drop BOS, and the include_next_token=False tail
        total += len(ids)
        if (i + 1) % 40 == 0:
            print(f"  ..{i+1}/{len(windows)}", flush=True)
    return ents, total


def bpp(ents, total, theta):
    n = sum(2 + int((e > theta).sum()) for e in ents)
    return total / n, n


def bisect(ents, total, target, lo=0.0, hi=20.0, iters=60):
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if bpp(ents, total, mid)[0] < target else (lo, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=4.53)
    ap.add_argument("--windows", default=f"{ROOT}/reports/bpb_windows.jsonl")
    ap.add_argument("--n_windows", type=int, default=320)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--attn_impl", default="xformers", choices=["xformers", "sdpa"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    patch_entropy_attn(args.attn_impl)
    # Read the checkpoint's TrainArgs directly instead of load_consolidated_model_and_tokenizer:
    # that helper unconditionally moves the 4.6B trunk to the GPU, and calibration never runs the
    # trunk -- only the ~100M entropy model. Skipping it keeps this to well under 1 GB so it can
    # share a box with a lingua eval.
    from bytelatent.args import TrainArgs
    train_cfg = TrainArgs.model_validate_json(open(os.path.join(CKPT, "params.json")).read())
    tok = train_cfg.data.tokenizer_args.build()
    pa = train_cfg.data.patcher_args.model_copy(deep=True)
    pa.realtime_patching = True
    pa.entropy_model_checkpoint_dir = ENTROPY_CKPT
    patcher = pa.build()
    released = float(pa.threshold)
    print(f"released threshold = {released:.6f}", flush=True)

    windows = [json.loads(l) for l in open(args.windows)][:args.n_windows]
    ents, total = collect(tok, patcher.entropy_model, windows, args.max_len)

    bpp_rel, np_rel = bpp(ents, total, released)
    theta = bisect(ents, total, args.target)
    bpp_cal, np_cal = bpp(ents, total, theta)
    print(f"released -> {bpp_rel:.4f} B/patch;  calibrated theta={theta:.6f} -> {bpp_cal:.4f}",
          flush=True)

    # The patcher thresholds bf16 entropies, so the target is generally not exactly reachable.
    step = 2.0 ** (math.floor(math.log2(abs(theta))) - 7) if theta else 0.0
    bracket = {"step": step,
               "below": {"theta": round(theta - step, 6), "bytes_per_patch": round(bpp(ents, total, theta - step)[0], 4)},
               "at":    {"theta": round(theta, 6), "bytes_per_patch": round(bpp_cal, 4)},
               "above": {"theta": round(theta + step, 6), "bytes_per_patch": round(bpp(ents, total, theta + step)[0], 4)}}
    allv = torch.cat(ents)
    row = {"model": "blt-1b (facebook/blt-1b, native bytelatent)",
           "attn_impl": args.attn_impl, "corpus": os.path.basename(args.windows),
           "n_windows": len(windows), "total_bytes": total,
           "target_bytes_per_patch": args.target,
           "released_threshold": released, "released_bytes_per_patch": round(bpp_rel, 4),
           "released_n_patches": np_rel,
           "calibrated_threshold": round(theta, 6), "calibrated_bytes_per_patch": round(bpp_cal, 4),
           "calibrated_n_patches": np_cal,
           "bf16_grid_bracket": bracket,
           "entropy_quantiles": {f"p{p}": round(float(torch.quantile(allv, p / 100)), 4)
                                 for p in (10, 25, 50, 75, 90, 99)},
           "rule": "patch start iff H[t] > theta, plus tokens 0 and 1; include_next_token=False"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

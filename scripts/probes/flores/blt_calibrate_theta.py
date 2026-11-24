#!/usr/bin/env python3
"""Calibrate BLT-1B's entropy patching threshold to a target bytes/patch on DCLM.

BLT's released threshold (1.335442) was fitted on BLT's own training mix; on the DCLM
held-out windows this repo uses everywhere else it produces a very different compression
rate, which makes "BLT vs the matched trio" a comparison at unmatched sequence-compression.
This script fits theta so BLT pools the same number of bytes per patch as Llama-3 tiktoken
puts in a token on the same data (4.5376 B/token, the repo's BPT constant).

BLT's rule (transformers BltPatcher.patch_lengths_from_entropies, monotonicity=False):
    patch starts = {token 0, token 1} U {t >= 1 : H[t] > theta}
so with L bytes and E = the entropy vector excluding BOS,
    n_patches(theta) = 2 + #{H in E[1:] : H > theta}      and   B/patch = L / n_patches.
n_patches is monotone decreasing in theta, so a bisection on theta is exact.

  python blt_calibrate_theta.py --target 4.53 --out reports_afterAAAIsub/flores/blt_theta.json
"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blt_hf_sliding_window as _sw

ROOT = "/mnt/ssd2/hyun2/AUNet"
MODEL_ID = "itazap/blt-1b-hf"


def load_patcher():
    from transformers import AutoTokenizer, BltConfig, BltForCausalLM
    _sw.enable()   # restore the 512-byte sliding window the conversion drops
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = BltConfig.from_pretrained(MODEL_ID)
    if not hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = cfg.global_config.num_hidden_layers
    # Load on CPU and move ONLY the entropy patcher to the GPU: calibration never runs the
    # 4.6B trunk, and the box's A5000s are usually busy with a lingua eval.
    model = BltForCausalLM.from_pretrained(MODEL_ID, config=cfg, dtype=torch.bfloat16).eval()
    model.model.patcher.cuda()
    return model, tok, cfg


@torch.inference_mode()
def collect_entropies(model, tok, windows, max_len):
    """Per-byte entropies (excluding BOS) for each window, in BOTH dtypes.

    The shipped patcher computes `Categorical(logits=logits).entropy()` on bf16 logits, so the
    entropy the model actually thresholds lives on the coarse bf16 grid (~0.016 spacing near 2.9).
    Enough probability mass piles onto single grid points that bytes/patch jumps over 4.53 between
    two adjacent representable thresholds -- so we also recompute entropy in fp32 from the same
    logits and calibrate there, then report what that theta realizes under the bf16 path.
    """
    ent_bf16, ent_fp32, total = [], [], 0
    patcher = model.model.patcher
    for i, w in enumerate(windows):
        ids = tok(w["text"], add_special_tokens=True)["input_ids"][:max_len]
        x = torch.tensor([ids], device="cuda")
        pe, _pl, lg = patcher(x, patch_size=None, threshold=1e9,
                              max_patch_length=None, patching_batch_size=1, device=x.device)
        # patch_lengths_from_entropies drops entropies[:, 0] (BOS) before thresholding
        ent_bf16.append(pe[0, 1:].float().cpu())
        ent_fp32.append(torch.distributions.Categorical(logits=lg[0, 1:].float()).entropy().cpu())
        total += len(ids)
        if (i + 1) % 40 == 0:
            print(f"  ..{i+1}/{len(windows)} windows", flush=True)
    return ent_bf16, ent_fp32, total


def bytes_per_patch(ents, total_bytes, theta):
    n_patches = sum(2 + int((e > theta).sum()) for e in ents)
    return total_bytes / n_patches, n_patches


def bisect(ents, total_bytes, target, lo=0.0, hi=20.0, iters=60):
    # B/patch increases with theta (fewer boundaries) -> plain bisection
    for _ in range(iters):
        mid = (lo + hi) / 2
        bpp, _ = bytes_per_patch(ents, total_bytes, mid)
        if bpp < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=4.53)
    ap.add_argument("--windows", default=f"{ROOT}/reports/bpb_windows.jsonl")
    ap.add_argument("--n_windows", type=int, default=320)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    windows = [json.loads(l) for l in open(args.windows)][:args.n_windows]
    model, tok, cfg = load_patcher()
    released = float(cfg.patching_threshold)
    print(f"released patching_threshold = {released:.6f}")

    ent_bf16, ent_fp32, total = collect_entropies(model, tok, windows, args.max_len)

    bpp_released, np_released = bytes_per_patch(ent_bf16, total, released)
    print(f"released theta -> {bpp_released:.4f} B/patch ({np_released} patches / {total} bytes)")

    # PRIMARY: calibrate on the bf16 entropies the patcher actually thresholds at inference.
    theta = bisect(ent_bf16, total, args.target)
    bpp, npat = bytes_per_patch(ent_bf16, total, theta)
    print(f"CALIBRATED theta = {theta:.6f} -> {bpp:.4f} B/patch ({npat} patches)")

    # bf16 has ~0.016 spacing near 2.9 and entropy mass piles onto single grid points, so the
    # target is generally not exactly reachable. Record the bracket the target falls in.
    def bf16_step(v):
        return 2.0 ** (math.floor(math.log2(abs(v))) - 7) if v else 0.0
    step = bf16_step(theta)
    bracket = {"step": step,
               "below": {"theta": round(theta - step, 6),
                         "bytes_per_patch": round(bytes_per_patch(ent_bf16, total, theta - step)[0], 4)},
               "at":    {"theta": round(theta, 6), "bytes_per_patch": round(bpp, 4)},
               "above": {"theta": round(theta + step, 6),
                         "bytes_per_patch": round(bytes_per_patch(ent_bf16, total, theta + step)[0], 4)}}
    print(f"  bf16 grid bracket: {json.dumps(bracket)}")

    # Diagnostic only: the same fit on fp32-recomputed entropies, which is continuous enough to
    # land on the target exactly. Reported to show the gap is a dtype-grid artifact, not a bad fit.
    theta32 = bisect(ent_fp32, total, args.target)
    bpp32, _ = bytes_per_patch(ent_fp32, total, theta32)

    allv = torch.cat(ent_bf16)
    q = {f"p{p}": round(float(torch.quantile(allv, p / 100)), 4) for p in (10, 25, 50, 75, 90, 99)}
    row = {"model": "blt-1b (itazap/blt-1b-hf)", "corpus": os.path.basename(args.windows),
           "n_windows": len(windows), "total_bytes": total, "target_bytes_per_patch": args.target,
           "entropy_dtype": "bf16 (as the shipped patcher runs it)",
           "released_threshold": released, "released_bytes_per_patch": round(bpp_released, 4),
           "calibrated_threshold": round(theta, 6), "calibrated_bytes_per_patch": round(bpp, 4),
           "bf16_grid_bracket": bracket,
           "fp32_diagnostic": {"theta": round(theta32, 6), "bytes_per_patch": round(bpp32, 4)},
           "entropy_quantiles": q, "rule": "patch start iff H[t] > theta, plus tokens 0 and 1"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(row, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

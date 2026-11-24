#!/usr/bin/env python3
"""Forward-latency benchmark for BLT-1B (ms/forward), companion to AUNet's
bench_model_forward / bench_llama_forward. BLT's per-step forward = entropy-model
patching + the ByteLatentTransformer forward, so we time both and report
base / entropy / total, plus ms per KB of bytes processed (the cross-model axis).

bsz x seq bytes/step is chosen to match the byte models' 32 KB/step: default
bsz=8 x seq=4096 = 32 KB (BLT's 4096-byte context halves the seq vs the 8192-byte
byte models, so we double the batch to keep bytes/step equal and comparable).
"""
import argparse, json, os, statistics as st
import torch
from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from run_blt_eval import build


def _time(fn, n_iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))   # ms
    return st.mean(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--entropy_ckpt_dir", required=True)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--n_iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real_text", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override entropy patch threshold (default 1.335 gives ~1.17x on DCLM; "
                         "~2.67 gives BLT design patch_size 4.5)")
    args = ap.parse_args()

    da = DistributedArgs(); da.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(da)
    model, tok, patcher = build(args.ckpt_dir, args.entropy_ckpt_dir, args.seq)
    model.eval()
    if args.threshold is not None:                 # measure at BLT's design compression
        patcher.threshold = args.threshold
        if getattr(patcher, "patcher_args", None) is not None:
            patcher.patcher_args.threshold = args.threshold

    # byte tokens (offset 4..259 for the 256 byte values). REAL DCLM text if --real_text, so the
    # entropy patcher compresses realistically (random bytes -> max entropy -> no compression).
    if args.real_text:
        wins = [json.loads(l)["text"] for l in open(args.real_text)]
        big = ("\n".join(wins)).encode("utf-8")
        rows = []
        for i in range(args.bsz):
            b = list(big[i*args.seq:(i+1)*args.seq])
            if len(b) < args.seq: b = (b*(args.seq//max(1,len(b))+1))[:args.seq]
            rows.append([x + 4 for x in b])
        toks = torch.tensor(rows, dtype=torch.long, device="cuda").long()
    else:
        g = torch.Generator(device="cuda").manual_seed(0)
        toks = (torch.randint(0, 256, (args.bsz, args.seq), generator=g, device="cuda") + 4).long()

    @torch.inference_mode()
    def patch_fn():
        patcher.patch(toks, include_next_token=False)

    pl, _ = patcher.patch(toks, include_next_token=False)  # for the base-forward timing
    _npatch = (pl > 0).sum(1).float().mean().item() if pl.dim() == 2 else float(pl.shape[-1])
    _compression = args.seq / max(1.0, _npatch)
    print(f"[BLT] {'REAL' if args.real_text else 'random'} text: mean patches {_npatch:.0f}/{args.seq} "
          f"(compression {_compression:.2f}x)", flush=True)

    @torch.inference_mode()
    def fwd_fn():
        model(toks, patch_lengths=pl)

    entropy_ms = _time(patch_fn, args.n_iters, args.warmup)
    base_ms = _time(fwd_fn, args.n_iters, args.warmup)
    total_ms = base_ms + entropy_ms
    kb = args.bsz * args.seq / 1024.0
    row = {"model": "BLT-1B", "bsz": args.bsz, "seq": args.seq, "kb_per_step": kb,
           "base_ms": round(base_ms, 2), "entropy_ms": round(entropy_ms, 2),
           "total_ms": round(total_ms, 2), "ms_per_kb": round(total_ms / kb, 3),
           "real_text": args.real_text, "compression": round(_compression, 3), "threshold": patcher.threshold}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[BLT-1B] bsz{args.bsz} seq{args.seq} ({kb:.0f}KB/step): base={base_ms:.1f} "
          f"+ entropy={entropy_ms:.1f} = {total_ms:.1f} ms/forward | {total_ms/kb:.2f} ms/KB", flush=True)


if __name__ == "__main__":
    main()

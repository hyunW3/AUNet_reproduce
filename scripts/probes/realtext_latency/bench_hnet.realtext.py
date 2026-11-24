#!/usr/bin/env python3
"""H-Net BPB (OOD, shared DCLM windows) and forward throughput (bps).

  bpb:        CUDA_VISIBLE_DEVICES=g python bench_hnet.py --mode bpb \
                --windows /mnt/ssd2/hyun2/AUNet/reports/bpb_windows.jsonl --out eval_out/hnet_bpb.json
  throughput: CUDA_VISIBLE_DEVICES=g python bench_hnet.py --mode throughput --out eval_out/hnet_tput.json
              (run on a FULLY IDLE A5000; matches the paper's 64 KB/forward saturating protocol)
"""
import argparse, json, math, os, time
import torch
from hnet_harness import HNetHarnessLM, build_hnet, BOS


def bpb(lm, windows_path, out):
    wins = [json.loads(l) for l in open(windows_path)]
    pairs = [("", w["text"]) for w in wins]
    total_lp = 0.0
    for i in range(0, len(pairs), lm.batch_size):
        for lp, _g in lm._score_batch(pairs[i:i + lm.batch_size]):
            total_lp += lp
    total_bytes = sum(w["n_bytes"] for w in wins)
    val = (-total_lp) / (math.log(2) * total_bytes)
    row = {"tag": "hnet_2stage_XL", "n_windows": len(wins), "total_bytes": total_bytes,
           "bpb": val, "note": "OOD (DCLM held-out; H-Net trained on other data)"}
    json.dump(row, open(out, "w"), indent=2)
    print(f"[hnet] windows={len(wins)} bytes={total_bytes} BPB={val:.4f} (OOD reference)")


def _real_toks(real_text, bsz, seqlen, dev):
    wins = [json.loads(l)["text"] for l in open(real_text)]
    big = ("\n".join(wins)).encode("utf-8")
    rows = []
    for i in range(bsz):
        b = list(big[i*seqlen:(i+1)*seqlen])
        if len(b) < seqlen: b = (b*(seqlen//max(1,len(b))+1))[:seqlen]
        rows.append(b)
    return torch.tensor(rows, dtype=torch.long, device=dev)


def throughput(model, out, bsz=8, seqlen=8192, iters=30, warmup=8, real_text=None, name="hnet"):
    dev = "cuda"
    if real_text:
        toks = _real_toks(real_text, bsz, seqlen, dev)   # REAL DCLM text -> learned chunker compresses
    else:
        toks = torch.randint(0, 254, (bsz, seqlen), dtype=torch.long, device=dev)
    mask = torch.ones((bsz, seqlen), dtype=torch.bool, device=dev)
    kb = bsz * seqlen / 1024.0
    with torch.inference_mode():
        for _ in range(warmup):
            model(toks, mask=mask)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(toks, mask=mask)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / iters * 1000.0
    bps = bsz * seqlen / (ms / 1000.0)
    row = {"tag": name, "batch": bsz, "seqlen": seqlen, "kb_per_fwd": kb, "real_text": real_text,
           "forward_ms": ms, "ms_per_kb": ms / kb, "bps_fwd": bps}
    json.dump(row, open(out, "w"), indent=2)
    print(f"[hnet] fwd={ms:.1f}ms  ms/KB={ms/kb:.2f}  bps(fwd)={bps/1000:.1f}k  "
          f"(batch {bsz}x{seqlen} = {kb:.0f} KB/fwd)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bpb", "throughput"], required=True)
    ap.add_argument("--name", default="hnet_2stage_XL")
    ap.add_argument("--windows", default="/mnt/ssd2/hyun2/AUNet/reports/bpb_windows.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--out", required=True)
    ap.add_argument("--real_text", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = build_hnet(args.name)
    if args.mode == "bpb":
        lm = HNetHarnessLM(model, args.batch_size, max_len=4096)
        bpb(lm, args.windows, args.out)
    else:
        throughput(model, args.out, bsz=args.batch_size, seqlen=args.seqlen, real_text=args.real_text, name=args.name)


if __name__ == "__main__":
    main()

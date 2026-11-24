#!/usr/bin/env python
"""Plan / verify the iso-FLOP-365 comparison: size the llama arm to match the byte model's
FLOPs/byte, and set its step count so both spend the SAME total training FLOPs.

Run from lingua/:  PYTHONPATH=$PWD python ../scripts/isoflop365/isoflop365_plan.py [--gpus 4]

Prints the byte reference (FLOP/byte, total FLOP), a small grid of llama candidates, and the
chosen llama config with the matched step count and the residual total-FLOP error. Use it to
re-derive steps if you change GPUs, batch, or the data budget.
"""
import argparse
import torch
from omegaconf import OmegaConf
from apps.aunet.hierarchical import HierarchicalArgs
from apps.main.transformer import LMTransformer, LMTransformerArgs, get_num_flop_per_token

NP = lambda m: sum(p.numel() for p in m.parameters())
BPT = 4.541  # llama3 bytes/token on DCLM (measured); re-measure if you change corpus/tokenizer

def byte_ref(cfg_path, gpus):
    c = OmegaConf.load(cfg_path)
    a = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(HierarchicalArgs()), c.get("model", {})))
    a.max_seqlens[0] = c["data"]["seq_len"]; a.vocab_size = 258
    fpb = a.flops_per_token(c["data"]["seq_len"])                      # FLOP/byte (1 tok = 1 byte)
    d = c["data"]
    nbytes = d["batch_size"] * gpus * c["grad_acc_steps"] * d["seq_len"] * c["steps"]
    return fpb, nbytes, fpb * nbytes

def llama_stats(dim, L, nh):
    a = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(LMTransformerArgs()),
                                            {"dim": dim, "n_layers": L, "n_heads": nh}))
    a.vocab_size = 128256; a.max_seqlen = 2048
    with torch.device("meta"): m = LMTransformer(a)
    ne = NP(m) - NP(m.tok_embeddings) - NP(m.output)
    return ne, get_num_flop_per_token(ne, L, dim, 2048)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--byte-config", default="apps/aunet/configs/bpebyte_100M_v4_root_greedy.yaml")
    ap.add_argument("--llama-config", default="apps/main/configs/llama_100M_cmp.yaml")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--heads", type=int, default=16)
    args = ap.parse_args()

    fpb_b, nbytes_b, flops_b = byte_ref(args.byte_config, args.gpus)
    print(f"BYTE reference  {args.byte_config}")
    print(f"  FLOP/byte {fpb_b/1e6:.1f} M | bytes-of-text {nbytes_b/1e9:.2f} B | TOTAL {flops_b:.3e} FLOP\n")

    print(f"LLAMA candidates (target FLOP/byte = {fpb_b/1e6:.1f} M):")
    for dim, L, nh in [(1024,18,16),(1024,19,16),(1088,16,17),(1152,16,18)]:
        ne, fpt = llama_stats(dim, L, nh)
        flag = "  <-- chosen" if (dim,L,nh)==(args.dim,args.layers,args.heads) else ""
        print(f"  dim{dim} L{L} h{nh}: non-embed {ne/1e6:5.1f}M | FLOP/tok {fpt/1e6:5.0f}M | "
              f"FLOP/byte {fpt/BPT/1e6:5.1f}M{flag}")

    lc = OmegaConf.load(args.llama_config); ld = lc["data"]
    ne, fpt = llama_stats(args.dim, args.layers, args.heads)
    tok_per_step = ld["batch_size"] * args.gpus * lc["grad_acc_steps"] * ld["seq_len"]
    steps = round(flops_b / (fpt * tok_per_step))
    tokens = tok_per_step * steps
    print(f"\nCHOSEN llama dim{args.dim}/L{args.layers}/h{args.heads}  "
          f"(non-embed {ne/1e6:.1f}M, FLOP/byte {fpt/BPT/1e6:.1f}M)")
    print(f"  matched steps = {steps}  (tok/step {tok_per_step/1e6:.3f}M -> {tokens/1e9:.2f}B tokens, "
          f"{tokens*BPT/1e9:.2f}B bytes-of-text)")
    print(f"  llama TOTAL {fpt*tokens:.3e} FLOP   vs byte {flops_b:.3e}  ->  "
          f"{100*(fpt*tokens-flops_b)/flops_b:+.2f}% error")
    print(f"  suggested warmup {round(steps*0.05)} | dump every {round(steps/2)}")

if __name__ == "__main__":
    main()

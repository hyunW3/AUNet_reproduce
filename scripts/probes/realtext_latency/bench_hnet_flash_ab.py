#!/usr/bin/env python3
"""H-Net forward latency with FlashAttention ON vs OFF, same protocol, same process.

The paper's forward-latency table times H-Net through its released code path, which
calls flash_attn_{varlen_,}qkvpacked_func, while AU-Net/BPEByte are timed with eager
SDPA. This A/B removes that kernel confound: run 1 is the released path, run 2 swaps
the two flash entry points for a PyTorch SDPA implementation of the same causal
(optionally windowed) attention and re-times the identical inputs.

  python bench_hnet_flash_ab.py --model hnet_1stage_XL --bsz 9 --seqlen 8192 \
      --real_text /path/to/dclm.jsonl --out flash_ab.json
"""
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F

HNET_REPO = os.environ.get("HNET_REPO", "/mnt/ssd2/hyun2/spacebyte.worktrees/h-net")
sys.path.insert(0, HNET_REPO)
sys.path.insert(0, "/mnt/ssd2/hyun2/eval_suite")


def sdpa_qkvpacked(qkv, softmax_scale=None, causal=True, window_size=(-1, -1), **kw):
    """(B,S,3,H,D) -> (B,S,H,D) with PyTorch SDPA; matches flash's causal/window semantics."""
    q, k, v = qkv.unbind(dim=2)
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))        # (B,H,S,D)
    S = q.shape[-2]
    mask = None
    w = window_size[0] if window_size else -1
    if w is not None and w >= 0:                             # sliding-window causal
        i = torch.arange(S, device=q.device).unsqueeze(1)
        j = torch.arange(S, device=q.device).unsqueeze(0)
        mask = (j <= i) & (j >= i - w)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=softmax_scale)
    else:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)                               # (B,S,H,D)


def sdpa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen, softmax_scale=None,
                          causal=True, window_size=(-1, -1), **kw):
    """(total,3,H,D) packed sequences -> same shape, looping per sequence."""
    outs = []
    cs = cu_seqlens.tolist()
    for a, b in zip(cs[:-1], cs[1:]):
        seg = qkv[a:b].unsqueeze(0)                          # (1,S,3,H,D)
        outs.append(sdpa_qkvpacked(seg, softmax_scale, causal, window_size)[0])
    return torch.cat(outs, dim=0)


def patch_flash_off():
    import hnet.mha as mha
    mha.flash_attn_qkvpacked_func = sdpa_qkvpacked
    mha.flash_attn_varlen_qkvpacked_func = sdpa_varlen_qkvpacked


def real_tokens(path, bsz, seqlen, dev):
    import json as _j
    buf = bytearray()
    with open(path) as f:
        for line in f:
            try: buf += _j.loads(line).get("text", "").encode()
            except Exception: continue
            if len(buf) > bsz * seqlen + 4096: break
    rows = []
    for i in range(bsz):
        b = list(buf[i * seqlen:(i + 1) * seqlen])
        if len(b) < seqlen:
            b = (b * (seqlen // max(1, len(b)) + 1))[:seqlen]
        rows.append(b)
    return torch.tensor(rows, dtype=torch.long, device=dev)


@torch.inference_mode()
def time_forward(model, toks, iters, warmup):
    mask = torch.ones_like(toks, dtype=torch.bool)
    for _ in range(warmup):
        model(toks, mask=mask)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(toks, mask=mask)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0       # ms/forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hnet_1stage_XL")
    ap.add_argument("--bsz", type=int, default=9)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--real_text", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from hnet_eval.harness import build_hnet, BOS                     # noqa: E402
    dev = "cuda"
    res = {"model": a.model, "batch": a.bsz, "seqlen": a.seqlen,
           "kb_per_fwd": a.bsz * a.seqlen / 1024.0,
           "real_text": bool(a.real_text), "iters": a.iters}

    toks = (real_tokens(a.real_text, a.bsz, a.seqlen, dev) if a.real_text
            else torch.randint(0, 254, (a.bsz, a.seqlen), dtype=torch.long, device=dev))

    m = build_hnet(a.model)
    res["flash_on_ms"] = time_forward(m, toks, a.iters, a.warmup)
    del m; torch.cuda.empty_cache()

    patch_flash_off()
    m = build_hnet(a.model)                                   # rebuild so the patch takes effect
    res["flash_off_ms"] = time_forward(m, toks, a.iters, a.warmup)
    res["ratio_off_over_on"] = res["flash_off_ms"] / res["flash_on_ms"]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

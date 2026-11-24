"""Per-component forward-time breakdown of AU-Net (word-pool) 1.3B: local byte transformers
(encoder/decoder), the global word-level trunk, and pooling. Mirrors profile_hnet.py."""
import sys, os, json, time, argparse
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fflm"))
from fflm_probe import build_generator          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/home/hyunwoong/AUNet_eval/runs/aunet2_1.3B/checkpoints/0000180000/consolidated")
ap.add_argument("--bsz", type=int, default=8)
ap.add_argument("--seqlen", type=int, default=8192)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=6)
args = ap.parse_args()

gen, tok = build_generator("aunet", args.ckpt, os.environ.get("AUNET_TOK"), args.seqlen)
model = gen.model
rp = gen.regex_pool

# real DCLM text -> byte sequences of exactly seqlen
wins = [json.loads(l)["text"] for l in open("/mnt/ssd2/hyun2/AUNet/reports/bpb_windows.jsonl")]
big = ("\n".join(wins)).encode("utf-8")
seqs, lvls = [], []
off = 0
for _ in range(args.bsz):
    b = list(big[off:off + args.seqlen]); off += args.seqlen
    if len(b) < args.seqlen:
        b = (b * (args.seqlen // len(b) + 1))[:args.seqlen]
    lm = rp.get_levels_mask(b)
    lm = lm.tolist() if torch.is_tensor(lm) else list(lm)
    seqs.append(b); lvls.append([int(x) for x in lm][:args.seqlen])

tokens = torch.tensor(seqs, dtype=torch.long, device="cuda")
level_mask = torch.tensor(lvls, dtype=torch.long, device="cuda")
target = torch.zeros((args.bsz, args.seqlen, 1), dtype=torch.long, device="cuda")
tlm = level_mask.clone()
print(f"pooled trunk positions (mean per seq) ~ {(level_mask>0).float().sum(1).mean().item():.0f} of {args.seqlen} bytes "
      f"(compression {args.seqlen/max(1,(level_mask>0).float().sum(1).mean().item()):.1f}x)")

COMP = {}
COMP["local enc (byte, 3L d512)"] = (model.encoders[0], "local")
COMP["local dec (byte, 3L d512)"] = (model.decoders[0], "local")
COMP[f"GLOBAL trunk ({len(model.trunk.layers)}L d2048)"] = (model.trunk, "global")

acc = {k: 0.0 for k in COMP}; starts = {}
def pre(l):
    def h(m, i): e = torch.cuda.Event(enable_timing=True); e.record(); starts[l] = e
    return h
def post(l):
    def h(m, i, o): e = torch.cuda.Event(enable_timing=True); e.record(); starts[l + "_e"] = e
    return h
for l, (m, c) in COMP.items():
    m.register_forward_pre_hook(pre(l)); m.register_forward_hook(post(l))

def fwd():
    return model(tokens, level_mask, target, tlm, return_logits=True, attn_impl="sdpa")

with torch.inference_mode():
    for _ in range(args.warmup): fwd()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(args.iters):
        fwd(); torch.cuda.synchronize()
        for l in COMP: acc[l] += starts[l].elapsed_time(starts[l + "_e"])
    total = (time.time() - t0) / args.iters * 1000.0

print(f"\nAU-Net (word-pool) 1.3B  forward  batch {args.bsz}x{args.seqlen} ({args.bsz*args.seqlen/1024:.0f} KB)")
print(f"total forward = {total:.1f} ms/iter\n")
per = {k: v / args.iters for k, v in acc.items()}; cat = {}
print(f"  {'component':34}{'ms':>8}{'% total':>9}")
for l, (m, c) in COMP.items():
    ms = per[l]; cat[c] = cat.get(c, 0) + ms
    print(f"  {l:34}{ms:8.2f}{100*ms/total:9.1f}")
for c in ("local", "global"):
    print(f"  {'CAT '+c:34}{cat.get(c,0):8.2f}{100*cat.get(c,0)/total:9.1f}")
cov = sum(cat.values())
print(f"  {'(pooling+embed/head+overhead)':34}{total-cov:8.2f}{100*(total-cov)/total:9.1f}")

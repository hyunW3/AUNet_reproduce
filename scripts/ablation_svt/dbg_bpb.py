"""Debug: score the same held-out windows via (a) the TRAINING forward path (tokenize() level_mask,
model.forward CE) and (b) the generator path used by bpb_windows_local.py / downstream eval."""
import json, math, sys
import numpy as np
import torch
sys.path.insert(0, ".")
from apps.aunet.generate import (load_consolidated_model_and_tokenizer,
                                 PackedHierarchicalCausalTransformerGenerator,
                                 PackedHierarchicalCausalTransformerGeneratorArgs)
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs
from apps.aunet.data.data import tokenize

ckpt, wins_path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
IMPL = sys.argv[4] if len(sys.argv) > 4 else "sdpa"
ONLY_A = len(sys.argv) > 5
model, tok, pool, cfg = load_consolidated_model_and_tokenizer(
    ckpt, model_cls=HierarchicalTransformer, model_args_cls=HierarchicalArgs)
wins = [json.loads(l) for l in open(wins_path)][:n]
regex = pool  # unused; training path rebuilds from cfg
from apps.aunet.data.regex_cutting import RegexArgs
from lingua.args import dataclass_from_dict
rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)

# (a) training path: pack docs back-to-back into one 8192+1 sequence exactly like training
T, M = [], []
for tn, _ in tokenize(((w, None) for w in wins), add_bos=True, add_eos=True, tokenizer_type="bytes", regex=rargs):
    T.extend(tn[0]); M.extend(tn[1])
L = int(cfg.data.seq_len)
assert len(T) >= L + 1, f"need >= {L+1} tokens, have {len(T)}; pass more windows"
toks = torch.tensor(T[:L + 1], dtype=torch.long); lm = torch.tensor(M[:L + 1], dtype=torch.long)
x, y = toks[:-1][None].repeat(2, 1).cuda(), toks[1:][None].repeat(2, 1).cuda()
lx, ly = lm[:-1][None].repeat(2, 1).cuda(), lm[1:][None].repeat(2, 1).cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x, lx, y, ly, attn_impl=IMPL)
ce = out[0].float().item()
print(f"(a) [{IMPL}] training-forward  mean CE = {ce:.4f} nats/byte -> BPB {ce / math.log(2):.4f} "
      f"(packed {L} tokens, patches {int((lm[:L] > 0).sum())})", flush=True)

if ONLY_A:
    sys.exit(0)
# (b) generator path
gen = PackedHierarchicalCausalTransformerGenerator(
    PackedHierarchicalCausalTransformerGeneratorArgs(temperature=0.0, max_gen_len=1, max_tokens=8192), model, tok, pool)
_, lls, _ = gen.generate([w["text"] for w in wins])
tot = sum(ll.sum().item() for ll in lls)
ntok_b = sum(ll.numel() for ll in lls)
nbytes = sum(w["n_bytes"] for w in wins)
print(f"(b) generator  sum ll = {tot:.1f} over {ntok_b} positions; bytes = {nbytes}; "
      f"BPB = {-tot / nbytes / math.log(2):.4f}; mean per position = {-tot / ntok_b:.4f} nats", flush=True)
ll0 = lls[0].float().cpu().numpy()
print("first window first 20 per-position nll:", np.round(-ll0[:20], 2).tolist())
print("first window last 5 per-position nll:", np.round(-ll0[-5:], 2).tolist(), "len", len(ll0), "n_bytes", wins[0]["n_bytes"])

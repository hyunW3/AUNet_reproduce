"""Held-out BPB through the TRAINING forward path (attn_impl="fmha", the kernel the model was trained
with). The 320 held-out DCLM windows (reports/bpb_windows.jsonl) are tokenized with the checkpoint's
own regex config (training tokenize(): BOS + bytes + EOS per window), packed back-to-back into
seq_len+1 sequences exactly like the training loader, and scored with model.forward CE.

Why not bpb_windows_local.py: on these 100M checkpoints the generator path returns ~2.8-3.5 BPB
(and attn_impl="sdpa" gives ~4.0), while the fmha forward gives ~1.35 on the same text.

BPB = mean CE (nats per predicted byte, incl. the ~0.05% BOS/EOS targets) / ln2.
Usage (from the lingua dir): python heldout_fmha.py <consolidated_dir> <windows.jsonl> <tag> <out.jsonl> [tok_override]
"""
import json, math, sys
import torch
sys.path.insert(0, ".")
from apps.aunet.generate import load_consolidated_model_and_tokenizer, _resolve_bpe_tokenizer_path
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs
from apps.aunet.data.data import tokenize
from apps.aunet.data.regex_cutting import RegexArgs
from lingua.args import dataclass_from_dict

ckpt, wins_path, tag, out_path = sys.argv[1:5]
tok_override = sys.argv[5] if len(sys.argv) > 5 else None
model, _, _, cfg = load_consolidated_model_and_tokenizer(
    ckpt, model_cls=HierarchicalTransformer, model_args_cls=HierarchicalArgs)
rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
if rargs.bpe_tokenizer_path is not None:
    rargs.bpe_tokenizer_path = _resolve_bpe_tokenizer_path(rargs.bpe_tokenizer_path, tok_override)

wins = [json.loads(l) for l in open(wins_path)]
T, M = [], []
for tn, _ in tokenize(((w, None) for w in wins), add_bos=True, add_eos=True, tokenizer_type="bytes", regex=rargs):
    T.extend(tn[0]); M.extend(tn[1])
L = int(cfg.data.seq_len)
n_seq = (len(T) - 1) // L
n_seq -= n_seq % 2                                   # pairs (bsz>=2: ConstantSumMask squeezes bsz 1)
toks = torch.tensor(T[:n_seq * L + 1], dtype=torch.long)
lm = torch.tensor(M[:n_seq * L + 1], dtype=torch.long)
nll = 0.0
npos = 0
for i in range(0, n_seq, 2):
    xs, ys, lxs, lys = [], [], [], []
    for j in (i, i + 1):
        a = j * L
        xs.append(toks[a:a + L]); ys.append(toks[a + 1:a + L + 1])
        lxs.append(lm[a:a + L]); lys.append(lm[a + 1:a + L + 1])
    x, y = torch.stack(xs).cuda(), torch.stack(ys).cuda()
    lx, ly = torch.stack(lxs).cuda(), torch.stack(lys).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ce = model(x, lx, y, ly, attn_impl="fmha")[0].float().item()
    nll += ce * y.numel()
    npos += y.numel()
patches = int((lm[:n_seq * L] > 0).sum())
row = {"tag": tag, "path": "train_forward_fmha", "n_seq": n_seq, "seq_len": L, "n_pred": npos,
       "bytes_per_patch": n_seq * L / patches, "ce_nats": nll / npos, "bpb": nll / npos / math.log(2)}
with open(out_path, "a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row), flush=True)

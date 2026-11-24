#!/usr/bin/env python3
"""Verify the boundary-dropout modes: does every emitted patch stay inside Prefix(R)?

free        -> expected to VIOLATE the closure (that is the point of the mode)
prefix_safe -> every patch must be a valid trie path
token_safe  -> every patch must additionally be a COMPLETE vocabulary entry
"""
import json
import sys

sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua")
from apps.aunet.data.regex_cutting import RegexArgs, RegexPool

TOK = "/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model"
WINDOWS = "/mnt/ssd2/hyun2/AUNet/reports/bpb_windows.jsonl"


def pool(p, mode):
    a = RegexArgs(strategy={"bpe_br": "1@1"})
    a.bpe_tokenizer_path = TOK
    a.bpe_online = True
    a.bpe_online_mode = "greedy"
    a.bpe_online_placement = "root"
    a.bpe_context_prefix = 0
    a.bpe_boundary_dropout = p
    a.bpe_boundary_dropout_mode = mode
    a.bpe_boundary_dropout_seed = 0
    return RegexPool(a)


def classify(trie, span):
    """(in_prefix_closure, is_complete_token) for one patch."""
    node = trie.root
    for b in span:
        node = node.children.get(int(b))
        if node is None:
            return False, False
    return True, node.is_token


def audit(rp, texts):
    trie = rp.byte_trie
    tot = inpref = istok = nbytes = 0
    for t in texts:
        bs = t.encode()
        m = rp.online_levels_mask(t, add_bos=0, add_eos=0)
        starts = [i for i, v in enumerate(m) if v > 0]           # root placement: patch starts
        spans = [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]
        spans.append((starts[-1], len(bs)))
        nbytes += len(bs)
        for s, e in spans:
            ok, tk = classify(trie, bs[s:e])
            tot += 1
            inpref += ok
            istok += tk
    return tot, inpref / tot, istok / tot, nbytes / tot


texts = [json.loads(l)["text"] for l in open(WINDOWS, encoding="utf-8")][:40]
print(f"{'mode':<12}{'p':>5}{'patches':>10}{'in Prefix(R)':>14}{'complete tok':>14}{'bytes/patch':>13}")
for mode, ps in (("free", (0.0, 0.1, 0.2)),
                 ("prefix_safe", (0.1, 0.2, 0.4)),
                 ("token_safe", (0.1, 0.2, 0.4))):
    for p in ps:
        tot, pref, tok, bpp = audit(pool(p, mode), texts)
        tag = "baseline" if p == 0.0 else mode
        print(f"{tag:<12}{p:>5.2f}{tot:>10,}{pref*100:>13.1f}%{tok*100:>13.1f}%{bpp:>13.3f}")

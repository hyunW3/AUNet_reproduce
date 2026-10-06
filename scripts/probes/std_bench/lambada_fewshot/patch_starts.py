#!/usr/bin/env python3
"""Patch starts of each model's parser on LAMBADA 5-shot prompts + target, standard vs GPT-3 format.

Same parsers as scripts/probes/patch_stats: Transformer/AUNet/BPEByte via compression_stability.build_parser
(CPU), BLT via the HF entropy patcher with its 512-byte window at the released threshold, H-Net via its
router's boundary mask. 500 items (random.seed(0)) of the exact std_bench prompts
(dump_prompts.py -> lambada_prompts.json["5"]; dump_gpt3_5.py -> gpt3_prompts5.json["5"]).

  runpy.sh patch_starts.py --model bpebyte|aunet|llama|blt    (lingua venv; GPU for blt)
  runext.sh hnet patch_starts.py --model hnet                  (H-Net env, GPU)
  -> <tmp>/starts_<model>.json {"ids": [...], "std": [[starts]...], "gpt3": [[starts]...], "tgt_byte": {...}}
"""
import argparse, json, random, sys
from pathlib import Path

PS = "/mnt/ssd2/hyun2/AUNet/scripts/probes/patch_stats"
sys.path[:0] = [PS, str(Path(PS).parent)]
T = sys.argv[sys.argv.index("--tmp") + 1] if "--tmp" in sys.argv else "/home/hyunw3/.claude/jobs/003c5e28/tmp"


def texts():
    std = json.load(open(f"{T}/lambada_prompts.json"))["5"]
    g3 = json.load(open(f"{T}/gpt3_prompts5.json"))["5"]
    random.seed(0)
    ids = sorted(random.sample(range(len(std)), 500))
    out = {"ids": ids, "std": [], "gpt3": [], "tgt_byte": {"std": [], "gpt3": []}}
    for f, P in (("std", std), ("gpt3", g3)):
        for i in ids:
            out[f].append(P[i]["ctx"] + P[i]["target"])
            out["tgt_byte"][f].append(len(P[i]["ctx"].encode()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("llama", "aunet", "bpebyte", "blt", "hnet"))
    ap.add_argument("--tmp", default=T)
    a = ap.parse_args()
    X = texts()
    res = {"ids": X["ids"], "tgt_byte": X["tgt_byte"]}
    if a.model in ("llama", "aunet", "bpebyte"):
        from compression_stability import build_parser
        parse = build_parser(a.model)
        for f in ("std", "gpt3"):
            res[f] = [list(parse(t)) for t in X[f]]
    elif a.model == "blt":
        import measure_blt as MB
        import torch
        patcher, thr = MB.load_patcher(torch.float32)
        for f in ("std", "gpt3"):
            res[f] = []
            for t in X[f]:
                ids = [MB.BOS] + [b + MB.OFFSET for b in t.encode()]
                ent = MB.entropies(patcher, [ids], 512)[0]
                res[f].append(MB.byte_starts(ent, len(ids) - 1, thr))
        res["threshold"] = thr
    else:
        import torch
        import measure_hnet as MH
        model = MH.build()
        for f in ("std", "gpt3"):
            res[f] = []
            for t in X[f]:
                ids = [MH.BOS] + list(t.encode())
                bm = MH.boundary_masks(model, [ids])
                res[f].append(torch.nonzero(bm[0, 1:len(ids)]).flatten().tolist())
    json.dump(res, open(f"{a.tmp}/starts_{a.model}.json", "w"))
    print("wrote", f"{a.tmp}/starts_{a.model}.json")


if __name__ == "__main__":
    main()

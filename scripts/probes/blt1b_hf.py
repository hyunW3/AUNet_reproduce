#!/usr/bin/env python3
"""Official BLT-1B (facebook/blt-1b via the transformers conversion itazap/blt-1b-hf)
on TKEval + the letter-cloze re-score + the BrittleBench-style grid.

Scoring matches the in-repo protocols exactly:
  * tkeval / cloze : teacher-forced greedy — same tuples as common.score():
                     (exact, n_correct_tok, n_tok, ll_sum) per (ctx, cont) pair.
  * brittle        : wraps the HF model as a loglikelihood(reqs) callable and calls
                     apps.aunet.eval_brittle_mc.run_brittle_mc, so items, perturbations,
                     seeds and aggregation are byte-identical to the lingua models.

CAVEAT: BLT-1B is trained by Meta on its own 1T-token mix — architecture AND data differ
from the in-repo 1.3B checkpoints (DCLM subset, 180k steps). Robustness deltas are
meaningful; absolute accuracies are confounded by training data.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import torch
from transformers import AutoTokenizer, BltForCausalLM

MODEL_ID = "itazap/blt-1b-hf"
ROOT = "/mnt/ssd2/hyun2/AUNet"
TKBASE = f"{ROOT}/benchmarks/TKEval"
TK_TASKS = {
    "cycled_letters":  ("complex_problem_solving/cycled_letters_all_data_0123_shots.json", "qa"),
    "unscramble":      ("complex_problem_solving/word_unscrambling_all_data_0123_shots.json", "qa"),
    "math_theorems":   ("complex_problem_solving/identify_math_theorems_all_data_0123_shots.json", "qa"),
    "common_substr":   ("token_structure_probing/test/multi_token_prob.common_substrings.all_data_0123_shots.json", "raw"),
    "lcsubseq":        ("token_structure_probing/test/multi_token_prob.longest_common_subsequences.all_data_0123_shots.json", "raw"),
    "lcsubstr":        ("token_structure_probing/test/multi_token_prob.longest_common_substrings.all_data_0123_shots.json", "raw"),
    "case_conversion": ("token_structure_probing/test/token_struct_prob.char_case_conversion.all_data_0123_shots.json", "raw"),
    "char_count":      ("token_structure_probing/test/token_struct_prob.character_count.all_data_0123_shots.json", "raw"),
    "nth_char":        ("token_structure_probing/test/token_struct_prob.nth_character.all_data_0123_shots.json", "raw"),
    "nth_char_end":    ("token_structure_probing/test/token_struct_prob.nth_character_from_end.all_data_0123_shots.json", "raw"),
}
CLOZE_TASKS = ["char_count", "nth_char", "nth_char_end"]

_model = None
_tok = None


def load():
    global _model, _tok
    if _model is None:
        from transformers import BltConfig
        _tok = AutoTokenizer.from_pretrained(MODEL_ID)
        cfg = BltConfig.from_pretrained(MODEL_ID)
        # transformers 5.x generic paths expect top-level num_hidden_layers; the
        # blt-1b-hf conversion only carries it in the sub-configs.
        if not hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = cfg.global_config.num_hidden_layers
        _model = BltForCausalLM.from_pretrained(
            MODEL_ID, config=cfg, dtype=torch.bfloat16).cuda().eval()
    return _model, _tok


@torch.inference_mode()
def score_pairs(pairs, log_every=2000):
    """common.score()-compatible: [(exact, n_correct, n_tok, ll_sum)] per (ctx, cont)."""
    model, tok = load()
    out = []
    for i, (ctx, cont) in enumerate(pairs):
        cids = tok(ctx, add_special_tokens=True)["input_ids"]
        kids = tok(cont, add_special_tokens=False)["input_ids"]
        if not kids:
            out.append((False, 0, 0, 0.0)); continue
        ids = torch.tensor([cids + kids], device="cuda")
        logits = model(ids).logits[0]          # [T, V]; logits[t] predicts ids[t+1]
        p = len(cids)
        sl = logits[p - 1:p - 1 + len(kids)].float()
        tgt = torch.tensor(kids, device="cuda")
        lp = torch.log_softmax(sl, dim=-1)
        ll = lp[torch.arange(len(kids)), tgt]
        greedy = (sl.argmax(-1) == tgt)
        out.append((bool(greedy.all().item()), int(greedy.sum().item()),
                    len(kids), float(ll.sum().item())))
        if log_every and (i + 1) % log_every == 0:
            print(f"  ..{i+1}/{len(pairs)}", flush=True)
    return out


def run_tkeval(limit, shots, seed, out_path):
    import random
    rows = []
    for name, (rel, cut) in TK_TASKS.items():
        data = json.load(open(f"{TKBASE}/{rel}"))[str(shots)]
        if limit and len(data) > limit:
            data = random.Random(seed).sample(data, limit)
        pairs, idx = [], []
        for i, x in enumerate(data):
            for t in x["target"]:
                pairs.append((x["input"].rstrip(" "), " " + t) if cut == "qa"
                             else (x["input"], t))
                idx.append(i)
        res = score_pairs(pairs)
        be, bf = {}, {}
        for ii, r in zip(idx, res):
            be[ii] = max(be.get(ii, 0), int(r[0]))
            bf[ii] = max(bf.get(ii, 0.0), r[1] / r[2] if r[2] else 0.0)
        row = {"tag": "blt1b_hf", "task": name, "shots": shots, "n": len(data),
               "acc": sum(be.values()) / len(data), "tok_frac": sum(bf.values()) / len(data)}
        print(f"[blt1b] {name:16s} n={len(data):4d} acc={row['acc']:.3f} tok_frac={row['tok_frac']:.3f}", flush=True)
        rows.append(row)
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_cloze(shots, out_path):
    cands = list(string.ascii_lowercase)
    rows = []
    for name in CLOZE_TASKS:
        rel, _ = TK_TASKS[name]
        data = json.load(open(f"{TKBASE}/{rel}"))[str(shots)]
        res = score_pairs([(x["input"], c) for x in data for c in cands])
        n = len(cands)
        corr = sum(int(cands[max(range(n), key=lambda j: res[i*n+j][3])] in x["target"])
                   for i, x in enumerate(data))
        row = {"tag": "blt1b_hf", "task": name, "shots": shots, "n": len(data),
               "cloze_acc": corr / len(data)}
        print(f"[blt1b] cloze {name:14s} acc={row['cloze_acc']:.3f}", flush=True)
        rows.append(row)
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_brittle(limit, out_path):
    import os
    os.environ.setdefault("BRITTLE_TASKS", "arc_easy,piqa,hellaswag")
    import sys
    sys.path.insert(0, f"{ROOT}/lingua")
    from apps.aunet.eval_brittle_mc import run_brittle_mc

    def loglikelihood(reqs):
        res = score_pairs([r.args for r in reqs])
        return [(r[3], r[0]) for r in res]

    results = run_brittle_mc(loglikelihood, limit=limit)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": results}, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["tkeval", "cloze", "brittle"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    O = f"{ROOT}/reports"
    if "tkeval" in args.suite:
        run_tkeval(args.limit, args.shots, args.seed, f"{O}/tkeval/results_blt1b.jsonl")
    if "cloze" in args.suite:
        run_cloze(args.shots, f"{O}/tkeval/cloze_blt1b.jsonl")
    if "brittle" in args.suite:
        run_brittle(args.limit, f"{O}/brittle/blt1b/results.json")

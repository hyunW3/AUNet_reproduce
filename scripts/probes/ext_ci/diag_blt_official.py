#!/usr/bin/env python3
"""Diagnose why the HF-port BLT scores differ from the paper-time bytelatent harness on short inputs.

Loads the official bytelatent ByteLatentTransformer / LMTransformer from the released facebook/blt-1b
and facebook/blt-entropy weights, then scores the same lm-eval items with the paper-time harness
(eval_suite/blt_eval/harness.py::BLTHarness) under:
  A  paper-time path       : batch 16, entropy model called with its default attention (no window)
  B  same, batch size 1    : isolates batch-mate / padding effects
  C  batch 16, entropy model forced to attn_impl=xformers (local_block_causal, window 512)
Writes accuracies and per-item acc/acc_norm to --out.

  cd blt_official && BLT_ALLOW_MISSING_FLEX_ATTENTION=1 BLT_SUPPRESS_ATTN_ERROR=1 PYTHONPATH=$PWD \
    ../blt_venv/bin/python ../diag_blt_official.py --weights ../blt_weights --limit 300 --out ../out/diag.json
"""
import argparse, json, os, sys

import torch


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--eval_suite", default=os.path.expanduser("~/aunet_ext/eval_suite"))
    ap.add_argument("--tasks", nargs="+", default=["hellaswag", "winogrande"])
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--conds", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--sniah_pairs", default=None, help="score S-NIAH pairs instead of lm-eval tasks")
    ap.add_argument("--sniah_per_cell", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from bytelatent.model.blt import ByteLatentTransformer
    from bytelatent.transformer import LMTransformer
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.data.patcher import PatcherArgs
    from bytelatent.distributed import DistributedArgs, setup_torch_distributed

    da = DistributedArgs(); da.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(da)

    model = ByteLatentTransformer.from_pretrained(f"{a.weights}/blt_1b", local_files_only=True)
    model = model.to("cuda", torch.bfloat16).eval()
    ent = LMTransformer.from_pretrained(f"{a.weights}/entropy", local_files_only=True).to("cuda", torch.bfloat16).eval()
    tok = BltTokenizer(vocab_size_unit_1=256, bpe_delim=False, add_bos=True, add_eos=False)
    patcher = PatcherArgs(threshold=1.335442066192627, realtime_patching=False, patching_device="cuda").build()
    patcher.entropy_model = ent
    print("model attn_impl:", getattr(model, "attn_impl", None), "| entropy attn_impl:", getattr(ent, "attn_impl", None),
          "sliding_window:", getattr(ent, "sliding_window", None), flush=True)

    sys.path.insert(0, a.eval_suite)
    from blt_eval.harness import BLTHarness  # noqa: E402
    from lm_eval import simple_evaluate  # noqa: E402

    orig_forward = ent.forward

    def windowed_forward(toks, *args, **kw):
        kw.setdefault("attn_impl", "xformers")
        return orig_forward(toks, *args, **kw)


    res = {}
    for cond in a.conds:
        ent.forward = windowed_forward if cond == "C" else orig_forward
        lm = BLTHarness(model, tok, patcher, batch_size=1 if cond == "B" else 16, max_len=4096)
        if a.sniah_pairs:
            from collections import defaultdict
            recs, seen = [], defaultdict(int)
            for l in open(a.sniah_pairs):
                rr = json.loads(l)
                if seen[rr["cell"]] < a.sniah_per_cell:
                    seen[rr["cell"]] += 1; recs.append(rr)
            lm.batch_size = 1 if cond == "B" else 8
            got = lm.score_pairs([(rr["prompt"], " " + rr["values"][0]) for rr in recs])
            per = defaultdict(list)
            for rr, (_lp, g) in zip(recs, got):
                per[rr["cell"]].append(int(g))
            res[cond] = {"per_cell": {c: sum(v) / len(v) for c, v in per.items()}, "per_item": dict(per)}
            print(cond, {c: round(v, 2) for c, v in sorted(res[cond]["per_cell"].items())}, flush=True)
            continue
        r = simple_evaluate(lm, tasks=a.tasks, num_fewshot=0, limit=a.limit, log_samples=True, bootstrap_iters=0)
        res[cond] = {"results": {t: {k: v for k, v in r["results"][t].items() if "acc" in k} for t in a.tasks},
                     "samples": {t: {"doc_id": [int(s["doc_id"]) for s in r["samples"][t]],
                                     "acc": [float(s["acc"]) for s in r["samples"][t]],
                                     "acc_norm": [float(s.get("acc_norm", s["acc"])) for s in r["samples"][t]]}
                                 for t in a.tasks}}
        print(cond, res[cond]["results"], flush=True)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()

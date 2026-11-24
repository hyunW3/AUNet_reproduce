#!/usr/bin/env python3
"""Held-out BPB + downstream for the scratchpad / updater pretraining arms.

Everything goes through the model's OWN `forward`, not a reimplementation: the arms differ inside
that forward (`scratchpad_k`, `updater_kind`), so any scorer that rebuilds the forward by hand
would silently score the baseline path for every arm.

  bpb        held-out byte NLL on fixed windows, the same protocol as reports/bpb_windows.jsonl
  hellaswag / arc_easy / boolq   loglikelihood MC, requests taken from lm-eval so the prompts match

Per-request logprobs are written out so two arms can be compared paired, item by item -- the
differences under test are far below the run-to-run noise of accuracy alone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lingua.args import dataclass_from_dict  # noqa: E402
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints  # noqa: E402
from apps.aunet.hierarchical import HierarchicalTransformer, HierarchicalArgs  # noqa: E402
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs  # noqa: E402

LN2 = math.log(2.0)
BOS = 256
TASKS = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "boolq": "acc"}


def load(ckpt_dir, tok_path=None):
    p = Path(ckpt_dir)
    if not (p / CONSOLIDATE_NAME).exists():
        p = Path(consolidate_checkpoints(str(p)))
    cfg = OmegaConf.load(p / "params.json")
    margs = dataclass_from_dict(HierarchicalArgs, cfg.model, strict=False)
    model = HierarchicalTransformer(margs)
    st = torch.load(p / CONSOLIDATE_NAME, weights_only=True, map_location="cpu")
    model.load_state_dict({k.replace("parametrizations", "_orig_mod.parametrizations")
                           if "transitions." in k and "parametrizations" in k else k: v
                           for k, v in st["model"].items()})
    del st
    model = model.to(dtype=torch.bfloat16).cuda().eval()
    rargs = dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False)
    if tok_path and getattr(rargs, "bpe_tokenizer_path", None):
        rargs.bpe_tokenizer_path = tok_path
    return model, RegexPool(rargs), cfg


def prep(text, pool, max_bytes, seqlen):
    clip = text.encode()[:max_bytes].decode("utf-8", errors="ignore")
    bs = list(clip.encode())
    n = len(bs)
    if n == 0 or n + 2 >= seqlen:
        return None
    ids_real = [BOS] + bs
    lm_real = pool.get_levels_mask(ids_real)
    ids = torch.zeros(seqlen, dtype=torch.long)
    lm = torch.zeros(seqlen, dtype=torch.long)
    ids[: n + 1] = torch.tensor(ids_real)
    lm[: n + 1] = torch.tensor(lm_real[: n + 1])
    lm[n + 1] = 1
    return ids, lm, n


@torch.no_grad()
def score_seqs(model, rows, bs):
    """rows: [(ids[T], lm[T], n_ctx, n_tot)] -> per-row (sum logprob over [n_ctx, n_tot), n bytes)"""
    out = [None] * len(rows)
    order = np.argsort([-r[3] for r in rows])
    for i in range(0, len(order), bs):
        sel = list(order[i : i + bs])
        while len(sel) < 2:
            sel.append(sel[-1])
        # rows are bucketed by length, so a batch can straddle two bucket sizes: pad up to the
        # batch's own max.  The pad tail is sealed into its own patch, exactly as prep() does.
        T = max(rows[j][0].shape[0] for j in sel)
        ids = torch.zeros(len(sel), T, dtype=torch.long)
        lm = torch.zeros(len(sel), T, dtype=torch.long)
        for b, j in enumerate(sel):
            t = rows[j][0].shape[0]
            ids[b, :t] = rows[j][0]; lm[b, :t] = rows[j][1]
            if t < T:
                lm[b, t] = 1
        ids, lm = ids.cuda(), lm.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True)
        for b, j in enumerate(sel):
            nc, nt = rows[j][2], rows[j][3]
            lp = F.log_softmax(logits[b, nc - 1 : nt - 1].float(), -1)
            s = lp.gather(-1, ids[b, nc:nt].unsqueeze(-1)).sum().item()
            if out[j] is None:
                out[j] = (s, nt - nc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--max_bytes", type=int, default=3840)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seqlen_cap", type=int, default=2048)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag/sp_eval"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, pool, cfg = load(args.ckpt, args.tokenizer_path)
    seqlen = int(model.transitions[0].seqlen_in)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    res = {"ckpt": args.ckpt,
           "scratchpad_k": int(getattr(model, "scratchpad_k", 0)),
           "updater_kind": str(getattr(model, "updater_kind", ""))}
    print(f"[{args.tag}] scratchpad_k={res['scratchpad_k']} updater={res['updater_kind'] or '-'}",
          flush=True)

    # ---- held-out BPB ------------------------------------------------------------------------
    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[: args.limit]
    rows, nll = [], []
    prepped = [p for p in (prep(w["text"], pool, args.max_bytes, seqlen) for w in wins) if p]
    t0 = time.time()
    for i in range(0, len(prepped), 2):
        ch = prepped[i : i + 2]
        if len(ch) < 2:
            ch = ch + [ch[-1]]
        ids = torch.stack([c[0] for c in ch]).cuda()
        lm = torch.stack([c[1] for c in ch]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, lm.clone(), ids, torch.zeros_like(lm), return_logits=True)
        for b in range(min(2, len(prepped) - i)):
            n = ch[b][2]
            nll.append(F.cross_entropy(logits[b, :n].float(), ids[b, 1 : n + 1],
                                       reduction="none").cpu().numpy())
    nll = np.concatenate(nll)
    res["bpb"] = float(nll.mean() / LN2)
    res["bpb_bytes"] = int(nll.size)
    np.save(outdir / f"{args.tag}_bpb.npy", nll.astype(np.float32))
    print(f"[{args.tag}] held-out BPB = {res['bpb']:.4f}  ({nll.size} bytes, "
          f"{time.time() - t0:.0f}s)", flush=True)

    # ---- downstream --------------------------------------------------------------------------
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    for task in args.tasks.split(","):
        t = get_task_dict([task], tm)[task]
        docs = list(t.eval_docs)
        if args.limit:
            docs = docs[: args.limit]
        rows, spans = [], []
        for d in docs:
            ctx, choices = t.doc_to_text(d), t.doc_to_choice(d)
            tgt = t.doc_to_target(d)
            gold = tgt if isinstance(tgt, int) else (choices.index(tgt) if tgt in choices
                                                    else int(tgt))
            spans.append((len(rows), len(choices), int(gold)))
            for c in choices:
                cont = str(c) if str(c).startswith(" ") else " " + str(c)
                cb = ctx.encode()[-(args.seqlen_cap - 64):]
                seq = [BOS] + list(cb) + list(cont.encode())
                T = min(seqlen, max(64, 2 ** math.ceil(math.log2(len(seq) + 1))))
                seq = seq[:T]
                ids = torch.zeros(T, dtype=torch.long); lm = torch.zeros(T, dtype=torch.long)
                ids[: len(seq)] = torch.tensor(seq)
                lv = pool.get_levels_mask(seq)
                lm[: len(seq)] = torch.tensor(lv[: len(seq)])
                if len(seq) < T:
                    lm[len(seq)] = 1
                rows.append((ids, lm, 1 + len(cb), len(seq)))
        t0 = time.time()
        sc = score_seqs(model, rows, args.bs)
        recs = []
        for off, n, gold in spans:
            lp = [sc[off + k][0] for k in range(n)]
            nb = [max(1, sc[off + k][1]) for k in range(n)]
            recs.append({"gold": gold, "lp": lp, "lp_norm": [a / b for a, b in zip(lp, nb)],
                         "pred": int(np.argmax(lp)),
                         "pred_norm": int(np.argmax([a / b for a, b in zip(lp, nb)]))})
        acc = float(np.mean([r["pred"] == r["gold"] for r in recs]))
        accn = float(np.mean([r["pred_norm"] == r["gold"] for r in recs]))
        res[task] = {"n": len(recs), "acc": acc, "acc_norm": accn,
                     "metric": TASKS.get(task, "acc")}
        with open(outdir / f"{args.tag}_{task}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"[{args.tag}] {task:11s} n={len(recs):5d} acc={acc:.4f} acc_norm={accn:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    key = [res[t][TASKS.get(t, "acc")] for t in args.tasks.split(",") if t in res]
    res["avg3"] = float(np.mean(key)) if key else None
    print(f"[{args.tag}] avg3 = {res['avg3']:.4f}" if res["avg3"] else "", flush=True)
    json.dump(res, open(outdir / f"{args.tag}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

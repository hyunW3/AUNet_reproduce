#!/usr/bin/env python3
"""Downstream multiple-choice accuracy for a frozen AU-Net with / without a patch-lag updater.

HellaSwag, ARC-Challenge and BoolQ are all pure loglikelihood tasks, so this scores them through
the SAME `split_forward` -> updater -> `decode_from_globals` path the updater was trained on,
rather than through the cached generate.py decoder.  Requests (context / continuation strings and
gold labels) come from lm-eval itself, so the prompt formatting matches the repo's existing tables.

Per-request logprobs are written out, so two runs can be compared PAIRED (same items, same order).
That matters here: the BPB differences under test are ~0.4%, far below the ~1pt run-to-run noise
of a 1.3B model on these sets, and only a paired view can say whether anything moved at all.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_ablation import load_model  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, decode_from_globals, PatchLagUpdater,
)

BOS = 256
TASKS = {"hellaswag": "acc_norm", "arc_challenge": "acc_norm", "boolq": "acc"}


def build_requests(task_name, limit=None, include_path=None):
    """-> list of dicts: {ctx, conts[], gold}. Uses lm-eval's own task construction."""
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(include_path=include_path) if include_path else TaskManager()
    task = get_task_dict([task_name], tm)[task_name]
    docs = list(task.eval_docs)
    if limit:
        docs = docs[:limit]
    out = []
    for d in docs:
        ctx = task.doc_to_text(d)
        choices = task.doc_to_choice(d)
        tgt = task.doc_to_target(d)
        gold = tgt if isinstance(tgt, int) else choices.index(tgt) if tgt in choices else int(tgt)
        out.append({"ctx": ctx, "conts": [f" {c}" if not str(c).startswith(" ") else str(c)
                                          for c in choices], "gold": int(gold)})
    return out


def encode(ctx, cont, max_bytes):
    """-> (ids, n_ctx, n_total). Byte model: the continuation is scored, the context is not."""
    cb = ctx.encode()[-max_bytes:]
    kb = cont.encode()
    ids = [BOS] + list(cb) + list(kb)
    return ids, 1 + len(cb), len(ids)


@torch.no_grad()
def score(model, pool, upd, pairs, seqlen_cap, bs, attn_impl="fmha", inject="global"):
    """pairs: [(ctx, cont)] -> [(sum_logprob, n_bytes)] over the continuation bytes only."""
    enc = [encode(c, k, seqlen_cap - 64) for c, k in pairs]
    order = np.argsort([-e[2] for e in enc])          # length-bucket so padding stays cheap
    res = [None] * len(enc)
    for i in range(0, len(order), bs):
        sel = order[i : i + bs]
        rows = [enc[j] for j in sel]
        if len(rows) < 2:
            rows = rows + [rows[-1]]
            sel = list(sel) + [sel[-1]]
        T = max(r[2] for r in rows) + 1
        T = min(seqlen_cap, max(64, int(2 ** math.ceil(math.log2(T)))))
        ids = torch.zeros(len(rows), T, dtype=torch.long)
        lm = torch.zeros(len(rows), T, dtype=torch.long)
        for b, (seq, nc, nt) in enumerate(rows):
            seq = seq[:T]
            ids[b, : len(seq)] = torch.tensor(seq)
            lvl = pool.get_levels_mask(seq)
            lm[b, : len(seq)] = torch.tensor(lvl[: len(seq)])
            if len(seq) < T:
                lm[b, len(seq)] = 1                   # seal the pad tail into its own patch
        ids, lm = ids.cuda(), lm.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = split_forward(model, ids, lm, attn_impl=attn_impl)
            g = f.glob.gather(1, f.repeat_idx.unsqueeze(-1).expand(-1, -1, f.glob.shape[-1]))
            local_delta = None
            if upd is not None:
                u, _ = upd(g, f.res, f.patch_offset)
                if inject == "local":
                    local_delta = u
                else:
                    g = g + u
            byte_idx = torch.arange(T, device=ids.device).unsqueeze(0).expand(len(rows), -1)
            logits = decode_from_globals(model, f, g, repeat_idx=byte_idx,
                                         local_delta=local_delta, attn_impl=attn_impl)
        for b, (seq, nc, nt) in enumerate(rows):
            nt = min(nt, T)
            lp = F.log_softmax(logits[b, nc - 1 : nt - 1].float(), dim=-1)
            tgt = ids[b, nc:nt]
            s = lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
            if res[sel[b]] is None:
                res[sel[b]] = (s, nt - nc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--updater", default=None, help="path to a *_updater.pt (omit = frozen model)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seqlen_cap", type=int, default=2048)
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag/downstream"))
    ap.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, pool, cfg = load_model(args.ckpt, args.tokenizer_path)
    trans = model.transitions[0]

    upd, inject = None, "global"
    if args.updater:
        ck = torch.load(args.updater, map_location="cpu", weights_only=False)
        a = ck["args"]
        upd = PatchLagUpdater(trans.dim_in, trans.dim_out, kind=a["kind"], summary=a["summary"],
                              gate=a["gate"], scope=a["scope"], use_gread=a["use_gread"],
                              use_offset=a.get("use_offset", False),
                              d_offset=a.get("d_offset", 32),
                              ctx=a["ctx"], d_head=a["d_head"], hidden=a.get("hidden", 0),
                              out_dim=trans.dim_in if a.get("inject") == "local" else trans.dim_out
                              ).cuda().to(torch.float32).eval()
        upd.load_state_dict(ck["state"])
        inject = a.get("inject", "global")
        print(f"[{args.tag}] updater {a['kind']}/{a['gate']} from step {ck['step']}", flush=True)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in args.tasks.split(","):
        reqs = build_requests(task, args.limit)
        pairs, spans = [], []
        for r in reqs:
            spans.append((len(pairs), len(r["conts"])))
            pairs += [(r["ctx"], c) for c in r["conts"]]
        t0 = time.time()
        sc = score(model, pool, upd, pairs, args.seqlen_cap, args.bs, inject=inject)
        rows = []
        for r, (off, n) in zip(reqs, spans):
            lp = [sc[off + k][0] for k in range(n)]
            nb = [max(1, sc[off + k][1]) for k in range(n)]
            norm = [lp[k] / nb[k] for k in range(n)]
            rows.append({"gold": r["gold"], "lp": lp, "lp_norm": norm,
                         "pred": int(np.argmax(lp)), "pred_norm": int(np.argmax(norm))})
        acc = float(np.mean([x["pred"] == x["gold"] for x in rows]))
        accn = float(np.mean([x["pred_norm"] == x["gold"] for x in rows]))
        n = len(rows)
        se = math.sqrt(acc * (1 - acc) / n)
        sen = math.sqrt(accn * (1 - accn) / n)
        summary[task] = {"n": n, "acc": acc, "acc_stderr": se,
                         "acc_norm": accn, "acc_norm_stderr": sen,
                         "metric": TASKS.get(task, "acc"), "sec": time.time() - t0}
        with open(outdir / f"{args.tag}_{task}.jsonl", "w") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
        print(f"[{args.tag}] {task:14s} n={n:5d} acc={acc:.4f}±{se:.4f} "
              f"acc_norm={accn:.4f}±{sen:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    key = [summary[t][TASKS.get(t, "acc")] for t in summary]
    summary["avg3"] = float(np.mean(key))
    print(f"[{args.tag}] avg3 = {summary['avg3']:.4f}", flush=True)
    with open(outdir / f"{args.tag}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()

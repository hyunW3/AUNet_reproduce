#!/usr/bin/env python3
"""Held-out BPB + downstream for the byte-matched llama (subword) baseline.

BPB is per BYTE, so a subword model's token-level loss cannot be converted with an assumed
bytes/token ratio -- it has to be measured: sum the token NLL over a window and divide by the
window's real UTF-8 byte count.  That makes the number directly comparable to the byte models
scored by eval_sp_models.py on the same windows.

Downstream uses the same lm-eval requests as the byte arms, so the prompts are identical; only the
tokenisation differs, which is the point of the comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))

from lingua.args import dataclass_from_dict  # noqa: E402
from lingua.checkpoint import CONSOLIDATE_NAME, consolidate_checkpoints  # noqa: E402
from lingua.tokenizer import build_tokenizer  # noqa: E402
from apps.main.transformer import LMTransformer, LMTransformerArgs  # noqa: E402

LN2 = math.log(2.0)
TASKS = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "boolq": "acc"}


def load(ckpt_dir, tok_path=None):
    p = Path(ckpt_dir)
    if not (p / CONSOLIDATE_NAME).exists():
        p = Path(consolidate_checkpoints(str(p)))
    cfg = OmegaConf.load(p / "params.json")
    model = LMTransformer(dataclass_from_dict(LMTransformerArgs, cfg.model, strict=False))
    st = torch.load(p / CONSOLIDATE_NAME, weights_only=True, map_location="cpu")
    model.load_state_dict(st["model"]); del st
    model = model.to(dtype=torch.bfloat16).cuda().eval()
    tk = cfg.data.tokenizer
    tok = build_tokenizer(tk.name, tok_path or tk.path)
    return model, tok, cfg


@torch.no_grad()
def score(model, rows, bs, max_len):
    """rows: [(ids, n_ctx)] -> [(sum logprob over ids[n_ctx:], n_scored_tokens)]"""
    out = [None] * len(rows)
    order = np.argsort([-len(r[0]) for r in rows])
    for i in range(0, len(order), bs):
        sel = list(order[i : i + bs])
        T = min(max_len, max(len(rows[j][0]) for j in sel))
        x = torch.zeros(len(sel), T, dtype=torch.long)
        for b, j in enumerate(sel):
            s = rows[j][0][:T]
            x[b, : len(s)] = torch.tensor(s)
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, attn_impl="sdpa")
        for b, j in enumerate(sel):
            ids, nc = rows[j][0], rows[j][1]
            nt = min(len(ids), T)
            if nt <= nc:
                out[j] = out[j] or (0.0, 0)
                continue
            # 128k vocab: materialising log_softmax over the whole slice costs hundreds of MB,
            # so take the gathered logit minus logsumexp instead -- same value, one column wide.
            sl = logits[b, nc - 1 : nt - 1].float()
            tgt = x[b, nc:nt]
            s = (sl.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - sl.logsumexp(-1)).sum().item()
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
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag/sp_eval"))
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model, tok, cfg = load(args.ckpt, args.tokenizer_path)
    max_len = int(cfg.model.max_seqlen)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    res = {"ckpt": args.ckpt, "family": "llama", "max_seqlen": max_len}
    print(f"[{args.tag}] llama max_seqlen={max_len}", flush=True)

    # ---- held-out BPB: token NLL summed, divided by the window's real byte count -------------
    wins = [json.loads(l) for l in open(args.windows)]
    if args.limit:
        wins = wins[: args.limit]
    tot_nll, tot_bytes, rows, byts = 0.0, 0, [], []
    for w in wins:
        clip = w["text"].encode()[: args.max_bytes].decode("utf-8", errors="ignore")
        ids = tok.encode(clip, add_bos=True, add_eos=False)
        if len(ids) < 2:
            continue
        rows.append((ids[:max_len], 1))          # score everything after BOS
        byts.append(len(clip.encode()))
    t0 = time.time()
    sc = score(model, rows, args.bs, max_len)
    trunc = 0
    for (s, n), (ids, _), nb in zip(sc, rows, byts):
        tot_nll += -s
        tot_bytes += nb
        trunc += int(len(ids) >= max_len)
    res["bpb"] = tot_nll / (LN2 * tot_bytes)
    res["bpb_bytes"] = tot_bytes
    res["windows_truncated"] = trunc
    print(f"[{args.tag}] held-out BPB = {res['bpb']:.4f}  ({tot_bytes} bytes, "
          f"{trunc}/{len(rows)} windows hit the {max_len}-token limit, {time.time() - t0:.0f}s)",
          flush=True)

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
                ci = tok.encode(ctx, add_bos=True, add_eos=False)
                full = tok.encode(ctx + cont, add_bos=True, add_eos=False)
                if len(full) > max_len:                  # keep the continuation, drop old context
                    full = full[-max_len:]
                    ci = ci[-(max_len - (len(full) - len(ci))):] if len(ci) else ci
                rows.append((full, max(1, len(ci))))
        t0 = time.time()
        sc = score(model, rows, args.bs, max_len)
        recs = []
        for off, n, gold in spans:
            lp = [sc[off + k][0] for k in range(n)]
            nb = [max(1, sc[off + k][1]) for k in range(n)]
            nrm = [a / b for a, b in zip(lp, nb)]
            recs.append({"gold": gold, "lp": lp, "lp_norm": nrm,
                         "pred": int(np.argmax(lp)), "pred_norm": int(np.argmax(nrm))})
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
    print(f"[{args.tag}] avg3 = {res['avg3']:.4f}", flush=True)
    json.dump(res, open(outdir / f"{args.tag}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

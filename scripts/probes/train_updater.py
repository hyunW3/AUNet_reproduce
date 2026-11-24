#!/usr/bin/env python3
"""Stage 1: train a lightweight patch-lag residual updater on a FROZEN AU-Net checkpoint.

Stage -1 (`patch_lag_oracle.py`) killed the distillation route: the Scratchpad teacher -- the trunk
output for a vector re-pooled at the current byte -- is out of distribution for a frozen trunk
(+0.62 BPB), and interpolating toward it is monotonically harmful from beta=0.  So there is no
usable teacher, and the objective here is the TASK loss alone (the plan's Sec 8.2), with the
updater free to find whatever direction actually helps.

The updater is zero-initialised and gated, and is applied ONLY where the read global vector is
patch_offset, so step 0 is the unmodified model bit-for-bit and every position whose vector is fresh is
untouched by construction.  Only the ~1-2M updater parameters get gradients; the backbone is
frozen and its encoder/trunk half runs under no_grad (the loss reaches the updater through the
up-projection and the 3-layer local decoder only).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_ablation import load_model, prep, LN2  # noqa: E402
from apps.aunet.patch_lag import (  # noqa: E402
    split_forward, decode_from_globals, teacher_globals, PatchLagUpdater,
)

SHINGLE = 64


def eval_shingles(wins):
    """Every 64-byte shingle of the eval windows, so training documents that overlap them can be
    dropped -- the eval set must stay held out."""
    out = set()
    for w in wins:
        b = w["text"].encode()
        for i in range(0, len(b) - SHINGLE, SHINGLE):
            out.add(hash(b[i : i + SHINGLE]))
    return out


def doc_stream(paths, shingles, win_bytes, seed=0):
    """Yield held-in text windows of <= win_bytes bytes, cut at UTF-8 boundaries."""
    rng = random.Random(seed)
    buf = b""
    files = list(paths)
    rng.shuffle(files)
    for path in files:
        with open(path, "rb") as fh:
            for line in fh:
                try:
                    txt = json.loads(line)["text"]
                except Exception:
                    continue
                b = txt.encode()
                if len(b) >= SHINGLE and any(
                    hash(b[i : i + SHINGLE]) in shingles
                    for i in range(0, min(len(b), 4096) - SHINGLE, SHINGLE)
                ):
                    continue                     # overlaps the eval set -> drop the document
                buf += b + b"\n"
                while len(buf) >= win_bytes:
                    cut = buf[:win_bytes].decode("utf-8", errors="ignore")
                    yield cut
                    buf = buf[win_bytes:]


def batchify(model, pool, texts, seqlen, max_bytes):
    rows = [p for p in (prep(t, pool, max_bytes, seqlen) for t in texts) if p is not None]
    if len(rows) < 2:
        return None
    ids = torch.stack([r[0] for r in rows]).cuda()
    lm = torch.stack([r[1] for r in rows]).cuda()
    return ids, lm, [r[2] for r in rows], rows


def teacher_at_random_offset(model, f, max_offset, gen, valid):
    """One Scratchpad teacher pass per step, at a single staleness sampled from the batch.

    The exact teacher needs one trunk pass per offset (15 of them), which is ~7x the cost of the
    whole training step.  Sampling ONE offset per step, weighted by how many bytes actually sit at
    that staleness, gives the same expected gradient for a per-position regression at 1/15 the
    cost -- every offset is covered over a few hundred steps.
    -> (g_teacher [B,T,d] at the selected positions, bool mask [B,T]) or (None, None)
    """
    counts = torch.stack([((f.patch_offset == r) & valid).sum()
                          for r in range(1, max_offset + 1)]).float()
    if counts.sum() == 0:
        return None, None
    r = int(torch.multinomial(counts / counts.sum(), 1, generator=gen).item()) + 1
    gt = teacher_globals(model, f, r)                       # [B,P,d]
    gt = gt.gather(1, f.repeat_idx.unsqueeze(-1).expand(-1, -1, gt.shape[-1]))
    return gt, ((f.patch_offset == r) & valid)


def repr_loss_fn(g_hat, g_teacher, g_read, sel, lambda_cos):
    """Normalised MSE + cosine on the CORRECTED vector vs the teacher, at `sel` only."""
    a = g_hat[sel].float()
    b = g_teacher[sel].float()
    scale = g_read[sel].float().pow(2).mean().clamp(min=1e-6)
    mse = (a - b).pow(2).mean() / scale
    cos = F.cosine_similarity(a, b, dim=-1).mean()
    return mse + lambda_cos * (1 - cos), float(mse), float(cos)


def run_batch(model, upd, ids, lm, ns, attn_impl="fmha", grad=True, cheat=False,
              objective="task", lambda_repr=1.0, lambda_cos=1.0, max_offset=15, gen=None,
              inject="global"):
    """-> (loss, per-byte nll, patch_offset, aux)"""
    with torch.no_grad():
        f = split_forward(model, ids, lm, attn_impl=attn_impl)
        T = f.res.shape[1]
        byte_idx = torch.arange(T, device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        g_base = f.glob.gather(1, f.repeat_idx.unsqueeze(-1).expand(-1, -1, f.glob.shape[-1]))
        extra = None
        if cheat:
            # POSITIVE CONTROL: hand the updater the byte it is about to predict.  If the loss does
            # not collapse, the wiring (gradient path, masking, decode) is broken -- so a null
            # result from the honest features would be uninterpretable.
            nxt = torch.roll(ids, -1, dims=1)
            extra = model.tok_embeddings(nxt).to(f.res.dtype)
        g_teacher, sel = (None, None)
        if objective != "task":
            # the zero-padded tail is sealed into its own patch by prep(); keep it out of the
            # regression or ~15 pad bytes per row would enter the target every step
            valid = (torch.arange(f.res.shape[1], device=ids.device).unsqueeze(0)
                     < torch.tensor(ns, device=ids.device).unsqueeze(1))
            g_teacher, sel = teacher_at_random_offset(model, f, max_offset, gen, valid)
    with torch.set_grad_enabled(grad):
        upd_vec, alpha = upd(g_base, f.res, f.patch_offset, extra=extra)
        g_hat = g_base + upd_vec if inject == "global" else g_base
        local_delta = upd_vec if inject == "local" else None
        nlls, tot, cnt = [], 0.0, 0
        task = None
        if objective != "distill" or not grad:
            logits = decode_from_globals(model, f, g_hat, repeat_idx=byte_idx,
                                         local_delta=local_delta, attn_impl=attn_impl)
            for i, n in enumerate(ns):
                ce = F.cross_entropy(logits[i, :n].float(), ids[i, 1 : n + 1], reduction="none")
                nlls.append(ce)
                tot = tot + ce.sum()
                cnt += n
            task = tot / cnt
        rep, mse, cos = None, float("nan"), float("nan")
        if inject == "local" and g_teacher is not None:
            g_teacher = None                 # no teacher for a local-space update
        if g_teacher is not None and bool(sel.any()):
            rep, mse, cos = repr_loss_fn(g_hat, g_teacher, g_base, sel, lambda_cos)
        if objective == "task":
            loss = task
        elif objective == "distill":
            loss = rep if rep is not None else task * 0.0
        else:
            loss = task + lambda_repr * (rep if rep is not None else 0.0)
    aux = {
        "repr_mse": mse, "teacher_cos": cos,
        "alpha": float(alpha.mean()),
        "upd_norm": float(upd_vec.detach().float().norm(dim=-1).mean()),
        "rel": float((upd_vec.detach().float().norm(dim=-1)
                      / (g_base if inject == "global" else f.res).float()
                      .norm(dim=-1).clamp(min=1e-6)).mean()),
    }
    return loss, nlls, f.patch_offset, aux


@torch.no_grad()
def evaluate(model, upd, prepped, attn_impl="fmha", bs=2, cheat=False, max_offset=0, gen=None,
             inject="global"):
    nll, st, coss = [], [], []
    for i in range(0, len(prepped), bs):
        rows = prepped[i : i + bs]
        if len(rows) < 2:
            rows = rows + [rows[-1]]
        ids = torch.stack([r[0] for r in rows]).cuda()
        lm = torch.stack([r[1] for r in rows]).cuda()
        ns = [r[2] for r in rows]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, nlls, patch_offset, aux = run_batch(
                model, upd, ids, lm, ns, attn_impl, grad=False, cheat=cheat,
                objective="task" if not max_offset else "combined",
                max_offset=max_offset or 15, gen=gen, inject=inject)
            if max_offset and aux["teacher_cos"] == aux["teacher_cos"]:
                coss.append(aux["teacher_cos"])
        for j in range(min(len(prepped) - i, bs)):
            nll.append(nlls[j].float().cpu().numpy())
            st.append(patch_offset[j, : ns[j]].cpu().numpy())
    if coss:
        return np.concatenate(nll), np.concatenate(st), float(np.mean(coss))
    return np.concatenate(nll), np.concatenate(st)


def report(nll, st, base_nll=None, prefix=""):
    b = nll.mean() / LN2
    line = f"{prefix}BPB={b:.4f}"
    if base_nll is not None:
        line += f" (base {base_nll.mean() / LN2:.4f}, delta {b - base_nll.mean() / LN2:+.4f})"
    m = st >= 1
    line += f" | offset>=1 BPB={nll[m].mean() / LN2:.4f}"
    if base_nll is not None:
        line += f" (delta {(nll[m].mean() - base_nll[m].mean()) / LN2:+.4f})"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/dclm_baseline_1.0_2shards_shuffled"),
                    help="ROOT/data by default; pass explicitly on a tree without a data/ dir")
    ap.add_argument("--data_glob", default="*chunk.1*.jsonl")
    ap.add_argument("--windows", default=str(ROOT / "reports/bpb_windows.jsonl"))
    ap.add_argument("--eval_windows", type=int, default=64)
    ap.add_argument("--win_bytes", type=int, default=3840)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--kind", default="linear", choices=["linear", "attn", "gread"])
    ap.add_argument("--summary", default="mean_last", choices=["last", "mean", "mean_last"])
    ap.add_argument("--gate", default="scalar", choices=["fixed", "scalar", "input", "vector"])
    ap.add_argument("--scope", default="patch", choices=["patch", "prefix"])
    ap.add_argument("--use_gread", action="store_true")
    ap.add_argument("--use_offset", action="store_true",
                    help="hand patch_offset to the head as an explicit feature (16+1 buckets), "
                         "not only as the mask and the attention window")
    ap.add_argument("--d_offset", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--d_head", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=0,
                    help="capacity knob: 0 = single projection, h > 0 = Linear-GELU-Linear")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default=str(ROOT / "reports/patch_lag"))
    ap.add_argument("--tokenizer_path", default=os.environ.get("TOKENIZER_PATH"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject", default="global", choices=["global", "local"],
                    help="where the update lands: on the patch vector (global) or after the "
                         "up-projection, beside the byte residual (local)")
    ap.add_argument("--objective", default="task", choices=["task", "distill", "combined"],
                    help="distill/combined regress the updater onto the Scratchpad teacher "
                         "residual g_teacher - g_read (plan Sec 8.1)")
    ap.add_argument("--lambda_repr", type=float, default=1.0)
    ap.add_argument("--lambda_cos", type=float, default=1.0)
    ap.add_argument("--max_offset", type=int, default=15)
    ap.add_argument("--cheat", action="store_true",
                    help="positive control: also feed the NEXT byte's embedding to the updater")
    ap.add_argument("--eval_only", default=None,
                    help="path to a saved *_updater.pt: score all --windows and dump an npz that "
                         "report_patch_lag.py can read (nll_full + nll_updater + patch_offset)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model, pool, cfg = load_model(args.ckpt, args.tokenizer_path)
    for p in model.parameters():
        p.requires_grad_(False)
    seqlen = int(model.transitions[0].seqlen_in)
    trans = model.transitions[0]

    upd = PatchLagUpdater(trans.dim_in, trans.dim_out, kind=args.kind, summary=args.summary,
                          gate=args.gate, scope=args.scope, use_gread=args.use_gread,
                          use_offset=args.use_offset, d_offset=args.d_offset,
                          ctx=args.ctx, d_head=args.d_head, hidden=args.hidden,
                          extra_dim=trans.dim_in if args.cheat else 0,
                          out_dim=trans.dim_in if args.inject == "local" else trans.dim_out
                          ).cuda().to(torch.float32)
    n_par = sum(p.numel() for p in upd.parameters())
    print(f"[{args.tag}] updater params={n_par / 1e6:.3f}M  kind={args.kind} gate={args.gate} "
          f"summary={args.summary} scope={args.scope} gread={args.use_gread} "
          f"offset={args.use_offset} inject={args.inject} out_dim={upd.out_dim}", flush=True)

    wins = [json.loads(l) for l in open(args.windows)]
    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        a = ck["args"]
        upd.load_state_dict(ck["state"]); upd.cuda().eval()
        allw = [p for p in (prep(w["text"], pool, args.win_bytes, seqlen) for w in wins)
                if p is not None]
        base_state = {k: v.clone() for k, v in upd.state_dict().items()}
        nll, st = evaluate(model, upd, allw, inject=a.get("inject", "global"))
        with torch.no_grad():                       # baseline = the same code path, delta = 0
            # `proj` is a Linear (hidden=0) or a Sequential ending in one; zero the OUTPUT layer
            (upd.proj[-1] if isinstance(upd.proj, torch.nn.Sequential) else upd.proj).weight.zero_()
            b_nll, b_st = evaluate(model, upd, allw, inject=a.get("inject", "global"))
        upd.load_state_dict(base_state)
        win = np.concatenate([np.full(w[2], i, np.int32) for i, w in enumerate(allw)])
        np.savez_compressed(outdir / f"{args.tag}_eval.npz", patch_offset=st.astype(np.int16),
                            window=win, nll_full=b_nll.astype(np.float32),
                            nll_updater=nll.astype(np.float32))
        print(f"[{args.tag}] step={ck['step']} " + report(nll, st, b_nll), flush=True)
        print(f"[{args.tag}] wrote {outdir / (args.tag + '_eval.npz')}", flush=True)
        return
    ev = [p for p in (prep(w["text"], pool, args.win_bytes, seqlen)
                      for w in wins[: args.eval_windows]) if p is not None]
    shing = eval_shingles(wins)
    files = sorted(Path(args.data).glob(args.data_glob))
    assert files, f"no training shards matched {args.data}/{args.data_glob}"
    print(f"[{args.tag}] train shards={[f.name for f in files]} eval windows={len(ev)}", flush=True)

    base_nll, base_st = evaluate(model, upd, ev, cheat=args.cheat, inject=args.inject)
    print(f"[{args.tag}] step 0 (zero-init, must equal baseline): "
          + report(base_nll, base_st), flush=True)

    opt = torch.optim.AdamW(upd.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup))
        * (0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    gen = torch.Generator(device="cuda"); gen.manual_seed(args.seed)
    stream = doc_stream(files, shing, args.win_bytes, seed=args.seed)
    log = open(outdir / f"{args.tag}_train.jsonl", "w")
    t0, step, seen = time.time(), 0, 0
    best = None
    while step < args.steps:
        texts = [next(stream) for _ in range(args.bs)]
        b = batchify(model, pool, texts, seqlen, args.win_bytes)
        if b is None:
            continue
        ids, lm, ns, _ = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _, aux = run_batch(
                model, upd, ids, lm, ns, cheat=args.cheat, objective=args.objective,
                lambda_repr=args.lambda_repr, lambda_cos=args.lambda_cos,
                max_offset=args.max_offset, gen=gen, inject=args.inject)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(upd.parameters(), args.clip)
        opt.step(); sched.step()
        step += 1; seen += sum(ns)
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss_bpb": float(loss) / LN2, "alpha": aux["alpha"],
                   "teacher_cos": aux["teacher_cos"], "repr_mse": aux["repr_mse"],
                   "upd_norm": aux["upd_norm"], "rel": aux["rel"], "gnorm": float(gn),
                   "lr": sched.get_last_lr()[0], "bytes": seen, "sec": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"  [{args.tag}] {step:6d} train_bpb={rec['loss_bpb']:.4f} "
                  f"alpha={rec['alpha']:.4f} |d|={rec['upd_norm']:.3f} rel={rec['rel']:.4f} "
                  f"cos={rec['teacher_cos']:.4f} gn={rec['gnorm']:.2e} {rec['sec']:.0f}s",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev_out = evaluate(model, upd, ev, cheat=args.cheat,
                              max_offset=args.max_offset if args.objective != "task" else 0,
                              gen=gen, inject=args.inject)
            nll, st = ev_out[0], ev_out[1]
            tcos = ev_out[2] if len(ev_out) > 2 else float("nan")
            print(f"  [{args.tag}] EVAL step {step}: teacher_cos={tcos:.4f} " + report(nll, st, base_nll), flush=True)
            log.write(json.dumps({"step": step, "eval_bpb": float(nll.mean() / LN2),
                                  "eval_bpb_stale": float(nll[st >= 1].mean() / LN2),
                                  "base_bpb": float(base_nll.mean() / LN2),
                                  "base_bpb_stale": float(base_nll[base_st >= 1].mean() / LN2)})
                      + "\n"); log.flush()
            if best is None or nll.mean() < best[0]:
                best = (nll.mean(), step)
                torch.save({"args": vars(args), "state": upd.state_dict(), "step": step},
                           outdir / f"{args.tag}_updater.pt")
    log.close()
    print(f"[{args.tag}] done. best eval nll={best[0]:.5f} at step {best[1]}; "
          f"{seen / 1e6:.1f}M bytes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

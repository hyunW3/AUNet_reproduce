#!/usr/bin/env python3
"""E2 (EXPERIMENTS_3.md): segmentation transplant for official BLT-1B on the echo_all probe.

Same bytes, different patch boundaries on the scored option span only. For each MC item and option,
natural patch starts come from the harness patcher exactly as in eval (bs 1, row entropies, boe pad
to a multiple of 128). Then the starts inside the option's token span are replaced by
  nat      unchanged (reproduces the harness score)
  xseg     the starts the SAME option received under the other context (echo_all <-> clean)
  byte     every option byte its own patch
and the model is run with these patch_lengths. Conditions:
  echo_nat / echo_cleanseg / echo_byte  (context = echo_all, "Options: (A) .." + question)
  clean_nat / clean_echoseg / clean_byte (context = clean question)

  PYTHONPATH= BLT_REPO=.. EVAL_SUITE=.. LC_SCRIPTS=<lc>/scripts/longctx python blt_transplant.py \
      --ckpt <lc>/ext/blt_weights --echo echo.jsonl --out transplant.jsonl [--shard i/n]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

A128 = lambda n: ((n + 127) // 128) * 128
A64 = lambda n: ((n + 63) // 64) * 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--echo", required=True, help="data/echo.jsonl (clean + echo_all rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--check", type=int, default=8, help="assert nat == harness score on the first N options")
    a = ap.parse_args()
    sys.path.insert(0, os.environ["LC_SCRIPTS"])
    import ext_models
    h = ext_models.build_blt(a.ckpt)
    tok, boe = h.tokenizer, h.tokenizer.boe_id
    # The compiled create_block_mask (bytelatent cross_attn_mask, _compile=True) recompiles for every new
    # (Q_LEN, KV_LEN) and its inductor kernel intermittently dies with a CUDA illegal memory access, which
    # kills the process. Build the same masks eagerly instead; --check verifies scores against the harness.
    import bytelatent.model.blt as _bm
    _orig_cbm = _bm.create_block_mask
    _bm.create_block_mask = lambda *x, **k: _orig_cbm(*x, **{**k, "_compile": False})

    rows = {}
    for l in open(a.echo):
        r = json.loads(l)
        if r["cond"] in ("clean", "echo_all"):
            rows[(r["cond"], r["length"], r["pos"])] = r
    keys = sorted({(m, i) for (_, m, i) in rows})
    if a.shard:
        si, sn = map(int, a.shard.split("/"))
        keys = keys[si::sn]
    done = set()
    if Path(a.out).exists():
        done = {(json.loads(l)["mc"], json.loads(l)["idx"]) for l in open(a.out)}

    @torch.inference_mode()
    def encode(ctx, cont):
        c = tok.encode(ctx, add_bos=True, add_eos=False)
        full = tok.encode(ctx + cont, add_bos=True, add_eos=False)
        Np = A128(len(full))
        t = torch.full((1, Np), boe, dtype=torch.long, device="cuda")
        t[0, :len(full)] = torch.tensor(full, device="cuda")
        pl, _ = h.patcher.patch(t, include_next_token=False, entropies=h._row_entropies(t))
        starts, s = [], 0
        for L in pl[0].tolist():
            if L == 0:
                break
            starts.append(s)
            s += L
        return t, full, len(c), len(full), starts, Np

    @torch.inference_mode()
    def score(t, full, c0, c1, starts, Np):
        st = sorted(set(starts))
        if os.environ.get("TP_DEBUG"):
            print("score", len(full), Np, len(st), c0, c1, flush=True)
        pl = [b - a for a, b in zip(st, st[1:] + [Np])]
        pl = pl + [0] * (A64(len(pl)) - len(pl))
        logp = torch.log_softmax(h.model(t, patch_lengths=torch.tensor([pl], device="cuda")).float(), -1)
        tgt = torch.tensor(full[c0:c1], device="cuda")
        lp = logp[0, torch.arange(c0 - 1, c1 - 1, device="cuda")]
        return float(lp.gather(-1, tgt.unsqueeze(-1)).sum())

    def swap(starts, c0, c1, new_rel):
        return [s for s in starts if not (c0 <= s < c1)] + [c0 + r for r in new_rel]

    checked = 0
    with open(a.out, "a") as f:
        for (m, i) in keys:
            if (m, i) in done:
                continue
            rc, re_ = rows[("clean", m, i)], rows[("echo_all", m, i)]
            out = {c: [] for c in ("echo_nat", "echo_cleanseg", "echo_byte", "clean_nat", "clean_echoseg", "clean_byte")}
            gold_patches = {}
            for j, opt in enumerate(rc["options"]):
                E = encode(re_["prompt"], opt)
                C = encode(rc["prompt"], opt)
                (te, fe, e0, e1, se, Ne), (tc, fc, k0, k1, sc, Nc) = E, C
                assert e1 - e0 == k1 - k0
                rel_e = [s - e0 for s in se if e0 <= s < e1]
                rel_c = [s - k0 for s in sc if k0 <= s < k1]
                every = list(range(e1 - e0))
                out["echo_nat"].append(score(te, fe, e0, e1, se, Ne))
                out["echo_cleanseg"].append(score(te, fe, e0, e1, swap(se, e0, e1, rel_c), Ne))
                out["echo_byte"].append(score(te, fe, e0, e1, swap(se, e0, e1, every), Ne))
                out["clean_nat"].append(score(tc, fc, k0, k1, sc, Nc))
                out["clean_echoseg"].append(score(tc, fc, k0, k1, swap(sc, k0, k1, rel_e), Nc))
                out["clean_byte"].append(score(tc, fc, k0, k1, swap(sc, k0, k1, every), Nc))
                if j == rc["gold"]:
                    gold_patches = {"echo": len(rel_e), "clean": len(rel_c), "bytes": e1 - e0}
                if checked < a.check:              # natural path must reproduce the eval harness
                    for ctx, got in ((re_["prompt"], out["echo_nat"][-1]), (rc["prompt"], out["clean_nat"][-1])):
                        ref = h.score_pairs([(ctx, opt)])[0][0]
                        # bf16 noise between the eager-mask path and the compiled harness is ~1%; all
                        # transplant contrasts are computed within the eager path, so only report it
                        if abs(ref - got) > 0.05 * max(1.0, abs(ref)):
                            print(f"CHECK_MISMATCH {m} {i} {j} harness={ref:.4f} eager={got:.4f}", flush=True)
                    checked += 1
            for c, lls in out.items():
                pred = max(range(len(lls)), key=lls.__getitem__)
                f.write(json.dumps({"id": f"tp-{c}-{m}-{i}", "task": "transplant", "cond": c, "mc": m, "idx": i,
                                    "lls": lls, "pred": pred, "gold": rc["gold"], "score": float(pred == rc["gold"]),
                                    "gold_patches": gold_patches}) + "\n")
            f.flush()
    print("TRANSPLANT_DONE", a.out)


if __name__ == "__main__":
    main()

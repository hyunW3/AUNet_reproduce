#!/usr/bin/env python3
"""Context-side segmentation transplant for official BLT-1B (follow-up to E2, blt_transplant.py).

echo_all prompt = LISTING ("Options: (A) .. (D) ..\\n") + the clean prompt ("Question: ..\\nAnswer:") + option.
Everything after the listing is byte-identical to the clean prompt, so its patch boundaries can be copied
from the clean prompt exactly. Conditions (same bytes, only patch_lengths differ):
  nat            natural entropy patching everywhere (= the harness echo_all score)
  sfx_clean      question + option boundaries copied from the clean prompt; listing natural
  sfx_list_byte  sfx_clean + every listing byte its own patch
  sfx_list_word  sfx_clean + listing patches split at spaces (one patch per word)
  sfx_list_opt   sfx_clean + one patch per listed option ("(A) gray ")
If BLT's echo loss / first-option bias survives all of them, BLT's segmentation is not the cause.

  PYTHONPATH=<extra> BLT_REPO=.. EVAL_SUITE=.. LC_SCRIPTS=<lc>/scripts/longctx python blt_transplant_ctx.py \
      --ckpt <blt_weights> --echo echo.jsonl --out out.jsonl [--shard i/n]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

A128 = lambda n: ((n + 127) // 128) * 128
A64 = lambda n: ((n + 63) // 64) * 64
CONDS = ("nat", "sfx_clean", "sfx_list_byte", "sfx_list_word", "sfx_list_opt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--echo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--check", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, os.environ["LC_SCRIPTS"])
    import ext_models
    h = ext_models.build_blt(a.ckpt)
    tok, boe = h.tokenizer, h.tokenizer.boe_id
    import bytelatent.model.blt as _bm          # eager block masks (see blt_transplant.py)
    _orig = _bm.create_block_mask
    _bm.create_block_mask = lambda *x, **k: _orig(*x, **{**k, "_compile": False})

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
        for L_ in pl[0].tolist():
            if L_ == 0:
                break
            starts.append(s)
            s += L_
        return t, full, len(c), len(full), starts, Np

    @torch.inference_mode()
    def score(t, full, c0, c1, starts, Np):
        st = sorted(set(starts))
        pl = [b - a_ for a_, b in zip(st, st[1:] + [Np])]
        pl = pl + [0] * (A64(len(pl)) - len(pl))
        logp = torch.log_softmax(h.model(t, patch_lengths=torch.tensor([pl], device="cuda")).float(), -1)
        tgt = torch.tensor(full[c0:c1], device="cuda")
        lp = logp[0, torch.arange(c0 - 1, c1 - 1, device="cuda")]
        return float(lp.gather(-1, tgt.unsqueeze(-1)).sum())

    checked = 0
    with open(a.out, "a") as f:
        for (m, i) in keys:
            if (m, i) in done:
                continue
            rc, re_ = rows[("clean", m, i)], rows[("echo_all", m, i)]
            assert re_["prompt"].endswith(rc["prompt"])
            listing = re_["prompt"][: len(re_["prompt"]) - len(rc["prompt"])]
            lb = listing.encode()
            Lb = len(lb)                                   # listing occupies tokens [1, 1+Lb)
            word = [1] + [1 + j for j in range(1, Lb) if lb[j - 1:j] == b" "]
            opt = [1] + [1 + j for j in range(Lb) if lb[j:j + 1] == b"(" and j + 2 < Lb and lb[j + 2:j + 3] == b")"]
            lists = {"nat": None, "byte": list(range(1, 1 + Lb)), "word": word, "opt": opt}
            out = {c: [] for c in CONDS}
            diag = []                                          # per option: boundaries changed by sfx_clean
            for opt_text in rc["options"]:
                te, fe, e0, e1, se, Ne = encode(re_["prompt"], opt_text)
                _, fc, k0, k1, sc, _ = encode(rc["prompt"], opt_text)
                n = len(fe)
                assert n == len(fc) + Lb
                sfx = [s + Lb for s in sc if 1 <= s < len(fc)]          # clean boundaries, shifted
                base = [s for s in se if s < 1 + Lb or s >= n]          # natural listing + padding
                nat_sfx = {s for s in se if 1 + Lb <= s < n}
                diag.append({"sfx_nat": len(nat_sfx), "sfx_clean": len(sfx), "changed": len(nat_sfx ^ set(sfx)),
                             "list_nat": len([s for s in se if 1 <= s < 1 + Lb])})
                out["nat"].append(score(te, fe, e0, e1, se, Ne))
                out["sfx_clean"].append(score(te, fe, e0, e1, base + sfx, Ne))
                pad = [s for s in se if s >= n]
                for c, key in (("sfx_list_byte", "byte"), ("sfx_list_word", "word"), ("sfx_list_opt", "opt")):
                    out[c].append(score(te, fe, e0, e1, [0] + lists[key] + sfx + pad, Ne))
                if checked < a.check:
                    ref = h.score_pairs([(re_["prompt"], opt_text)])[0][0]
                    if abs(ref - out["nat"][-1]) > 0.05 * max(1.0, abs(ref)):
                        print(f"CHECK_MISMATCH {m} {i} harness={ref:.4f} eager={out['nat'][-1]:.4f}", flush=True)
                    checked += 1
            for c, lls in out.items():
                pred = max(range(len(lls)), key=lls.__getitem__)
                f.write(json.dumps({"id": f"tpc-{c}-{m}-{i}", "task": "transplant_ctx", "cond": c, "mc": m,
                                    "idx": i, "lls": lls, "pred": pred, "gold": rc["gold"],
                                    "score": float(pred == rc["gold"]), "seg": diag}) + "\n")
            f.flush()
    print("TRANSPLANT_CTX_DONE", a.out)


if __name__ == "__main__":
    main()

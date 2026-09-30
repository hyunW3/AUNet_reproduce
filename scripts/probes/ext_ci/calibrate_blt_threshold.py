#!/usr/bin/env python3
"""Find the BLT entropy threshold that gives a target bytes/patch on the main-table B/patch inputs.

Target definition = the B/patch column of tab:main_13b: pooled bytes / patches over the clean
inputs (context + gold answer, BOS excluded) of HellaSwag, ARC-E, ARC-C, PIQA, WinoGrande
(reports/patch_stats/texts.jsonl, group "downstream"). Entropies come from the official entropy
model through the eval harness call (BLTHarness._row_entropies, xformers, 512-byte window), computed
once; the threshold is then bisected offline with the eval patching rule (a patch starts at byte
p+1 when entropy[p] > theta; BOS and the first byte always start one). DCLM windows are reported
at the chosen threshold for reference.

  cd blt_official && PYTHONPATH=$PWD EVAL_SUITE=../eval_suite AUNET_LINGUA=../lingua \
    ../blt_venv/bin/python ../calibrate_blt_threshold.py --texts ../texts_clean5.jsonl \
      --windows ../bpb_windows.jsonl --target 4.65 --out ../out/blt_threshold_calibration.json
"""
import argparse, json, os, sys

import torch


def entropies(lm, text):
    ids = lm.tokenizer.encode(text, add_bos=True, add_eos=False)[-4096:]   # eval left-truncation
    n = len(ids)
    toks = torch.full((1, ((n + 127) // 128) * 128), lm.tokenizer.boe_id, dtype=torch.long, device="cuda")
    toks[0, :n] = torch.tensor(ids, device="cuda")
    return lm._row_entropies(toks)[0, :n].float().cpu()


def patches(ent, theta):
    """#patches starting on a real byte (token 1 .. n-1) under the eval rule."""
    n = ent.numel()
    return 1 + int((ent[1:n - 1] > theta).sum())      # token 1 always starts; p>=1 -> start at p+1 <= n-1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--target", type=float, default=4.65)
    ap.add_argument("--weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_ext import build_blt_official
    lm = build_blt_official(a.weights, max_len=4096)
    T5 = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande")
    rows = [json.loads(l) for l in open(a.texts)]
    rows = [r for r in rows if r["group"] == "downstream" and r["bench"] in T5]
    E = [entropies(lm, r["text"]) for r in rows]
    nbytes = sum(e.numel() - 1 for e in E)
    W = [entropies(lm, json.loads(l)["text"]) for l in open(a.windows)]
    wbytes = sum(e.numel() - 1 for e in W)
    bpp = lambda es, nb, th: nb / sum(patches(e, th) for e in es)

    lo, hi = 0.5, 6.0                  # bytes/patch rises monotonically with theta
    for _ in range(40):
        mid = (lo + hi) / 2
        if bpp(E, nbytes, mid) < a.target:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2
    per = {t: bpp([e for e, r in zip(E, rows) if r["bench"] == t],
                  sum(e.numel() - 1 for e, r in zip(E, rows) if r["bench"] == t), theta) for t in T5}
    res = {"target": a.target, "theta": theta,
           "clean5_bytes_per_patch": bpp(E, nbytes, theta),
           "clean5_per_task": per,
           "dclm_bytes_per_patch": bpp(W, wbytes, theta),
           "released_theta": 1.335442066192627,
           "clean5_at_released": bpp(E, nbytes, 1.335442066192627),
           "dclm_at_released": bpp(W, wbytes, 1.335442066192627),
           "n_items": len(E), "n_dclm_windows": len(W)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

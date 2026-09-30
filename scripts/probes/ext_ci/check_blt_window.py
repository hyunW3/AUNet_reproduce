#!/usr/bin/env python3
"""Verify the official BLT entropy patcher runs with its 512-byte sliding window in the evaluation path.

Uses exactly the calls the eval harness makes (eval_suite/blt_eval/harness.py::BLTHarness):
_row_entropies(toks) -> patcher.patch(toks, include_next_token=False, entropies=...), on the 320
DCLM held-out windows (3,840 B each). Reports
  - bytes/patch over real bytes (BOS patch and padding excluded)
  - mean next-byte entropy and share above the threshold by position bin (flat past 512 => window on)
for the default path (attn_impl from the released config) and, as a control, with the window
removed (attn_impl="sdpa" + BLT_SUPPRESS_ATTN_ERROR=1 -> plain causal, the paper-time behaviour).

  cd blt_official && PYTHONPATH=$PWD EVAL_SUITE=../eval_suite MASTER_ADDR=127.0.0.1 MASTER_PORT=29868 \
    BLT_SUPPRESS_ATTN_ERROR=1 ../blt_venv/bin/python ../check_blt_window.py --windows ../bpb_windows.jsonl
"""
import argparse, json, os, sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_ext import build_blt_official
    lm = build_blt_official(a.weights, max_len=4096)
    ent = lm.patcher.entropy_model
    print(f"entropy model: attn_impl={ent.attn_impl} attn_bias_type={ent.attn_bias_type} "
          f"sliding_window={ent.sliding_window}  threshold={lm.patcher.threshold}", flush=True)
    orig = ent.forward
    wins = [json.loads(l)["text"] for l in open(a.windows)]
    bins = [(1, 512), (512, 768), (768, 1024), (1024, 2048), (2048, 3840)]
    report = {}
    for label, impl in (("default (released config)", None), ("control: window removed (sdpa)", "sdpa")):
        ent.forward = (lambda t, *x, **k: orig(t, *x, **{**k, "attn_impl": impl})) if impl else orig
        tb = tp = 0
        H = torch.zeros(len(bins)); A = torch.zeros(len(bins)); C = torch.zeros(len(bins))
        for w in wins:
            ids = lm.tokenizer.encode(w, add_bos=True, add_eos=False)
            n = len(ids)
            toks = torch.full((1, ((n + 127) // 128) * 128), lm.tokenizer.boe_id, dtype=torch.long, device="cuda")
            toks[0, :n] = torch.tensor(ids, device="cuda")
            e = lm._row_entropies(toks)
            pl, _ = lm.patcher.patch(toks, include_next_token=False, entropies=e)
            pl = pl[0].tolist()
            # patch starts over token positions; count patches that start on a real byte (1 .. n-1)
            starts, pos = [], 0
            for L in pl:
                if L > 0:
                    starts.append(pos)
                pos += L
            real = [s for s in starts if 1 <= s < n]
            tb += n - 1; tp += len(real)
            for j, (lo, hi) in enumerate(bins):
                seg = e[0, lo:min(hi, n)]
                H[j] += seg.sum().item(); A[j] += (seg > lm.patcher.threshold).sum().item(); C[j] += seg.numel()
        r = {"bytes_per_patch": tb / tp,
             "by_position": {f"{lo}-{hi}": {"mean_entropy": round((H[j] / C[j]).item(), 3),
                                             "frac_above_threshold": round((A[j] / C[j]).item(), 3)}
                             for j, (lo, hi) in enumerate(bins)}}
        report[label] = r
        print(f"\n[{label}] DCLM bytes/patch = {r['bytes_per_patch']:.3f}", flush=True)
        for k, v in r["by_position"].items():
            print(f"   pos {k:>9s}: mean H {v['mean_entropy']:.3f}   >thr {v['frac_above_threshold']:.3f}")
    ent.forward = orig
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

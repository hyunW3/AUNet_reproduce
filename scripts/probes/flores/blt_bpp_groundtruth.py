#!/usr/bin/env python3
"""Decisive bytes/patch check: call the SAME patcher.patch() the eval harness calls.

The repo disagrees with itself on what theta=1.335442 realizes on the DCLM windows:
    4.071  reports_afterAAAIsub/flores/raw/blt_theta_native{,_v2}.json   (rule re-implemented)
    1.151  reports/latency_realtext/blt_windows_patchstats.json          (in-repo EntropyPatcher)
    1.151  reports_afterAAAIsub/flores/raw/blt_theta.json                (itazap conversion)
    1.134  reports/latency_realtext/blt_th1.335442.json                  ("compression")
Both 4.071 numbers come from code that re-derives `starts = {0,1} u {t : H[t] > theta}` by hand.
That is exactly the step that could be wrong, so re-running it proves nothing. This script instead
drives `patcher.patch()` -- the call eval_suite/blt_eval/harness.py makes -- and counts the patch
lengths it returns. Whatever this says is what the model actually saw.

  $BLT_PY blt_bpp_groundtruth.py --attn xformers --out gt.json
"""
import argparse, json, os, sys
import torch

ROOT = "/mnt/ssd2/hyun2/AUNet"
BLT_REPO = os.environ.get("BLT_REPO", "/mnt/ssd2/hyun2/lzspacebyte/blt")
sys.path.insert(0, BLT_REPO)
CKPT = f"{BLT_REPO}/hf-weights/blt_1b"
ENTROPY_CKPT = f"{BLT_REPO}/hf-weights/entropy_model"


def patch_attn(impl):
    import json as _json
    import bytelatent.data.patcher as _p
    from bytelatent.transformer import LMTransformer, LMTransformerArgs

    def load_entropy_model(d, sd, device="cpu"):
        with open(os.path.join(d, "params.json")) as fr:
            r = _json.loads(fr.read())
        torch.set_default_dtype(torch.bfloat16)
        mp = r["entropy_model"]
        a = LMTransformerArgs(dim=mp["dim"], n_layers=mp["n_layers"], n_heads=mp["n_heads"],
                              max_seqlen=mp["max_seqlen"], ffn_dim_multiplier=mp["ffn_dim_multiplier"],
                              vocab_size=mp["vocab_size"], attn_bias_type="local_block_causal",
                              attn_impl=impl, sliding_window=512)
        m = LMTransformer(a)
        m.load_state_dict(torch.load(sd, map_location=device)["model"], strict=False)
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
        return m, a
    _p.load_entropy_model = load_entropy_model
    print(f"[native] entropy attn -> {impl}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="xformers", choices=["xformers", "sdpa"])
    ap.add_argument("--windows", default=f"{ROOT}/reports/bpb_windows.jsonl")
    ap.add_argument("--n_windows", type=int, default=320)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--thresholds", default="1.335442066192627,1.46875")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    patch_attn(args.attn)
    from bytelatent.args import TrainArgs
    train_cfg = TrainArgs.model_validate_json(open(os.path.join(CKPT, "params.json")).read())
    tok = train_cfg.data.tokenizer_args.build()
    windows = [json.loads(l) for l in open(args.windows)][:args.n_windows]

    out = {"attn_impl": args.attn, "n_windows": len(windows),
           "method": "patcher.patch(include_next_token=False), patch_lengths counted directly",
           "arms": {}}

    for th in [float(x) for x in args.thresholds.split(",")]:
        pa = train_cfg.data.patcher_args.model_copy(deep=True)
        pa.realtime_patching = True
        pa.entropy_model_checkpoint_dir = ENTROPY_CKPT
        pa.threshold = th
        patcher = pa.build()

        tot_b = tot_p = 0
        lens = []
        with torch.inference_mode():
            for i, w in enumerate(windows):
                ids = tok.encode(w["text"], add_bos=True, add_eos=False)[:args.max_len]
                x = torch.tensor([ids], device="cuda")
                pl, _ = patcher.patch(x, include_next_token=False)
                row = pl[0]
                nz = int((row > 0).sum())           # trailing zero-length patches are padding
                assert int(row.sum()) == len(ids), f"rows must sum to seq_len: {int(row.sum())} vs {len(ids)}"
                tot_b += len(ids)
                tot_p += nz
                lens.extend(row[row > 0].tolist())
                if (i + 1) % 80 == 0:
                    print(f"  θ={th:.6f}  ..{i+1}/{len(windows)}  running B/patch={tot_b/tot_p:.4f}", flush=True)
        lens_t = torch.tensor(lens, dtype=torch.float)
        out["arms"][f"{th:.6f}"] = {
            "threshold": th, "total_bytes": tot_b, "n_patches": tot_p,
            "bytes_per_patch": round(tot_b / tot_p, 4),
            "patch_len_mean": round(float(lens_t.mean()), 4),
            "patch_len_median": round(float(lens_t.median()), 4),
            "patch_len_p90": round(float(lens_t.quantile(0.9)), 4),
            "patch_len_max": int(lens_t.max()),
            "frac_len1": round(float((lens_t == 1).float().mean()), 4),
        }
        print(f"θ={th:.6f} -> {out['arms'][f'{th:.6f}']['bytes_per_patch']} B/patch "
              f"({tot_p} patches / {tot_b} bytes)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

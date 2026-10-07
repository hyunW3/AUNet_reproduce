#!/usr/bin/env python3
"""BLT-1B robustness (Noise / Typo / Despace / PBP) on HellaSwag + ARC-Easy.

Runs through Meta's NATIVE `bytelatent` code path, not the `itazap/blt-1b-hf` conversion.
That choice is forced: the conversion's entropy model builds an ordinary causal mask while
the real one runs `attn_bias_type="local_block_causal"` with `sliding_window=512`, and the
two disagree so badly on DCLM (1.15 vs 4.07 bytes/patch at the released threshold) that
every threshold-dependent number from the conversion is an artifact of the port.
See reports_afterAAAIsub/flores/summary.md, "CORRECTION".

Two further traps this script has to step around:

* This BLT checkout hardcodes `attn_impl="sdpa"` in `bytelatent/entropy_model.py` (a local
  workaround for a venv whose xformers had no CUDA kernels). With `local_block_causal` that
  path is only exact below 512 bytes and silently mis-patches above it -- 1.39 vs 3.6 B/patch
  on a 1.7 kB prompt. We rebind `load_entropy_model` to the upstream xformers setting at
  runtime, leaving the shared checkout untouched (same trick as blt_native_flores.py).

* BLT's accuracy depends on BATCH COMPOSITION. Rows are padded to a common 128-multiple byte
  length, that padding goes through the entropy patcher, and the resulting segmentation moves:
  reports/verify_blt_1335 measured ARC-Easy at 0.6873 (bs16) vs 0.6498 (bs8) on identical
  items. Any batch size > 1 therefore mixes a batching artifact into the clean-vs-perturbed
  delta this script exists to measure. Default is batch_size=1 and it should stay there.

  $BLT_PY blt_robustness_native.py --threshold 1.46875 --limit 500 --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch

ROOT = "/mnt/ssd2/hyun2/AUNet"
BLT_REPO = os.environ.get("BLT_REPO", "/mnt/ssd2/hyun2/lzspacebyte/blt")
EVAL_SUITE = os.environ.get("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite")
CKPT = f"{BLT_REPO}/hf-weights/blt_1b"
ENTROPY_CKPT = f"{BLT_REPO}/hf-weights/entropy_model"


def _patch_entropy_attn(attn_impl="xformers"):
    """Rebind load_entropy_model with the upstream attn_impl, leaving the checkout untouched.

    patcher.py does `from bytelatent.entropy_model import load_entropy_model` at import time,
    so the name has to be replaced on the patcher module, not on entropy_model.
    """
    import json as _json
    import bytelatent.data.patcher as _patcher
    from bytelatent.transformer import LMTransformer, LMTransformerArgs

    def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
        with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
            reloaded = _json.loads(fr.read())
        torch.set_default_dtype(torch.bfloat16)
        mp = reloaded["entropy_model"]
        args = LMTransformerArgs(
            dim=mp["dim"], n_layers=mp["n_layers"], n_heads=mp["n_heads"],
            max_seqlen=mp["max_seqlen"], ffn_dim_multiplier=mp["ffn_dim_multiplier"],
            vocab_size=mp["vocab_size"],
            attn_bias_type="local_block_causal", attn_impl=attn_impl, sliding_window=512,
        )
        m = LMTransformer(args)
        m.load_state_dict(torch.load(state_dict_path, map_location=device)["model"], strict=False)
        m.to(device).eval()
        for prm in m.parameters():
            prm.requires_grad = False
        return m, args

    _patcher.load_entropy_model = load_entropy_model
    print(f"[native] entropy model attn_impl -> {attn_impl}", flush=True)


def build(threshold, max_len):
    """Native BLT + realtime entropy patcher, with the threshold optionally overridden."""
    from bytelatent.distributed import DistributedArgs, setup_torch_distributed
    from bytelatent.generate import load_consolidated_model_and_tokenizer

    da = DistributedArgs()
    da.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(da)
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(CKPT)
    pa = train_cfg.data.patcher_args.model_copy(deep=True)
    pa.realtime_patching = True
    pa.entropy_model_checkpoint_dir = ENTROPY_CKPT
    released = float(pa.threshold)
    if threshold is not None:
        pa.threshold = float(threshold)
    patcher = pa.build()
    print(f"[blt] threshold: released={released:.6f} -> in use={float(patcher.threshold):.6f}",
          flush=True)
    return model, tokenizer, patcher, released


def make_counting_harness(BLTHarness):
    """BLTHarness + a running (real bytes / real patches) tally.

    Has to be a genuine BLTHarness SUBCLASS, not a wrapper: lm_eval.simple_evaluate rejects
    anything that is not an lm_eval.api.model.LM subclass, so a delegating proxy fails the
    isinstance check on the noise/typo axes. BLTHarness is only importable after the BLT
    sys.path dance, hence the factory.

    'real' bytes/patches exclude the boe padding added to reach the 128-multiple, so the number
    is the compression actually applied to the eval prompts -- which is what the threshold is
    set to control, and is NOT the same as the DCLM calibration figure.
    """

    class CountingBLTHarness(BLTHarness):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.n_bytes = 0
            self.n_patches = 0

        @torch.inference_mode()
        def _tally(self, batch):
            a128 = lambda n: ((n + 127) // 128) * 128
            tok = self.tokenizer
            enc = [tok.encode(c + x, add_bos=True, add_eos=False)[: self.max_len]
                   for c, x in batch]
            Np = a128(max(len(f) for f in enc))
            toks = torch.full((len(enc), Np), tok.boe_id, dtype=torch.long, device="cuda")
            for i, f in enumerate(enc):
                toks[i, : len(f)] = torch.tensor(f, device="cuda")
            pl, _ = self.patcher.patch(toks, include_next_token=False)
            for i, f in enumerate(enc):
                starts = torch.cumsum(pl[i], 0) - pl[i]   # patch start offsets
                self.n_patches += int((starts < len(f)).sum())
                self.n_bytes += len(f)

        def score_pairs(self, pairs):
            out = []
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                self._tally(chunk)
                out.extend(self._score_batch(chunk))
            return out

        def bytes_per_patch(self):
            return round(self.n_bytes / self.n_patches, 4) if self.n_patches else None

    return CountingBLTHarness


# --------------------------------------------------------------------------- axes --
# All four axes are pinned to HellaSwag + ARC-Easy. The paper used a different dataset per
# experiment (appendix.tex:210); this unifies them so one table can be read across axes.
TASKS = ("hellaswag", "arc_easy")
_MK = "acc_norm,none"


def _acc(res, k):
    r = res.get(k) or {}
    return r.get(_MK, r.get("acc_norm"))


def axis_noise(lm, limit):
    expand_noise_tasks, summarize_noise = AU["expand_noise_tasks"], AU["summarize_noise"]
    from lm_eval import simple_evaluate
    from common.compat import safe_task_manager
    tl = expand_noise_tasks(["hellaswag", "hellaswag_noise", "arc_easy", "arc_easy_noise"])
    res = simple_evaluate(lm, tasks=tl, num_fewshot=0, task_manager=safe_task_manager(),
                          limit=limit, log_samples=False, bootstrap_iters=0)["results"]
    res.update(summarize_noise(res))
    out = {}
    for t in TASKS:
        c, n = _acc(res, t), _acc(res, f"{t}_noise_avg")
        out[t] = {"clean": c, "noise_avg": n,
                  "delta_pp": round(100 * (n - c), 2) if (c is not None and n is not None) else None}
    out["variants"] = {k: _acc(res, k) for k in sorted(res) if "_noise_" in k}
    d = [v["delta_pp"] for t, v in out.items() if t in TASKS and v["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_typo(lm, limit):
    expand_typo_tasks, summarize_typo = AU["expand_typo_tasks"], AU["summarize_typo"]
    expand_typo_ds_tasks, summarize_typo_ds = AU["expand_typo_ds_tasks"], AU["summarize_typo_ds"]
    from lm_eval import simple_evaluate
    from common.compat import safe_task_manager
    tl = expand_typo_tasks(["hellaswag", "hellaswag_typo", "arc_easy", "arc_easy_typo"])
    tl = expand_typo_ds_tasks(tl)
    res = simple_evaluate(lm, tasks=tl, num_fewshot=0, task_manager=safe_task_manager(),
                          limit=limit, log_samples=False, bootstrap_iters=0)["results"]
    res.update(summarize_typo(res))
    res.update(summarize_typo_ds(res))
    out = {}
    for t in TASKS:
        c, n = _acc(res, t), _acc(res, f"{t}_typo_avg")
        out[t] = {"clean": c, "typo_avg": n,
                  "delta_pp": round(100 * (n - c), 2) if (c is not None and n is not None) else None}
    out["variants"] = {k: _acc(res, k) for k in sorted(res) if "_typo_" in k}
    d = [v["delta_pp"] for t, v in out.items() if t in TASKS and v["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_despace(lm, limit):
    res = AU["run_despace_mc"](lm.loglikelihood, limit=limit)
    out = {"raw": {k: v.get("acc") for k, v in res.items() if isinstance(v, dict)}}
    for t in TASKS:
        c = (res.get(f"despace_mc_{t}_clean") or {}).get("acc")
        a = (res.get(f"despace_mc_{t}_all100") or res.get(f"despace_mc_{t}_despaceall") or {}).get("acc")
        out[t] = {"clean": c, "all100": a,
                  "delta_pp": round(100 * (a - c), 2) if (c is not None and a is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


def axis_pbp(lm, limit):
    res = AU["run_pbp_mc"](lm.loglikelihood, limit=limit)
    out = {"raw": {k: v.get("acc") for k, v in res.items() if isinstance(v, dict)}}
    for t in list(TASKS) + ["curated"]:
        c = (res.get(f"pbp_mc_{t}_canonical") or {}).get("acc")
        s = (res.get(f"pbp_mc_{t}_space") or {}).get("acc")
        out[t] = {"canonical": c, "space": s,
                  "delta_pp": round(100 * (s - c), 2) if (c is not None and s is not None) else None}
    d = [out[t]["delta_pp"] for t in TASKS if out[t]["delta_pp"] is not None]
    out["delta_mean_pp"] = round(st.mean(d), 2) if d else None
    return out


AXES = {"noise": axis_noise, "typo": axis_typo, "despace": axis_despace, "pbp": axis_pbp}

# The BLT repo ships its OWN top-level `apps` package (blt/apps/__init__.py), and eval_suite's
# harness pushes BLT_REPO to the front of sys.path when it is imported. Whichever `apps` binds
# first wins for the whole process, so lingua's apps.aunet.* must be imported BEFORE any BLT
# import -- otherwise every axis dies with "No module named 'apps.aunet'".
AU = {}


def _preload_aunet():
    from apps.aunet.eval_noise import expand_noise_tasks, summarize_noise
    from apps.aunet.eval_typo import expand_typo_tasks, summarize_typo
    from apps.aunet.eval_typo_ds import expand_typo_ds_tasks, summarize_typo_ds
    from apps.aunet.eval_despace_mc import run_despace_mc, DESPACE_TASKS
    from apps.aunet.eval_pbp_mc import run_pbp_mc, MC_TASKS
    AU.update(locals())
    print(f"[aunet] despace tasks={DESPACE_TASKS}  pbp_mc tasks={MC_TASKS}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="entropy patching threshold; omit for the released 1.335442")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="KEEP AT 1 -- see the module docstring on batch-composition sensitivity")
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--axes", default="noise,typo,despace,pbp")
    ap.add_argument("--attn_impl", default="xformers", choices=["xformers", "sdpa"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # env the perturbation modules read AT IMPORT TIME -- must be set before they load
    os.environ["DESPACE_TASKS"] = ",".join(TASKS)
    os.environ["PBP_MC_TASKS"] = ",".join(TASKS)
    # 논문 표의 despace 셀은 공백 100% 제거 엔드포인트(`despaceall`)다. 기본 사다리는 10/40/70%
    # 까지만 만들므로 DESPACE_FULL 로 켠다. DESPACE_ANSWER 는 보기 텍스트까지 despace 하는 arm.
    os.environ["DESPACE_FULL"] = "1"
    os.environ["DESPACE_ANSWER"] = "1"

    sys.path.insert(0, BLT_REPO)
    sys.path.insert(0, EVAL_SUITE)
    sys.path.insert(0, f"{ROOT}/lingua")
    _preload_aunet()          # must precede every BLT import -- see the AU comment above

    _patch_entropy_attn(args.attn_impl)
    from blt_eval.harness import BLTHarness

    model, tok, patcher, released = build(args.threshold, args.max_len)
    lm = make_counting_harness(BLTHarness)(model, tok, patcher, args.batch_size, args.max_len)

    row = {"model": "BLT-1B (facebook/blt-1b, native bytelatent)",
           "attn_impl": args.attn_impl, "batch_size": args.batch_size, "limit": args.limit,
           "tasks": list(TASKS), "released_threshold": released,
           "threshold": float(patcher.threshold), "axes": {}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    for name in args.axes.split(","):
        name = name.strip()
        if not name:
            continue
        t0 = time.time()
        print(f"\n===== axis {name} (threshold {float(patcher.threshold):.6f}) =====", flush=True)
        b0, p0 = lm.n_bytes, lm.n_patches
        try:
            row["axes"][name] = AXES[name](lm, args.limit)
        except Exception as e:
            import traceback
            traceback.print_exc()
            row["axes"][name] = {"error": f"{type(e).__name__}: {e}"}
        d_b, d_p = lm.n_bytes - b0, lm.n_patches - p0
        row["axes"][name]["_minutes"] = round((time.time() - t0) / 60, 1)
        row["axes"][name]["_bytes_per_patch"] = round(d_b / d_p, 4) if d_p else None
        json.dump(row, open(args.out, "w"), indent=2)   # checkpoint after every axis
        print(f"[{name}] {row['axes'][name]['_minutes']} min, "
              f"{row['axes'][name]['_bytes_per_patch']} B/patch", flush=True)

    row["bytes_per_patch_overall"] = lm.bytes_per_patch()
    json.dump(row, open(args.out, "w"), indent=2)
    print("\nWROTE", args.out)
    print(json.dumps({k: v for k, v in row.items() if k != "axes"}, indent=2))


if __name__ == "__main__":
    main()

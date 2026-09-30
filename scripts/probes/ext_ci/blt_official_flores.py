#!/usr/bin/env python3
"""FLORES-200 X->en for BLT-1B via the official bytelatent path (BLT_execution_guide.md).

Protocol identical to the matched trio (reports_afterAAAIsub/flores/summary.md): dev split, first
200 sentences per language, 5-shot with exemplars from devtest, greedy, stop at "\\n", 288-byte
generation budget (= ceil(96*6/2), what apps.aunet.eval gives the in-repo byte models).

Config per the guide: official facebookresearch/blt + released weights + attn_impl=xformers (so all
three 512-byte windows apply natively) + batch size 1 + released threshold 1.335442066192627.

Validated on this box before use:
  * teacher-forced BPB, first 4 DCLM windows: 1.0461 (official xformers 1.0462, unwindowed ~1.675)
  * bytes/patch, 320 DCLM windows: 4.075 (unwindowed 1.151)
  * generation vs teacher forcing: is_greedy=True on 4/4 FLORES prompts, i.e. the incremental
    re-patching path reproduces the teacher-forced argmax
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("EVAL_SUITE", "/mnt/ssd2/hyun2/eval_suite")
from run_ext import build_blt_official, _lm_eval_compat          # noqa: E402

LANGS = ["de", "fr", "es", "it", "nl", "ko"]
RELEASED = 1.335442066192627


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blt_weights", default="/mnt/ssd2/hyun2/AUNet/runs/ext_ci_snu55/blt_weights")
    ap.add_argument("--tasks_dir", default="/mnt/ssd2/hyun2/AUNet/lingua/eval_tasks")
    ap.add_argument("--split", choices=["dev", "devtest"], default="dev")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--gen_bytes", type=int, default=288)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--threshold", type=float, default=RELEASED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    _lm_eval_compat()
    sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua")
    sys.modules.pop("apps", None)
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    langs = [x.strip() for x in a.langs.split(",") if x.strip()]
    pre = "flores_dev_" if a.split == "dev" else "flores_"
    tasks = [f"{pre}{l}-en" for l in langs]

    lm = build_blt_official(a.blt_weights, a.max_len, a.threshold)
    print(f"[blt_official] threshold={a.threshold} batch_size={lm.batch_size} "
          f"released={'yes' if a.threshold == RELEASED else 'NO (patch-size-matched variant)'}",
          flush=True)

    t0 = time.time()
    res = simple_evaluate(lm, tasks=tasks, num_fewshot=a.num_fewshot,
                          task_manager=TaskManager(include_path=a.tasks_dir),
                          limit=a.limit, log_samples=True, bootstrap_iters=1000,
                          gen_kwargs=f"max_gen_toks={a.gen_bytes}")
    r = res["results"]
    per = {l: {"bleu": r[f"{pre}{l}-en"]["bleu,none"],
               "bleu_stderr": r[f"{pre}{l}-en"]["bleu_stderr,none"],
               "chrf": r[f"{pre}{l}-en"]["chrf,none"]} for l in langs}
    b = [per[l]["bleu"] for l in langs]
    c = [per[l]["chrf"] for l in langs]
    lat = [l for l in langs if l != "ko"]
    row = {"model": "BLT-1B (official bytelatent, xformers, bs1)", "family": "blt-entropy",
           "split": a.split, "n": a.limit, "num_fewshot": a.num_fewshot, "decoding": "greedy",
           "gen_bytes": a.gen_bytes, "threshold": a.threshold,
           "threshold_kind": "released" if a.threshold == RELEASED else "patch-size-matched",
           "langs": langs, "per_lang": per, "elapsed_s": round(time.time() - t0, 1),
           "avg_bleu": round(sum(b) / len(b), 3), "avg_chrf": round(sum(c) / len(c), 3),
           "avg_latin_bleu": round(sum(per[l]["bleu"] for l in lat) / len(lat), 3) if lat else None,
           "avg_latin_chrf": round(sum(per[l]["chrf"] for l in lat) / len(lat), 3) if lat else None}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(row, open(a.out, "w"), indent=2)
    json.dump(res.get("samples", {}), open(a.out.replace(".json", "_samples.json"), "w"), default=str)
    print(json.dumps({k: v for k, v in row.items() if k != "per_lang"}, indent=2))
    for l in langs:
        print(f"  {l}: BLEU {per[l]['bleu']:6.2f}  chrF {per[l]['chrf']:6.2f}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()

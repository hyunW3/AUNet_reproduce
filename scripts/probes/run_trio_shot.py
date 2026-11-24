#!/usr/bin/env python3
"""n-shot downstream for a trio model (Llama/AU-Net/BPEByte) via build_generator +
the codebase's EvalHarnessLM, so it matches BLT's protocol. Runs the same task set
as the BLT 3-shot (hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq).
"""
import argparse, json, os, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fflm"))
from fflm_probe import build_generator, DEFAULT_TOK   # noqa: E402
from lm_eval import simple_evaluate                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["subword", "aunet"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num_fewshot", type=int, default=3)
    ap.add_argument("--tasks", nargs="+",
                    default=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "boolq"])
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--tok_path", default=DEFAULT_TOK)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gen, tok = build_generator(args.family, args.ckpt, args.tok_path, args.max_tokens)
    if args.family == "subword":
        from apps.main.eval import EvalHarnessLM
    else:
        from apps.aunet.eval import EvalHarnessLM
    lm = EvalHarnessLM(gen)
    res = simple_evaluate(lm, tasks=args.tasks, num_fewshot=args.num_fewshot,
                          limit=args.limit, log_samples=False, bootstrap_iters=0)
    r = res["results"]
    conv = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm",
            "piqa": "acc_norm", "winogrande": "acc", "boolq": "acc"}
    out = {"tag": args.tag, "num_fewshot": args.num_fewshot}
    for t in args.tasks:
        k = t if t in r else next((x for x in r if x.startswith(t)), None)
        m = conv.get(t, "acc")
        out[t] = round(100 * r[k].get(f"{m},none", r[k].get(m)), 2) if k else None
    vals = [out[t] for t in args.tasks if out.get(t) is not None]
    out["avg"] = round(st.mean(vals), 2) if vals else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(f"[{args.tag}] {args.num_fewshot}-shot: " +
          " ".join(f"{t}={out.get(t)}" for t in args.tasks) + f" | avg={out['avg']}", flush=True)


if __name__ == "__main__":
    main()

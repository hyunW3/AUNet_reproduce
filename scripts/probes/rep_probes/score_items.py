#!/usr/bin/env python3
"""Score a probe item file with one model, per item (shared by repetition probes A-D).

Item formats (jsonl, one per line; every model scores byte-identical strings):
  {"type": "mc",  "key", "probe", "cond", "task", "ctx", "choices": [continuations...], "gold": int}
      -> acc (argmax summed loglik) and acc_norm (loglik / len(choice) in characters, as lm-eval)
  {"type": "gen", "key", "probe", "cond", "task", "prompt", "value"}
      -> exact: greedy reproduction of " " + value given the prompt (teacher forced; = S-NIAH scoring)

Families and environments:
  llama | aunet | bpebyte   lingua venv  (PYTHONPATH=lingua; fflm build_generator + EvalHarnessLM)
  blt                       bytelatent env (run_ext.build_blt_official: windows native, batch 1, <=4096 B)
  hnet                      H-Net env      (run_ext.build_hnet_lm, max_len 8192)

  python score_items.py --family bpebyte --items items.jsonl --out scored_bpebyte.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

L = Path(os.environ.get("AUNET_ROOT", "/mnt/ssd2/hyun2/AUNet"))
CKPT = {"llama": "main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated",
        "aunet": "main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated",
        "bpebyte": "main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"}


class Req:
    def __init__(self, ctx, cont):
        self.args = (ctx, cont)


def build(family, a):
    if family in CKPT:
        sys.path.insert(0, str(L / "lingua"))
        sys.path.insert(0, str(L / "scripts/fflm"))
        from fflm_probe import build_generator, DEFAULT_TOK
        fam = "subword" if family == "llama" else "aunet"
        gen, _ = build_generator(fam, str(Path(a.ckpt_root or L) / CKPT[family]), DEFAULT_TOK, 8192)
        if fam == "subword":
            from apps.main.eval import EvalHarnessLM
        else:
            from apps.aunet.eval import EvalHarnessLM
        return EvalHarnessLM(gen)
    sys.path.insert(0, os.environ.get("RUN_EXT_DIR", str(L / "scripts/probes/ext_ci")))
    import run_ext
    if family == "blt":
        return run_ext.build_blt_official(a.blt_weights, 4096, a.threshold)
    return run_ext.build_hnet_lm(batch_size=8, max_len=8192)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["llama", "aunet", "bpebyte", "blt", "hnet"], required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_root", default=None)
    ap.add_argument("--blt_weights", default=os.path.expanduser("~/aunet_ext/blt_weights"))
    ap.add_argument("--threshold", type=float, default=1.335442066192627)
    ap.add_argument("--chunk", type=int, default=2000, help="requests per loglikelihood call")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items)]
    lm = build(a.family, a)
    reqs, owners = [], []
    for i, it in enumerate(items):
        if it["type"] == "mc":
            for c in it["choices"]:
                reqs.append(Req(it["ctx"], c)); owners.append(i)
        else:
            reqs.append(Req(it["prompt"], " " + it["value"])); owners.append(i)
    t0, res = time.time(), []
    for s in range(0, len(reqs), a.chunk):
        res.extend(lm.loglikelihood(reqs[s:s + a.chunk]))
        print(f"  {min(s + a.chunk, len(reqs))}/{len(reqs)} requests  {time.time() - t0:.0f}s", flush=True)
    by = {}
    for (ll, greedy), o, r in zip(res, owners, reqs):
        by.setdefault(o, []).append((float(ll), bool(greedy), r.args[1]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i, it in enumerate(items):
            rec = {k: it.get(k) for k in ("key", "probe", "cond", "task", "depth", "length", "fmt", "dist")}
            if it["type"] == "mc":
                lls = [x[0] for x in by[i]]
                norm = [x[0] / max(1, len(x[2])) for x in by[i]]
                rec["acc"] = int(max(range(len(lls)), key=lls.__getitem__) == it["gold"])
                rec["acc_norm"] = int(max(range(len(norm)), key=norm.__getitem__) == it["gold"])
            else:
                rec["exact"] = int(by[i][0][1])
            rec["family"] = a.family
            f.write(json.dumps(rec) + "\n")
    print("wrote", a.out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

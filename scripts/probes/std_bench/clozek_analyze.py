#!/usr/bin/env python3
"""Analyze fmt_robust.py --mode clozek: LAMBADA cue effect acc("____. Answer:") - acc("____. ->") as a function
of the number of per-item-drawn demonstrations k, per model, per seed and averaged over the 3 seeds
(paired item bootstrap over the seed-averaged effect), plus the BPEByte - Transformer / - AU-Net differences.

  python3 clozek_analyze.py --root <dir with <model>/clozek_n1000.jsonl> --n 1000
"""
import argparse, json, random

M = ["llama_1.3b", "aunet_1.3b", "bpebyte_1.3b"]
S = {"llama_1.3b": "Transf", "aunet_1.3b": "AUNet", "bpebyte_1.3b": "BPEByte"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = {}
    for m in M:
        R[m] = {}
        for l in open(f"{a.root}/{m}/clozek_n{a.n}.jsonl"):
            r = json.loads(l)
            R[m][(r["seed"], r["k"], r["cue"])] = [x["correct"] for x in r["rows"]]
    seeds = sorted({s for s, _, _ in R[M[0]]}); ks = sorted({k for _, k, _ in R[M[0]]})
    n = a.n
    rng = random.Random(0)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(a.B)]
    lo, hi = int(.025 * a.B), int(.975 * a.B) - 1

    def eff(m, k, idx, seed=None):
        ss = seeds if seed is None else [seed]
        return 100 * sum(R[m][(s, k, "Answer:")][i] - R[m][(s, k, "->")][i] for s in ss for i in idx) / (len(ss) * len(idx))

    def acc(m, k, cue):
        return 100 * sum(sum(R[m][(s, k, cue)]) for s in seeds) / (len(seeds) * n)

    out = {}
    print("accuracy (mean over seeds)  '->' / 'Answer:'")
    print(f"{'k':>3s} " + "".join(f"{S[m]:>16s}" for m in M))
    for k in ks:
        print(f"{k:3d} " + "".join(f"{acc(m, k, '->'):7.1f} /{acc(m, k, 'Answer:'):6.1f}" for m in M))
    print("\ncue effect Answer: - ->  (seed-mean [95% CI]; per-seed values)")
    for k in ks:
        line = f"k={k:<2d}"
        out[k] = {}
        for m in M:
            pt = eff(m, k, range(n)); bs = sorted(eff(m, k, b) for b in boots)
            per = [eff(m, k, range(n), s) for s in seeds]
            out[k][m] = {"effect": pt, "ci": [bs[lo], bs[hi]], "per_seed": per}
            line += f"  {S[m]} {pt:+5.1f} [{bs[lo]:+5.1f},{bs[hi]:+5.1f}] ({'/'.join(f'{x:+.0f}' for x in per)})"
        print(line)
        for r in ("llama_1.3b", "aunet_1.3b"):
            d = sorted(eff("bpebyte_1.3b", k, b) - eff(r, k, b) for b in boots)
            out[k][f"bpe_minus_{r}"] = [eff("bpebyte_1.3b", k, range(n)) - eff(r, k, range(n)), d[lo], d[hi]]
        print(f"      BPEByte-Transf {out[k]['bpe_minus_llama_1.3b'][0]:+5.1f} [{out[k]['bpe_minus_llama_1.3b'][1]:+5.1f},{out[k]['bpe_minus_llama_1.3b'][2]:+5.1f}]"
              f"   BPEByte-AUNet {out[k]['bpe_minus_aunet_1.3b'][0]:+5.1f} [{out[k]['bpe_minus_aunet_1.3b'][1]:+5.1f},{out[k]['bpe_minus_aunet_1.3b'][2]:+5.1f}]")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

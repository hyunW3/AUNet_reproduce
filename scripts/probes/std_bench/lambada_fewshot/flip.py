import json, re
import numpy as np
T = "/home/hyunw3/.claude/jobs/003c5e28/tmp/"
SB = "/mnt/ssd2/hyun2/AUNet/reports/std_bench/"
P = json.load(open(T + "lambada_prompts.json"))
M = {"Transformer": "llama_1.3b", "AUNet": "aunet_1.3b", "BPEByte": "bpebyte_1.3b", "BLT": "blt_1b", "H-Net": "hnet_1stage_XL"}


def bits(reg, k):
    j = json.load(open(f"{SB}{reg}/ds{k}_seed1234_lambada.json"))
    d = {int(i.split(":")[1]): b for i, b in zip(j["items"]["lambada_openai"], j["bits"]["lambada_openai"])}
    return np.array([d[i] for i in range(len(d))])


ctx0 = [x["ctx"] for x in P["0"]]
tgt = [x["target"].strip() for x in P["0"]]
inctx = np.array([re.search(r"\b" + re.escape(t) + r"\b", c) is not None for c, t in zip(ctx0, tgt)])
for k in ("3", "5"):
    L = np.array([len(x["ctx"].encode()) for x in P[k]])
    q = np.quantile(L, [.25, .5, .75])
    binid = np.digitize(L, q)
    # target word occurs in a demonstration (the part of the k-shot prompt before the scored passage)
    demo = [x["ctx"][: len(x["ctx"]) - len(c)] for x, c in zip(P[k], ctx0)]
    assert all(x["ctx"].endswith(c) for x, c in zip(P[k], ctx0))
    indemo = np.array([re.search(r"\b" + re.escape(t) + r"\b", d) is not None for d, t in zip(demo, tgt)])
    print(f"\n=== {k}-shot  (prompt-length quartile edges {q.astype(int)} B; target in demos: {indemo.mean():.1%})")
    print(f"{'model':12s} {'0-shot':>7s} {k+'-shot':>7s} {'drop':>6s} | drop by prompt-length quartile Q1..Q4 | in-ctx  not-in-ctx | tgt∈demos  ∉demos | 0✓→k✗  0✗→k✓")
    for name, reg in M.items():
        b0, bk = bits(reg, 0), bits(reg, int(k))
        d = 100 * (b0 - bk)
        qs = " ".join(f"{d[binid == i].mean():5.1f}" for i in range(4))
        print(f"{name:12s} {100*b0.mean():7.1f} {100*bk.mean():7.1f} {d.mean():6.1f} | {qs}              | {d[inctx].mean():6.1f} {d[~inctx].mean():8.1f}   | {d[indemo].mean():8.1f} {d[~indemo].mean():7.1f}  | {int(((b0==1)&(bk==0)).sum()):5d} {int(((b0==0)&(bk==1)).sum()):6d}")

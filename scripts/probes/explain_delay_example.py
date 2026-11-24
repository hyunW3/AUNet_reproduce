#!/usr/bin/env python3
"""Worked example: which patch vector each position reads under full / delay1 / delay_bnd, and the
per-byte loss that follows.

A patch vector is pooled at the patch's START position and the trunk is causal, so patch p's vector
summarises the text UP TO AND INCLUDING its boundary byte -- it knows nothing about the rest of its
own patch.  The read column is therefore labelled by the text the vector was pooled through, not by
the byte span the patch covers.

  --score  additionally runs the checkpoint on this one sentence and adds per-byte BPB columns
           (single short sentence -> illustrative, not a statistic; corpus numbers are in
           reports/global_ablation/summary_delay.md).
"""
import argparse, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lingua"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omegaconf import OmegaConf                                        # noqa: E402
from lingua.args import dataclass_from_dict                            # noqa: E402
from apps.aunet.data.regex_cutting import RegexPool, RegexArgs         # noqa: E402

LN2 = math.log(2.0)
MODES = ("full", "delay1", "delay_bnd", "local_only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "main/main/1.3B/aunet2_1.3B/checkpoints/0000180000/consolidated"))
    ap.add_argument("--text", default="Jesus sent out 72 disciples to go")
    ap.add_argument("--rows", type=int, default=22)
    ap.add_argument("--ctx", type=int, default=14, help="chars of pooled-through text to show")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = OmegaConf.load(Path(a.ckpt) / "params.json")
    pool = RegexPool(dataclass_from_dict(RegexArgs, cfg.data.regex, strict=False))
    ids = [256] + list(a.text.encode())
    lm = np.array(pool.get_levels_mask(ids)); lm[0] = 1                # force_first, as the model does
    P = np.cumsum(lm) - 1                                              # patch each position belongs to
    st = np.flatnonzero(lm)                                            # pooling position of each patch

    def pooled_through(p):
        """What patch p's vector actually contains: the text up to and including its boundary byte."""
        end = st[p] + 1
        s = bytes(b for b in ids[1:end] if b < 256).decode("utf8", "replace").replace(" ", "␣")
        return "<BOS>" + (("…" + s[-a.ctx:]) if len(s) > a.ctx else s)

    nll = {}
    if a.score:
        import torch
        from global_ablation import load_model, Ablation, prep, score_batch
        model, pool2, _ = load_model(a.ckpt)
        seqlen = int(model.transitions[0].seqlen_in)
        p = prep(a.text, pool2, len(a.text.encode()), seqlen)
        assert p is not None
        pid, plm, n, _ = p
        for m in MODES:
            with Ablation(model, m) as abl:
                out = score_batch(model, pid.unsqueeze(0).repeat(2, 1), plm.unsqueeze(0).repeat(2, 1),
                                  [n, n], "fmha", abl.needs_cpi)
            nll[m] = out[0].numpy()

    ch = lambda b: "<BOS>" if b == 256 else repr(chr(b))
    head = ("| j | input | boundary? | target | pos-in-patch | read: full (pooled through) | "
            "read: d1 | read: d_bnd |")
    if a.score:
        head += (" BPB full | BPB delay1 | BPB delay_bnd | BPB local_only |"
                 " Δ d1 | Δ d_bnd | Δ local |")
    out = [f"text: {a.text!r}", "",
           "patches (span → vector pooled through): " + " ; ".join(
               f"p{i}:{bytes(b for b in ids[s:e] if b < 256).decode('utf8', 'replace')!r}→{pooled_through(i)}"
               for i, (s, e) in enumerate(zip(st, list(st[1:]) + [len(ids)]))), "",
           head, "|---" * (head.count("|") - 1) + "|"]
    for j in range(min(a.rows, len(ids) - 1)):
        t = j + 1
        s = t
        while s > 0 and lm[s] == 0:
            s -= 1
        rf, rd = P[j], max(P[j] - 1, 0)
        rb = rd if lm[j] else rf
        row = (f"| {j} | {ch(ids[j])} | {'**yes**' if lm[j] else '·'} | {ch(ids[t])} | {t - s} | "
               f"p{rf} = {pooled_through(rf)} | "
               f"{('**p%d**' % rd) if rd != rf else 'p%d' % rf} | "
               f"{('**p%d**' % rb) if rb != rf else 'p%d (unchanged)' % rf} |")
        if a.score:
            f_, d1, db, lo = (nll[m][j] / LN2 for m in MODES)
            row += (f" {f_:.2f} | {d1:.2f} | {db:.2f} | {lo:.2f} | "
                    f"{d1 - f_:+.2f} | {db - f_:+.2f} | {lo - f_:+.2f} |")
        out.append(row)
    if a.score:
        tot = {m: nll[m].mean() / LN2 for m in MODES}
        out += ["", "whole sentence (%d B): " % len(nll["full"])
                + " | ".join(f"{m} {tot[m]:.3f}" for m in MODES) + " BPB"]
    txt = "\n".join(out)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""trunk 이 버린 complexity 가 어디로 가는가 — U-Net skip 가설의 결정적 측정.

ib_compression.md §3 에서 BPEByte 의 trunk 는 I(X;Z) 를 0.204(L0) -> 0.035(L24) 로 83% 버린다.
가설: **버려도 되기 때문에 버린다.** 이 아키텍처에는 byte encoder 출력을 trunk 바깥으로 돌려
decoder 에 바로 더하는 skip 이 있다 (hierarchical.py: `h = res + h`, config `residuals=[True]`).
그렇다면 바이트 정보는 trunk 를 통과할 필요가 없고, decoder 입력에서 되살아나 있어야 한다.

측정 지점 (전부 바이트 해상도라 좌표 정렬이 1:1 — searchsorted 가 필요 없다):

  res      encoders[0] 출력            skip 이 나르는 것 자체
  h        dec_in - res                trunk 가 up-projection 을 거쳐 기여한 것
  dec_in   decoders[0] 입력 = res + h  decoder 가 실제로 받는 것

Transformer 는 skip 이 없으므로 대응점이 없다. 참조로 embedding 과 최종 은닉상태를 같은
좌표계에서 잰다.

예측: skip 가설이 맞다면 I(X;res) 가 크고 I(X;dec_in) 도 크며, I(X;h) 는 trunk L24 수준으로
작아야 한다. 틀리다면 dec_in 의 complexity 가 trunk 출력 수준으로 낮을 것이다.

  python ib_skip.py --out runs/ib_compression/skip.json
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/fflm"))
sys.path.insert(0, str(ROOT / "lingua"))

from ib_extract import ngram_labels, backoff_labels, MAX_ORDER, WINDOWS   # noqa: E402

N_ANCHOR = 100
D_REF = 2048      # eps 를 이 차원 기준으로 잡고 나머지는 sqrt 규칙으로 환산한다


def eps_for(dim, eps_ref, calib=True):
    """차원이 다른 지점을 비교하려면 eps 를 차원에 맞춰야 한다.

    단위구 위 두 무작위 벡터의 코사인은 std = 1/sqrt(d) 다. logit = cos/eps 의 스케일을
    차원 무관하게 두려면 eps 는 1/sqrt(d) 에 비례해야 한다. 논문의 Appendix E.1.1 보정이
    노리는 것이 이것인데 값이 본문에 없어 닫힌 형태로 쓴다.
    """
    return eps_ref * (D_REF / dim) ** 0.5 if calib else eps_ref


def shannon(p):
    p = np.asarray(p, np.float64); p = p / p.sum()
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / np.log(N_ANCHOR))


class _NoLoad:
    """체크포인트를 읽지 않고 모델을 새로 초기화한다.

    consolidated.pth 가 21GB 라 무작위 체크포인트를 디스크에 만들 여유가 없다(여유 124GB).
    대신 torch.load 와 load_state_dict 를 무력화하고, 로더가 돌려준 모델에 init_weights() 를
    부른다. 구조·config·dtype 은 학습본과 완전히 같고 가중치만 초기값이다.
    """

    def __enter__(self):
        import torch
        self.t = torch
        self._load, self._lsd = torch.load, torch.nn.Module.load_state_dict
        torch.load = lambda *a, **k: {"model": {}}
        torch.nn.Module.load_state_dict = lambda self_, *a, **k: None
        return self

    def __exit__(self, *e):
        self.t.load, self.t.nn.Module.load_state_dict = self._load, self._lsd


class Points:
    """모델 하나에서 여러 지점의 (바이트정렬된) 은닉상태를 한 번의 forward 로 뽑는다."""

    def __init__(self, arm, ckpt, tok, random_init=False):
        import torch
        from fflm_probe import build_generator, DEFAULT_TOK
        self.t, self.arm = torch, arm
        fam = "subword" if arm == "llama" else "aunet"
        self.fam = fam
        if random_init:
            with _NoLoad():
                self.gen, self.tok = build_generator(
                    fam, ckpt, tok or os.environ.get("AUNET_TOK", DEFAULT_TOK), 8192)
            m0 = self.gen.model
            torch.manual_seed(1234)
            m0.init_weights()
            pdt = next(p.dtype for p in m0.parameters())
            self.gen.model = m0.cuda().eval()
            for prm in self.gen.model.parameters():
                prm.data = prm.data.to(dtype=pdt)
        else:
            self.gen, self.tok = build_generator(
                fam, ckpt, tok or os.environ.get("AUNET_TOK", DEFAULT_TOK), 8192)
        self.gen.max_gen_len = 1
        self.cap = {}
        m = self.gen.model
        if fam == "aunet":
            # 생성기는 각 스택을 PackedInternalCausalTransformerGenerator 로 감싸고 .prefill() 을
            # 부르므로 encoders[0].forward 는 실행되지 않는다. 실제로 도는 것은 그 안의 layer 들이라
            # 층 단위로 건다. BaseTransformer.forward 는 마지막에 norm 을 걸지 않으므로
            # encoders[0].layers[-1] 출력이 곧 hierarchical.forward 가 저장하는 `res` 다.
            m.encoders[0].layers[-1].register_forward_hook(
                lambda _m, _i, o: self.cap.__setitem__("res", (o[0] if isinstance(o, tuple) else o).detach()[0].float().cpu().numpy()))
            m.decoders[0].layers[0].register_forward_pre_hook(
                lambda _m, i: self.cap.__setitem__("dec_in", i[0].detach()[0].float().cpu().numpy()))
            m.trunk.layers[-1].register_forward_hook(
                lambda _m, _i, o: self.cap.__setitem__("trunk_out", (o[0] if isinstance(o, tuple) else o).detach()[0].float().cpu().numpy()))
            trans = m.transitions[-1]
            orig = trans.down

            def down(x, mask, freq_cis, mask_idx=None, position=None, _o=orig):
                out = _o(x, mask, freq_cis, mask_idx, position)
                self.cap["_ref"] = (mask_idx.detach().flatten().cpu().numpy()
                                    if mask_idx is not None else None)
                self.cap["trunk_in"] = out.detach()[0].float().cpu().numpy()
                return out
            trans.down = down
        else:
            m.tok_embeddings.register_forward_hook(
                lambda _m, _i, o: self.cap.__setitem__("emb", o.detach()[0].float().cpu().numpy()))
            m.layers[-1].register_forward_hook(
                lambda _m, _i, o: self.cap.__setitem__("last", (o[0] if isinstance(o, tuple) else o).detach()[0].float().cpu().numpy()))

    def run(self, text):
        """-> dict[name] = (nb, dim) 바이트 정렬 배열"""
        self.cap.clear()
        with self.t.no_grad():
            self.gen.generate([text])
        nb = len(text.encode())
        out = {}
        if self.fam == "aunet":
            # encoder/decoder 는 [BOS]+bytes 순서 -> 바이트 i 는 행 i+1
            for k in ("res", "dec_in"):
                a = self.cap[k]
                out[k] = a[1:1 + nb] if len(a) >= nb + 1 else a[:nb]
            out["h"] = out["dec_in"] - out["res"]
            ref = self.cap["_ref"].astype(np.int64) - 1
            keep = ref >= 0
            order = np.argsort(ref[keep], kind="stable")
            sel = np.flatnonzero(keep)[order]
            closing0 = ref[keep][order]
            slot = np.clip(np.searchsorted(closing0, np.arange(nb), "left"), 0, int(keep.sum()) - 1)
            for nm in ("trunk_out", "trunk_in"):
                gg = self.cap[nm][: len(ref)][sel]
                out[nm] = gg[slot]
            # 해상도 통제: trunk_out 은 패치당 1벡터를 ~4.6 바이트에 뿌린 것이고 res/dec_in 은
            # 바이트당 1벡터다. 그 차이만으로 I(X;Z) 가 낮아 보일 수 있으므로, res 를 패치가
            # 닫히는 바이트에서만 뽑아 똑같이 뿌린 판본을 같이 잰다. 이게 여전히 trunk_out 보다
            # 훨씬 높으면 하락은 해상도 때문이 아니다.
            closing = ref[keep][order]
            rc = out["res"][np.clip(closing, 0, nb - 1)]
            out["res_pooled"] = rc[slot]

            # 풀링 readout ablation: 같은 byte encoder 상태에서 "패치당 1벡터" 를 만드는 방식을
            # 바꿔 가며 문맥 정보가 얼마나 남는지 본다. 학습이 필요 없다 — 지금 trunk 가 받는
            # 0.1303 이 "패치당 1벡터" 의 한계인지, 아니면 이 풀링 함수의 한계인지 가른다.
            R = out["res"]
            start = np.concatenate(([0], closing[:-1] + 1))
            end = np.clip(closing, 0, nb - 1) + 1
            cs = np.concatenate([np.zeros((1, R.shape[1])), np.cumsum(R, 0)])
            mean = (cs[np.clip(end, 0, nb)] - cs[np.clip(start, 0, nb)]) / np.maximum(
                (end - start)[:, None], 1)
            out["pool_mean"] = mean[slot]
            first = R[np.clip(start, 0, nb - 1)]
            out["pool_firstlast"] = np.concatenate([first, rc], 1)[slot]
            mx = np.stack([R[s_:max(e_, s_ + 1)].max(0) for s_, e_ in zip(start, end)])
            out["pool_max"] = mx[slot]
            # 패치당 2벡터: 중간과 끝. 바이트에는 offset 절반 기준으로 나눠 준다.
            mid = R[np.clip((start + end) // 2, 0, nb - 1)]
            half = start + (end - start) // 2
            use_mid = np.arange(nb) < half[slot]
            out["pool_k2"] = np.where(use_mid[:, None], mid[slot], rc[slot])
            return out, nb
        ids = self.tok.encode(text, add_bos=False, add_eos=False)
        parts = self.tok.tkt_model.decode_tokens_bytes(ids)
        en = np.cumsum(np.fromiter((len(p) for p in parts), np.int64, len(parts)))
        slot = np.clip(np.searchsorted(en - 1, np.arange(nb), "left"), 0, len(ids) - 1)
        for k, src in (("emb", "emb"), ("last", "last")):
            out[k] = self.cap[src][1:1 + len(ids)][slot]
        return out, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/ib_compression/skip.json"))
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--min_count", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--arms", default="bpebyte,llama")
    ap.add_argument("--no_dim_calib", action="store_true",
                    help="eps 를 차원별로 환산하지 않는다(초기 실행 재현용)")
    ap.add_argument("--random", action="store_true",
                    help="체크포인트를 읽지 않고 init_weights() 로 초기화한 모델을 쓴다")
    a = ap.parse_args()

    wins = [json.loads(l) for l in open(WINDOWS)]
    if a.limit:
        wins = wins[: a.limit]
    blobs = [w["text"].encode() for w in wins]

    labels = {}
    for side, s in (("X", "left"), ("Y", "right")):
        per = [[ngram_labels(b, k, s) for k in range(1, MAX_ORDER + 1)] for b in blobs]
        counts = []
        for k in range(MAX_ORDER):
            c = {}
            for pw in per:
                u, n = np.unique(pw[k][pw[k] >= 0], return_counts=True)
                for uu, nn in zip(u.tolist(), n.tolist()):
                    c[uu] = c.get(uu, 0) + nn
            counts.append(c)
        labels[side] = [backoff_labels(pw, counts, a.min_count) for pw in per]

    W = {}
    res = {}
    want = set(a.arms.split(","))
    for arm, ck in (("bpebyte", ROOT / "main/main/1.3B/bpebyte_br_greedy_root_1.3B/checkpoints/0000180000/consolidated"),
                    ("llama",   ROOT / "main/main/1.3B/llama_1.8B_paper/checkpoints/0000060000/consolidated")):
        if arm not in want:
            continue
        P = Points(arm, str(ck), a.tok, random_init=a.random)
        g, lab = {}, {"X": {}, "Y": {}}
        for wi, w in enumerate(wins):
            pts, nb = P.run(w["text"])
            for name, h in pts.items():
                d = h.shape[1]
                if d not in W:
                    rng = np.random.default_rng(1000 * d)
                    ww = rng.standard_normal((d, N_ANCHOR))
                    W[d] = ww / np.linalg.norm(ww, axis=0, keepdims=True)
                zb = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
                lg = zb @ W[d] / eps_for(d, a.eps, not a.no_dim_calib)
                lg -= lg.max(1, keepdims=True)
                z = np.exp(lg); zhat = z / z.sum(1, keepdims=True)
                g.setdefault(name, np.zeros(N_ANCHOR))
                g[name] += zhat.sum(0)
                for side in ("X", "Y"):
                    L = labels[side][wi][:nb]
                    m = L >= 0
                    dd = lab[side].setdefault(name, {})
                    u, inv = np.unique(L[m], return_inverse=True)
                    S = np.zeros((len(u), N_ANCHOR)); np.add.at(S, inv, zhat[m])
                    C = np.bincount(inv, minlength=len(u))
                    for i, uu in enumerate(u.tolist()):
                        if uu in dd:
                            dd[uu][0] += S[i]; dd[uu][1] += C[i]
                        else:
                            dd[uu] = [S[i], int(C[i])]
            if (wi + 1) % 40 == 0:
                print(f"  {arm} {wi+1}/{len(wins)}", flush=True)
        for name in g:
            HZ = shannon(g[name])
            row = {"H(Z)": HZ, "dim": None}
            for side in ("X", "Y"):
                dd = lab[side][name]
                c = np.array([v[1] for v in dd.values()], float)
                hh = np.array([shannon(v[0]) for v in dd.values()])
                row[f"H(Z|{side})"] = float((c / c.sum() * hh).sum())
                row[f"I({side};Z)"] = HZ - row[f"H(Z|{side})"]
            row["optimality"] = row["I(Y;Z)"] / row["I(X;Z)"] if row["I(X;Z)"] > 1e-9 else None
            res[f"{arm}.{name}"] = row
        del P

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}\n")
    print("| point | H(Z) | I(X;Z) | I(Y;Z) | optimality |")
    print("|---|---:|---:|---:|---:|")
    for k, v in res.items():
        o = v["optimality"]
        print(f"| {k} | {v['H(Z)']:.4f} | {v['I(X;Z)']:.4f} | {v['I(Y;Z)']:.4f} | "
              f"{'—' if o is None else f'{o:.3f}'} |")


if __name__ == "__main__":
    main()

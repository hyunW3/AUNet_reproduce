#!/usr/bin/env python3
"""Information-Bottleneck 측정 — Conklin et al., "Learning is Forgetting" (arXiv:2604.07569).

논문의 방법:
  complexity   I(X;Z) = H(Z) - sum_x P(x) H(Z|X=x)      X = 앞선 문맥
  expressivity I(Y;Z) = H(Z) - sum_y P(y) H(Z|Y=y)      Y = 뒤따르는 토큰
  optimality   = I(Y;Z) / I(X;Z)
  IB 한계선    = I(Y;Z) = I(X;Z) (대각선). 그 위로는 갈 수 없다.

엔트로피 추정(soft-entropy):
  1. 은닉상태를 단위구로 정규화
  2. 구에서 균일하게 뽑은 앵커 n=100 개에 코사인 투영
  3. z_hat = softmax(z_bar . W / eps)  -> 앵커 위 범주분포
  4. H = Shannon(그룹 내 z_hat 평균) / log(n)   -> [0,1], 차원 무관
  5. 층마다 구해 평균

이 구현이 논문과 다른 점, 그리고 왜:

* **byte 좌표계** (기본). 논문은 각 모델의 자기 토큰으로 X/Y 를 만든다. 우리는 subword 와
  byte 모델을 비교하므로 그러면 X 의 알파벳 자체가 달라져 I(X;Z) 가 비교 불가다. 그래서
  X/Y 를 **바이트 n-gram** 으로 두고, 두 모델의 각 벡터를 같은 바이트 좌표에 올린다
  (바이트 i 는 causal read position 이 i 이상인 첫 슬롯이 대표 — whybpebyte Experiment E 와
  같은 규칙). 이러면 라벨·그룹 크기·희소성이 두 arm 에서 완전히 같다.
  `--units own` 으로 논문 그대로(자기 토큰 단위)도 뽑을 수 있으나 cross-arm 비교용은 아니다.

* **층 선택**. 논문은 전 층 평균이다. BPEByte 는 local encoder(3층 512d) / trunk(25층 2048d) /
  decoder(3층 512d) 가 해상도와 차원이 다르므로 전부 평균내면 두 arm 이 다른 것을 재게 된다.
  `--stack trunk` (기본) 는 양쪽 다 25층 x 2048d 인 global trunk 만 본다 — 구조가 정확히 맞는
  유일한 지점이다. `--stack all` 은 논문 그대로 전 층 평균.

* **앵커 W 는 두 arm 이 공유한다** (같은 seed, 같은 차원). 논문은 공유 여부를 명시하지 않지만,
  공유하지 않으면 arm 간 차이에 앵커 뽑기 운이 섞인다.

* **eps**. 논문의 보정 절차(Appendix E.1.1)는 본문에 값이 없다. trunk 차원이 양쪽 2048 로
  같으므로 차원 보정 자체가 필요 없고, 여기서는 **한 값을 두 모델에 똑같이** 쓴다. 기본값은
  포화를 피하는 범위에서 고르고(--eps), analyze 쪽에서 sweep 을 함께 보고한다.

* **backoff**. 논문은 Katz 류 backoff 를 쓴다고만 한다. 여기서는 order k 의 n-gram 이
  `--min_count` 미만이면 k-1 로 내린다. 그룹이 작으면 H(Z|X=x) 가 과소추정돼 I 가 부풀지만,
  **라벨이 두 arm 에서 동일하므로 그 편향도 동일하다**. min_count 는 sweep 한다.

출력: npz 하나. 층별 (전역 분포, 라벨별 분포 합, 라벨별 개수) 만 저장하므로 은닉상태를
전부 들고 있지 않아도 된다.

  python ib_extract.py --arm llama   --ckpt <consolidated> --out x.npz
  python ib_extract.py --arm bpebyte --ckpt <consolidated> --out y.npz
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/fflm"))
sys.path.insert(0, str(ROOT / "scripts/probes/whybpebyte"))
sys.path.insert(0, str(ROOT / "lingua"))

WINDOWS = ROOT / "reports/bpb_windows.jsonl"
FAMILY = {"llama": "subword", "aunet": "aunet", "bpebyte": "aunet"}
MAX_ORDER = 4


# --------------------------------------------------------------------- 라벨
def ngram_labels(data: bytes, order: int, side: str) -> np.ndarray:
    """바이트 위치마다 order 길이의 문맥을 정수 하나로 접는다.

    side='left'  : 위치 i 의 라벨은 data[i-order:i]  (X, 앞선 문맥)
    side='right' : 위치 i 의 라벨은 data[i+1:i+1+order] (Y, 뒤따르는 바이트)
    경계에서 창이 모자라면 -1 (그 위치는 그 order 에서 제외).
    """
    a = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    n = len(a)
    lab = np.zeros(n, np.int64)
    ok = np.ones(n, bool)
    for j in range(order):
        if side == "left":
            idx = np.arange(n) - order + j
        else:
            idx = np.arange(n) + 1 + j
        good = (idx >= 0) & (idx < n)
        lab = lab * 256 + np.where(good, a[np.clip(idx, 0, n - 1)], 0)
        ok &= good
    return np.where(ok, lab, -1)


def backoff_labels(per_order: list[np.ndarray], counts: list[dict], min_count: int) -> np.ndarray:
    """가장 높은 order 부터 내려가며 count >= min_count 인 첫 라벨을 쓴다.

    order 를 섞어 쓰므로 라벨 공간이 겹치지 않게 (order, label) 을 하나의 키로 접는다.
    """
    n = len(per_order[0])
    out = np.full(n, -1, np.int64)
    todo = np.ones(n, bool)
    for k in range(len(per_order), 0, -1):
        lab = per_order[k - 1]
        cnt = counts[k - 1]
        cand = todo & (lab >= 0)
        if not cand.any():
            continue
        c = np.fromiter((cnt.get(int(v), 0) for v in lab[cand]), np.int64, int(cand.sum()))
        take = np.zeros(n, bool)
        take[np.flatnonzero(cand)[c >= min_count]] = True
        out[take] = lab[take] * (MAX_ORDER + 1) + k     # order 를 키에 섞는다
        todo &= ~take
    return out


# ---------------------------------------------------------------- 추출기
class TrunkExtractor:
    """윈도 하나를 forward 해서 (층 x 슬롯 x dim) 은닉상태와 바이트 read position 을 준다."""

    def __init__(self, arm, ckpt, tok_path, stack, max_tokens=8192):
        import torch
        from fflm_probe import build_generator, DEFAULT_TOK
        self.t = torch
        self.arm, self.family, self.stack = arm, FAMILY[arm], stack
        self.gen, self.tok = build_generator(
            self.family, ckpt, tok_path or os.environ.get("AUNET_TOK", DEFAULT_TOK), max_tokens)
        self.gen.max_gen_len = 1
        self.cap = {}
        self._install()

    def _stacks(self):
        m = self.gen.model
        if self.family == "subword":
            return [("trunk", m.layers)]
        out = [("trunk", m.trunk.layers)]
        if self.stack == "all":
            out = ([(f"enc{i}", e.layers) for i, e in enumerate(m.encoders)] + out
                   + [(f"dec{i}", d.layers) for i, d in enumerate(m.decoders)])
        return out

    def _install(self):
        cap = self.cap
        for tag, layers in self._stacks():
            for li, layer in enumerate(layers):
                def hook(_m, _i, o, key=f"{tag}.{li}"):
                    h = o[0] if isinstance(o, tuple) else o
                    cap[key] = h.detach()[0].float().cpu().numpy()
                layer.register_forward_hook(hook)
        if self.family == "aunet":
            trans = self.gen.model.transitions[-1]
            orig = trans.down

            def down(x, mask, freq_cis, mask_idx=None, position=None, _o=orig):
                out = _o(x, mask, freq_cis, mask_idx, position)
                cap["_ref"] = (mask_idx.detach().flatten().cpu().numpy()
                               if mask_idx is not None else None)
                return out
            trans.down = down
        self.keys = [f"{tag}.{li}" for tag, layers in self._stacks() for li in range(len(layers))]

    def run(self, text: str):
        """-> (H: dict[key] = (S,h) 배열, ref: (S,) 각 슬롯이 읽은 마지막 바이트)"""
        self.cap.clear()
        with self.t.no_grad():
            self.gen.generate([text])
        nb = len(text.encode())
        if self.family == "aunet":
            ref = self.cap["_ref"].astype(np.int64) - 1        # prefill 은 [BOS]+bytes
            keep = ref >= 0
            order = np.argsort(ref[keep], kind="stable")
            sel = np.flatnonzero(keep)[order]
            H = {}
            for k in self.keys:
                if not k.startswith("trunk"):
                    H[k] = self.cap[k]                          # byte 해상도 스택
                    continue
                h = self.cap[k]
                H[k] = h[: len(ref)][sel]
            return H, ref[keep][order], nb
        ids = self.tok.encode(text, add_bos=False, add_eos=False)
        parts = self.tok.tkt_model.decode_tokens_bytes(ids)
        en = np.cumsum(np.fromiter((len(p) for p in parts), np.int64, len(parts)))
        H = {k: self.cap[k][1: 1 + len(ids)] for k in self.keys}   # BOS 행 제거
        return H, en - 1, nb


# ------------------------------------------------------------------ 집계
class Accum:
    """층별로 전역 분포와 라벨별 분포 합을 모은다. 은닉상태는 들고 있지 않는다."""

    def __init__(self, keys, n_anchor, epss):
        self.keys, self.n, self.epss = list(keys), n_anchor, list(epss)
        self.keys = [f"{e:g}|{k}" for e in self.epss for k in list(keys)]
        self.g = {k: np.zeros(n_anchor, np.float64) for k in self.keys}
        self.gn = 0
        self.lab = {"X": {}, "Y": {}}      # side -> {label: (n_anchor 합, count)}

    def add_global(self, key, zhat):
        self.g[key] += zhat.sum(0)

    def add_labeled(self, side, key, labels, zhat):
        d = self.lab[side].setdefault(key, {})
        uniq, inv = np.unique(labels, return_inverse=True)
        sums = np.zeros((len(uniq), self.n), np.float64)
        np.add.at(sums, inv, zhat)
        cnts = np.bincount(inv, minlength=len(uniq))
        for i, u in enumerate(uniq):
            u = int(u)
            if u in d:
                d[u][0] += sums[i]; d[u][1] += cnts[i]
            else:
                d[u] = [sums[i], int(cnts[i])]

    def save(self, path, meta):
        out = {"keys": np.array(self.keys), "g_n": self.gn,
               "meta": np.array(json.dumps(meta))}
        for k in self.keys:
            out[f"g/{k}"] = self.g[k]
        for side in ("X", "Y"):
            for k, d in self.lab[side].items():
                labs = np.fromiter(d.keys(), np.int64, len(d))
                out[f"{side}lab/{k}"] = labs
                out[f"{side}sum/{k}"] = np.stack([d[int(l)][0] for l in labs])
                out[f"{side}cnt/{k}"] = np.fromiter((d[int(l)][1] for l in labs), np.int64, len(labs))
        np.savez_compressed(path, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(FAMILY))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--stack", default="trunk", choices=["trunk", "all"])
    ap.add_argument("--units", default="byte", choices=["byte", "own"])
    ap.add_argument("--anchors", type=int, default=100)
    # eps 는 콤마로 여러 값을 준다. 투영(zb @ W)은 한 번만 하고 softmax 만 값마다 다시 하므로
    # 비용이 거의 늘지 않는다. 포화(H->1)와 원핫(H->0) 사이의 값을 고르기 위한 보정용이다.
    ap.add_argument("--eps", default="0.002,0.005,0.01,0.02,0.05")
    ap.add_argument("--anchor_seed", type=int, default=0)
    ap.add_argument("--min_count", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    wins = [json.loads(l) for l in open(WINDOWS)]
    if a.limit:
        wins = wins[: a.limit]

    ex = TrunkExtractor(a.arm, a.ckpt, a.tok, a.stack)

    # 라벨은 모델과 무관하게 먼저 만든다 — 두 arm 이 같은 라벨을 쓰도록.
    # 1) 전 윈도의 n-gram count, 2) backoff 로 위치별 유효 라벨
    all_bytes = [w["text"].encode() for w in wins]
    labels = {}
    for side, s in (("X", "left"), ("Y", "right")):
        per_win = [[ngram_labels(b, k, s) for k in range(1, MAX_ORDER + 1)] for b in all_bytes]
        counts = []
        for k in range(MAX_ORDER):
            c = {}
            for pw in per_win:
                u, n = np.unique(pw[k][pw[k] >= 0], return_counts=True)
                for uu, nn in zip(u.tolist(), n.tolist()):
                    c[uu] = c.get(uu, 0) + nn
            counts.append(c)
        labels[side] = [backoff_labels(pw, counts, a.min_count) for pw in per_win]

    epss = [float(x) for x in str(a.eps).split(",")]
    W = {}          # dim -> (dim, n_anchor) 공유 앵커
    acc = None

    for wi, w in enumerate(wins):
        H, ref, nb = ex.run(w["text"])
        if acc is None:
            acc = Accum(H.keys(), a.anchors, epss)
        for key, h in H.items():
            d = h.shape[1]
            if d not in W:
                g = np.random.default_rng(a.anchor_seed + 1000 * d).standard_normal((d, a.anchors))
                W[d] = g / np.linalg.norm(g, axis=0, keepdims=True)
            zb = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
            proj = zb @ W[d]                     # eps 와 무관한 부분: 한 번만
            for e in epss:
                logit = proj / e
                logit -= logit.max(1, keepdims=True)
                z = np.exp(logit)
                zhat = z / z.sum(1, keepdims=True)
                ek = f"{e:g}|{key}"
                if a.units == "byte" and key.startswith("trunk"):
                    # 바이트 i <- causal read position 이 i 이상인 첫 슬롯
                    slot = np.searchsorted(ref, np.arange(nb), side="left")
                    slot = np.clip(slot, 0, len(zhat) - 1)
                    zz = zhat[slot]
                else:
                    zz = zhat
                acc.add_global(ek, zz)
                if a.units == "byte" and key.startswith("trunk"):
                    for side in ("X", "Y"):
                        lab = labels[side][wi][:nb]
                        m = lab >= 0
                        acc.add_labeled(side, ek, lab[m], zz[m])
        acc.gn += 1
        if (wi + 1) % 40 == 0:
            print(f"  {a.arm} {wi + 1}/{len(wins)}", flush=True)

    acc.save(a.out, vars(a))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

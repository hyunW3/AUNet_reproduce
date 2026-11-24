#!/usr/bin/env python3
"""ib_extract.py 의 npz 들을 complexity / expressivity / optimality 로 접는다.

  H(Z)      = Shannon(층별 전역 앵커분포) / log n,  층 평균
  H(Z|X)    = sum_x P(x) Shannon(그룹 x 의 평균 앵커분포) / log n,  층 평균
  I(X;Z)    = H(Z) - H(Z|X)          complexity
  I(Y;Z)    = H(Z) - H(Z|Y)          expressivity
  optimality= I(Y;Z) / I(X;Z)        1 이 IB 한계선(대각선)

  python ib_analyze.py out/*.npz
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np


def shannon(p, n):
    p = np.asarray(p, np.float64)
    p = p / p.sum()
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum() / np.log(n))


def read(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    n = meta["anchors"]
    allk = z["keys"].tolist()
    epss = sorted({k.split("|")[0] for k in allk}, key=float)
    out = {}
    for e in epss:
        out[e] = _one(z, n, [k for k in allk if k.startswith(f"{e}|") and "|trunk" in k])
    return meta, out


def _one(z, n, keys):
    hz, hzx, hzy, groups = [], [], [], {}
    for k in keys:
        hz.append(shannon(z[f"g/{k}"], n))
        for side, dst in (("X", hzx), ("Y", hzy)):
            s, c = z[f"{side}sum/{k}"], z[f"{side}cnt/{k}"]
            P = c / c.sum()
            h = np.array([shannon(s[i], n) for i in range(len(c))])
            dst.append(float((P * h).sum()))
            groups[side] = (len(c), float(c.mean()))
    hz, hzx, hzy = map(lambda v: float(np.mean(v)), (hz, hzx, hzy))
    return dict(n_layers=len(keys), HZ=hz, HZX=hzx, HZY=hzy,
                IXZ=hz - hzx, IYZ=hz - hzy, groups=groups)


def main(paths):
    data = [(Path(p).stem,) + read(p) for p in paths]
    m = data[0][1]
    any_e = next(iter(data[0][2]))
    print("## Information Bottleneck (arXiv:2604.07569 방법)\n")
    print(f"설정: trunk {data[0][2][any_e]['n_layers']}층, 앵커 {m['anchors']}, units {m['units']}, "
          f"min_count {m['min_count']}, 윈도 {m.get('limit') or 320}개\n")
    print("| eps | model | H(Z) | H(Z\\|X) | H(Z\\|Y) | complexity I(X;Z) | expressivity I(Y;Z) | optimality |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for e in sorted(data[0][2], key=float):
        for lab, _mm, res in data:
            r = res[e]
            opt = r["IYZ"] / r["IXZ"] if r["IXZ"] > 1e-9 else float("nan")
            print(f"| {e} | {lab} | {r['HZ']:.4f} | {r['HZX']:.4f} | {r['HZY']:.4f} | "
                  f"{r['IXZ']:.4f} | {r['IYZ']:.4f} | {opt:.4f} |")
    g = data[0][2][any_e]["groups"]
    print(f"\n그룹 수 / 평균 크기 — X: {g['X'][0]} / {g['X'][1]:.0f}, Y: {g['Y'][0]} / {g['Y'][1]:.0f}")
    print("(라벨은 모델과 무관하게 먼저 만들었으므로 모든 arm 이 동일하다)")


if __name__ == "__main__":
    main(sys.argv[1:])

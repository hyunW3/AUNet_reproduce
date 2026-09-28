#!/usr/bin/env python3
"""논문 학습본 1.3B robustness — 5태스크 |Δ| + paired bootstrap 95% CI, item limit 별.

지표 규약은 robustness_paper_vs_retrain.py 와 같다:
  noise/typo : acc_norm (BoolQ 는 acc) — clean 대비 변형 평균, 태스크 macro 평균
  despace    : despace_mc 의 acc, clean 대비 despaceall(공백 100% 제거, 보기 포함)

limit N 은 저장된 2000-item 결과의 앞 N 문항이다. lm-eval `limit` 과 despace_mc 의 `_apply_limit`
둘 다 앞에서부터 자르고, despace 의 per-item seed 도 문항 순번이라 재실행과 같은 문항·같은 변형이 된다.
bootstrap 은 태스크별로 문항을 복원추출하되 세 모델에 같은 인덱스를 쓴다(paired).

입력: runs/robustness_paper1p3b{,_ext}/<arm>/results.json (noise/typo samples)
      runs/robustness_despace_bits/<arm>/results.json   (despace bits, DESPACE_BITS=1)
  python robustness_bootstrap_ci.py [--B 2000] [--out reports/robustness_ci]
"""
import argparse, json, os
import numpy as np

L = "/mnt/ssd2/hyun2/AUNet"
ARMS = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte"}
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
AXES = ("noise", "typo", "despace")
LIMITS = (500, 1000, 1500, 2000)


def _samples(arm):
    S = {}
    for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
        S.update(json.load(open(f"{L}/runs/{d}/{arm}/results.json"))["samples"])
    return S


def _bits(recs, metric):
    return {int(r["doc_id"]): float(r[metric] if metric in r else r["acc"]) for r in recs}


def load(arm):
    """{task: {"clean_<ax>": vec, ax: matrix[variants, items]}} — 문항 순서(doc_id) 정렬."""
    S = _samples(arm)
    D = json.load(open(f"{L}/runs/robustness_despace_bits/{arm}/results.json"))["results"]
    out = {}
    for t in TASKS:
        m = "acc" if t == "boolq" else "acc_norm"
        c = _bits(S[t], m); ids = sorted(c)
        cell = {"clean_noise": np.array([c[i] for i in ids])}
        cell["clean_typo"] = cell["clean_noise"]
        for ax in ("noise", "typo"):
            vs = [k for k in S if k.startswith(f"{t}_{ax}_") and not k.endswith("_avg")]
            cell[ax] = np.array([[_bits(S[k], m)[i] for i in ids] for k in vs])
        cell["clean_despace"] = np.array(D[f"despace_mc_{t}_clean"]["bits"]["acc"], float)
        cell["despace"] = np.array([D[f"despace_mc_{t}_despaceall"]["bits"]["acc"]], float)
        out[t] = cell
    return out


def axis_abs(cells, ax, idx):
    """|태스크 macro 평균 Δ| (pp). idx: {task: item index array}."""
    return abs(np.mean([100 * (cells[t][ax][:, idx[t]].mean() - cells[t][f"clean_{ax}"][idx[t]].mean())
                        for t in TASKS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{L}/reports/robustness_ci")
    a = ap.parse_args()
    data = {arm: load(arm) for arm in ARMS}
    rng = np.random.default_rng(a.seed)
    rows = []
    for N in LIMITS:
        for ax in AXES:
            n_t = {t: min(N, data["llama"][t][f"clean_{ax}"].size) for t in TASKS}
            assert all(data[arm][t][f"clean_{ax}"].size == data["llama"][t][f"clean_{ax}"].size
                       for arm in ARMS for t in TASKS), "arms disagree on item count"
            idx = {t: np.arange(n_t[t]) for t in TASKS}
            pt = {arm: axis_abs(data[arm], ax, idx) for arm in ARMS}
            bs = {arm: np.empty(a.B) for arm in ARMS}
            for b in range(a.B):
                ri = {t: rng.integers(0, n_t[t], n_t[t]) for t in TASKS}
                for arm in ARMS:
                    bs[arm][b] = axis_abs(data[arm], ax, ri)
            row = {"limit": N, "axis": ax, "n_items": n_t}
            for arm in ARMS:
                row[arm] = {"abs_delta": round(pt[arm], 3),
                            "ci95": [round(x, 3) for x in np.percentile(bs[arm], [2.5, 97.5])]}
            for other in ("llama", "aunet"):
                d = bs["bpebyte"] - bs[other]
                row[f"bpebyte_minus_{other}"] = {"point": round(pt["bpebyte"] - pt[other], 3),
                                                 "ci95": [round(x, 3) for x in np.percentile(d, [2.5, 97.5])]}
            rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/robustness_ci.json", "w") as f:
        json.dump({"B": a.B, "seed": a.seed, "tasks": TASKS, "rows": rows}, f, indent=1)

    md = [f"# 1.3B robustness |Δ| (pp), 5 tasks, paired bootstrap 95% CI (B={a.B})\n",
          "| limit | axis | Transformer | AUNet | BPEByte | BPEByte−Transformer | BPEByte−AUNet |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        c = [f"{r[arm]['abs_delta']:.2f} [{r[arm]['ci95'][0]:.2f}, {r[arm]['ci95'][1]:.2f}]" for arm in ARMS]
        d = [f"{r[k]['point']:+.2f} [{r[k]['ci95'][0]:+.2f}, {r[k]['ci95'][1]:+.2f}]"
             for k in ("bpebyte_minus_llama", "bpebyte_minus_aunet")]
        md.append(f"| {r['limit']} | {r['axis']} | " + " | ".join(c + d) + " |")
    open(f"{a.out}/robustness_ci.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

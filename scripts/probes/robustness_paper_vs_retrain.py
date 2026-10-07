#!/usr/bin/env python3
"""논문 학습본 vs 재학습본 robustness 비교 (통일 기준 5태스크).

§12.1 표는 실행 두 개의 합이다: base(hellaswag, arc_easy) + ext(arc_challenge, piqa, boolq).
그래서 한 arm 을 읽으려면 디렉터리 두 개를 합쳐야 한다.

지표 규약은 agg_robustness.py 와 같다:
  noise/typo  : acc_norm (BoolQ 는 acc_norm 이 없어 acc) — clean 대비 변형 평균
  despace/pbp : 전용 센티널 태스크의 acc

  python robustness_paper_vs_retrain.py
"""
import json, os, statistics as st, sys

L = "/mnt/ssd2/hyun2/AUNet"
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
AXES = ("pbp", "noise", "typo", "despace")
ARMS = ("llama", "aunet", "bpebyte")
NAME = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte"}
SETS = {  # 라벨 -> (base 디렉터리, ext 디렉터리)
    "paper":   (f"{L}/runs/robustness_paper1p3b",   f"{L}/runs/robustness_paper1p3b_ext"),
    "retrain": (f"{L}/runs/robustness_retrain1p3b", f"{L}/runs/robustness_retrain1p3b_ext"),
}


def load(dirs, arm):
    """base/ext 두 results.json 을 하나의 results 딕셔너리로 합친다."""
    res = {}
    for d in dirs:
        f = os.path.join(d, arm, "results.json")
        if os.path.exists(f):
            res.update(json.load(open(f))["results"])
    return res or None


def _norm(res, k):
    """noise/typo 기준선. acc_norm 이 있으면 그것, 없으면 acc (BoolQ)."""
    r = res.get(k)
    if not isinstance(r, dict):
        return None
    for key in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
        if r.get(key) is not None:
            return 100 * r[key]
    return None


def _acc(res, k):
    r = res.get(k)
    if not isinstance(r, dict):
        return None
    for key in ("acc,none", "acc"):
        if r.get(key) is not None:
            return 100 * r[key]
    return None


def deltas(res):
    """{task: {axis: Δpp}}"""
    out = {}
    for t in TASKS:
        cell = {
            "noise":   (_norm(res, t), _norm(res, f"{t}_noise_avg")),
            "typo":    (_norm(res, t), _norm(res, f"{t}_typo_avg")),
            "despace": (_acc(res, f"despace_mc_{t}_clean"),
                        _acc(res, f"despace_mc_{t}_despaceall") or _acc(res, f"despace_mc_{t}_all100")),
            "pbp":     (_acc(res, f"pbp_mc_{t}_canonical"), _acc(res, f"pbp_mc_{t}_space")),
        }
        out[t] = {ax: (None if c is None or p is None else round(p - c, 2))
                  for ax, (c, p) in cell.items()}
    return out


def axis_mean(d, ax):
    xs = [d[t][ax] for t in TASKS if d[t][ax] is not None]
    return round(st.mean(xs), 2) if len(xs) == len(TASKS) else None


def fmt(x, sign=True):
    return "—" if x is None else (f"{x:+.2f}" if sign else f"{x:.2f}")


def main():
    data = {}
    for label, dirs in SETS.items():
        for arm in ARMS:
            res = load(dirs, arm)
            if res:
                data[(label, arm)] = deltas(res)

    have = sorted({a for (_, a) in data})
    print("## 축 평균 (5태스크, Δpp — 0 에 가까울수록 견고)\n")
    print("| arm | 축 | 논문본 | 재학습본 | 차이 |")
    print("|---|---|---:|---:|---:|")
    for arm in have:
        for ax in AXES:
            p = data.get(("paper", arm), {})
            r = data.get(("retrain", arm), {})
            pv = axis_mean(p, ax) if p else None
            rv = axis_mean(r, ax) if r else None
            diff = None if (pv is None or rv is None) else round(rv - pv, 2)
            print(f"| {NAME[arm]} | {ax} | {fmt(pv)} | {fmt(rv)} | {fmt(diff)} |")
        # macro |Δ|
        row = []
        for label in ("paper", "retrain"):
            d = data.get((label, arm))
            vs = [axis_mean(d, ax) for ax in AXES] if d else []
            row.append(round(st.mean([abs(v) for v in vs]), 2) if vs and all(v is not None for v in vs) else None)
        dm = None if None in row else round(row[1] - row[0], 2)
        print(f"| {NAME[arm]} | **macro \\|Δ\\|** | {fmt(row[0], False)} | {fmt(row[1], False)} | {fmt(dm)} |")

    print("\n## 태스크별 (Δpp)\n")
    for ax in AXES:
        print(f"### {ax}\n")
        print("| task | " + " | ".join(f"{NAME[a]} 논문 | {NAME[a]} 재학습 | Δ차" for a in have) + " |")
        print("|---" * (1 + 3 * len(have)) + "|")
        for t in TASKS:
            cells = []
            for a in have:
                pv = data.get(("paper", a), {}).get(t, {}).get(ax)
                rv = data.get(("retrain", a), {}).get(t, {}).get(ax)
                dd = None if (pv is None or rv is None) else round(rv - pv, 2)
                cells += [fmt(pv), fmt(rv), fmt(dd)]
            print(f"| {t} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()

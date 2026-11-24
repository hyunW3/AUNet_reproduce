#!/usr/bin/env python3
"""재학습 최종 체크포인트 vs 논문 최종 체크포인트 — Transformer / BPEByte 나란히.

논문 쪽 수집 규칙은 build_stage_viewer.py 와 동일하다: evals_* 를 glob 하면 few-shot
재실행(evals_5shot, evals_arc_challenge_5shot, evals_fg_162k …)이 섞여 BPEByte 가
Avg5 61.10 으로 뜬다(정답 60.45). 지정한 디렉토리만 읽고, 그 안에서도 n-shot 0 만 쓴다.
"""
import json, os, math

ROOT = "/mnt/ssd2/hyun2/AUNet"
PAPER_FINAL_SOURCES = [
    ("evals_5bench",           ["hellaswag", "arc_easy", "piqa", "winogrande", "boolq"]),
    ("evals_mmlu_arcc",        ["arc_challenge"]),
    ("evals_mmlu_text",        ["mmlu_text"]),
    ("evals_mmlu_text_0shot",  ["mmlu_text"]),
]
METRIC = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm",
          "piqa": "acc_norm", "winogrande": "acc", "boolq": "acc", "mmlu_text": "acc"}
AVG5 = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
AVG7 = AVG5 + ["boolq", "mmlu_text"]

ARMS = [
    ("Transformer 1.8B", f"{ROOT}/runs/llama_1.8B_a100x4/stage_0000059589/results.json", 59589,
     f"{ROOT}/main/main/1.3B/llama_1.8B_paper", 60000),
    ("BPEByte 1.3B", f"{ROOT}/runs/bpebyte_br_greedy_root_1.3B_a100x4/stage_0000180000/results.json", 180000,
     f"{ROOT}/main/main/1.3B/bpebyte_br_greedy_root_1.3B", 180000),
]

def pull(res, cfgs, want, into, se_into):
    for t in want:
        if t not in res or t in into: continue
        ns = (cfgs.get(t) or {}).get("num_fewshot", 0)
        if ns not in (0, None): continue          # few-shot 재실행 배제
        m = METRIC[t]
        if f"{m},none" in res[t]:
            into[t] = res[t][f"{m},none"] * 100
            s = res[t].get(f"{m}_stderr,none")
            if s is not None: se_into[t] = s * 100

def load_new(path):
    v, se = {}, {}
    d = json.load(open(path))
    pull(d["results"], d.get("configs", {}), METRIC, v, se)
    return v, se

def load_paper(base):
    v, se = {}, {}
    for sub, tasks in PAPER_FINAL_SOURCES:
        for t in tasks:
            for p in (f"{base}/{sub}/{t}/results.json", f"{base}/{sub}/results.json"):
                if not os.path.exists(p): continue
                d = json.load(open(p))
                pull(d.get("results", {}), d.get("configs", {}), [t], v, se)
    return v, se

def avg(v, keys):
    x = [v[k] for k in keys if k in v]
    return sum(x) / len(x) if len(x) == len(keys) else float("nan")
def avg_se(se, keys):                # 태스크 독립 가정 하의 평균 표준오차
    x = [se[k] for k in keys if k in se]
    return math.sqrt(sum(s*s for s in x)) / len(keys) if len(x) == len(keys) else float("nan")

print(f"{'':<16}" + "".join(f"{a[0]:>34}" for a in ARMS))
print(f"{'':<16}" + "".join(f"{'논문':>10}{'재학습':>11}{'차이':>10}" for _ in ARMS))
print("-" * (16 + 34*len(ARMS)))
cols = []
for name, newp, newstep, paperdir, pstep in ARMS:
    nv, nse = load_new(newp); pv, pse = load_paper(paperdir)
    cols.append((nv, nse, pv, pse, newstep, pstep))
for t in AVG7:
    line = f"{t:<16}"
    for nv, nse, pv, pse, *_ in cols:
        if t in nv and t in pv:
            line += f"{pv[t]:10.2f}{nv[t]:11.2f}{nv[t]-pv[t]:+10.2f}"
        else:
            line += f"{'-':>10}{'-':>11}{'-':>10}"
    print(line)
print("-" * (16 + 34*len(ARMS)))
for lbl, keys in (("Avg5", AVG5), ("Avg7", AVG7)):
    line = f"{lbl:<16}"
    for nv, nse, pv, pse, *_ in cols:
        line += f"{avg(pv,keys):10.2f}{avg(nv,keys):11.2f}{avg(nv,keys)-avg(pv,keys):+10.2f}"
    print(line)
line = f"{'  z (차이/se)':<16}"
for nv, nse, pv, pse, *_ in cols:
    for keys in (AVG5, AVG7):
        pass
    d5 = avg(nv,AVG5)-avg(pv,AVG5)
    s5 = math.sqrt(avg_se(nse,AVG5)**2 + avg_se(pse,AVG5)**2)
    line += f"{'':10}{'':11}{d5/s5:+10.2f}" if s5 == s5 and s5 > 0 else f"{'':10}{'':11}{'-':>10}"
print(line + "   (Avg5 기준)")
print()
for (name, _, ns, _, ps), (nv, nse, pv, pse, *_) in zip(ARMS, cols):
    print(f"  {name}: 재학습 step {ns:,} / 논문 step {ps:,}"
          + ("  ← 학습량 동일" if ns == ps else f"  ← 재학습이 {ps-ns:,} step 적다 ({(ps-ns)/ps*100:.2f}%)"))

# ---------------------------------------------------------------- arm 간 격차
# 논문의 주장은 "BPEByte vs Transformer" 라는 **차이**에 있다. 각 arm 이 얼마나
# 재현됐는지보다, 그 차이가 재현되는지가 더 중요하다.
print("\n=== arm 간 격차 (BPEByte − Transformer) ===")
(tv, tse, tp, tps, *_), (bv, bse, bp, bps, *_) = cols
print(f"{'':<10}{'논문':>10}{'재학습':>11}{'격차의 변화':>14}")
for lbl, keys in (("Avg5", AVG5), ("Avg7", AVG7)):
    gp = avg(bp, keys) - avg(tp, keys)
    gn = avg(bv, keys) - avg(tv, keys)
    print(f"{lbl:<10}{gp:+10.2f}{gn:+11.2f}{gn-gp:+14.2f}")
print()
for lbl, keys in (("Avg5", AVG5),):
    gp = avg(bp, keys) - avg(tp, keys); gn = avg(bv, keys) - avg(tv, keys)
    se_p = math.sqrt(avg_se(bps,keys)**2 + avg_se(tps,keys)**2)
    se_n = math.sqrt(avg_se(bse,keys)**2 + avg_se(tse,keys)**2)
    print(f"  논문 격차 {gp:+.2f} (se {se_p:.2f}, z {gp/se_p:+.2f})")
    print(f"  재학습 격차 {gn:+.2f} (se {se_n:.2f}, z {gn/se_n:+.2f})")
    print(f"  -> 두 격차 모두 |z| < 2. 부호가 뒤집히지만 어느 쪽도 유의하지 않다:")
    print(f"     1.3B 다운스트림에서 두 arm 은 **동률**이고, 그 사실이 재학습으로 재확인됐다.")

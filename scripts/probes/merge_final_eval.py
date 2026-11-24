#!/usr/bin/env python3
"""BPEByte 최종(180,000) 평가 병합.

평가를 태스크별 잡으로 쪼개 돌렸으므로(stage_0000180000_{a,mmlu,hsgen,aegen}) 결과
JSON 이 여러 개다. 뷰어와 이후 분석은 stage_<pad>/results.json 하나를 읽으므로 합친다.

처음엔 a/b 둘로 갈랐는데, 생성 태스크(hellaswag_gen/arc_easy_gen)는 문서당 비용이
판별형의 100배 수준이라 b 가 29시간 넘게 안 끝났다. 비용 종류가 다른 태스크를 한 잡에
섞지 않는다 — 그래서 지금은 mmlu / hsgen / aegen 을 따로 돌린다.
"""
import json, os, sys, shutil

RUN = "/mnt/ssd2/hyun2/AUNet/runs/bpebyte_br_greedy_root_1.3B_a100x4"
PAD = "0000180000"
dst = os.path.join(RUN, f"stage_{PAD}")
HALVES = ("a", "mmlu", "hsgen", "aegen")
parts = [os.path.join(RUN, f"stage_{PAD}_{h}") for h in HALVES]

have = [p for p in parts if os.path.isfile(os.path.join(p, "results.json"))]
if not have:
    sys.exit("결과가 하나도 없다")
for p in parts:
    if p not in have:
        print(f"  아직 없음(건너뜀): {os.path.basename(p)}")

out = None
for p in have:
    d = json.load(open(os.path.join(p, "results.json")))
    if out is None:
        out = d; continue
    for key in ("results", "configs", "n-samples", "versions", "higher_is_better"):
        if key in d:
            dup = set(out.get(key, {})) & set(d[key])
            if dup:
                print(f"  경고: {key} 키 중복 {sorted(dup)[:5]}")
            out.setdefault(key, {}).update(d[key])

os.makedirs(dst, exist_ok=True)
json.dump(out, open(os.path.join(dst, "results.json"), "w"), indent=2)
for p in have:                       # per-doc 샘플은 paired/McNemar 검정에 쓴다
    for f in os.listdir(p):
        if f.startswith("samples_") and not os.path.exists(os.path.join(dst, f)):
            shutil.copy2(os.path.join(p, f), os.path.join(dst, f))

M = {'hellaswag': 'acc_norm,none', 'arc_easy': 'acc_norm,none', 'arc_challenge': 'acc_norm,none',
     'piqa': 'acc_norm,none', 'winogrande': 'acc,none', 'boolq': 'acc,none', 'mmlu_text': 'acc,none'}
r = out["results"]
v = {k: r[k][m] * 100 for k, m in M.items() if k in r}
for k in M:
    print(f"  {k:14s} " + (f"{v[k]:6.2f}" if k in v else "     -"))
if len(v) >= 5:
    print(f"  Avg5 {sum(v[k] for k in list(M)[:5])/5:.2f}")
if len(v) == 7:
    print(f"  Avg7 {sum(v.values())/7:.2f}")
# 생성형 MC 는 exact_match 로 채점된다(acc 아님). 4지선다라 우연 수준이 25.0 이고,
# 45k~162k 다섯 마일스톤이 전부 25.1~25.5 였다 — BPEByte 가 이 형식을 수행하지 못한다는
# 바닥값이며 학습이 진행돼도 움직이지 않는다. Avg5/Avg7 에는 들어가지 않는다.
for k in ("hellaswag_gen", "arc_easy_gen"):
    xs = [x * 100 for n, x in r.get(k, {}).items()
          if n.startswith(("exact_match", "acc")) and "stderr" not in n]
    if xs: print(f"  {k:14s} {max(xs):6.2f}   (우연 25.0)")
print("합침 ->", os.path.join(dst, "results.json"))

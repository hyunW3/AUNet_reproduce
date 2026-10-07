# Robustness 평가: 논문 4태스크 기준 스크립트 위치

**프로토콜**: HellaSwag / ARC-E / ARC-C / PIQA, 태스크당 limit 2000, perturbation seed 1234, 문맥과 보기를 모두 변형.
BoolQ는 2026-10-06에 제외했습니다(yes/no 보기는 변형할 수 없음). PBP만 5태스크 정의가 남아 있습니다.

**지표**: HellaSwag/ARC는 `acc_norm`, PIQA는 `acc`, PBP는 전부 `acc`입니다.
축 값은 4태스크 평균 (변형 − clean)의 |Δ|입니다.

## 표 생성

`scripts/probes/paper_robustness_tables.py --overleaf paper_overleaf`
→ `table_appendix/robustness_detail.tex`, `tables/robustness_category.tex`

## 축별 실행 스크립트 → 결과 위치

| 축 | Transformer / AUNet / BPEByte | BLT (θ=1.34, 1.61) · H-Net |
|---|---|---|
| Noise (5전략, `_both`) | `lingua/scripts/eval/robustness/queue_robustness_paper.sh` (+ `_ext` config) → `run_robustness_local.sh` → `runs/robustness_paper1p3b{,_ext}/<arm>/` | `scripts/probes/ext_ci/run_ext.py --axis noise` → `reports/ext_ci/blt_official/{,t1609_}noise_*.json`, `reports/ext_ci/hnet/noise.json` |
| Typo (typoboth, 4편집 × char/word) | `run_typoleet_both.sh` `trio_typo` → `runs/robustness_typoleet_both/trio_typo_<arm>/` | `run_ext.py --axis typoboth` → `runs/robustness_typoleet_both/blt{1335,1609}_typoboth_*.json`, `hnet_typoboth.json` |
| Despace (`despaceall`, 공백 100% 제거) | `runs/robustness_despace_bits/queue.sh` (`DESPACE_BITS=1 DESPACE_PROBS=""`) → `runs/robustness_despace_bits/<arm>/` | `run_ext.py --axis despace` → `reports/ext_ci/blt_official/{,t1609_}despace.json`, `reports/ext_ci/hnet/despace.json` |
| Leet (`nla_leet_both`) | `run_typoleet_both.sh` `trio_leet` → `scripts/probes/format_mc/run_format_lingua.py` → `runs/robustness_typoleet_both/trio_leet_<arm>/` | `scripts/probes/format_mc/run_format_ext.py` → `blt{1335,1609}_leetboth_*.json`, `hnet_leetboth.json` |
| PBP | `runs/robustness_paper1p3b{,_ext}`의 `pbp_mc` | BLT θ=1.34: `reports/robustness_ext/blt_pbp_*.json`. θ=1.61은 정의상 Δ=0 |

- 변형 코드는 `lingua/apps/aunet/eval_{noise,typo_ds,despace_mc,pbp_mc}.py`입니다. 외부 모델도 같은 코드를 seed 1234로 씁니다.
- 체크포인트(논문 학습본)는 `main/main/1.3B/{llama_1.8B_paper@60000, aunet2_1.3B@180000, bpebyte_br_greedy_root_1.3B@180000}`입니다.
- BLT 설정: 공식 `bytelatent` 경로, xformers 512 window, **batch_size=1**.
  단독 실행기는 `scripts/probes/blt_robustness_native.py`, H-Net은 `scripts/probes/hnet_robustness.py`입니다.

## 주의

- Typo/Leet 실행: `bash scripts/probes/run_typoleet_both.sh runs/robustness_typoleet_both "0 1 2 3"` (`typo-leet-both` 브랜치를 main에 merge, lingua `eval_typo_ds` typoboth 포함).
- `runs/`는 gitignore 대상이라 Despace 큐 `runs/robustness_despace_bits/queue.sh`는 추적되지 않습니다. 실행 당시(2026-09-28)에만 있던 `lingua/run_robustness_local.sh` 사본을 호출하므로, 재실행할 때는 `lingua/scripts/eval/robustness/run_robustness_local.sh`를 쓰면 됩니다. 결과는 이미 있어 표 생성에는 영향이 없습니다.
- CI 스크립트(`robustness_bootstrap_ci.py`, `ci_main_table.py`, `format_mc/robust_metric_table.py`)는 아직 5태스크(BoolQ 포함)와 이전 Typo 기준이라 논문 표와 다릅니다.

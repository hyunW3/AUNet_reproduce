# BLT 약점 probe — 결론 (2026-10-02)

후보와 근거는 `CANDIDATES.md`, 전체 표는 `RESULTS.md` (`scripts/probes/blt_weak/report.py`로 생성), 그림은
`dist512.png`. 4개 모델(Llama-1.8B / AU-Net / BPEByte-rg 1.3B / BLT-1B official)이 byte-identical prompt
10,200개를 같은 runner(`run_longctx.py`, greedy + substr / rank)로 채점. BLT는 bs 1, ≤4096 B.

## 한 줄 요약

**가설(후보 1·2)은 대부분 반증됐다.** BLT patching의 메커니즘 차이(반복 구간 병합, 512 B window에 따른
segmentation 변화)는 patch probe에서 그대로 확인됐지만, **정확도로는 BLT가 거의 모든 probe에서 BPEByte와
같거나 앞선다.** BPEByte가 앞선 셀은 near-collision MK-NIAH `rand` K=2(0.96 vs 0.85, 95% CI가 겨우 분리)와
dist512 hash d=2048(0.74 vs 0.67, CI 겹침) 정도이고, 같은 probe의 다른 K / d에서는 BLT가 앞서므로 일관된 우위가 아니다.

## 주요 수치

| probe | BPEByte | BLT-1B | 비고 |
|---|---:|---:|---|
| HashHop 3584 B | 0.25 | **0.92** | BPEByte는 길이에 따라 0.77→0.25로 감소, BLT는 ≤2 KB에서 1.00 |
| MK-NIAH K=8 (word key) | 0.61 | **0.98** | |
| near-collision MK-NIAH, Δ vs rand (평균 nc_*) | −0.34 | −0.33 | 하락 폭 동일, 절대값은 BLT가 높음 (nc_last K=8: 0.30 vs 0.57) |
| S-NIAH-1/2/3 `distract` Δ | −0.20 | **−0.04** | BLT는 무관 distractor에 거의 영향 없음 |
| S-NIAH-1/2/3 `key_nc` Δ | −0.61 | **−0.41** | |
| S-NIAH-1/2/3 `keyval_nc` Δ | −0.65 | −0.53 | |
| count (few-shot, 20 options) | 0.04–0.08 | 0.08–0.10 | **전 모델 chance(0.05) 근처** → 판별력 없음 |
| copy rand L=128 | 0.44 | **1.00** | |
| copy periodic_mut L=32 / 64 | 0.36 / 0.04 | 0.42 / 0.14 | 전 모델 붕괴; Llama 0.54 / 0.11 |
| dist512 hash, d ≤ 512 → d > 512 | 0.90 → 0.74 | 0.99 → 0.86 | BLT는 d ≥ 2048에서만 0.67–0.72 (BPEByte 0.74 / 0.60과 동급) |

## 메커니즘은 확인됨 (patch probe, §8)

- **반복 병합**: count `word`(평균 48 B) 구간이 BLT는 2.9 patch, BPEByte는 7.4 patch이다. copy에서 BLT가 생성하는
  답(128 B random)은 **2.1 patch**에 담긴다. 앞선 copy 결과가 1.00이므로 긴 patch가 정확한 복사를 막지 못한다.
  local decoder가 patch 내부의 byte를 충분히 복원한다.
- **512 B window**: dist512 hash에서 BLT query key(12 B)의 patch 수는 d가 커지면서 1.1(d ≤ 448) → 4.5(512–768)
  → 11.5(≥ 2048)로 바뀌고, 정답 span도 1.7 → 7 → 11.9로 바뀐다. 같은 문자열의 segmentation이 거리에 따라 달라진다는
  예측은 정확히 맞다. BPEByte는 모든 d에서 6.7 patch로 일정하다. 다만 BLT 정확도 하락은 d ≥ 2048에서만 나타나고
  (1.00 → 0.67–0.72), 512 경계 자체에서 생기는 하락은 작다(0.94).
- **near-collision**: `nc_last`에서 BLT query key는 1.4 patch로 병합된다(`rand`는 5.2). 그런데도 정확도 하락 폭은
  다른 모델과 같다. 병합은 일어나지만 손해로 이어지지 않는다.

## 부수 발견 (논문에 쓸 만한 것)

1. **BLT 시스템 취약성**: official BLT는 copy의 반복성 입력(`periodic`, `words_mut`)에서만 94/600 row(16%)가
   `CUDA illegal memory access`로 crash했다. `rand`와 `periodic_mut`(entropy가 높은 입력)에서는 0건이다. 매우 긴
   patch가 생기는 입력에서 kernel이 깨지는 것으로 보인다. crash row는 표에서 제외했으므로 해당 셀의 BLT n은
   76–90이다. 정확도 약점이 아니라 robustness / serving 약점이다(후보 6과 같은 축).
2. **우리 byte 모델의 거리 약점**: hash needle에서 **AU-Net은 d ≥ 448부터 0.00**이고(d=384에서 0.64),
   BPEByte도 d=448에서 0.96 → 0.72로 떨어진다. num needle에서는 모두 1.00이다. 즉 단어 경계가 없는 12-byte
   random 문자열이 needle에서 ~450 B 이상 떨어지면 trio의 byte 단계가 복사하지 못한다. local(byte) attention
   window 쪽 한계로 보인다. 같은 hierarchical 구조이고 경계가 448–512 사이라는 점이 근거다. BLT의 약점이 아니라
   **우리 모델의 약점**이다.
3. HashHop / MK-NIAH의 BLT arm: 기존 trio 결과(`reports/niah/hashhop_1p3b`)와 같은 생성기로 만들었다. BLT가 크게
   앞선다(1.3B matched 비교에서 BLT는 외부 reference이므로 공정성 각주 필요: 학습 데이터·토큰 수가 다름).

## 다음에 해볼 만한 것

- 남은 후보 3(echo / priming)과 4(OOD 입력에서 patch 수 폭주 → **compute / latency** 비교)는 정확도 대신
  **효율 축**이다. BLT가 정확도로는 강하므로 BPEByte의 우위는 "같은 정확도에서 예측 가능한 patch 수와 결정성"
  쪽에서 주장하는 편이 현실적이다.
- count probe는 1B base 모델에서 floor다. 쓰려면 0–5 범위의 쉬운 셋으로 다시 만들거나 rank 대신 비교형
  ("A가 B보다 많나?")으로 바꿔야 한다.
- 위 부수 발견 2(trio hash-needle 거리 붕괴)는 local window 설정을 확인한 뒤 따로 파고들 가치가 있다.

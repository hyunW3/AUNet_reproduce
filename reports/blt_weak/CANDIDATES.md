# BLT가 약할 만한 perturbation / task — BPEByte 우위 후보

작성 2026-10-02. **결론은 `FINDINGS.md`** (후보 1·2 반증; 3은 유의한 BLT 약점, 4·6은 비용 예측성·결정성 축에서 확인, 5는 미미 — 2차 섹션 참고). 비교 대상: BLT-1B (official bytelatent + xformers, entropy 512 window, released threshold
1.3354, bs 1) vs. matched 1.3B trio (Llama / AU-Net / BPEByte-rg). 실험 결과는 `RESULTS.md`, 코드는
`scripts/probes/blt_weak/`.

## 먼저 피할 방향 (BLT가 이미 강한 축)

| 축 | 근거 |
|---|---|
| 공백 제거 (despace) | `reports/despace/summary.md`: all100 macro acc BLT 0.514 vs BPEByte 0.428 / Llama 0.414 / AU-Net 0.345 |
| typo / 문자 조작 / 철자 (CUTE류) | BLT 논문 강점 축. ext_ci 결과 Noise 13.0 / Typo 4.2 / Despace 5.9 (Δ) |
| digit 산술 | 숫자는 entropy가 높아 BLT가 거의 1-byte patch → 산술엔 오히려 유리 |

## 후보 (기대 효과 순)

### 1. 반복 / 카운팅 — 예측 가능 구간이 하나의 긴 patch로 병합
- **메커니즘**: BLT patch 경계 = entropy model의 next-byte entropy > threshold. 반복 구간은 entropy가 낮아
  긴 patch 하나가 되고, released 설정은 `max_patch_length=None`이라 상한도 없다. global model이 반복 횟수·위치를
  볼 수 없다. BPEByte는 경계가 bytes의 순수 함수(BPE root)라 반복 1회당 같은 수의 patch.
- **probe**: `count` (few-shot, word / char / digit / chunk_sep / chunk_nosep, gold 2..15, rank over " 1\n".." 20\n"),
  `copy` (rand / periodic / periodic_mut / words_mut, L=32/64/128, exact).
- **smoke에서 이미 보인 것**: `apple` ×12 (83 B) → BLT 2 patches (최장 82 B), BPEByte 13 patches.

### 2. Segmentation inconsistency — 같은 문자열이 문맥/거리에 따라 다르게 쪼개짐
- **메커니즘**: entropy model은 512-byte sliding window. key가 처음 나오면 잘게(1 B), 512 B 안에서 다시 나오면
  (induction) 굵게, 512 B 밖에서 다시 나오면 다시 잘게 쪼개진다 → query와 needle에서 같은 key의 표현이 다름.
  공통 prefix를 공유하는 near-collision key는 prefix 뒤 suffix가 낮은 entropy로 긴 patch에 묻힌다.
- **probe**:
  - `hashhop` (hops=1, 16-letter KEY = 'VAL', 512–3584 B) — BLT arm 추가
  - `mkniah_word` (RULER MK-NIAH, K=1/2/4/8, 2 KB) — BLT arm 추가
  - `mkniah_nc` — 8-char code key; distractor key = random(`rand`) 또는 target key의 **한 글자만** 바꾼 것
    (`nc_last` / `nc_mid` / `nc_first`), K=2/4/8
  - `sniah{1,2,3}_nc` — S-NIAH-1/2/3 + distractor needle 3개: `clean` / `distract`(무관 key) /
    `key_nc`(target key에서 1글자 다른 key) / `keyval_nc`(key_nc + 값도 tail만 다름), 1024/2048/3200 B
  - `dist512` — 고정 ~3.5 KB essay, needle과 query 사이 거리 d ∈ {64..3072} (512 경계 sweep), hash / num needle
- **mechanism probe**: `patch_probe.py` — needle key / query key / 정답 span이 각 모델에서 몇 patch인지.

### 3. Echo / priming perturbation (BLT 전용 confound)
- 질문에 선택지 문자열을 미리 넣으면(`A) … B) …`) 정답 continuation의 entropy가 떨어져 patch가 굵어지고
  loglik scoring이 바뀐다. 의미는 같은데 BLT만 segmentation이 바뀐다.
- probe 예: 질문 2회 반복, 선택지 사전 나열, 오답만 echo. 지표: Δacc + 정답 선택지 patch 수 변화. → `echo` probe (2차)

### 4. Entropy model 분포 밖 입력 — patch 크기 폭주 / 붕괴
- random / hex / base64 / UUID haystack → 거의 1-byte patch → global seq 폭증, latency·effective context 악화.
  boilerplate / 표 / 로그처럼 반복이 많으면 반대로 과도하게 긴 patch.
- BPEByte는 random에서도 1.39 B/patch (Llama 1.33)로 예측 가능.
- `reports/compression_stability/summary.md`의 BLT 행이 전부 NOT MEASURED → 채우면 figure 하나. → `ood` probe + latency (2차)

### 5. Entropy만 흔드는 adversarial insertion
- zero-width space, soft hyphen, NBSP, 희귀 이모지 삽입 → 의미는 그대로, BLT는 entropy spike로 주변 경계가
  연쇄적으로 바뀌고 3–8gram hash embedding도 교란. BPE도 쪼개지므로 차이가 작을 수 있어 우선순위 낮음. → `insert` probe (2차)

### 6. 결정성 / batch invariance (시스템 강점)
- BLT는 batch / dtype에 따라 threshold 근처 경계가 flip (`reports/blt_batch_invariance`: bs>1에서 task당 1–3% item
  flip). BPEByte는 entropy model이 없어 경계가 입력만으로 결정 → prefix caching, incremental decoding, 재현성.

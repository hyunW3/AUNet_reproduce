# Whitespace / format robustness (1B, 2026-10-02 ~ 10-04)

공백·구두점·포맷 변형에 대한 1B 모델 robustness를 논문 robustness 프로토콜(5태스크 × ≤2000 items)과 같은
아이템·스코어러로 측정한 실험이다. 원자료와 전체 표는 [`format_robustness/`](format_robustness/) 에 있다.

## 요약

- **공백은 빈도가 높을수록 모든 모델이 단조롭게 나빠진다.** 단어 사이 공백을 2–4칸/5칸으로 늘리면(p=1.0)
  2.3–6.3pt 하락. 공백을 *없애는* despace(7–22pt)보다는 훨씬 작다.
- **H-Net만 `\n\n`에 붕괴한다.** 모든 단어 사이를 `\n\n`으로 바꾸면 H-Net −13.8pt(ARC-E −22.5, BoolQ −15.1),
  나머지 세 모델은 −4.5~−5.3. 단일 `\n`(line_split)에서는 H-Net −1.8로 오히려 가장 작다.
- **구두점·특수기호 제거는 거의 무해하다** (punct_drop p=1.0: −0.3~−2.1, symbol_drop ≈0).
- **NL-Augmenter Leet Transformation이 가장 강한 교란** (−12.0~−16.1, BLT 최악). 논문 메인 표에 Leet 열로 반영.
- **FormatSpread spread**(기본 + 10개 형식의 최대−최소): AU-Net 5.0, H-Net 5.5, BPEByte 9.1, BLT 9.2.
  설명어와 본문 사이에 `' \n\t'`, `'\n    '` 같은 공백 제어문자 구분자가 들어가는 형식에서 크게 떨어진다.
- AU-Net은 BPEByte보다 탭·구두점·대소문자 변형에 강하고, 뚜렷하게 약한 셀은 없다
  (공백을 *제거*하는 despace에서만 크게 약하다 — 별도 실험).

## 프로토콜

| 항목 | 내용 |
|---|---|
| 태스크 | HellaSwag / ARC-E / ARC-C / PIQA / BoolQ, 각 앞 2000개 (ARC-C 1172, PIQA 1838 전량). 아이템은 `format_robustness/items/` 에 json으로 고정 |
| 채점 | lm-eval `doc_to_text` 문맥 + `" <option>"` 연속의 loglikelihood. acc = 총 logprob argmax, acc_norm = 바이트 정규화 |
| 기본 지표 | **논문 규약**: HS/ARC는 acc_norm, PIQA·BoolQ는 acc. (acc 전용·acc_norm 전용 표도 함께 제공) |
| 교란 범위 | 질문 블록만. 마지막 `Answer:` cue 줄과 보기는 바이트 단위로 clean과 동일. HellaSwag는 cue가 없어 문맥 전체를 교란하고 뒤쪽 패딩은 넣지 않음 |
| 빈도 | p ∈ {0.5, 1.0}; 적용 가능한 각 위치를 아이템별 시드(1234+idx)로 독립 적용, p=0.5 위치 ⊂ p=1.0 위치 |
| Δ와 CI | 같은 아이템의 clean 대비 perturbed, 5태스크 macro; 95% paired item bootstrap |

| 모델 | 체크포인트 / 실행 |
|---|---|
| AU-Net 2 1.3B | `main/main/1.3B/aunet2_1.3B` step 180000 (논문본) |
| BPEByte rg 1.3B | `main/main/1.3B/bpebyte_br_greedy_root_1.3B` step 180000 (논문본, 전체 md5 검증 사본) |
| BLT-1B | official bytelatent, xformers, 공개 threshold θ=1.34 (Leet만 θ=1.61도), batch 1 |
| H-Net 1-stage XL | eval_suite harness, batch 8 |
| Transformer (Llama 1.8B) | Leet만 측정 (메인 표용) |

## 변형 (35개)

| family | 변형 | 동작 |
|---|---|---|
| 공백 | `space_run` | 단어 사이 공백 → 2–4칸 |
| | `pad_space` | 블록 앞뒤 4칸 + 단어 사이 공백 → 5칸 |
| | `pad_newline` | 블록 앞뒤 `\n\n` + 단어 사이 공백 → `\n\n` |
| | `punct_space` | `, . ; : ? !` 앞에 공백 |
| 제거 | `punct_drop` | `. , ; : ? ! ' "`(곡선 따옴표·말줄임표 포함) 삭제 |
| | `symbol_drop` | 그 밖의 비영숫자·비공백 문자(`( ) - / & % $` …) 삭제 |
| ReCode식 | `line_split` | 단어 사이 공백 → `\n` (LineSplit) |
| | `tab` | 단어 사이 공백 → `\t` (Tab-Indent) |
| | `nl_insert` | 줄바꿈·문장 끝·cue 앞에 빈 줄 (NewlineInsert) |
| NL-Augmenter | `nla_*` 6종 | 원본 코드·기본 파라미터·고정 시드: whitespace_perturbation, underscore_trick, butter_fingers, change_char_case, swap_characters, **Leet Transformation** |
| FormatSpread | `fs00`–`fs09` | 원본 문법 목록에서 (설명어 대소문자 × 구분자 × 필드 연결자) 10개 샘플(seed 42) |

ReCode는 코드 생성(pass@k) 벤치마크라 1B 베이스 모델로는 의미 있는 점수가 안 나와, 포맷 변환만 MC 프롬프트에 적용했다.
NL-Augmenter의 모델 기반 변환(패러프레이즈, 구두점 복원)은 제외했다.

## 결과 (논문 규약, 5태스크 macro Δ pt; 전체 표·CI는 [`format_robustness/summary_paper.md`](format_robustness/summary_paper.md))

clean: AU-Net 58.6 / BPEByte 59.5 / BLT 62.7 / H-Net 57.8

| 변형 | AU-Net | BPEByte | BLT-1B | H-Net |
|---|---|---|---|---|
| space_run p0.5 → p1.0 | −1.4 → −2.9 | −2.1 → −3.8 | −1.1 → −2.3 | −2.6 → −4.3 |
| pad_space p0.5 → p1.0 | −2.2 → −4.6 | −3.5 → −6.3 | −1.7 → −4.0 | −1.5 → −4.4 |
| pad_newline p0.5 → p1.0 | −3.0 → −4.5 | −3.4 → −5.0 | −2.3 → −5.3 | −2.5 → **−13.8** |
| punct_space p1.0 | −0.1 | −1.2 | −0.0 | +0.3 |
| punct_drop p1.0 | −0.3 | −2.1 | −1.2 | −0.7 |
| symbol_drop p1.0 (22–25% 아이템만 해당) | +0.4 | +0.3 | −0.0 | −0.6 |
| line_split p1.0 | −3.1 | −3.5 | −5.5 | −1.8 |
| tab p1.0 | −3.4 | −3.5 | −3.2 | −4.6 |
| nl_insert p1.0 | −0.5 | −0.1 | −0.5 | −0.6 |
| NL-Aug whitespace | −2.5 | −2.5 | −2.4 | −3.2 |
| NL-Aug underscore | −0.3 | −0.8 | −0.6 | −0.7 |
| NL-Aug butter_fingers | −2.4 | −3.8 | −3.2 | −2.6 |
| NL-Aug change_char_case | −0.2 | −1.4 | −2.3 | −2.0 |
| NL-Aug swap_characters | −2.9 | −3.6 | −2.6 | −3.8 |
| **NL-Aug Leet** | −12.7 | −12.9 | −16.1 | −12.0 |
| FormatSpread spread (max−min) | 5.0 | 9.1 | 9.2 | 5.5 |

H-Net `pad_newline_p100` 태스크별: HS −15.3, ARC-E −22.5, ARC-C −12.5, PIQA −3.5, BoolQ −15.1.

## 논문 반영 (paper_overleaf main)

- **Leet 열** (`cea9336`, `12b687d`, `d41d486`): 메인 표 Robustness에 Leet 축, 부록 `tab:robustness_detail`에 Leet 블록,
  캡션은 "Leet Transformation of NL-Augmenter~\citep{dhole2023nl}". Transformer와 BLT θ=1.61은 Leet만 추가 측정
  (`format_robustness/leet_extra/`).
- **지표 규약 통일** (`db59d25`): Noise/Typo/Despace/Leet는 HS/ARC acc_norm, PIQA·BoolQ acc; PBP는 모든 태스크 acc.
  메인 표 값·CI·굵게·Avg(5축 평균), 부록 두 표, 5절 본문 수치, 부록 지표 설명, 레이더 그림 갱신.

| 메인 표 |ΔAcc| | PBP | Noise | Typo | Despace | Leet | Avg |
|---|---|---|---|---|---|---|
| Transformer | 10.65 | 14.2 | 5.8 | 14.1 | 14.3 | 11.8 |
| AUNet | 0.05 | 11.0 | 3.9 | 21.9 | 12.7 | 9.9 |
| BPEByte | 0.14 | 11.4 | 3.8 | 13.8 | 12.9 | 8.4 |
| BLT θ=1.34 | 0.00 | 12.7 | 4.0 | 7.3 | 16.1 | 8.0 |
| BLT θ=1.61 | 0.00 | 12.1 | 3.5 | 6.7 | 15.1 | 7.5 |
| H-Net | 0.00 | 11.8 | 4.8 | 12.5 | 12.0 | 8.2 |

**PBP를 acc로 남긴 이유**: trailing-space 이동은 바이트 모델의 모든 보기 logprob을 같은 상수만큼 옮기므로 acc 변화가
정확히 0이어야 한다(BLT·H-Net 재측정에서 0.00 재현). acc_norm은 보기마다 정규화 길이가 1바이트씩 달라져 경계에 둔감한
모델도 0.18–0.26pt 움직인다 — PBP가 아니라 정규화 artifact (`format_robustness/pbp_metric_variants.json`).
Despace는 텍스트 내용 자체를 바꾸는 교란이라 이 논리가 적용되지 않으므로 Noise/Typo와 같은 규약으로 맞췄다.

## 검증

- 첫 20개 PIQA/BoolQ의 clean 정답 비트가 논문 despace 실행과 일치: AU-Net·BLT 40/40, H-Net·BPEByte 39/40
  (bf16 배치 구성 차이로 1개씩).
- BLT 샤드 A/B의 clean 비트가 5태스크 모두 완전히 동일(bs 1 결정적).
- `robust_metric_table.py`에 기존 규약(BoolQ만 acc)을 넣으면 논문의 기존 Noise/Typo/Leet 값이 그대로 재현됨.
- Leet 추가 측정(snu55)의 clean 비트는 논문 실행과 0.3–1.6% 다름(하드웨어·배치 차이). Δ는 같은 실행 안에서 짝지어
  계산하므로 영향 없음.

## 주의점

- BoolQ는 yes/no 2지선다라 프롬프트 변화가 사전확률을 움직인다. AU-Net은 공백 변형에서 BoolQ가 오히려 +1~+2.6pt 오른다.
- symbol_drop은 해당 기호가 있는 아이템이 22–25%뿐이라 효과가 희석된다(변경된 아이템만의 Δ는 `summary*.json`의 `delta_changed`).
- NL-Augmenter 변환 중 swap_characters를 뺀 5개는 원본처럼 매 호출 고정 시드라 모든 아이템이 같은 난수열을 쓴다.
- ece-agpu18의 long-context suite가 쓰던 `~/AUNet_lc/ckpt/bpebyte`는 9월 재학습본(a100x4)을 가리키고 있었다.
  이 실험은 처음부터 검증된 논문본 사본을 썼고, 링크는 2026-10-04에 논문본으로 교체했다.

## 파일

| 경로 | 내용 |
|---|---|
| `format_robustness/summary_paper.md` · `summary.md` · `summary_acc_norm.md` | 35변형 × 4모델 전체 표 (논문 규약 / acc / acc_norm), 태스크별 Δ, FormatSpread spread |
| `format_robustness/raw/` | 실행별 결과 json (per-item 비트 포함) |
| `format_robustness/items/` | 고정 아이템 |
| `format_robustness/leet_main.json`, `leet_extra/` | 메인 표 Leet 열(6행), Transformer·BLT θ=1.61 추가 측정 |
| `format_robustness/robust_piqa_acc.json` | 논문 규약 메인 표 값 + CI |
| `format_robustness/pbp_accnorm/`, `pbp_metric_variants.json` | PBP acc/acc_norm 재측정 |
| `../scripts/probes/format_mc/` | 변형·스코어러(`format_mc.py`), 드라이버, 런처, 집계 스크립트 |

재현:
```
PYTHONPATH=lingua python scripts/probes/format_mc/dump_items.py --items_dir reports/format_robustness/items
bash scripts/probes/format_mc/run_ece.sh <gpu> aunet|bpebyte|hnet <out>
bash scripts/probes/format_mc/run_ece.sh <gpu> blt <out> 2000 <task> A|B
python scripts/probes/format_mc/report.py --in_dir reports/format_robustness/raw --out_dir reports/format_robustness --metric paper
python scripts/probes/format_mc/robust_metric_table.py --acc boolq,piqa
```

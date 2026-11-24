# Patch-Lag Mitigation with a Lightweight Residual Updater

## 1. 연구 목표

계층형 바이트 레벨 트랜스포머에서 발생하는 **patch lag**를, 기존 백본을 재학습하지 않고 경량 모듈만 추가 학습하여 완화한다.

핵심 방향은 다음과 같다.

- 기존 hierarchical byte-level transformer는 고정한다.
- 가장 최근에 확정된 글로벌 임베딩은 그대로 유지한다.
- 아직 완료되지 않은 현재 패치의 정보를 이용해 경량 모듈이 residual update `delta`를 예측한다.
- 보정된 임베딩은 `기존 글로벌 임베딩 + gate * delta`로 구성한다.
- 실제 글로벌 임베딩이 제공되는 바이트 위치에서는 경량 모듈을 사용하지 않는다.
- 프리트레이닝 단계에 모듈을 통합하는 실험은 현재 범위에서 제외한다.

---

## 2. 문제 정의

패치 `p`의 마지막 바이트가 입력되기 전에는 현재 패치의 글로벌 임베딩 `g_p`가 아직 계산되지 않는다. 따라서 패치 내부의 중간 바이트 위치에서는 직전 패치의 글로벌 임베딩 `g_{p-1}`을 계속 사용하게 된다.

이로 인해 현재 패치에서 새로 들어온 정보가 글로벌 경로에 즉시 반영되지 않는 **patch lag**가 발생한다.

Scratchpad 방식은 현재까지 입력된 부분 패치에 대해 추가 forward를 수행하여 임시 글로벌 임베딩을 계산하지만, 그만큼 추론 연산량이 증가한다.

본 실험에서는 추가적인 전체 forward 대신, 작은 선형 또는 선형 어텐션 모듈로 임시 글로벌 임베딩을 근사한다.

---

## 3. 제안 구조

현재 시점 `t`가 패치 `p` 내부에 있다고 하자.

- `g_{p-1}`: 가장 최근에 완료된 패치의 글로벌 임베딩
- `H_t`: 현재 시점까지 사용할 수 있는 frozen byte-level hidden states
- `U_theta`: 추가 학습할 경량 업데이트 모듈
- `delta_t`: 현재 패치 정보로부터 예측한 residual update
- `alpha_t`: residual의 크기를 조절하는 gate
- `m_t`: updater 적용 여부를 나타내는 마스크

```text
delta_t = U_theta(H_t, g_{p-1})
alpha_t = Gate_theta(H_t, g_{p-1})
```

실제 글로벌 임베딩이 아직 없는 위치에서는 다음과 같이 보정한다.

```text
g_hat_t = g_{p-1} + alpha_t * delta_t
```

현재 패치의 실제 글로벌 임베딩 `g_p`가 제공되는 위치에서는 updater를 사용하지 않는다.

```text
if exact_global_embedding_is_available:
    g_hat_t = g_p
else:
    g_hat_t = g_{p-1} + alpha_t * delta_t
```

이를 하나의 식으로 쓰면 다음과 같다.

```text
g_hat_t = m_t * (g_{p-1} + alpha_t * delta_t)
          + (1 - m_t) * g_p
```

여기서 `m_t = 1`은 patch-lag가 존재하는 패치 내부 위치이고, `m_t = 0`은 실제 글로벌 임베딩이 삽입되는 위치이다.

---

## 4. Gate의 역할

Gate는 예측된 residual이 기존 글로벌 임베딩을 과도하게 변경하지 않도록 크기를 조절한다.

기본 형태는 다음과 같다.

```text
alpha_t = sigmoid(a_t)
```

비교할 gate 형태는 다음과 같다.

| Gate 유형 | 정의 | 목적 |
|---|---|---|
| Fixed gate | `alpha = 1` | gate가 실제로 필요한지 확인하는 대조군 |
| Global scalar | 모든 위치에서 하나의 학습 가능한 스칼라 사용 | 가장 단순하고 안정적인 설정 |
| Input-dependent scalar | 현재 입력으로부터 위치별 스칼라 예측 | 패치 내용에 따라 보정 강도를 조절 |
| Vector gate | 임베딩 차원별 gate 사용 | 표현력은 높지만 파라미터와 불안정성 증가 가능 |

초기 실험에서는 **global scalar** 또는 **input-dependent scalar**를 우선 사용한다.

안정적인 시작을 위해 다음 중 하나를 적용한다.

- gate logit을 음수로 초기화하여 `alpha`가 작은 값에서 시작하도록 한다.
- residual head의 마지막 projection을 zero initialization한다.
- `||alpha_t * delta_t|| / ||g_{p-1}||` 비율을 모니터링한다.

---

## 5. 모델 어블레이션

### 5.1 Linear Residual Head

가장 단순한 기준 모델이다.

```text
s_t = Pool(H_t)
delta_t = W_delta * LN(s_t) + b_delta
```

Input-dependent gate를 사용할 경우 다음과 같이 구성한다.

```text
alpha_t = sigmoid(w_gate^T * LN(s_t) + b_gate)
```

장점:

- 파라미터 수가 작다.
- 연산량과 지연 증가가 매우 작다.
- 구현과 디버깅이 쉽다.

단점:

- 입력을 단일 벡터로 pooling하므로 순서 정보가 많이 손실될 수 있다.

### 5.2 Causal Linear-Attention Updater

현재까지 입력된 byte hidden states를 causal하게 누적하는 경량 모듈이다.

예시 상태 업데이트는 다음과 같다.

```text
S_t = S_{t-1} + phi(k_t) * v_t^T
Z_t = Z_{t-1} + phi(k_t)

c_t = (phi(q_t)^T * S_t) / (phi(q_t)^T * Z_t + epsilon)
delta_t = W_o * c_t
```

특징:

- full softmax attention을 사용하지 않는다.
- 상태를 streaming 방식으로 업데이트할 수 있다.
- 현재 패치 내부의 바이트 순서를 선형 head보다 잘 반영할 수 있다.
- 상태를 어디에서 reset하는지에 따라 입력 범위를 제어할 수 있다.

---

## 6. 입력 범위 어블레이션

반드시 다음 두 가지 입력 범위를 비교한다.

### 6.1 Current-Patch Only

현재 진행 중인 패치의 hidden states만 updater에 입력한다.

```text
H_t = {h_{p,1}, ..., h_{p,j}}
```

- 패치가 시작될 때 updater state를 reset한다.
- 현재 패치의 최신 정보에 집중한다.
- 직전까지의 오래된 정보는 기존 글로벌 임베딩 `g_{p-1}`이 담당한다고 가정한다.
- 제안 아이디어의 핵심 가정과 가장 잘 맞는 설정이다.

Pooling 후보:

- Last hidden state
- Mean pooling
- Mean + Last concatenation

### 6.2 Full-Input / Full-Prefix

현재 시점까지의 전체 입력 hidden states를 updater에 사용한다.

```text
H_t = {h_1, ..., h_t}
```

- Linear head에서는 cumulative mean, cumulative last, 또는 cached pooled state를 사용한다.
- Linear-attention updater에서는 sequence 시작부터 상태를 유지한다.
- 전체 prefix를 매 시점마다 다시 forward하지 않고, cached hidden state 또는 streaming state를 사용한다.
- 현재 패치 정보 외에 장기 문맥이 updater 자체에 필요한지 검증한다.

### 6.3 선택적 보조 입력: Previous Global + Current Patch

다음 입력도 보조 어블레이션으로 사용할 수 있다.

```text
s_t = concat(g_{p-1}, Pool(H_current_patch))
```

이 설정은 current-patch-only와 full-prefix 사이의 절충안이다.

---

## 7. 학습 범위

기본 원칙은 **backbone retraining 없이 추가 모듈만 학습**하는 것이다.

### 7.1 Head Only

- 기존 hierarchical byte-level transformer 전체를 freeze한다.
- residual updater와 gate만 학습한다.
- 가장 먼저 수행할 기본 설정이다.

### 7.2 LoRA Only

- residual updater를 사용하지 않는다.
- 기존 모델의 제한된 projection에만 LoRA를 부착한다.
- 추가 파라미터만으로 patch lag를 간접적으로 보상할 수 있는지 확인하는 대조군이다.

권장 부착 위치:

- 글로벌 임베딩이 local representation과 결합되는 projection
- 해당 fusion block의 `Q`, `V`, 또는 output projection
- 필요할 경우 해당 block의 FFN projection

초기 rank 후보:

```text
r in {4, 8, 16}
```

### 7.3 LoRA + Residual Updater

- residual updater와 gate를 유지한다.
- 글로벌 임베딩 결합부에 낮은 rank의 LoRA를 추가한다.
- updater가 생성한 새로운 분포에 frozen backbone이 적응할 필요가 있는지 확인한다.

공정한 비교를 위해 다음을 함께 보고한다.

- 학습 파라미터 수
- 추론 시 추가 파라미터 수
- 추가 FLOPs
- latency
- 성능 대비 파라미터 효율

---

## 8. 학습 타깃과 목적 함수

### 8.1 Teacher Residual Distillation

가장 직접적인 학습 방식이다.

Scratchpad 방식의 추가 forward로 계산한 임시 글로벌 임베딩을 teacher로 사용한다.

- `g_teacher_t`: 현재 부분 패치에 대해 Scratchpad 추가 forward가 생성한 임시 글로벌 임베딩
- `g_{p-1}`: 직전 완료 패치의 글로벌 임베딩

학습할 residual target은 다음과 같다.

```text
delta_target_t = g_teacher_t - g_{p-1}
```

모델은 다음을 근사한다.

```text
g_hat_t = g_{p-1} + alpha_t * delta_t
           approximately g_teacher_t
```

Representation loss 예시는 다음과 같다.

```text
L_mse = MSE(g_hat_t, g_teacher_t)
L_cos = 1 - cosine_similarity(g_hat_t, g_teacher_t)
L_repr = L_mse + lambda_cos * L_cos
```

장점:

- updater가 해야 할 역할이 명확하다.
- Scratchpad의 성능 이득을 작은 모듈에 distill할 수 있다.
- teacher 계산은 학습 데이터에 대해 offline cache할 수 있으므로, 추론 시 추가 forward는 필요 없다.

### 8.2 Task Loss Only

보정된 글로벌 임베딩을 사용해 기존 next-byte 또는 language-model objective를 직접 최적화한다.

```text
L_task = CrossEntropy(logits(g_hat_t), target_t)
```

장점:

- 최종 성능을 직접 최적화한다.
- teacher embedding의 표현 공간을 그대로 모사할 필요가 없다.

단점:

- 작은 head만 학습할 때 gradient가 약하거나 불안정할 수 있다.
- updater가 실제로 patch lag를 해결했는지 해석하기 어렵다.

### 8.3 Combined Objective

권장 기본 objective이다.

```text
L = L_task
    + lambda_repr * L_repr
    + lambda_reg * ||alpha_t * delta_t||_2^2
```

비교할 objective:

1. Distillation only
2. Task loss only
3. Distillation + task loss

실제 글로벌 임베딩이 제공되는 위치에서는 updater loss를 계산하지 않거나 loss mask를 0으로 설정한다.

---

## 9. 학습 안정화

다음 항목을 기본적으로 적용하거나 어블레이션한다.

- Backbone parameter freeze
- Residual output projection zero initialization
- Gate의 작은 초기값
- Gradient clipping
- Residual norm regularization
- Updater에만 독립적인 learning rate 적용
- LoRA와 updater에 서로 다른 learning rate 적용
- Patch boundary 위치 loss masking

권장 모니터링 값:

```text
mean(alpha_t)
std(alpha_t)
||delta_t||
||alpha_t * delta_t|| / ||g_{p-1}||
cosine(g_hat_t, g_teacher_t)
MSE(g_hat_t, g_teacher_t)
```

주의:

- Task loss를 통해 head를 학습할 때, backbone 파라미터는 freeze하되 `g_hat_t` 이후의 연산 그래프까지 `torch.no_grad()`로 감싸면 안 된다.
- 파라미터 gradient는 막되, loss에서 updater까지의 gradient path는 유지해야 한다.
- Distillation-only 학습에서는 frozen feature와 teacher embedding을 미리 cache하여 학습 비용을 줄일 수 있다.

---

## 10. Baseline

다음 baseline을 비교한다.

| ID | 방법 | 설명 |
|---|---|---|
| B0 | Original | 직전 패치 글로벌 임베딩을 그대로 사용 |
| B1 | Scratchpad | 부분 패치에 대해 추가 forward를 수행하여 임시 글로벌 임베딩 계산 |
| B2 | Oracle/Teacher upper bound | teacher 임베딩을 직접 사용하여 updater 근사의 상한 확인 |
| B3 | LoRA only | updater 없이 LoRA만 학습 |
| M1 | Linear updater | 제안한 linear residual head |
| M2 | Linear-attention updater | 제안한 causal linear-attention updater |
| M3 | Linear updater + LoRA | 선형 updater와 LoRA 혼합 |
| M4 | Linear-attention updater + LoRA | 선형 어텐션 updater와 LoRA 혼합 |

가능하면 학습 파라미터 수를 맞춘 parameter-matched control도 추가한다.

---

## 11. 단계별 실험 계획

전체 조합을 처음부터 모두 수행하지 않고, 다음 순서로 좁혀간다.

### Stage 0. 구현 검증

고정 설정:

- Linear residual head
- Current-patch only
- Mean + Last pooling
- Global scalar gate
- Distillation + task loss
- Backbone frozen
- 글로벌 임베딩 삽입 위치에서는 updater skip

검증 항목:

- residual이 실제로 0이 아닌 값을 학습하는가
- gate가 0으로 collapse하지 않는가
- teacher embedding과의 거리가 감소하는가
- patch 내부 위치의 loss가 개선되는가

### Stage 1. 핵심 2 x 2 비교

| Updater | Input scope |
|---|---|
| Linear | Current patch only |
| Linear | Full input |
| Linear attention | Current patch only |
| Linear attention | Full input |

이 단계에서 가장 중요한 질문은 다음과 같다.

1. 단순 linear head로 충분한가
2. 바이트 순서를 모델링하는 linear attention이 필요한가
3. 현재 패치 정보만으로 충분한가
4. updater에도 전체 문맥이 필요한가

### Stage 2. 입력 요약 및 Gate 어블레이션

Stage 1의 상위 모델에 대해 다음을 비교한다.

입력 요약:

- Last
- Mean
- Mean + Last
- Previous global + Current patch summary

Gate:

- Fixed `alpha = 1`
- Global scalar
- Input-dependent scalar

### Stage 3. Objective 어블레이션

- Distillation only
- Task loss only
- Combined objective

### Stage 4. LoRA 비교

동일한 데이터와 평가 조건에서 다음을 비교한다.

- Head only
- LoRA only
- Head + LoRA

LoRA rank는 우선 `{4, 8, 16}`으로 제한한다.

---

## 12. 평가 지표

### 12.1 모델 품질

- Bits per byte 또는 byte-level NLL
- Perplexity
- Downstream task accuracy
- Long-context task 성능
- Needle-in-a-haystack 또는 위치 민감형 retrieval 성능

### 12.2 Patch-Lag 직접 측정

전체 평균만 보면 patch lag 개선 여부가 가려질 수 있으므로, 반드시 패치 내부 위치별로 분석한다.

- Byte offset within patch별 NLL
- 패치 시작, 중간, 마지막 위치별 성능
- Patch length bin별 성능
- 실제 글로벌 임베딩 삽입 직전 위치의 성능
- Original 대비 Scratchpad gap 중 제안 방법이 회복한 비율

예시:

```text
Recovery Ratio =
    (Score_method - Score_original)
    / (Score_scratchpad - Score_original)
```

### 12.3 Representation 품질

- `MSE(g_hat_t, g_teacher_t)`
- `cosine(g_hat_t, g_teacher_t)`
- `||g_hat_t - g_{p-1}||`
- 패치가 진행됨에 따라 teacher와의 거리가 감소하는지

### 12.4 효율성

- 추가 파라미터 수
- 추가 FLOPs per byte
- Wall-clock latency per byte
- Throughput
- Peak memory
- Streaming state memory
- Scratchpad 추가 forward 대비 연산 절감률

### 12.5 안정성

- 긴 컨텍스트에서 updater state drift
- Gate collapse 여부
- Residual norm 폭증 여부
- Patch boundary에서 embedding discontinuity
- 입력 길이에 따른 성능 저하

---

## 13. 최소 구현 의사코드

```python
# Backbone parameters are frozen.
for param in backbone.parameters():
    param.requires_grad = False

# frozen_features may be cached or detached before the updater.
local_hidden, previous_global, exact_global, is_exact_global_pos = \
    backbone.extract_features(input_bytes)

summary_or_state = updater_input_builder(
    local_hidden=local_hidden,
    previous_global=previous_global,
    scope=input_scope,  # "current_patch" or "full_input"
)

delta = residual_updater(summary_or_state)
alpha = gate(summary_or_state)

predicted_global = previous_global + alpha * delta

# Never use the head where an exact global embedding is already available.
global_for_model = torch.where(
    is_exact_global_pos,
    exact_global,
    predicted_global,
)

# Keep autograd from the loss to delta and alpha even though backbone weights are frozen.
logits = backbone.forward_from_global(
    local_hidden=local_hidden,
    global_embedding=global_for_model,
)

task_loss = cross_entropy(logits, targets)

valid_update_mask = ~is_exact_global_pos
repr_loss = representation_loss(
    predicted_global[valid_update_mask],
    teacher_global[valid_update_mask],
)

reg_loss = (
    alpha[valid_update_mask] * delta[valid_update_mask]
).pow(2).mean()

loss = task_loss + lambda_repr * repr_loss + lambda_reg * reg_loss
loss.backward()
optimizer.step()
```

Linear-attention updater에서 input scope는 state reset 정책으로 구현할 수 있다.

```python
if input_scope == "current_patch" and is_patch_start:
    updater.reset_state()

# full_input 설정에서는 sequence 시작 이후 state를 유지한다.
delta, updater_state = updater.step(current_hidden, updater_state)
```

---

## 14. 구현 체크리스트

- [ ] Backbone 전체 freeze
- [ ] 현재 패치와 full-input 입력 범위 구현
- [ ] 패치 시작 위치와 글로벌 임베딩 삽입 위치 마스크 생성
- [ ] 실제 글로벌 임베딩 위치에서 updater 완전 skip
- [ ] Linear residual head 구현
- [ ] Causal linear-attention updater 구현
- [ ] Fixed, scalar, input-dependent gate 구현
- [ ] Scratchpad teacher embedding 생성 또는 cache
- [ ] Distillation, task, combined objective 구현
- [ ] LoRA-only 및 LoRA+head 설정 구현
- [ ] 패치 내부 byte offset별 metric logging
- [ ] Gate와 residual norm logging
- [ ] FLOPs, latency, memory 측정
- [ ] Original, Scratchpad, Oracle baseline 비교

---

## 15. 주요 가설

### H1. 경량 residual updater는 patch lag를 완화한다.

패치 내부 위치에서 Original보다 낮은 NLL을 보이고, Scratchpad가 제공하는 성능 이득의 일부를 회복할 것이다.

### H2. Current-patch-only 입력만으로도 충분할 수 있다.

과거 정보는 이미 `g_{p-1}`에 요약되어 있으므로, residual updater는 현재 미완료 패치의 정보만 처리해도 충분할 수 있다.

### H3. Linear attention은 긴 패치에서 더 유리하다.

패치 길이가 길거나 내부 순서가 중요한 경우, 단순 pooling 기반 linear head보다 linear-attention updater가 더 좋은 teacher 근사와 task 성능을 보일 수 있다.

### H4. Gate는 학습 안정성과 원본 성능 보존에 기여한다.

Gate를 제거하면 residual이 과도하게 커지거나 기존 글로벌 표현을 훼손할 가능성이 있다.

### H5. LoRA는 updater와 결합할 때 가장 유용할 수 있다.

LoRA-only보다 updater가 patch lag를 직접 완화하고, LoRA는 frozen backbone이 보정된 글로벌 임베딩을 더 잘 활용하도록 적응시키는 역할을 할 수 있다.

---

## 16. 현재 연구 범위에서 제외하는 항목

다음 항목은 초기 결과가 확보된 이후의 후속 연구로 남긴다.

- 프리트레이닝 시작부터 updater를 공동 학습하는 설정
- Backbone 전체 fine-tuning
- 모든 레이어에 updater를 삽입하는 대규모 구조 변경
- 복수 패치의 상세 정보를 updater state에 장기간 보존하는 복잡한 메모리 구조

현재 단계의 목표는 다음 한 문장으로 요약한다.

> 기존 체크포인트를 유지한 상태에서, 작은 residual updater만 학습하여 Scratchpad의 추가 forward를 얼마나 저렴하게 근사할 수 있는지 검증한다.

# AU-Net / BPEByte Online 학습 및 평가 가이드

이 문서는 AU-Net / BPEByte Online 모델의 환경 설정, 학습, 평가 방법을 정리한 가이드입니다.

---

## 0. CUDA / Driver 호환성 확인

먼저 학습 또는 평가를 실행할 서버의 GPU와 드라이버 정보를 확인합니다.

```bash
nvidia-smi
```

확인할 항목은 다음과 같습니다.

* `CUDA Version`
* GPU 모델

예시 기준 서버:

* `snu55`
* NVIDIA driver: `535`
* 지원 CUDA 버전: 약 CUDA 12.2
* GPU: Ampere A5000, `sm_86`

환경 선택 기준은 다음과 같습니다.

* Driver가 **CUDA 12.6 미만**만 지원하는 경우
  → 아래의 **cu121 환경 설정**을 사용합니다.

* Driver가 **CUDA 12.6 / 12.8 이상**을 지원하는 경우
  → cu121로 다운그레이드할 필요 없이 기본 환경 설정을 사용하면 됩니다.

```bash
bash setup/create_env.sh
```

기본 환경 설정은 Torch 2.7+cu128 환경을 사용합니다.

---

## 1. cu121 환경 설정

`snu55`처럼 드라이버가 오래되어 CUDA 12.6/12.8을 지원하지 않는 서버에서는 cu121 환경을 사용합니다.

아래 방법은 기존 conda 환경을 clone하지 않고, 새 환경을 처음부터 생성하는 방식입니다. 따라서 재실행하기 쉽습니다.

```bash
# conda 설치 경로에 맞게 수정
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -y -n aunet_eval_cu121 python=3.11
conda activate aunet_eval_cu121
```

먼저 드라이버와 호환되는 PyTorch 및 xFormers를 설치합니다.

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 xformers==0.0.28.post3
```

그 다음 나머지 dependency를 설치합니다.

```bash
pip install ninja
pip install -r requirements.txt
pip install lm-eval==0.4.12
```

CUDA가 정상적으로 동작하는지 확인합니다.

```bash
python -c "import torch, xformers; print(torch.__version__, torch.version.cuda, xformers.__version__); torch.zeros(1).cuda(); print('CUDA OK')"
```

정상적으로 설정되었다면 다음과 같은 출력이 포함되어야 합니다.

```text
CUDA OK
```

---

## 2. 학습 설정

학습 전에 아래 config 파일의 경로들을 현재 서버 환경에 맞게 수정해야 합니다.

```text
apps/aunet/configs/bpebyte_br_bt_online_1.3B.yaml
```

수정이 필요한 주요 항목은 다음과 같습니다.

```yaml
dump_dir: ...
root_dir: ...
bpe_tokenizer_path: ...
include_path: ...
```

각 항목의 의미는 다음과 같습니다.

* `dump_dir`: 학습 결과와 checkpoint가 저장될 경로
* `root_dir`: 데이터셋 또는 프로젝트 기준 경로
* `bpe_tokenizer_path`: 사용할 BPE tokenizer 경로
* `include_path`: 학습에 포함할 데이터 또는 config 관련 경로

---

## 3. 학습 실행

다음 명령어로 학습을 실행합니다.

```bash
cd lingua
source .venv/bin/activate

torchrun --nproc-per-node 4 --master-port 29501 \
    -m apps.aunet.train \
    config=apps/aunet/configs/bpebyte_br_bt_online_1.3B.yaml
```

여기서 `--nproc-per-node 4`는 GPU 4장을 사용하는 설정입니다. 사용하는 GPU 수에 맞게 조정할 수 있습니다.

---

## 4. Single-Letter Likelihood 평가

Single-letter likelihood 평가는 내부적으로 **방식 2**라고 부르는 평가 방식입니다.

평가 config 파일은 다음 위치에 있습니다.

```text
lingua/apps/aunet/configs/eval_gen_mc_ll_b200.yaml
```

이 config는 HellaSwag와 ARC-Easy에 대해 선택지 letter들의 P(letter | prompt)를 계산해 argmax로 정답을
고르는 방식입니다 (`lingua/eval_tasks/gen_mc/`에 정의된 `hellaswag_gen_ll` / `arc_easy_gen_ll` task). 생성을
하지 않으므로 online bt 디코딩이 불안정해도 동작합니다. Section 5의 평가 스크립트가 사용하는 config와 동일합니다.

---

## 5. 학습된 모델 평가

학습이 완료된 checkpoint만 평가하려면 다음 스크립트를 사용합니다.

```bash
cd lingua
bash eval_bpebyte_online.sh ${trained_model_ckpt_path}
```

예시는 다음과 같습니다.

```bash
cd lingua
bash eval_bpebyte_online.sh runs/bpebyte_br_bt_online_1.3B/checkpoints/0000180000
```

> conda 환경을 사용하는 서버(Section 1)에서는 먼저 `conda activate aunet_eval_cu121`로 환경을 활성화해야 합니다.
> 위 checkpoint 경로는 예시이며, 실제로는 사용하는 서버에 존재하는 checkpoint 디렉터리(`params.json`이 있는
> `.../checkpoints/<step>` 또는 그 안의 `consolidated/`)를 지정해야 합니다.

평가 스크립트에서 사용하는 config는 다음 경로에 있습니다.

```text
lingua/apps/aunet/configs/eval_gen_mc_ll_b200.yaml
```

---

## 6. 주의사항

* `snu55`처럼 오래된 드라이버를 사용하는 서버에서는 cu121 환경을 사용해야 합니다.
* `torch`와 `xformers`는 반드시 `requirements.txt` 설치보다 먼저 설치해야 합니다.
* 그렇지 않으면 `requirements.txt` 설치 과정에서 CUDA 버전이 맞지 않는 PyTorch가 설치될 수 있습니다.
* `lm-eval==0.4.12`는 `snu55` 환경과 동일한 평가 결과를 맞추기 위해 고정합니다.
* CUDA 12.6/12.8 이상을 지원하는 최신 드라이버 서버에서는 cu121 환경을 만들 필요 없이 기본 `setup/create_env.sh`를 사용하는 것이 좋습니다.

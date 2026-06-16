# AU-Net / BPEByte Online Training & Evaluation Setup

This guide explains how to set up the environment, train the BPEByte online model, and run evaluation.

## 0. Check CUDA / Driver Compatibility

Before creating the environment, check the target machine:

```bash
nvidia-smi
```

Look at:

* `CUDA Version`
* GPU model

Example reference machine:

* `snu55`
* NVIDIA driver: `535`
* CUDA compatibility: around `12.2`
* GPU: Ampere A5000, `sm_86`

Use the following rule:

* If the driver supports **CUDA < 12.6**, use the **cu121 setup** below.
* If the driver supports **CUDA 12.6 / 12.8 or higher**, you can skip the cu121 downgrade and use the default setup:

```bash
bash setup/create_env.sh
```

This default setup uses the stock Torch 2.7+cu128 environment.

---

## 1. Environment Setup for cu121

Use this setup for machines similar to `snu55`, where the driver is older and does not support CUDA 12.6/12.8.

This creates a fresh conda environment instead of cloning an existing one, so it is safe to rerun.

```bash
# Adjust this path if your conda installation is elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda create -y -n aunet_eval_cu121 python=3.11
conda activate aunet_eval_cu121
```

Install driver-matched PyTorch and xFormers first:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 xformers==0.0.28.post3
```

Install the remaining dependencies:

```bash
pip install ninja
pip install -r requirements.txt
pip install lm-eval==0.4.12
```

Verify that CUDA works:

```bash
python -c "import torch, xformers; print(torch.__version__, torch.version.cuda, xformers.__version__); torch.zeros(1).cuda(); print('CUDA OK')"
```

Expected output should include:

```text
CUDA OK
```

---

## 2. Training

Before training, update the following paths in:

```text
apps/aunet/configs/bpebyte_br_bt_online_1.3B.yaml
```

Required fields to check:

```yaml
dump_dir: ...
root_dir: ...
bpe_tokenizer_path: ...
include_path: ...
```

Then start training:

```bash
cd lingua
source .venv/bin/activate

torchrun --nproc-per-node 4 --master-port 29501 \
    -m apps.aunet.train \
    config=apps/aunet/configs/bpebyte_br_bt_online_1.3B.yaml
```

---

## 3. Single-Letter Likelihood Evaluation

The single-letter likelihood evaluation is also referred to as **method 2**.

The eval config is located at:

```text
lingua/apps/aunet/configs/eval_gen_mc_ll_b200.yaml
```

It scores P(letter | prompt) over the option letters (argmax) for both HellaSwag and ARC-Easy —
the `hellaswag_gen_ll` / `arc_easy_gen_ll` tasks defined under `lingua/eval_tasks/gen_mc/`. Because
it never generates, it works even when the online bt decode is flaky. This is the same config the
evaluation script in Section 4 uses.

---

## 4. Evaluation Only

To evaluate a trained checkpoint, use:

```bash
cd lingua
bash eval_bpebyte_online.sh ${trained_model_ckpt_path}
```

Example:

```bash
cd lingua
bash eval_bpebyte_online.sh runs/bpebyte_br_bt_online_1.3B/checkpoints/0000180000
```

> On a conda host (Section 1), activate the env first: `conda activate aunet_eval_cu121`.
> The checkpoint path above is illustrative — point it at a checkpoint dir that exists on your
> machine (one holding `params.json`, e.g. `.../checkpoints/<step>` or its `consolidated/` subdir).

The evaluation config used by the script is:

```text
lingua/apps/aunet/configs/eval_gen_mc_ll_b200.yaml
```

### Evaluating a checkpoint trained on another host

A checkpoint bakes the absolute `bpe_tokenizer_path` from its training config into `params.json`.
If you evaluate a checkpoint trained on a **different** machine (e.g. a B200 checkpoint with a
`/NHNHOME/...` path) on this host, the model load fails because that path does not exist here.
Override it with the local tokenizer path (anything after `--` is passed through to the eval):

```bash
bash eval_bpebyte_online.sh ${trained_model_ckpt_path} -- \
    regex_bpe_tokenizer_path=$HOME/AUNet_eval/tokenizer/llama3/tokenizer.model
```

A checkpoint trained on **this** host (Section 2, where you set `bpe_tokenizer_path`) does not need
the override.

---

## Notes

* Use the cu121 setup only when the target machine has an older driver, such as `snu55`.
* Install `torch` and `xformers` before `requirements.txt`; otherwise, dependencies may pull incompatible CUDA builds.
* `lm-eval==0.4.12` is pinned to match the `snu55` evaluation environment.
* For machines with newer drivers supporting CUDA 12.6/12.8, prefer the default setup script instead of the cu121 environment.

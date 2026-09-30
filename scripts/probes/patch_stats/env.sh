# Source before running the lingua-venv measurement scripts: the venv's torch needs its bundled
# NVIDIA libs (libcusparseLt etc.) on LD_LIBRARY_PATH, as in lingua/run_robustness_local.sh.
NV=/mnt/ssd2/hyun2/AUNet/lingua/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(ls -d $NV/*/lib 2>/dev/null | paste -sd: -):${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

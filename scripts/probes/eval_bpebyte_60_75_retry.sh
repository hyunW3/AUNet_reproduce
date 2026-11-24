#!/bin/bash
# 60/75 재시도. 25%(45,000) 평가가 GPU 4장을 쓰고 있으므로 그게 끝난 뒤에 시작한다.
set -u
L=/mnt/ssd2/hyun2/AUNet
while pgrep -f "[e]val_bpebyte_25.sh" >/dev/null; do sleep 300; done
exec bash $L/scripts/probes/eval_bpebyte_60_75.sh

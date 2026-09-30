# SVT ablation (Stride · Vocab · Tokenizer), 100M

Plan: `plans/ablation_100M_stride_vocab_tokenizer.md`. Launched 2026-09-30 on **ece-agpu18 GPUs 5,6**.
ece-agpu11 had no free GPU at launch (all 8 held by other users).

Remote tree `~/AUNet_svt` on ece-agpu18: a snapshot of `~/AUNet/lingua` (`apps/`, `lingua/`, `setup/`).
Provenance is in `lingua/SNAPSHOT_*`. The snapshot is patched with `patch_stride.py`; data and tokenizer
are symlinked to `~/AUNet`. It uses the venv at `~/AUNet/lingua/.venv`.

| arm | regex change vs `lb_rg_100M` | bytes/patch (DCLM, 200 docs) |
|---|---|---:|
| (baseline) `lb_rg_100M` | llama3 128K, online greedy root | 4.566 |
| `stride4p57` | `strategy: {stride: 4.57@1}`, offline, no tokenizer | ≈4.57 (4.5 → 4.499 measured) |
| `rg_llama3_V32k` | `tokenizer_extra/llama3_V32k.model` (first 32,768 ranks) | 4.224 |
| `rg_gpt2` | `tokenizer_extra/gpt2/tokenizer.json` | 4.397 |
| `rg_qwen2` | `tokenizer/qwen2/tokenizer.json` | 4.481 |

All other hyperparameters are copied from the **resolved** `lb_rg_100M/config.yaml` (`gen_svt_cfgs.py`):
98.6M params, 53,504 steps × global batch 48 × 8192 B, 2 GPUs (bs 12 × ga 2), LR 3.4e-3, seed 777.

Pre-launch checks:
- `test_svt.py` runs unit tests of the stride offsets and measures bytes/patch through the real `tokenize()`.
- A 100-step smoke of the unchanged baseline config on the snapshot tracks `lb_rg_100M` losses within ±0.05 at steps 10–100, and runs at the same speed (0.267 vs 0.270 s/step).

Throughput is ~0.27 s/step, so each arm takes ~4 h. The queue runs sequentially: stride, V32k, gpt2, qwen2.

```bash
ssh ece-agpu18 'cat ~/AUNet_svt/logs/queue.log; for a in stride4p57 rg_llama3_V32k rg_gpt2 rg_qwen2; do echo $a $(tail -n1 ~/AUNet_svt/runs/$a/metrics.jsonl 2>/dev/null | cut -c1-40); done'
# restart / resume (skips .DONE arms, resumes from checkpoints):
ssh ece-agpu18 'cd ~/AUNet_svt && setsid nohup ./svt_queue.sh >> logs/queue.log 2>&1 < /dev/null &'
```

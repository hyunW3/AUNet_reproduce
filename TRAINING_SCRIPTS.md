# Training & eval pipeline scripts

Canonical location of every script below: **`lingua/`** (git-tracked on `main`, commit `d99983e`+).
Run from the `lingua/` dir; all use `torchrun --nproc-per-node 4` on the 4×B200 box and the
`lingua/.venv`. Checkpoints land in `runs/<name>/checkpoints/`; eval results in `runs/<name>/evals_*/`.
Long-running drivers send Telegram via `../alert_knock` and gate on each other through `/tmp/*.log`
markers (so the GPUs are never double-booked: **evals can share GPU, a 1.3B train cannot** — it needs
~153 GB/GPU standalone).

## Per-model: training script → config → run dir
| model (scale) | train script | config | run dir |
|---|---|---|---|
| Llama 760M | `train_llama_760M.sh` | `apps/main/configs/llama_760M_b200.yaml` | `runs/llama_760M` |
| AU-Net2 760M | `train_aunet2_760M.sh` | `apps/aunet/configs/aunet2_760M_b200.yaml` | `runs/aunet2_760M` |
| BPEByte root_greedy 760M (new code) | `train_bpebyte_root_greedy_760M.sh` / `redo_root_greedy_760M.sh` | `apps/aunet/configs/bpebyte_root_greedy_760M_b200.yaml` | `runs/bpebyte_root_greedy_760M` |
| BPEByte root_greedy 760M (OLD code @4ed607e) | `train_oldcode760M.sh` | `bpebyte_root_greedy_760M_oldcode.yaml` (in `/var/tmp/aunet_4ed607e` worktree) | `runs/bpebyte_root_greedy_760M_oldcode_4ed607e` |
| Llama 1.8B | `train_llama_1.8B_b200.sh` | `apps/main/configs/` llama 1.8B | `runs/llama_1.8B_paper` |
| AU-Net2 1.3B | `train_aunet2_1.3B.sh` | `apps/aunet/configs/aunet2_1.3B_b200.yaml` | `runs/aunet2_1.3B` |
| BPEByte root_greedy 1.3B (new code) | `train_bpebyte_root_greedy_1.3B.sh` (or via `arc_1p3b_then_retrain.sh`) | `apps/aunet/configs/bpebyte_root_greedy_1.3B_b200.yaml` (bs24·ga2, 180k) | `runs/bpebyte_br_greedy_root_1.3B` |
| BPEByte online-bt 1.3B | `train_bpebyte_online_bt_1.3B.sh` | online-bt config | `runs/bpebyte_online_bt_1.3B` |
| ByteFlow 760M / 1.3B | `train_byteflow_760M.sh` / `train_byteflow_1.3B.sh` | `apps/aunet/configs/byteflow_*` | `runs/byteflow_*` |

Recipe notes: byte models ~29.2k steps @760M / 180k @1.3B (iso-text Chinchilla); 1.3B uses
**bs24·ga2** (doubled micro-batch, same global batch as bs12·ga4). Paper scaling laws applied for
LR/BSZ/steps. `dump.every=2500 keep=1` at 1.3B (milestones preserved as non-multiples).

## Eval drivers
- **Full downstream suite (7 groups: 5bench / mmlu+arc_c / cute / noise / pbp / typo_ds / typo):**
  `run_eval_<model>_<scale>_full.sh` (e.g. `run_eval_bpebyte_root_greedy_1.3B_full.sh`). All use
  `EVAL_GROUPS` array (NOT `GROUPS` — that's a bash special var = the user's GIDs 1999/2357, which
  silently produced empty results; fixed `d99983e`).
- **ARC few-shot:** `arc_fewshot_parallel.sh` (760M, parallel with refs), arc handled in
  `arc_1p3b_then_retrain.sh` PHASE A (1.3B existing ckpts).
- **Few-shot matrix (3/5-shot × benchmarks + MMLU-text 0/3/5):** `eval_1p3b_fewshot_all.sh`.
- **MMLU-text (cloze):** `run_eval_mmlu_text.sh`. **PBP:** `run_eval_pbp.sh` / `rerun_pbp.sh`.
- **Generation-framed MC:** `run_eval_gen*.sh`, `run_eval_gen_mc*.sh`.

## Session orchestration drivers (the 760M/1.3B comparison)
Run order, each gated on the previous via `/tmp/*.log` markers:
1. `run_760M_chain.sh` — train Llama+AU-Net2+BPEByte-rg 760M → `eval_760M_after_chain.sh` (downstream).
2. `train_oldcode760M.sh` — BPEByte rg 760M on the 1.3B code point (4ed607e) + its eval (uses
   `declare -A G`, so it dodged the GROUPS bug) → signals `OLDCODE760M COMPLETE`.
3. `eval_newcode_760M.sh` — recovery re-run of the new-code 760M downstream (after the GROUPS fix).
4. `refs_then_1p3b.sh` / `parallel_refs_1p3b.sh` / `run_1p3b_gated.sh` — aunet2+llama 760M refs eval,
   then the 1.3B new-code rerun. (`parallel_refs_1p3b.sh` proved the 1.3B can't share GPU → fell back
   to sequential `run_1p3b_gated.sh`.)
5. `arc_1p3b_then_retrain.sh` — **current live driver**: PHASE A arc 3/5-shot on existing 1.3B ckpts
   (oldseg-rg / aunet2 / llama_1.8B) → PHASE B restart 1.3B new-code training (sends `TRAINING
   STARTED` / `TRAINING ENDED` pings) → PHASE C re-eval + arc for the new-code 1.3B.
6. `eval_1p3b_fewshot_all.sh` — gated on (5)'s `ALL COMPLETE`: few-shot matrix for the new-code 1.3B
   (letter-MMLU skipped; MMLU-text 0/3/5 instead).

## Notes / gotchas baked into the scripts
- Single-process debug: unset `LOCAL_RANK` → rank0/world1; the non-torchrun path picks a deterministic
  port (`random.Random(-1)` → 28805) so back-to-back runs collide — pin a unique `MASTER_PORT`.
- Smoke (25 steps) hits a benign `IndexError` at `train.py:661` (`existing_saves[-1]`) — end-of-run
  eval with no checkpoint; the grep-based smoke check ignores it and the full run saves so it's fine.
- `dump.every keep=1` → only the latest checkpoint is retained at 1.3B.

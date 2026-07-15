# Held / paused training

**Last updated: 2026-07-15.** Training runs deliberately **on hold** (config + data kept, ready to resume). Complements `PROJECT_STATUS.md`; each entry has why-held + exact resume command.

| Run | Scale | Held since | Reason | Resume-ready |
|---|---|---|---|---|
| **AU-Net-3 2.5B baseline** (SuperBPE level-2 study) | 2.5B | 2026-07-15 | user: "we do not run 2.5B right now" | ✅ config + tokenizer + data |
| **AU-Net-law 300M scaling points** (rg + AU-Net word) | 300M | 2026-07-15 | user: "do not run law-recipe 300M runs right now"; the adhoc-γ10 300M runs serve the scaling ladder instead | ⚠️ only step-40,128 ckpt saved (dump.every=40128) |
| **Multilingual B4 pilot** (en/fi/zh/code × 4 families) | 100M r40 | 2026-06-30 | user pause; final losses anomalous (~0.025), unverified | ✅ data/configs on ece |

---

## 1. AU-Net-3 2.5B baseline — SuperBPE level-2 scale reference

**What.** The AU-Net-3 (word/2-word) baseline at the paper's `2B_2level.yaml` scale, as the **scale reference** for the SuperBPE-superword level-2 study. The 100M part of that study is **complete** (offline + causal trios, BPB + 6-bench → `reports/superbpe_level2_100M.html`); this 2.5B run was the next step (does the causal learned-superword win hold at scale / do the near-chance 100M benches separate).

**Why held.** User decided not to run it now (2026-07-15). Watcher stopped; it never started. B200 is running the user's own `hybrid_760M_aunetlaw` job.

**Config** — `lingua/apps/aunet/configs/aunet3_2p5B_baseline_2level.yaml` (verified 2.50B params):
- arch = **exact** `2B_2level.yaml`: `dimensions [512,2048,3456]`, `layers [3,6,12]`, `word1:1@1` / `word2:2@1` (static 2-word pooling), tokenizer `bytes`.
- **global batch 336** = `batch_size 21` (per-GPU) × `grad_acc_steps 4` × 4 GPUs.
- `steps: 300000` (from 2B_2level) → **multi-WEEK on 4×B200. Confirm/shorten the budget before launching.**
- dump_dir `runs/superword_l3/aunet3_2p5B_baseline`.

**Resume** (4×B200, when the GPUs are free):
```bash
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/lingua && source .venv/bin/activate
TMPDIR=/var/tmp torchrun --nproc-per-node 4 --master-port 29620 \
  -m apps.aunet.train config=apps/aunet/configs/aunet3_2p5B_baseline_2level.yaml
```
Memory note: 3-stage byte model — fits 4×B200 at micro 21 (estimate ~145 GB/GPU); if it OOMs, drop to `data.batch_size=14 grad_acc_steps=6` (holds global 336) or enable `selective_activation_checkpointing`. The prior auto-gated launcher (`run_2p5b_b200.sh`, 1-hour-idle gate) is gone; relaunch or run the command directly.

**Related (all complete, not held):** SuperBPE tokenizer `tokenizer/llama3_superbpe/`; 100M trio configs `bpebyte_100M_{superword,static2subword,superword_rg,static2subword_rg,rootgreedy_2stage_ref}_l3.yaml` + `aunet3_100M_static2word_baseline.yaml`; results in `runs/superword_l3/` **on ece-agpu11** (`/home/hwbae/AUNet/`), report `reports/superbpe_level2_100M.html`.

---

## 2. Multilingual B4 pilot

Paused 2026-06-30 (user). Training done but final losses anomalous (~0.025, unverified); downstream eval + fertility figure + 760M follow-on not proceeding. Data/configs kept on ece for resume. See `PROJECT_STATUS.md` §1 and `reports/b4_pilot_estimate.md`.

---

## 3. AU-Net-law 300M scaling points (rg + AU-Net word)

**What.** The **AU-Net-law-recipe** twins of the 300M scaling-ladder points — `bpebyte_rg_300M_aunetlaw` and `aunet_300M_aunetlaw` (batch 8, LR 2.9e-3, **80,256 steps**, γ10.4). Intended as the law-recipe complement to the leaderboard, matching the 100M/300M hybrid law ablation. (A law `llama_300M` @17,830 is also partial.)

**Why held.** User: "do not run law-recipe 300M runs right now" (2026-07-15). The scaling ladder (BPB + downstream in `reports/scaling_bpb.csv` / `scaling_data.csv` / `dashboard_bpebyte_rg`) is served by the **converged adhoc-γ10** 300M runs instead — `small/cmp_g10/{rg,aunet}_300M` @9,900, giving the canonical **rg 1.0142 / AU-Net 1.0157** held-out BPB. So the law 300M is not on the critical path.

**⚠️ Provenance note.** These partial law runs were briefly (and wrongly) pointed at by `reports/scaling_bpb.csv`, producing a non-monotonic 300M (BPB > 100M) because they were undertrained. Reverted 2026-07-15 to the converged adhoc runs; do **not** repoint the ladder here until they finish.

**State (stopped, not deleted):**
- `bpebyte_rg_300M_aunetlaw` — last metric step **44,630 / 80,256** (~56%), stopped 2026-07-13. grad_acc 8.
- `aunet_300M_aunetlaw` — last metric step **60,710 / 80,256** (~76%), stopped 2026-07-08. grad_acc 4.
- Both `dump.every=40128`, so the **only saved checkpoint is step 40,128** (resuming loses 40,128→last progress). Weights + configs on **ece** at `/home/hwbae/AUNet/runs/poc/portable_aunetlaw/{rgllaw_300M,aunetllaw_300M}`; data `/home/hwbae/AUNet/data`.

**Resume** (on ece, when GPUs free — finishes 40,128 → 80,256; consider lowering `dump.every` to 10000 for resilience):
```bash
cd /home/hwbae/AUNet/lingua && source .venv/bin/activate
# BPEByte-rg (2-GPU, grad_acc 8):
TMPDIR=/var/tmp torchrun --nproc-per-node 2 --master-port 29560 -m apps.aunet.train \
  config=/home/hwbae/AUNet/runs/poc/portable_aunetlaw/rgllaw_300M/config.yaml
# AU-Net word (grad_acc 4): same, config=.../aunetllaw_300M/config.yaml --master-port 29564
```
Each resumes automatically from `checkpoints/0000040128`. When both reach 80,256, recompute their tail-mean BPB and (only then) repoint the 300M rows of `reports/scaling_bpb.csv` from the `*_adhoc` runs to these, and regenerate `scaling_bpb.png` + `scaling_dashboard.html`.

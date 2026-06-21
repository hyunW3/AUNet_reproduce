# 1.3B BPB — iso-byte comparison (canonical baseline = Llama 1.8B)

> **Superseded framing.** This file originally truncated at 275 GB because the wrong Llama
> (`llama_1B_dm10`, 275 GB) was used as the baseline. The canonical subword baseline is
> **`llama_1.8B_paper`** (dim2048/25L; paper Table 10/11 recipe: seq 4096, lr 3e-3, 60k steps), which
> trained to **286.2 GB** — it *overshoots* the byte models' 283.1 GB. So no Llama extension is needed
> (it already passed 283 GB); the iso-byte budget is **283 GB** (the byte endpoint). Truncated figure:
> `runs/bpb_all_1.3B_283G.png`; full marked figure: `runs/bpb_all_1.3B.png`.

| model | run dir | steps | training text |
|-------|---------|-------|---------------|
| Llama 1.8B (subword, **baseline**) | `llama_1.8B_paper` | 60,000 | **286.2 GB** (62.9B tok × 4.5483 B/tok, seq 4096) |
| BPEByte offline-bt | `bpebyte_br_bt_1.3B` | 180,000 | 283.1 GB |
| BPEByte online-bt | `bpebyte_br_bt_online_1.3B` | 180,000 | 283.1 GB |
| BPEByte root_greedy | `bpebyte_br_greedy_root_1.3B` | 180,000 | 283.1 GB |
| AU-Net2 (pure byte) | `aunet2_1.3B` | 180,000 | 283.1 GB |

## Final train BPB at the common 283 GB budget (EMA α=0.01)

| model | BPB @283 GB (iso-byte) |
|-------|------------------------|
| BPEByte offline-bt (leaky) | **0.780** |
| BPEByte online-bt (leaky) | 0.787 |
| **Llama 1.8B (subword)** | **0.839** |
| BPEByte root_greedy (leak-free) | 0.860 |
| AU-Net2 (pure byte) | 0.866 |

(The 1.8B's *endpoint* is 286.2 GB / 0.840 EMA; at the common 283 GB it is 0.839. The raw last-step
loss/out reads ~0.87 BPB but that is a single noisy step, not the EMA curve value.)

**Conclusions:**
- Leaky `bt` variants (0.78–0.79) sit well below everything — boundary lookahead not realizable at
  generation (see `BPEByte_root_greedy_method.md`).
- The **leak-free byte models** (root_greedy 0.860, AU-Net2 0.866) have **higher** train BPB than the
  1.8B subword baseline (0.839). The byte advantage is on the character/robustness axes (CUTE, PBP,
  typo/noise), not raw train BPB.
- Using the correct 1.8B baseline (vs the old `llama_1B_dm10`, also ~0.839 EMA) does not change the
  ranking; it just makes the comparison iso-byte at 283 GB without any Llama extension.

## Was a Llama→283 GB extension needed? No.

The 1.8B baseline already trained to 286.2 GB, past the byte models' 283.1 GB, so the comparison is
read directly at 283 GB (the 1.8B's 283→286 tail is dropped in the truncated plot). The earlier
attempt to extend `llama_1B_dm10` (275→283 GB) was on the wrong model and is abandoned/cleaned up.
(For the record, that extension also surfaced a lingua resume gotcha: auto-resume restores the peak
base LR and resets the scheduler → re-warms a converged model; a correct continuation needs
model-only `init_ckpt_path` + a fresh optimizer at constant `lr_min`. Not needed here.)

*Companion: `BPEByte_root_greedy_method.md`, `runs/bpb_all_1.3B.png`, `runs/bpb_all_1.3B_283G.png`.*

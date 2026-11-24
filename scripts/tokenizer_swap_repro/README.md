# Tokenizer-swap replication bundle

Reproduces the eval-time **tokenizer-swap** study: swap ONLY the patch-boundary
tokenizer on a fixed byte-in/byte-out checkpoint (`bpebyte_br_greedy_root_1.3B` @180k),
no weight updates, and measure BPB + downstream. Full results & analysis:
[`../../PoC/tokenizer_swap_all_results.md`](../../PoC/tokenizer_swap_all_results.md).

**Finding replicated:** the swap ALWAYS hurts (zh BPB +0.21, code +0.34, te +0.47,
English downstream −2.3 acc_norm) — boundary-**distribution** binding, not coverage/granularity.

## Files (run in order)
| script | what |
|--------|------|
| `00_env.sh` | sourced by all — paths, venv, checkpoint, caches (edit `BASE` if the tree moves) |
| `01_download_tokenizers.sh` | fetch `tokenizer.json` for qwen2 / bloom / starcoder2 / gemma |
| `02_stage_data.sh` | stage BPB corpora as `*.val.jsonl` (zh/te wiki, multi-lang code) |
| `10_bpb_swap.sh <domain> <arm>` | BPB under a swap — the sensitive metric (§1-3) |
| `20_downstream_swap.sh <arm> <tasks> [lim]` | downstream/cloze harness under a swap (§3-5) |
| `30_augment_not_swap.sh` | ADD BLOOM-Telugu into llama3 trie (add ≠ swap; doesn't help) (§2 note) |
| `40_subword_ref.sh {bpb\|down} …` | subword `llama_1.8B` reference (`apps.main.eval`; not swappable) |
| `50_continue_pretrain.sh <lang> <arm>` | **continue-pretrain** the fixed ckpt on a new lang/domain — the fix the swap study points to |
| `51_gate_continue.sh <lang:arm> …` | idle-gate GPUs, then run continue-pretrain pairs sequentially |
| `run_all.sh` | runs every eval arm end-to-end (continue-pretrain is opt-in, see below) |

## Continue-pretrain (the follow-up)
An eval-time swap can't re-target a fixed checkpoint; **continue-pretraining with the new
boundaries can** — the trunk learns the new pooling distribution. `50_continue_pretrain.sh`
generalizes the two committed Chinese pilot configs
(`bpebyte_1.3B_zh_continue_{llama3,qwen2}.yaml`) to any language×tokenizer, generating the
config on the fly (init@180k, 6000 steps, LR 1e-4 + 1k re-warmup, 60% target-lang / 40% DCLM
English replay). Natural pairings: `zh→qwen2`, `te→bloom`, `code→starcoder2`
(`llama3` = keep-original control). Needs a train corpus at `data/<lang>_wiki_train`.
After training, re-run `10_bpb_swap.sh` against the NEW checkpoint to test whether adaptation
closed the swap gap — the open question. Launch idle-gated:
`nohup ./51_gate_continue.sh zh:qwen2 te:bloom code:starcoder2 > gate_cont.log 2>&1 &`.
Status: dormant this session — GPUs never freed (PoC training held them).

## Mechanism
- **Swap:** `regex_bpe_tokenizer_path=<tok> force_bpe_online_mode=greedy`.
  `force_*` is REQUIRED — a bare override runs offline/leaky and tanks BPB.
- **Add-not-swap:** `regex_bpe_augment_tokenizer_path=<tok> regex_bpe_augment_unicode_range=[LO,HI]`.
- **BPB run:** `harness.tasks=[]` → only `eval_on_val` (per-byte NLL). BPB = |nll|/bytes/ln2.
- The byte-trie builder auto-detects each tokenizer's byte-recovery family
  (byte-level BPE vs SentencePiece metaspace) from `tokenizer.json` — no per-tokenizer flags.

## Prereqs / caveats
- The 1.3B checkpoint and the DCLM val data must exist locally (not shipped here).
- `02_stage_data.sh` is BEST-EFFORT re-staging from public HF mirrors; exact doc counts /
  the cmmlu parquet repack and the custom cloze/MC task defs (`mmlu_cloze`, `humaneval_mc`,
  `belebele_cloze`) were built on ece — port them into `$TASKS` if absent (note at end of `02`).
- Config load tolerates deprecated checkpoint keys (e.g. `coding_rate.ema_decay`) via the
  committed non-strict recursive merge in `lingua/lingua/args.py` — old checkpoints just load.
- `NPROC=4` (B200). On shared ece nodes use `NPROC=2`. `LIMIT=50 ./run_all.sh` for a smoke test.

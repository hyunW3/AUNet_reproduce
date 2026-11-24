#!/usr/bin/env python3
"""Emit matched pretraining configs (baseline / scratchpad1 / scratchpad1frac) from an hp json.

Everything except the scratchpad flags is copied from the recorded run, so the three arms differ
only in the architecture under test.  What is NOT copied is stated in the header of each config:
the recorded runs used 4 GPUs with grad_acc 24 (6.29 MB/step, 21 GB total), which is ~57 h per run
on one A100 with the scratchpad path.  With three arms to compare before a deadline the global
batch is cut instead, identically for every arm; the comparison is internal, so the absolute BPB
will sit well above the recorded 1.13 and must not be quoted against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

ARMS = {
    "base":            dict(scratchpad_k=0, scratchpad_shift=1.0),
    "scratchpad1":     dict(scratchpad_k=1, scratchpad_shift=1.0),
    "scratchpad1frac": dict(scratchpad_k=1, scratchpad_shift=0.5),
    # the other direction: no second trunk pass at all, a small trained head instead
    "l3o_local":       dict(updater_kind="attn", updater_use_gread=True,
                            updater_use_offset=True, updater_gate="input"),
    # same head, made wide and two-layer.  On a FROZEN backbone the gain was flat across a 63x
    # parameter sweep, but that says nothing about a head the backbone can co-adapt to, which is
    # what this arm tests.
    "l3o_big":         dict(updater_kind="attn", updater_use_gread=True,
                            updater_use_offset=True, updater_gate="input",
                            updater_hidden=4096),
    # ~50 M added -- half the size of the 100M model it is bolted onto.  Any gain here has to be
    # read against a parameter-matched baseline before it can be called a patch-lag effect.
    "l3o_50m":         dict(updater_kind="attn", updater_use_gread=True,
                            updater_use_offset=True, updater_gate="input",
                            updater_hidden=3840, updater_depth=4),
    # ---- input ablation at the same ~50 M capacity ------------------------------------------
    # l3o_50m sees three things: the current patch's bytes, the patch vector it is already
    # reading, and how far into the patch it is.  These drop one at a time so the -0.0197 can be
    # split into what the INFORMATION buys and what the CAPACITY buys.
    "l3o_50m_gread":   dict(updater_kind="gread", updater_use_offset=True,   # no current bytes
                            updater_gate="input", updater_hidden=3840, updater_depth=4),
    "l3o_50m_attn":    dict(updater_kind="attn", updater_use_gread=False,    # bytes only
                            updater_use_offset=False, updater_gate="input",
                            updater_hidden=3840, updater_depth=4),
    "l3o_50m_nooff":   dict(updater_kind="attn", updater_use_gread=True,     # no offset
                            updater_use_offset=False, updater_gate="input",
                            updater_hidden=3840, updater_depth=4),
    # PARAMETER-MATCHED baseline: no updater at all, the ~50 M spent on trunk depth instead
    # (10 -> 17 layers, 148.15 M vs l3o_50m's 149.25 M).  It answers the half that an input
    # ablation cannot: is 50 M spent on a patch-lag head better than 50 M spent on a bigger model?
    # FLOPs are NOT matched -- the extra layers run on ~1800 patch slots, the head on 8192 bytes.
    "base_150m":       dict(layers=[3, 17]),
    # ---- the two follow-ups the offset decomposition asked for ------------------------------
    # The offset split showed scratchpad1 (0 added params) and base_150m (+50 M of trunk) buy
    # almost exactly the same thing at patch_offset >= 1 (-0.0162 vs -0.0165); base_150m's whole
    # advantage sits at offset 0, which scratchpad cannot touch.  If those two axes are really
    # orthogonal, running them together should land near the sum; if they overlap, this ties with
    # base_150m and the patch-lag ceiling is confirmed.
    "scratchpad1_150m": dict(scratchpad_k=1, scratchpad_shift=1.0, layers=[3, 17]),
    # Same 50 M head as l3o_50m, same inputs, same dense compute -- pointed at patch_offset == 0
    # instead of >= 1.  That is where 49.3% of the held-out loss actually is (BPB 3.06 vs 0.82),
    # and it is the half of the problem no patch-lag mechanism can reach.
    # ---- looped trunk: one checkpoint, many depths (PoC/loopedBPEByte) ----------------------
    # 3 shared trunk layers applied 1-6 times.  Mean depth over the training cycle is 3.2 x 3 =
    # 9.6 layer-applications, i.e. about the 10-layer baseline's trunk cost, but the depth is a
    # knob at inference.  That is what makes the marginal-return-per-difficulty curve affordable:
    # one training run instead of one per depth.
    "looped3":         dict(layers=[3, 3], trunk_loops=4, trunk_loops_train="1,2,3,4,6"),
    "l3o_50m_start":   dict(updater_kind="attn", updater_use_gread=True,
                            updater_use_offset=True, updater_gate="input",
                            updater_hidden=3840, updater_depth=4, updater_act="start"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", required=True)
    ap.add_argument("--run_key", default=None, help="key inside runs{} (omit for a flat json)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="/home/hwbae/AUNet_patchlag",
                    help="where data/ and tokenizer/ live")
    ap.add_argument("--dump_root", default=None, help="where runs/ go (default: --root)")
    ap.add_argument("--source", default="dclm_baseline_1.0_2shards_shuffled")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_acc", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--lr", type=float, default=None, help="default: sqrt-scaled from the recipe")
    ap.add_argument("--outdir", default=str(ROOT / "lingua/apps/aunet/configs"))
    args = ap.parse_args()

    hp = json.load(open(args.hp))
    rec = hp["runs"][args.run_key] if args.run_key else hp

    rec_global = rec["data"]["batch_size"] * rec["grad_acc_steps"] * \
        rec["distributed"]["dp_replicate"] * rec["data"]["seq_len"]
    new_global = args.batch_size * args.grad_acc * args.gpus * rec["data"]["seq_len"]
    lr = args.lr if args.lr is not None else \
        round(rec["optim"]["lr"] * (new_global / rec_global) ** 0.5, 6)
    warmup = max(1, round(rec["optim"]["warmup"] / rec["steps"] * args.steps))

    out = []
    for arm in args.arms.split(","):
        flags = ARMS[arm]
        name = f"sp_{args.tag}_{arm}"
        model = dict(rec["model"]); model.update(flags)
        cfg = {
            "name": name,
            "dump_dir": f"{args.dump_root or args.root}/runs/scratchpad/{name}",
            "steps": args.steps,
            "grad_acc_steps": args.grad_acc,
            "seed": rec["seed"],
            "probe_freq": None,
            "optim": {**rec["optim"], "lr": lr, "warmup": warmup},
            "distributed": {**rec["distributed"], "dp_replicate": args.gpus, "dp_shard": 1,
                            # the scratchpad path is hand-written sdpa; compile is off for every
                            # arm so the three stay comparable
                            "compile": False},
            "model": model,
            "data": {**rec["data"], "root_dir": f"{args.root}/data",
                     "sources": {args.source: 1.0},
                     "batch_size": args.batch_size,
                     "regex": _fix_regex(rec["regex"], args.root)},
            "checkpoint": {"dump": {"every": args.steps, "keep": 1},
                           "eval": {"every": args.steps, "keep": 0}},
            "logging": {"freq": 20},
        }
        p = Path(args.outdir) / f"{name}.yaml"
        p.write_text(
            f"# AUTOGENERATED from {Path(args.hp).name}"
            + (f" [{args.run_key}]" if args.run_key else "") + "\n"
            f"# arm: {arm}  ({', '.join(f'{k}={v}' for k, v in flags.items())})\n"
            f"# DEVIATIONS from the recorded run, applied IDENTICALLY to every arm:\n"
            f"#   global batch {rec_global / 1e6:.2f} MB/step -> {new_global / 1e6:.2f} MB/step "
            f"({args.gpus} GPU, grad_acc {rec['grad_acc_steps']} -> {args.grad_acc})\n"
            f"#   steps {rec['steps']} -> {args.steps}, warmup {rec['optim']['warmup']} -> {warmup}"
            f" (same fraction)\n"
            f"#   lr {rec['optim']['lr']} -> {lr} (sqrt of the batch ratio)\n"
            f"#   data source -> {args.source} (the recorded dclm_r10lr_2chunk is not on this host)\n"
            f"#   compile true -> false (the scratchpad path is hand-written sdpa)\n"
            f"#   total bytes {rec_global * rec['steps'] / 1e9:.1f} GB -> "
            f"{new_global * args.steps / 1e9:.2f} GB -- ABSOLUTE BPB IS NOT COMPARABLE to the\n"
            f"#   recorded {rec.get('train_BPB')}; only the arms are comparable to each other.\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        out.append(str(p))
    for p in out:
        print(p)


def _fix_regex(rx, root):
    if not rx:
        return rx
    rx = dict(rx)
    if rx.get("bpe_tokenizer_path"):
        rx["bpe_tokenizer_path"] = f"{root}/tokenizer/llama3/tokenizer.model"
    return rx


if __name__ == "__main__":
    main()

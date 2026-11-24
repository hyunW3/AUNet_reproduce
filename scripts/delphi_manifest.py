#!/usr/bin/env python
"""Build reports/delphi/manifest.jsonl -- the single source of truth for the
Delphi-style downstream-prediction sweep (BPEByte-rg only).

One row per checkpoint: parameter counts read from the weights themselves, D and the
LR state read from the run's metrics.jsonl, analytic FLOPs/byte recovered as
median(speed/FLOPS / speed/wps) (train.py logs FLOPS = wps * model.flops_per_token(),
so this ratio IS flops_per_token -- analytic, not measured), and the role each
checkpoint plays in the fit.

Roles
  fit        annealed (lr == lr_min) endpoint at N <= 760M -> Stage-1 scaling law
  link       <=760M, kept by the 1.15x-in-D de-dup rule -> Stage-2 sigmoid pool only
  holdout    every 1.3B checkpoint -> never enters either fit

`fit` rows are also link-pool members; `link` marks pool-only rows.

Usage:  python scripts/delphi_manifest.py [--out reports/delphi/manifest.jsonl]
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

ROOT = "/mnt/ssd2/hyun2/AUNet"

# (run_dir, step, role) -- explicit rather than globbed so a stray checkpoint can
# never silently join the fit set.
REGISTRY = [
    # --- 99M -------------------------------------------------------------------
    ("runs/small/ablation_100M/v4_root_greedy",                  1672,   "fit"),
    ("main/main/100M/rg_100M_g10",                             3344,   "fit"),      # gamma=10 rung
    ("runs/small/cmp_100M/v4_root_greedy_ot",                    6688,   "fit"),
    ("runs/others/superseded/cmp_100M_ratio20/v4_root_greedy",   6688,   "control"),  # same D as the row above, different seed
    ("runs/others/superseded/cmp_100M_ratio20/v4_root_greedy",   8360,   "link"),
    ("runs/others/superseded/cmp_100M_ratio20/v4_root_greedy",  10032,   "link"),
    ("runs/others/superseded/cmp_100M_ratio20/v4_root_greedy",  11704,   "link"),
    ("runs/others/superseded/cmp_100M_ratio20/v4_root_greedy",  13376,   "fit"),
    # --- 296M ------------------------------------------------------------------
    ("main/main/300M/rg_300M_g10",                             9900,   "fit"),      # gamma=10 rung
    # --- 680M ------------------------------------------------------------------
    ("main/main/others/760M/bpebyte_root_greedy_760M_bpp101",           8760,   "link"),
    ("main/main/others/760M/bpebyte_root_greedy_760M_bpp101",          17520,   "link"),
    ("main/main/others/760M/bpebyte_root_greedy_760M_bpp101",          26280,   "link"),
    ("main/main/others/760M/bpebyte_root_greedy_760M_bpp101",          29200,   "fit"),
    ("main/main/760M/rg_760M",                                  18180,   "link"),
    ("main/main/760M/rg_760M",                                  36360,   "link"),
    ("main/main/760M/rg_760M",                                  54540,   "link"),
    ("main/main/760M/rg_760M",                                  60600,   "fit"),      # gamma=10 rung
    # --- 1.3B: held out --------------------------------------------------------
    ("main/main/1.3B/bpebyte_br_greedy_root_1.3B",              54000,   "holdout"),
    ("main/main/1.3B/bpebyte_br_greedy_root_1.3B",             108000,   "holdout"),
    ("main/main/1.3B/bpebyte_br_greedy_root_1.3B",             162000,   "holdout"),
    ("main/main/1.3B/bpebyte_br_greedy_root_1.3B",             180000,   "holdout"),  # gamma=10 rung, headline
]

# The canonical gamma=10 ladder rungs (~210 bytes/param), for the Delphi-faithful panel.
GAMMA10 = {
    ("main/main/100M/rg_100M_g10", 3344),
    ("main/main/300M/rg_300M_g10", 9900),
    ("main/main/760M/rg_760M", 60600),
    ("main/main/1.3B/bpebyte_br_greedy_root_1.3B", 180000),
}


def git_provenance():
    """Provenance for BOTH repos. The eval and model code lives in the `lingua`
    submodule, whose working-tree changes do NOT appear in the parent's `git diff`."""
    def probe(cwd):
        def run(*a):
            return subprocess.run(a, cwd=cwd, capture_output=True, text=True).stdout
        diff = run("git", "diff", "HEAD")
        return {
            "head": run("git", "rev-parse", "HEAD").strip(),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "diff_lines": len(diff.splitlines()),
        }
    return {"repo": probe(ROOT), "lingua": probe(os.path.join(ROOT, "lingua"))}


def read_metrics(run_dir):
    """-> (per-step dict, analytic flops/byte). Robust to partial lines."""
    path = os.path.join(ROOT, run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return {}, None
    by_step, fpb = {}, []
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            f, w = d.get("speed/FLOPS"), d.get("speed/wps")
            if f and w:
                fpb.append(f / w)
            s = d.get("global_step")
            if s is not None and "optim/total_tokens" in d:
                by_step[s] = d
    return by_step, (statistics.median(fpb) if fpb else None)


def nearest(by_step, step):
    """Row at the largest logged step <= `step`; falls back to the nearest one."""
    if not by_step:
        return None
    le = [s for s in by_step if s <= step]
    return by_step[max(le)] if le else by_step[min(by_step, key=lambda s: abs(s - step))]


def param_counts(ckpt_dir):
    """(total, non-embedding) parameters, counted from the weights on the meta device."""
    import torch
    sd = torch.load(os.path.join(ckpt_dir, "consolidated", "consolidated.pth"),
                    map_location="meta", weights_only=True)
    sd = sd.get("model", sd)
    total = emb = 0
    for k, v in sd.items():
        n = 1
        for d in v.shape:
            n *= d
        total += n
        # byte model: `tok_embeddings` in, `vocab` out -- both scale with vocab_size 258
        if "tok_embeddings" in k or k.startswith("vocab.") or ".vocab." in k:
            emb += n
    return total, total - emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/delphi/manifest.jsonl"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    prov = git_provenance()
    cache, rows, problems = {}, [], []

    for run_dir, step, role in REGISTRY:
        if run_dir not in cache:
            cache[run_dir] = read_metrics(run_dir)
        by_step, fpb = cache[run_dir]

        ckpt = os.path.join(ROOT, run_dir, "checkpoints", f"{step:010d}")
        cons = os.path.join(ckpt, "consolidated", "consolidated.pth")
        if not os.path.exists(cons):
            problems.append(f"MISSING consolidated weights: {run_dir}@{step}")
            continue

        m = nearest(by_step, step)
        if m is None:
            problems.append(f"MISSING metrics.jsonl: {run_dir}@{step}")
            continue
        D = m.get("optim/total_tokens")
        lr = m.get("optim/lr")

        params = json.load(open(os.path.join(ckpt, "consolidated", "params.json")))
        regex = params["data"]["regex"]
        lr_max = params.get("optim", {}).get("lr")
        total_p, nonemb_p = param_counts(ckpt)

        lr_frac = (lr / lr_max) if (lr and lr_max) else None
        # cosine with lr_min_ratio 0.01 -> annealed iff we are at the floor
        annealed = lr_frac is not None and lr_frac <= 0.0105

        rows.append({
            "id": f"{run_dir.replace('/', '_')}__{step:010d}",
            "run_dir": run_dir,
            "step": step,
            "role": role,
            "gamma10": (run_dir, step) in GAMMA10,
            "held_out": role == "holdout",
            "annealed": annealed,
            "N_total": total_p,
            "N_nonembed": nonemb_p,
            "D_bytes": D,
            "bytes_per_param": (D / nonemb_p) if D else None,
            "flops_per_byte": fpb,
            "C_flops": (fpb * D) if (fpb and D) else None,
            "lr_at_step": lr,
            "lr_max": lr_max,
            "lr_frac_of_max": lr_frac,
            "train_loss_at_step": m.get("loss/out"),
            "metrics_step_used": m.get("global_step"),
            "ckpt_path": os.path.join(ckpt, "consolidated"),
            "dump_dir": os.path.join(ROOT, run_dir, f"delphi_{step:010d}"),
            "per_doc": os.path.join(
                ROOT, "reports/delphi/per_doc",
                "holdout_1p3b" if role == "holdout" else "le760M",
                f"{run_dir.replace('/', '_')}__{step:010d}.jsonl"),
            "regex": {k: regex.get(k) for k in
                      ("bpe_online", "bpe_online_mode", "bpe_online_placement",
                       "bpe_online_committed_view", "bpe_context_prefix")},
            "model_dimensions": params["model"].get("dimensions"),
            "model_layers": params["model"].get("layers"),
            "provenance": prov,
            "boundary_check": None,
            "eval_seconds": None,
            "status": "pending",
        })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    by_role = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    print(f"wrote {len(rows)} rows -> {args.out}")
    for repo, pv in prov.items():
        print(f"{repo:>7}: {pv['head'][:9]}  worktree diff {pv['diff_lines']:>5} lines  "
              f"sha {pv['diff_sha256'][:12]}")
    for role in ("fit", "control", "link", "holdout"):
        rs = by_role.get(role, [])
        print(f"\n[{role}] n={len(rs)}")
        for r in sorted(rs, key=lambda x: (x["N_nonembed"], x["D_bytes"])):
            flag = " *g10" if r["gamma10"] else ""
            ann = "annealed" if r["annealed"] else f"lr {r['lr_frac_of_max']:.3f}"
            print(f"   N={r['N_nonembed']/1e6:8.2f}M  D={r['D_bytes']/1e9:7.2f}GB  "
                  f"B/p={r['bytes_per_param']:6.1f}  C={r['C_flops']:.3e}  {ann:>10}  "
                  f"loss={r['train_loss_at_step']:.4f}  {r['run_dir']}@{r['step']}{flag}")
    if problems:
        print("\nPROBLEMS:", *problems, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Resumable 4-lane sweep: score every checkpoint in reports/delphi/manifest.jsonl on
the 6-bench suite with log_samples=true, verify the boundary scheme, extract the compact
per-doc table, and gzip the raw blob.

  python scripts/delphi_sweep.py                     # everything still pending
  python scripts/delphi_sweep.py --roles fit         # just the Stage-1 fit set
  python scripts/delphi_sweep.py --ids A,B --lanes 2
  python scripts/delphi_sweep.py --dry-run

Detach it -- the sweep outlives the shell and this box has other tenants:
  setsid nohup python scripts/delphi_sweep.py > reports/delphi/logs/sweep.log 2>&1 &
Stop it by process group (never pkill): the pgid is written to logs/sweep.pgid.

Per-lane state lives in <dump_dir>/{.running,results.json[.gz]}; the manifest is never
rewritten in place, results are appended to reports/delphi/sweep_status.jsonl.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from queue import Queue

ROOT = "/mnt/ssd2/hyun2/AUNet"
LINGUA = os.path.join(ROOT, "lingua")
CONFIG = "apps/aunet/configs/delphi/eval_6bench_local.yaml"
TOKENIZER = os.path.join(ROOT, "tokenizer/llama3/tokenizer.model")
MANIFEST = os.path.join(ROOT, "reports/delphi/manifest.jsonl")
STATUS = os.path.join(ROOT, "reports/delphi/sweep_status.jsonl")
LOGDIR = os.path.join(ROOT, "reports/delphi/logs")
CORE_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq"

# r10lr_downstream.sh's threshold. Stacking an eval onto a card that already holds a
# training job died with "tried to allocate 254 MiB, 244 MiB free" -- the generator's
# create_block_mask at max_tokens=16384 is far hungrier than a training forward.
MAX_USED_MB = 2000

_print_lock = threading.Lock()
_status_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def env_for(gpu):
    e = dict(os.environ)
    nvidia = os.path.join(LINGUA, ".venv/lib/python3.12/site-packages/nvidia")
    libs = [os.path.join(nvidia, d, "lib") for d in sorted(os.listdir(nvidia))
            if os.path.isdir(os.path.join(nvidia, d, "lib"))]
    e["LD_LIBRARY_PATH"] = ":".join(libs + [e.get("LD_LIBRARY_PATH", "")]).strip(":")
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e["TMPDIR"] = "/var/tmp"
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    e["ENABLE_INTRA_NODE_COMM"] = "0"
    # Redirect the tokenizer path baked into params.json by the training host.
    # NEVER use regex_bpe_tokenizer_path= on the CLI: apps/aunet/eval.py:617-628 rebuilds
    # RegexPool from RegexArgs defaults and silently resets bpe_online/mode/placement.
    e["TOKENIZER_PATH"] = TOKENIZER
    return e


def gpu_free_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    used = {}
    for line in out.strip().splitlines():
        i, m = [x.strip() for x in line.split(",")]
        used[int(i)] = int(m)
    return used


def wait_for_gpu(gpu, poll=120):
    warned = False
    while True:
        used = gpu_free_mb().get(gpu, 10 ** 9)
        if used < MAX_USED_MB:
            return
        if not warned:
            log(f"lane {gpu}: waiting for the card ({used} MiB in use, need < {MAX_USED_MB})")
            warned = True
        time.sleep(poll)


def verify_boundaries(row, dump_dir):
    """eval_meta.json vs the checkpoint's params.json['data']['regex'].

    r10lr_downstream.sh does this because a whole round of results was silently
    invalidated by skipping it: a mismatched boundary scheme scores near chance and
    raises nothing.
    """
    meta = json.load(open(os.path.join(dump_dir, "eval_meta.json")))
    want = row["regex"]
    bad = []
    for k in ("bpe_online", "bpe_online_mode", "bpe_online_placement",
              "bpe_online_committed_view", "bpe_context_prefix"):
        if meta.get(k) != want.get(k):
            bad.append(f"{k}: eval={meta.get(k)!r} ckpt={want.get(k)!r}")
    return ("OK" if not bad else "MISMATCH: " + "; ".join(bad))


def record(entry):
    with _status_lock:
        os.makedirs(os.path.dirname(STATUS), exist_ok=True)
        with open(STATUS, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def already_done(dump_dir):
    for n in ("results.json", "results.json.gz"):
        p = os.path.join(dump_dir, n)
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return True
    return False


def run_one(row, gpu, dry_run):
    dump = row["dump_dir"]
    tag = f"{row['run_dir']}@{row['step']}"
    if already_done(dump):
        log(f"lane {gpu}: skip (done) {tag}")
        return "skipped"

    running = os.path.join(dump, ".running")
    if os.path.exists(running):
        log(f"lane {gpu}: skip (.running marker) {tag} -- delete it if the run really died")
        return "locked"

    os.makedirs(dump, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    logfile = os.path.join(LOGDIR, f"{row['id']}.log")
    cmd = [
        os.path.join(LINGUA, ".venv/bin/python"), "-m", "torch.distributed.run",
        "--nproc-per-node", "1", "--master-port", str(29600 + gpu),
        "-m", "apps.aunet.eval",
        f"config={CONFIG}",
        f"ckpt_dir={row['ckpt_path']}",
        f"dump_dir={dump}",
    ]
    if dry_run:
        log(f"lane {gpu}: DRY {tag}\n    {' '.join(cmd)}")
        return "dry"

    wait_for_gpu(gpu)
    open(running, "w").write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
    t0 = time.time()
    log(f"lane {gpu}: START {tag}")
    try:
        with open(logfile, "w") as lf:
            rc = subprocess.run(cmd, cwd=LINGUA, env=env_for(gpu),
                                stdout=lf, stderr=subprocess.STDOUT).returncode
    finally:
        if os.path.exists(running):
            os.remove(running)
    dt = time.time() - t0

    if rc != 0 or not already_done(dump):
        log(f"lane {gpu}: FAILED rc={rc} after {dt/60:.1f} min -- {tag} (see {logfile})")
        record({"id": row["id"], "status": "failed", "rc": rc, "eval_seconds": dt})
        return "failed"

    bcheck = verify_boundaries(row, dump)
    if not bcheck.startswith("OK"):
        log(f"lane {gpu}: BOUNDARY {bcheck} -- {tag} QUARANTINED")
        record({"id": row["id"], "status": "quarantined", "boundary_check": bcheck,
                "eval_seconds": dt})
        return "quarantined"

    # compact per-doc table, then shrink the 58 MB raw blob (kept: it is the audit trail)
    ex = subprocess.run(
        ["python3", os.path.join(ROOT, "scripts/delphi_extract.py"),
         "--dump_dir", dump, "--tasks", CORE_TASKS,
         "--out", row["per_doc"],
         "--summary", row["per_doc"].replace(".jsonl", ".summary.json")],
        capture_output=True, text=True)
    if ex.returncode != 0:
        log(f"lane {gpu}: EXTRACT FAILED {tag}\n{ex.stdout}\n{ex.stderr}")
        record({"id": row["id"], "status": "extract_failed", "eval_seconds": dt,
                "boundary_check": bcheck})
        return "extract_failed"

    raw = os.path.join(dump, "results.json")
    if os.path.exists(raw):
        with open(raw, "rb") as fi, gzip.open(raw + ".gz", "wb", compresslevel=9) as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(raw)

    log(f"lane {gpu}: DONE {tag} in {dt/60:.1f} min\n{ex.stdout.rstrip()}")
    record({"id": row["id"], "status": "done", "eval_seconds": dt,
            "boundary_check": bcheck, "dump_dir": dump, "per_doc": row["per_doc"],
            "finished_at": datetime.now(timezone.utc).isoformat()})
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--roles", default="fit,control,link,holdout")
    ap.add_argument("--ids", help="comma-separated manifest ids (overrides --roles)")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest)]
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    else:
        roles = set(a.roles.split(","))
        rows = [r for r in rows if r["role"] in roles]
    # biggest first, and interleave scales so no lane collects all four 1.3B rows
    rows.sort(key=lambda r: (-r["N_nonembed"], -r["D_bytes"]))

    os.makedirs(LOGDIR, exist_ok=True)
    open(os.path.join(LOGDIR, "sweep.pgid"), "w").write(f"{os.getpgrp()}\n")
    log(f"{len(rows)} checkpoint(s), {a.lanes} lane(s), pgid {os.getpgrp()}")

    q = Queue()
    for r in rows:
        q.put(r)
    tally = {}

    def worker(gpu):
        while True:
            try:
                r = q.get_nowait()
            except Exception:
                return
            try:
                res = run_one(r, gpu, a.dry_run)
            except Exception as e:                       # keep the lane alive
                log(f"lane {gpu}: EXCEPTION on {r['id']}: {e}")
                res = "exception"
            with _status_lock:
                tally[res] = tally.get(res, 0) + 1
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True)
               for g in range(a.lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"sweep finished: {tally}")


if __name__ == "__main__":
    main()

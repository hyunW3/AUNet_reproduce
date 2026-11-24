#!/usr/bin/env python3
"""Build a self-contained HTML viewer of downstream benchmark scores vs training progress.

Scans milestone eval results for the re-trained 1.3B/1.8B arms and the paper
checkpoints they are meant to reproduce, then writes a single static HTML file
with the data embedded.  Re-runnable: just run it again as new evals land.

    python3 scripts/probes/build_stage_viewer.py [-o OUT]

Read-only with respect to experiment data.  Never touches runs/*/checkpoints/.
"""

from __future__ import annotations

import argparse
import codecs
import datetime as _dt
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "reports_afterAAAIsub", "stage_viewer.html")

# ---------------------------------------------------------------- metric spec

TASKS = [
    ("hellaswag",     "HellaSwag",     "acc_norm,none"),
    ("arc_easy",      "ARC-e",         "acc_norm,none"),
    ("arc_challenge", "ARC-c",         "acc_norm,none"),
    ("piqa",          "PIQA",          "acc_norm,none"),
    ("winogrande",    "WinoGrande",    "acc,none"),
    ("boolq",         "BoolQ",         "acc,none"),
    ("mmlu_text",     "MMLU (text)",   "acc,none"),
]
TASK_METRIC = {k: m for k, _, m in TASKS}
TASK_LABEL = {k: l for k, l, _ in TASKS}
AVG5 = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"]
AVG7 = AVG5 + ["boolq", "mmlu_text"]

MILESTONES = [25, 30, 50, 60, 75, 90, 100]

# ------------------------------------------------------------------- families

FAMILIES = [
    dict(id="transformer", label="Transformer", short="Transformer",
         run="llama_1.8B_a100x4", total_steps=59589, slot=1,
         paper_ckpt="llama_1.8B_paper", paper_total_steps=60000,
         blurb="1.8B dense baseline, BPE tokenizer"),
    dict(id="bpebyte", label="BPEByte", short="BPEByte",
         run="bpebyte_br_greedy_root_1.3B_a100x4", total_steps=180000, slot=2,
         paper_ckpt="bpebyte_br_greedy_root_1.3B", paper_total_steps=180000,
         blurb="1.3B, byte backbone with greedy-root BPE boundaries"),
    dict(id="aunet2", label="AU-Net 2", short="AU-Net 2",
         run="aunet2_1.3B_a100x4", total_steps=180000, slot=3,
         paper_ckpt="aunet2_1.3B", paper_total_steps=180000,
         blurb="1.3B, 2-stage autoregressive U-Net"),
    dict(id="aunet3", label="AU-Net 3", short="AU-Net 3",
         run="aunet3_1.3B_a100x4", total_steps=102857, slot=4,
         paper_ckpt=None,
         blurb="1.3B, 3-stage autoregressive U-Net"),
]

# Paper (previous submission) 0-shot headline numbers are assembled from three
# eval directories; each contributes a disjoint set of tasks.
PAPER_SOURCES = [
    ("evals_5bench",        ["hellaswag", "arc_easy", "piqa", "winogrande", "boolq"]),
    ("evals_mmlu_arcc",     ["arc_challenge"]),
    ("evals_mmlu_text_0shot", ["mmlu_text"]),
]
PAPER_HEADLINE_AVG5 = {"transformer": 60.0, "bpebyte": 60.4, "aunet2": 60.1}

# Milestone evals of the paper run nest one directory per task; the final
# checkpoint does not.  Both are pinned to these named directories on purpose:
# the same checkpoint dirs also hold few-shot and variant reruns
# (evals_5shot, evals_hellaswag_3shot, evals_arc_challenge_5shot,
# evals_oldeval_*, eval_gr90_fg, evals_fg_162k, ...) whose numbers look
# plausible but do not reproduce the published table — globbing evals_* gives
# the final checkpoint an Avg5 of 61.10 instead of the correct 0-shot 60.45.
# Never glob.  scrape_tasks() additionally drops anything not recorded at
# n-shot 0.
PAPER_MILESTONE_SOURCES = [
    ("evals_5bench",    ["hellaswag", "arc_easy", "piqa", "winogrande", "boolq"]),
    ("evals_mmlu_arcc", ["arc_challenge"]),
    ("evals_mmlu_text", ["mmlu_text"]),
]

STAGE_RE = re.compile(r"^stage_(\d{10})(?:_(.+))?$")
PAPER_STAGE_RE = re.compile(r"^stage_(\d+)$")

# ------------------------------------------------------------------ json load


HEAD_SCAN_LIMIT = 64 << 20  # stop hunting for a head key past this much text


def load_head_blocks(path, keys):
    """Decode the named top-level values of an lm-eval results.json.

    These files carry per-sample logs and can run to hundreds of MB, but every
    key we need (`results`, then `n-shot`) is written before the giant `samples`
    array.  Walk forward once, decoding each requested value in file order, and
    stop as soon as the last one is in hand.  `keys` must be given in the order
    lm-eval writes them.  A key that never turns up is simply absent from the
    returned dict.
    """
    dec = json.JSONDecoder()
    out = {}
    with open(path, "rb") as fh:
        inc = codecs.getincrementaldecoder("utf-8")("replace")
        state = {"buf": inc.decode(fh.read(1 << 16)), "read": 1 << 16}

        def more():
            chunk = fh.read(1 << 21)
            if not chunk:
                return False
            state["buf"] += inc.decode(chunk)
            state["read"] += len(chunk)
            return True

        for key in keys:
            pat = '"%s"' % key
            i = state["buf"].find(pat)
            while i < 0:
                if state["read"] >= HEAD_SCAN_LIMIT or not more():
                    break
                i = state["buf"].find(pat)
            if i < 0:
                continue
            # the colon may sit just past the current buffer edge
            while state["buf"].find(":", i + len(pat)) < 0 and more():
                pass
            j = state["buf"].index(":", i + len(pat))
            state["buf"] = state["buf"][j + 1:].lstrip()
            while True:
                try:
                    obj, end = dec.raw_decode(state["buf"])
                    break
                except ValueError:
                    if not more():
                        raise
            out[key] = obj
            state["buf"] = state["buf"][end:]
    return out


def scrape_tasks(path, warnings, keep=None):
    """Extract the tracked 0-shot task scores (percent) held in one results.json.

    `keep` optionally restricts which task keys are accepted.  Any task whose
    recorded `n-shot` is not 0 is dropped on the floor: several eval directories
    beside the ones we read hold few-shot reruns whose numbers look plausible but
    are not the published 0-shot ones.
    """
    try:
        head = load_head_blocks(path, ["results", "n-shot"])
    except Exception as exc:  # noqa: BLE001 - report and keep going
        warnings.append(f"could not parse {rel(path)}: {type(exc).__name__}: {exc}")
        return {}
    res = head.get("results") or {}
    nshot = head.get("n-shot") or {}
    out = {}
    for key, metric in TASK_METRIC.items():
        if keep is not None and key not in keep:
            continue
        node = res.get(key)
        if not (isinstance(node, dict) and metric in node):
            continue
        shots = nshot.get(key)
        if shots is not None and shots != 0:
            warnings.append(f"{rel(path)}: ignored {key} — recorded n-shot={shots}, not 0")
            continue
        try:
            out[key] = round(100.0 * float(node[metric]), 4)
        except (TypeError, ValueError):
            warnings.append(f"non-numeric {key}.{metric} in {rel(path)}")
    return out


def rel(path):
    return os.path.relpath(path, ROOT)


def averages(scores):
    def avg(keys):
        if all(k in scores and scores[k] is not None for k in keys):
            return round(sum(scores[k] for k in keys) / len(keys), 4)
        return None
    return avg(AVG5), avg(AVG7)


def merge_scores(target, incoming, source, provenance, warnings, label):
    for key, val in incoming.items():
        if key in target and abs(target[key] - val) > 1e-9:
            warnings.append(
                f"{label}: {key} disagrees between {provenance[key]} "
                f"({target[key]:.2f}) and {rel(source)} ({val:.2f}); keeping the first"
            )
            continue
        if key not in target:
            target[key] = val
            provenance[key] = rel(source)


# ------------------------------------------------------------------- scanning


def scan_family(fam, warnings):
    run_dir = os.path.join(ROOT, "runs", fam["run"])
    total = fam["total_steps"]
    out = dict(
        run_dir=rel(run_dir),
        run_dir_exists=os.path.isdir(run_dir),
        milestones=[],
        expected=[],
        skipped=[],
    )
    by_step = {}
    if out["run_dir_exists"]:
        for name in sorted(os.listdir(run_dir)):
            full = os.path.join(run_dir, name)
            if not os.path.isdir(full):
                continue
            m = STAGE_RE.match(name)
            if not m:
                continue
            step = int(m.group(1))
            rj = os.path.join(full, "results.json")
            if not os.path.isfile(rj):
                out["skipped"].append(dict(dir=name, step=step, why="no results.json"))
                continue
            by_step.setdefault(step, []).append((name, rj))

    for step in sorted(by_step):
        scores, prov, dirs = {}, {}, []
        # bare stage_<step> first, so it wins any disagreement with a suffixed rerun
        entries = sorted(by_step[step], key=lambda e: (len(e[0]), e[0]))
        for name, rj in entries:
            got = scrape_tasks(rj, warnings)
            if not got:
                out["skipped"].append(dict(dir=name, step=step,
                                           why="results.json has none of the tracked tasks"))
                continue
            dirs.append(name)
            merge_scores(scores, got, rj, prov, warnings, f"{fam['label']} @ step {step}")
        if not dirs:
            continue
        a5, a7 = averages(scores)
        pct = 100.0 * step / total
        nominal = min(MILESTONES, key=lambda p: abs(round(total * p / 100.0) - step))
        off = abs(round(total * nominal / 100.0) - step) > 2
        out["milestones"].append(dict(
            step=step, pct=round(pct, 3), nominal=None if off else nominal,
            dirs=dirs, provenance=prov,
            scores={k: scores.get(k) for k, _, _ in TASKS},
            missing=[k for k, _, _ in TASKS if k not in scores],
            avg5=a5, avg7=a7,
        ))

    # checkpoint step names only (never their contents) - lets us separate
    # "training has not got there yet" from "checkpoint is sitting there un-evaluated"
    ckpt_steps = set()
    ck = os.path.join(run_dir, "checkpoints")
    if os.path.isdir(ck):
        for name in os.listdir(ck):
            if name.isdigit():
                ckpt_steps.add(int(name))
    out["latest_ckpt"] = max(ckpt_steps) if ckpt_steps else None

    have = {m["nominal"] for m in out["milestones"]}
    skipped_steps = {s["step"] for s in out["skipped"]}
    for pct in MILESTONES:
        step = int(round(total * pct / 100.0))
        if pct in have:
            reason = "present"
        elif not out["run_dir_exists"]:
            reason = "run directory does not exist"
        elif step in skipped_steps:
            reason = "stage dir exists but holds no usable results.json"
        elif step in ckpt_steps:
            reason = "checkpoint exists, milestone eval not run yet"
        elif out["latest_ckpt"] is None:
            reason = "no checkpoints at all under this run"
        else:
            reason = ("training has not reached this step (latest checkpoint %s)"
                      % f"{out['latest_ckpt']:,}")
        out["expected"].append(dict(pct=pct, step=step, present=pct in have, reason=reason))
    return out


def scan_paper(fam, warnings):
    if not fam.get("paper_ckpt"):
        return None
    base = os.path.join(ROOT, "main", "main", "1.3B", fam["paper_ckpt"])
    if not os.path.isdir(base):
        warnings.append(f"{fam['label']}: paper checkpoint dir missing: {rel(base)}")
        return None
    scores, prov = {}, {}
    used, absent = [], []
    for sub, keys in PAPER_SOURCES:
        rj = os.path.join(base, sub, "results.json")
        if not os.path.isfile(rj):
            absent.append(sub)
            continue
        got = scrape_tasks(rj, warnings)
        got = {k: v for k, v in got.items() if k in keys}
        for k in keys:
            if k not in got:
                warnings.append(f"{fam['label']} paper: {k} absent from {rel(rj)}")
        if got:
            used.append(rel(rj))
            merge_scores(scores, got, rj, prov, warnings, f"{fam['label']} paper")
    if not scores:
        return None
    a5, a7 = averages(scores)
    headline = PAPER_HEADLINE_AVG5.get(fam["id"])
    check = None
    if headline is not None and a5 is not None:
        note = ""
        if abs(round(a5, 1) - headline) >= 0.05:
            note = ("re-rounding boundary: the exact mean is %.4f, which rounds to %.1f, "
                    "while averaging the per-task numbers at 1 dp gives %.1f"
                    % (a5, round(a5, 1), round(sum(round(scores[k], 1) for k in AVG5) / 5.0, 1)))
        check = dict(expected=headline, got=round(a5, 2), note=note,
                     ok=abs(a5 - headline) <= 0.1)
    return dict(ckpt_dir=rel(base), sources=used, missing_dirs=absent,
                provenance=prov,
                scores={k: scores.get(k) for k, _, _ in TASKS},
                missing=[k for k, _, _ in TASKS if k not in scores],
                avg5=a5, avg7=a7, headline_check=check)


def scan_paper_milestones(fam, warnings):
    """Intermediate milestone evals of the *paper* run.

    Layout differs from that run's final checkpoint: milestones nest one
    directory per task under each eval directory,
    `stage_<step>/<evaldir>/<task>/results.json`, while the final checkpoint
    writes a single `<evaldir>/results.json`.
    """
    base = os.path.join(ROOT, "main", "main", "1.3B", fam["paper_ckpt"])
    total = fam.get("paper_total_steps") or fam["total_steps"]
    pts = []
    if not os.path.isdir(base):
        return pts
    for name in sorted(os.listdir(base)):
        m = PAPER_STAGE_RE.match(name)
        if not m or not os.path.isdir(os.path.join(base, name)):
            continue
        step = int(m.group(1))
        scores, prov, used, absent = {}, {}, [], []
        for sub, keys in PAPER_MILESTONE_SOURCES:
            for task in keys:
                rj = os.path.join(base, name, sub, task, "results.json")
                if not os.path.isfile(rj):
                    absent.append(f"{sub}/{task}")
                    continue
                got = scrape_tasks(rj, warnings, keep=[task])
                if not got:
                    warnings.append(f"{fam['label']} paper @ step {step}: "
                                    f"{task} not usable in {rel(rj)}")
                    continue
                used.append(rel(rj))
                merge_scores(scores, got, rj, prov, warnings,
                             f"{fam['label']} paper @ step {step}")
        if not scores:
            continue
        a5, a7 = averages(scores)
        nominal = min(MILESTONES, key=lambda p: abs(round(total * p / 100.0) - step))
        if abs(round(total * nominal / 100.0) - step) > 2:
            nominal = None
        pts.append(dict(
            step=step, pct=round(100.0 * step / total, 3), nominal=nominal,
            kind="milestone", dir=name, sources=used, missing_files=absent,
            provenance=prov,
            scores={k: scores.get(k) for k, _, _ in TASKS},
            missing=[k for k, _, _ in TASKS if k not in scores],
            avg5=a5, avg7=a7))
    pts.sort(key=lambda p: p["step"])
    return pts


def metric_view(rec):
    """Flatten one point to {metric_key: value} over Avg5, Avg7 and every task."""
    out = {"avg5": rec["avg5"], "avg7": rec["avg7"]}
    for k, _, _ in TASKS:
        out[k] = rec["scores"].get(k)
    return out


def paper_deltas(run_points, paper_points):
    """Re-trained minus paper at every nominal milestone both runs reached."""
    by_run = {m["nominal"]: m for m in run_points if m.get("nominal") is not None}
    by_pap = {m["nominal"]: m for m in paper_points if m.get("nominal") is not None}
    rows = []
    for pct in MILESTONES:
        if pct not in by_run or pct not in by_pap:
            continue
        a, b = by_run[pct], by_pap[pct]
        va, vb = metric_view(a), metric_view(b)
        rows.append(dict(
            pct=pct, run_step=a["step"], paper_step=b["step"],
            paper_kind=b.get("kind", "milestone"),
            run=va, paper=vb,
            delta={k: (None if va[k] is None or vb[k] is None
                       else round(va[k] - vb[k], 4)) for k in va},
        ))
    return rows


def collect():
    warnings = []
    fams = []
    for fam in FAMILIES:
        rec = dict(fam)
        rec.pop("paper_ckpt", None)
        scanned = scan_family(fam, warnings)
        rec.update(scanned)
        rec["paper"] = scan_paper(fam, warnings)
        rec["paper_total_steps"] = fam.get("paper_total_steps") or fam["total_steps"]
        curve = scan_paper_milestones(fam, warnings) if fam.get("paper_ckpt") else []
        rec["paper_n_milestones"] = len(curve)
        if rec["paper"]:
            p = rec["paper"]
            curve = curve + [dict(
                step=rec["paper_total_steps"], pct=100.0, nominal=100, kind="final",
                dir=p["ckpt_dir"], sources=p["sources"], missing_files=p["missing_dirs"],
                provenance=p["provenance"], scores=p["scores"], missing=p["missing"],
                avg5=p["avg5"], avg7=p["avg7"])]
        rec["paper_curve"] = curve
        rec["paper_shared"] = paper_deltas(rec["milestones"], curve)
        n = len(rec["milestones"])
        if n == 0:
            rec["status"] = "none"
            rec["status_text"] = "no data yet"
        elif n == len(MILESTONES):
            rec["status"] = "complete"
            rec["status_text"] = f"all {n} milestones"
        else:
            rec["status"] = "partial"
            rec["status_text"] = f"{n} of {len(MILESTONES)} milestones"
        fams.append(rec)
    return dict(
        generated=_dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        root=ROOT,
        milestones=MILESTONES,
        tasks=[dict(key=k, label=l, metric=m) for k, l, m in TASKS],
        avg5=AVG5, avg7=AVG7,
        paper_sources=[dict(dir=d, tasks=t) for d, t in PAPER_SOURCES],
        paper_milestone_sources=[dict(dir=d, tasks=t) for d, t in PAPER_MILESTONE_SOURCES],
        families=fams,
        warnings=warnings,
    )


# ----------------------------------------------------------------------- HTML

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Downstream score vs training progress</title>
<style>
:root {
  color-scheme: light;
  --ground: #fbfcfd;
  --panel: #f2f4f8;
  --panel-2: #e9edf4;
  --ink: #161922;
  --ink-soft: #4b5364;
  --ink-mute: #737d8f;
  --border: #dde2ea;
  --border-soft: #e8ecf2;
  --accent: #3a56d4;
  --accent-soft: #eaeeff;
  --grid: #dfe4ec;
  --axis: #c3cad6;
  --ok: #0f8a68;
  --ok-soft: #e2f4ee;
  --warn: #b26a08;
  --warn-soft: #fbf0dd;
  --idle: #7c8598;
  --idle-soft: #edeff3;
  --shadow: 0 1px 2px rgba(22,25,34,.05), 0 8px 24px -16px rgba(22,25,34,.28);
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  /* series palette - validated with dataviz/scripts/validate_palette.py
     --mode light --pairs all --surface #f2f4f8 : all six checks PASS */
  --s1: #024e9a;
  --s2: #8f144c;
  --s3: #059a94;
  --s4: #7b6c01;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #0e1015;
    --panel: #161a22;
    --panel-2: #1d222c;
    --ink: #e7eaf1;
    --ink-soft: #aab3c4;
    --ink-mute: #7d8698;
    --border: #262c38;
    --border-soft: #1f242e;
    --accent: #8098f5;
    --accent-soft: #1a2140;
    --grid: #242a35;
    --axis: #39414f;
    --ok: #3fc79c;
    --ok-soft: #10291f;
    --warn: #d99a3c;
    --warn-soft: #2b2113;
    --idle: #8d97ab;
    --idle-soft: #1c212b;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
    /* --mode dark --pairs all --surface #161a22 : all six checks PASS */
    --s1: #2470cb;
    --s2: #a75073;
    --s3: #04a19b;
    --s4: #7b6c01;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #0e1015;
  --panel: #161a22;
  --panel-2: #1d222c;
  --ink: #e7eaf1;
  --ink-soft: #aab3c4;
  --ink-mute: #7d8698;
  --border: #262c38;
  --border-soft: #1f242e;
  --accent: #8098f5;
  --accent-soft: #1a2140;
  --grid: #242a35;
  --axis: #39414f;
  --ok: #3fc79c;
  --ok-soft: #10291f;
  --warn: #d99a3c;
  --warn-soft: #2b2113;
  --idle: #8d97ab;
  --idle-soft: #1c212b;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  --s1: #2470cb;
  --s2: #a75073;
  --s3: #04a19b;
  --s4: #7b6c01;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.55;
  overflow-x: hidden;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 72px; display: flex; flex-direction: column; gap: 26px; }

.eyebrow {
  font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--ink-mute); margin: 0;
}
h1 { font-size: 26px; line-height: 1.2; margin: 6px 0 0; letter-spacing: -.01em; text-wrap: balance; }
h2 { font-size: 16px; margin: 0; letter-spacing: -.005em; }
p  { margin: 0; }
.lede { color: var(--ink-soft); max-width: 68ch; margin-top: 10px; }

header.page { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.stamp { font-family: var(--mono); font-size: 11px; color: var(--ink-mute); text-align: right; line-height: 1.7; }

button { font: inherit; color: inherit; cursor: pointer; }
.ghost {
  background: var(--panel); border: 1px solid var(--border); border-radius: 7px;
  padding: 5px 11px; font-size: 12px; color: var(--ink-soft);
}
.ghost:hover { background: var(--panel-2); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }

.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  box-shadow: var(--shadow); padding: 20px 22px; display: flex; flex-direction: column; gap: 16px;
}
.card > .head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.card > .head .note { font-size: 12.5px; color: var(--ink-mute); }

/* run status rail */
.rail { display: grid; grid-template-columns: repeat(auto-fit, minmax(232px, 1fr)); gap: 12px; }
.runchip {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; display: flex; flex-direction: column; gap: 5px; box-shadow: var(--shadow);
  border-left: 3px solid var(--sc);
}
.runchip .name { display: flex; align-items: center; gap: 8px; font-weight: 620; font-size: 14px; }
.runchip .swatch { width: 10px; height: 10px; border-radius: 3px; background: var(--sc); flex: none; }
.runchip .meta { font-family: var(--mono); font-size: 11.5px; color: var(--ink-mute); word-break: break-all; }
.pill {
  display: inline-flex; align-items: center; gap: 5px; align-self: flex-start;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 999px; border: 1px solid transparent;
}
.pill.complete { background: var(--ok-soft); color: var(--ok); border-color: color-mix(in srgb, var(--ok) 32%, transparent); }
.pill.partial  { background: var(--warn-soft); color: var(--warn); border-color: color-mix(in srgb, var(--warn) 32%, transparent); }
.pill.none     { background: var(--idle-soft); color: var(--idle); border-color: color-mix(in srgb, var(--idle) 32%, transparent); }

/* controls */
.controls { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.seg { display: flex; flex-wrap: wrap; gap: 4px; background: var(--panel-2); border: 1px solid var(--border); border-radius: 9px; padding: 4px; }
.seg button {
  background: transparent; border: 0; border-radius: 6px; padding: 5px 11px;
  font-size: 12.5px; color: var(--ink-soft); white-space: nowrap;
}
.seg button:hover { background: color-mix(in srgb, var(--accent) 10%, transparent); }
.seg button[aria-pressed="true"] { background: var(--panel); color: var(--ink); font-weight: 600; box-shadow: var(--shadow); }
label.check { display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--ink-soft); }
label.check input { accent-color: var(--accent); width: 15px; height: 15px; }
.seg button.constit { box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 42%, transparent); }
.seg button[aria-pressed="true"].constit { box-shadow: var(--shadow), inset 0 0 0 1px color-mix(in srgb, var(--accent) 42%, transparent); }

/* metric definition panel - always visible, sits with the selector */
.metricdef {
  display: flex; flex-direction: column; gap: 7px;
  background: var(--panel-2); border: 1px solid var(--border-soft);
  border-radius: 10px; padding: 11px 13px;
}
.metricdef .row { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; font-size: 12.5px; color: var(--ink-soft); }
.metricdef .tag {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--ink-mute); min-width: 46px; flex: none;
}
.metricdef .rule { font-size: 12.5px; color: var(--ink-mute); max-width: 80ch; }
.metricdef .rule b { color: var(--ink-soft); font-weight: 640; }
.chip {
  display: inline-flex; align-items: baseline; gap: 6px;
  border: 1px solid var(--border); background: var(--panel); border-radius: 6px;
  padding: 2px 8px; font-size: 12px; color: var(--ink-soft); white-space: nowrap;
}
.chip .m { font-family: var(--mono); font-size: 10px; color: var(--ink-mute); }
.chip.in { border-color: color-mix(in srgb, var(--accent) 48%, transparent); background: var(--accent-soft); color: var(--ink); font-weight: 620; }
.chip.in .m { color: var(--ink-soft); }
.chip.out { opacity: .5; }

td.dpos { color: var(--ok); }
td.dneg { color: var(--warn); }
.fine { font-size: 12.5px; color: var(--ink-mute); max-width: 88ch; }

/* chart */
.chartbox { position: relative; }
svg.chart { display: block; width: 100%; height: auto; overflow: visible; }
svg.chart text { font-family: var(--sans); fill: var(--ink-mute); }
svg.chart .axis-title { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; font-family: var(--mono); }
svg.chart .tick { font-size: 11px; font-family: var(--mono); }
svg.chart .endlabel { font-size: 12px; font-weight: 600; fill: var(--ink-soft); }
svg.chart .baselabel { font-size: 10.5px; font-family: var(--mono); fill: var(--ink-mute); }

.tip {
  position: absolute; pointer-events: none; opacity: 0; transition: opacity .1s ease;
  background: var(--panel); border: 1px solid var(--border); border-radius: 9px;
  box-shadow: var(--shadow); padding: 10px 12px; min-width: 208px; z-index: 5;
}
.tip.on { opacity: 1; }
.tip .t-head { display: flex; align-items: center; gap: 7px; font-weight: 640; font-size: 13px; margin-bottom: 2px; }
.tip .t-sub { font-family: var(--mono); font-size: 11px; color: var(--ink-mute); margin-bottom: 8px; }
.tip table { border-collapse: collapse; width: 100%; }
.tip td { padding: 1.5px 0; font-size: 12px; }
.tip td.v { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); padding-left: 14px; }
.tip td.k { color: var(--ink-soft); }
.tip tr.sep td { border-top: 1px solid var(--border); padding-top: 5px; }
.tip tr.emph td { font-weight: 640; }

.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; }
.legend .item { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--ink-soft); }
.legend .item.off { opacity: .5; }
.legend .key { width: 22px; height: 0; border-top: 2px solid var(--sc); flex: none; }
.legend .key.dash { border-top-style: dashed; }
.legend .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--sc); flex: none; }

/* matrix */
.matrix { overflow-x: auto; }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--border-soft); white-space: nowrap; }
th { font-family: var(--mono); font-size: 10.5px; letter-spacing: .07em; text-transform: uppercase; color: var(--ink-mute); font-weight: 500; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
td.na { color: var(--ink-mute); }
tr.paper td { background: color-mix(in srgb, var(--accent) 6%, transparent); font-style: italic; }
tr.famhead td { padding-top: 16px; font-weight: 640; font-style: normal; border-bottom: 1px solid var(--border); }
td .fam { display: inline-flex; align-items: center; gap: 8px; }
td .fam .dot { width: 9px; height: 9px; border-radius: 3px; background: var(--sc); flex: none; }
.mark { display: inline-block; width: 15px; text-align: center; font-family: var(--mono); }
.mark.yes { color: var(--ok); }
.mark.no  { color: var(--ink-mute); }

.scroller { overflow-x: auto; -webkit-overflow-scrolling: touch; }

.notes { display: flex; flex-direction: column; gap: 10px; font-size: 13px; color: var(--ink-soft); }
.notes ul { margin: 0; padding-left: 18px; display: flex; flex-direction: column; gap: 6px; }
.notes code, code.path { font-family: var(--mono); font-size: 12px; background: var(--panel-2); padding: 1px 5px; border-radius: 4px; color: var(--ink-soft); word-break: break-all; }
.flag { border-left: 3px solid var(--warn); background: var(--warn-soft); border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 12.5px; color: var(--ink); }
.flag b { font-weight: 640; }

@media (max-width: 640px) {
  .wrap { padding: 22px 14px 56px; }
  h1 { font-size: 21px; }
  .stamp { text-align: left; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">

  <header class="page">
    <div>
      <p class="eyebrow">Re-training milestones vs. AAAI submission checkpoints</p>
      <h1>Downstream score as a function of training progress</h1>
      <p class="lede">Every milestone eval found under <code class="path">runs/&lt;run&gt;/stage_&lt;step&gt;/results.json</code>,
      placed on a shared 0&ndash;100% training-progress axis. The paper checkpoints are a
      <em>different</em> training run of the same model: where that run kept and evaluated intermediate checkpoints they are drawn
      as a second, dashed curve in the same hue &mdash; never merged into the re-trained line, never a separate model family.</p>
    </div>
    <div class="stamp">
      <div id="stamp-time"></div>
      <div><button class="ghost" id="theme" type="button">Switch theme</button></div>
    </div>
  </header>

  <section class="rail" id="rail"></section>

  <section class="card">
    <div class="head">
      <h2>Score vs progress</h2>
      <span class="note" id="chart-note"></span>
    </div>
    <div class="controls">
      <div class="seg" id="metric-seg" role="group" aria-label="Metric"></div>
      <label class="check"><input type="checkbox" id="show-paper" checked> Show paper run</label>
    </div>
    <div class="metricdef" id="metricdef"></div>
    <div class="chartbox">
      <svg class="chart" id="chart" role="img" aria-label="Benchmark score against percent of training completed"></svg>
      <div class="tip" id="tip" role="status" aria-live="polite"></div>
    </div>
    <div class="legend" id="legend"></div>
  </section>

  <section class="card">
    <div class="head">
      <h2>Re-trained vs. paper run at the same point in training</h2>
      <span class="note" id="delta-note"></span>
    </div>
    <div class="scroller"><table id="deltatable"></table></div>
    <p class="fine" id="delta-fine"></p>
  </section>

  <section class="card">
    <div class="head">
      <h2>Milestone coverage</h2>
      <span class="note">Which of the seven scheduled milestones (25 / 30 / 50 / 60 / 75 / 90 / 100%) actually have an eval.</span>
    </div>
    <div class="matrix"><table id="coverage"></table></div>
  </section>

  <section class="card">
    <div class="head">
      <h2>Every number</h2>
      <span class="note">Percent of tracked metric, 0-shot. Blank cells are absent from the results file, never zero.</span>
    </div>
    <div class="scroller"><table id="bigtable"></table></div>
  </section>

  <section class="card">
    <div class="head"><h2>Provenance &amp; caveats</h2></div>
    <div class="notes" id="notes"></div>
  </section>

</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("payload").textContent);
  const SLOT = { 1: "var(--s1)", 2: "var(--s2)", 3: "var(--s3)", 4: "var(--s4)" };
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
  const fmt = (v, d) => (v == null || Number.isNaN(v)) ? null : v.toFixed(d == null ? 2 : d);
  const int = (n) => n.toLocaleString("en-US");

  /* ---------------- metric model ---------------- */
  const METRICS = [
    { key: "avg5", label: "Avg5" },
    { key: "avg7", label: "Avg7" },
  ].concat(D.tasks.map((t) => ({ key: t.key, label: t.label })));
  let metric = "avg5";
  let showPaper = true;

  const valueOf = (rec) => {
    if (!rec) return null;
    if (metric === "avg5") return rec.avg5;
    if (metric === "avg7") return rec.avg7;
    const v = rec.scores[metric];
    return v == null ? null : v;
  };
  const metricLabel = () => (METRICS.find((m) => m.key === metric) || {}).label;

  /* ---------------- theme ---------------- */
  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const dark = cur ? cur === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    draw();
  });
  $("#stamp-time").textContent = "built " + D.generated;

  /* ---------------- run status rail ---------------- */
  (function rail() {
    const host = $("#rail");
    D.families.forEach((f) => {
      const c = el("div", "runchip");
      c.style.setProperty("--sc", SLOT[f.slot]);
      const nm = el("div", "name");
      nm.appendChild(el("span", "swatch"));
      nm.appendChild(el("span", null, f.label));
      c.appendChild(nm);
      c.appendChild(el("span", "pill " + f.status, f.status_text));
      c.appendChild(el("div", "meta", f.run_dir + (f.run_dir_exists ? "" : "  (missing)")));
      c.appendChild(el("div", "meta", int(f.total_steps) + " total steps"
        + (f.latest_ckpt != null ? " · latest ckpt " + int(f.latest_ckpt) + " (" + (100 * f.latest_ckpt / f.total_steps).toFixed(1) + "%)" : " · no checkpoints")));
      c.appendChild(el("div", "meta", f.blurb));
      host.appendChild(c);
    });
  })();

  /* ---------------- controls ---------------- */
  const segBtn = {};
  (function seg() {
    const host = $("#metric-seg");
    METRICS.forEach((m) => {
      const b = el("button", null, m.label);
      b.type = "button";
      b.dataset.key = m.key;
      b.setAttribute("aria-pressed", String(m.key === metric));
      b.addEventListener("click", () => {
        metric = m.key;
        host.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        draw();
      });
      segBtn[m.key] = b;
      host.appendChild(b);
    });
  })();

  /* ---------------- metric definitions (always visible) ---------------- */
  const taskByKey = {};
  D.tasks.forEach((t) => { taskByKey[t.key] = t; });

  // every observed value of one task, across every point this page holds
  function spanOf(key) {
    const vals = [];
    D.families.forEach((f) => {
      f.milestones.forEach((m) => { if (m.scores[key] != null) vals.push(m.scores[key]); });
      (f.paper_curve || []).forEach((c) => { if (c.scores[key] != null) vals.push(c.scores[key]); });
    });
    return vals.length ? [Math.min.apply(null, vals), Math.max.apply(null, vals)] : null;
  }

  function drawMetricDef() {
    const host = $("#metricdef");
    host.innerHTML = "";
    const inAvg = metric === "avg5" ? D.avg5 : (metric === "avg7" ? D.avg7 : null);

    const row = (tag) => { const r = el("div", "row"); r.appendChild(el("span", "tag", tag)); host.appendChild(r); return r; };
    const chips = (r, keys) => keys.forEach((k) => {
      const t = taskByKey[k];
      if (!t) return;
      const cls = inAvg ? (inAvg.indexOf(k) >= 0 ? "chip in" : "chip out")
                        : (metric === k ? "chip in" : "chip");
      const c = el("span", cls);
      c.appendChild(el("span", null, t.label));
      c.appendChild(el("span", "m", t.metric.replace(/,none$/, "")));
      r.appendChild(c);
    });

    const r5 = row("Avg5"); chips(r5, D.avg5);
    r5.appendChild(el("span", "rule", "mean of these five"));

    const extra = D.avg7.filter((k) => D.avg5.indexOf(k) < 0);
    const r7 = row("Avg7"); chips(r7, extra);
    r7.appendChild(el("span", "rule", "the five above plus these two — mean of all seven"));

    const rule = row("Rule");
    const p1 = el("span", "rule");
    p1.innerHTML = "An average is computed <b>only when every constituent task is present</b> at that point; "
      + "otherwise the cell is blank, never a partial mean. The small type on each chip is the lm-eval metric key "
      + "actually read (all at <b>n-shot 0</b>).";
    rule.appendChild(p1);

    const mm = spanOf("mmlu_text");
    const rp = row("Read");
    const p2 = el("span", "rule");
    p2.innerHTML = "<b>Avg5 is the primary indicator.</b> MMLU barely moves at this scale"
      + (mm ? " (observed " + fmt(mm[0], 1) + "&ndash;" + fmt(mm[1], 1) + " across every point on this page, against a 25% chance floor)" : "")
      + ", so Avg7 dilutes the same movement rather than adding to it.";
    rp.appendChild(p2);

    Object.keys(segBtn).forEach((k) => {
      segBtn[k].classList.toggle("constit", !!inAvg && inAvg.indexOf(k) >= 0);
    });
  }
  $("#show-paper").addEventListener("change", (e) => { showPaper = e.target.checked; draw(); });

  /* ---------------- chart ---------------- */
  const SVGNS = "http://www.w3.org/2000/svg";
  const svg = $("#chart");
  const tip = $("#tip");
  const W = 980, H = 470, M = { t: 18, r: 186, b: 58, l: 62 };
  const PW = W - M.l - M.r, PH = H - M.t - M.b;
  let hot = [];

  const mk = (tag, attrs, txt) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (txt != null) n.textContent = txt;
    return n;
  };
  const surface = () => getComputedStyle(document.body).getPropertyValue("--panel").trim() || "#fff";

  function niceDomain(vals) {
    if (!vals.length) return [0, 100, 20];
    let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * 0.12;
    lo -= pad; hi += pad;
    const raw = (hi - lo) / 5;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= raw) || 10 * mag;
    return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step, step];
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    hot = [];
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

    const series = D.families.map((f) => {
      const pp = (showPaper ? (f.paper_curve || []) : [])
        .map((c) => ({ m: c, v: valueOf(c) })).filter((p) => p.v != null);
      // two or more points is a curve; a lone point (a run whose only paper
      // artefact is its final checkpoint) stays a horizontal reference
      const asCurve = pp.length > 1;
      return {
        f,
        pts: f.milestones.map((m) => ({ m, v: valueOf(m) })).filter((p) => p.v != null),
        ppts: asCurve ? pp : [],
        base: asCurve ? null : (pp.length ? pp[0].v : null),
        baseRec: asCurve ? null : (pp.length ? pp[0].m : null),
      };
    });

    const vals = [];
    series.forEach((s) => {
      s.pts.forEach((p) => vals.push(p.v));
      s.ppts.forEach((p) => vals.push(p.v));
      if (s.base != null) vals.push(s.base);
    });
    const [y0, y1, ystep] = niceDomain(vals);
    const X = (pct) => M.l + (pct / 100) * PW;
    const Y = (v) => M.t + PH - ((v - y0) / (y1 - y0)) * PH;

    const withData = series.filter((s) => s.pts.length).length;
    $("#chart-note").textContent = withData
      ? metricLabel() + " · " + withData + " of " + D.families.length + " families have milestone evals"
      : "no family has data for " + metricLabel();

    if (!vals.length) {
      svg.appendChild(mk("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "tick" },
        "No milestone has all tasks required for " + metricLabel() + "."));
      buildLegend(series);
      drawMetricDef();
      drawDeltas();
      return;
    }

    // y grid + ticks
    for (let v = y0; v <= y1 + 1e-9; v += ystep) {
      const y = Y(v);
      svg.appendChild(mk("line", { x1: M.l, x2: M.l + PW, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1 }));
      svg.appendChild(mk("text", { x: M.l - 10, y: y + 4, "text-anchor": "end", class: "tick" }, v.toFixed(ystep < 1 ? 1 : 0)));
    }
    // x axis
    svg.appendChild(mk("line", { x1: M.l, x2: M.l + PW, y1: M.t + PH, y2: M.t + PH, stroke: "var(--axis)", "stroke-width": 1 }));
    D.milestones.concat([0]).sort((a, b) => a - b).forEach((p) => {
      const x = X(p);
      svg.appendChild(mk("line", { x1: x, x2: x, y1: M.t + PH, y2: M.t + PH + 5, stroke: "var(--axis)", "stroke-width": 1 }));
      svg.appendChild(mk("text", { x: x, y: M.t + PH + 19, "text-anchor": "middle", class: "tick" }, p));
    });
    svg.appendChild(mk("text", { x: M.l + PW / 2, y: H - 10, "text-anchor": "middle", class: "axis-title" }, "Training progress (% of total steps)"));
    svg.appendChild(mk("text", { x: 14, y: M.t + PH / 2, "text-anchor": "middle", class: "axis-title",
      transform: "rotate(-90 14 " + (M.t + PH / 2) + ")" }, metricLabel() + " (%)"));

    // ---- paper baselines (drawn under the live curves) -------------------
    const gutter = [];   // right-margin labels, de-collided together below
    const bases = series.filter((s) => s.base != null)
      .map((s) => ({ s: s, y: Y(s.base), c: SLOT[s.f.slot] }))
      .sort((a, b) => a.y - b.y);

    bases.forEach((b) => {
      const y = b.y, c = b.c;
      svg.appendChild(mk("line", { x1: M.l, x2: M.l + PW, y1: y, y2: y, stroke: c,
        "stroke-width": 2, "stroke-dasharray": "2 6", "stroke-linecap": "round", opacity: .85 }));
      const d = 5;
      svg.appendChild(mk("path", { d: "M" + X(100) + " " + (y - d) + "L" + (X(100) + d) + " " + y
        + "L" + X(100) + " " + (y + d) + "L" + (X(100) - d) + " " + y + "Z",
        fill: surface(), stroke: c, "stroke-width": 2 }));
      gutter.push({ y: y, ly: y, c: c, text: b.s.f.short + " (paper) " + fmt(b.s.base), paper: true });
      hot.push({ x: X(100), y: y, fam: b.s.f, rec: b.s.baseRec, v: b.s.base, paper: true });
    });

    // ---- paper-run curves -------------------------------------------------
    // Same hue as the re-trained arm on purpose: same model family, different
    // training run. Dashed stroke + hollow markers are the only channels that
    // separate them, so the two are never read as different models.
    series.forEach((s) => {
      if (!s.ppts.length) return;
      const c = SLOT[s.f.slot];
      const dstr = s.ppts.map((p, i) => (i ? "L" : "M") + X(p.m.pct) + " " + Y(p.v)).join("");
      svg.appendChild(mk("path", { d: dstr, fill: "none", stroke: c, "stroke-width": 2,
        "stroke-dasharray": "7 5", "stroke-linecap": "round", "stroke-linejoin": "round", opacity: .9 }));
      s.ppts.forEach((p) => {
        svg.appendChild(mk("circle", { cx: X(p.m.pct), cy: Y(p.v), r: 4.2,
          fill: surface(), stroke: c, "stroke-width": 2 }));
        hot.push({ x: X(p.m.pct), y: Y(p.v), fam: s.f, rec: p.m, v: p.v, paper: true });
      });
      const lastp = s.ppts[s.ppts.length - 1];
      gutter.push({ y: Y(lastp.v), ly: Y(lastp.v), c: c,
                    text: s.f.short + " (paper) " + fmt(lastp.v), paper: true });
    });

    // ---- live curves ------------------------------------------------------
    series.forEach((s) => {
      if (!s.pts.length) return;
      const c = SLOT[s.f.slot];
      const dstr = s.pts.map((p, i) => (i ? "L" : "M") + X(p.m.pct) + " " + Y(p.v)).join("");
      svg.appendChild(mk("path", { d: dstr, fill: "none", stroke: c, "stroke-width": 2,
        "stroke-linecap": "round", "stroke-linejoin": "round" }));
      s.pts.forEach((p) => {
        svg.appendChild(mk("circle", { cx: X(p.m.pct), cy: Y(p.v), r: 4.5, fill: c,
          stroke: surface(), "stroke-width": 2 }));
        hot.push({ x: X(p.m.pct), y: Y(p.v), fam: s.f, rec: p.m, v: p.v, paper: false });
      });
      const last = s.pts[s.pts.length - 1];
      if (last.m.pct >= 88) {
        // would run into the right margin: label it in the gutter with the baselines
        gutter.push({ y: Y(last.v), ly: Y(last.v), c: c, text: s.f.short, paper: false });
      } else {
        const lx = X(last.m.pct) + 12, ly = Y(last.v) + 4;
        svg.appendChild(mk("circle", { cx: lx, cy: ly - 4, r: 4, fill: c }));
        svg.appendChild(mk("text", { x: lx + 9, y: ly, class: "endlabel" }, s.f.short));
      }
    });

    // ---- de-collide every right-gutter label ------------------------------
    const GAP = 15;
    gutter.sort((a, b) => a.y - b.y);
    for (let i = 1; i < gutter.length; i++)
      if (gutter[i].ly - gutter[i - 1].ly < GAP) gutter[i].ly = gutter[i - 1].ly + GAP;
    const over = gutter.length ? gutter[gutter.length - 1].ly - (M.t + PH) : 0;
    if (over > 0) gutter.forEach((g) => { g.ly -= over; });
    for (let i = gutter.length - 2; i >= 0; i--)
      if (gutter[i + 1].ly - gutter[i].ly < GAP) gutter[i].ly = gutter[i + 1].ly - GAP;

    gutter.forEach((g) => {
      const x0 = M.l + PW + 8, x1 = M.l + PW + 18;
      if (Math.abs(g.ly - g.y) > 1)
        svg.appendChild(mk("path", { d: "M" + x0 + " " + g.y + "L" + x1 + " " + g.ly,
          fill: "none", stroke: g.c, "stroke-width": 1, opacity: .5 }));
      if (g.paper) {
        svg.appendChild(mk("line", { x1: x1 + 2, x2: x1 + 12, y1: g.ly, y2: g.ly, stroke: g.c,
          "stroke-width": 2, "stroke-dasharray": "2 3", "stroke-linecap": "round" }));
        svg.appendChild(mk("text", { x: x1 + 18, y: g.ly + 4, class: "baselabel" }, g.text));
      } else {
        svg.appendChild(mk("circle", { cx: x1 + 6, cy: g.ly, r: 4, fill: g.c }));
        svg.appendChild(mk("text", { x: x1 + 18, y: g.ly + 4, class: "endlabel" }, g.text));
      }
    });

    buildLegend(series);
    drawMetricDef();
    drawDeltas();
  }

  function buildLegend(series) {
    const host = $("#legend");
    host.innerHTML = "";
    series.forEach((s) => {
      const i = el("span", "item" + (s.pts.length ? "" : " off"));
      i.style.setProperty("--sc", SLOT[s.f.slot]);
      i.appendChild(el("span", "key"));
      i.appendChild(el("span", null, s.f.label + (s.pts.length ? " — " + s.pts.length + " milestone" + (s.pts.length === 1 ? "" : "s") : " — no data yet")));
      host.appendChild(i);
    });
    if (showPaper && series.some((s) => s.ppts.length || s.base != null)) {
      const n = series.reduce((a, s) => a + s.ppts.length, 0);
      const i = el("span", "item");
      i.style.setProperty("--sc", "var(--ink-mute)");
      i.appendChild(el("span", "key dash"));
      i.appendChild(el("span", null, n
        ? "paper run — same family, previous training run (dashed, hollow markers)"
        : "paper run — final checkpoint only (dashed reference)"));
      host.appendChild(i);
    }
  }

  /* ---------------- hover ---------------- */
  function nearest(mx, my) {
    let best = null, bd = Infinity;
    hot.forEach((h) => {
      const d = (h.x - mx) * (h.x - mx) + (h.y - my) * (h.y - my);
      if (d < bd) { bd = d; best = h; }
    });
    return bd <= 40 * 40 ? best : null;
  }

  function showTip(h, clientX, clientY) {
    const box = svg.parentNode.getBoundingClientRect();
    tip.innerHTML = "";
    const head = el("div", "t-head");
    const dot = el("span");
    dot.style.cssText = "width:10px;height:10px;border-radius:3px;flex:none;background:" + SLOT[h.fam.slot];
    head.appendChild(dot);
    head.appendChild(el("span", null, h.fam.label + (h.paper ? " · paper run" : " · re-trained")));
    tip.appendChild(head);
    const totalFor = h.paper ? h.fam.paper_total_steps : h.fam.total_steps;
    tip.appendChild(el("div", "t-sub",
      h.rec.pct.toFixed(1) + "% of training · step " + int(h.rec.step) + " / " + int(totalFor)
      + (h.paper ? (h.rec.kind === "final" ? " · final ckpt" : " · milestone") : "")));

    const tbl = document.createElement("table");
    D.tasks.forEach((t) => {
      const tr = tbl.insertRow();
      tr.insertCell().outerHTML = '<td class="k">' + t.label + "</td>";
      const v = h.rec.scores[t.key];
      tr.insertCell().outerHTML = '<td class="v">' + (v == null ? "—" : fmt(v)) + "</td>";
    });
    [["Avg5", h.rec.avg5], ["Avg7", h.rec.avg7]].forEach((row, i) => {
      const tr = tbl.insertRow();
      tr.className = "emph" + (i === 0 ? " sep" : "");
      tr.insertCell().outerHTML = '<td class="k">' + row[0] + "</td>";
      tr.insertCell().outerHTML = '<td class="v">' + (row[1] == null ? "—" : fmt(row[1])) + "</td>";
    });
    tip.appendChild(tbl);
    tip.classList.add("on");

    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = clientX - box.left + 16, y = clientY - box.top - th / 2;
    if (x + tw > box.width) x = clientX - box.left - tw - 16;
    y = Math.max(0, Math.min(box.height - th, y));
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (W / r.width);
    const my = (e.clientY - r.top) * (H / r.height);
    const h = nearest(mx, my);
    if (h) showTip(h, e.clientX, e.clientY); else tip.classList.remove("on");
  });
  svg.addEventListener("pointerleave", () => tip.classList.remove("on"));

  /* ---------------- re-trained vs paper, same progress ---------------- */
  function drawDeltas() {
    const t = $("#deltatable");
    t.innerHTML = "";
    const rows = [];
    D.families.forEach((f) => (f.paper_shared || []).forEach((r) => rows.push({ f: f, r: r })));
    const usable = rows.filter((x) => x.r.run[metric] != null && x.r.paper[metric] != null);

    $("#delta-note").textContent = metricLabel()
      + " · re-trained minus paper run, at every milestone both runs reached";

    const head = t.createTHead().insertRow();
    ["Family", "Progress", "Re-trained step", "Paper step", "Re-trained",
     "Paper run", "\u0394 (re-trained \u2212 paper)"].forEach((h) => head.appendChild(el("th", null, h)));
    const tb = t.createTBody();

    if (!usable.length) {
      const c = tb.insertRow().insertCell();
      c.colSpan = 7;
      c.className = "na";
      c.textContent = rows.length
        ? "No shared milestone has " + metricLabel() + " on both runs."
        : "No family yet has a milestone the paper run also evaluated.";
      return;
    }

    usable.forEach((x) => {
      const tr = tb.insertRow();
      const c0 = tr.insertCell();
      const fam = el("span", "fam");
      fam.style.setProperty("--sc", SLOT[x.f.slot]);
      fam.appendChild(el("span", "dot"));
      fam.appendChild(el("span", null, x.f.label));
      c0.appendChild(fam);

      const pc = tr.insertCell(); pc.className = "num"; pc.textContent = x.r.pct + "%";
      const s1 = tr.insertCell(); s1.className = "num"; s1.textContent = int(x.r.run_step);
      const s2 = tr.insertCell(); s2.className = "num";
      s2.textContent = int(x.r.paper_step) + (x.r.paper_kind === "final" ? " (final)" : "");
      const v1 = tr.insertCell(); v1.className = "num"; v1.textContent = fmt(x.r.run[metric]);
      const v2 = tr.insertCell(); v2.className = "num"; v2.textContent = fmt(x.r.paper[metric]);

      const d = x.r.delta[metric];
      const dc = tr.insertCell();
      dc.className = "num " + (d == null ? "na" : (d >= 0 ? "dpos" : "dneg"));
      dc.textContent = d == null ? "—" : (d >= 0 ? "+" : "\u2212") + Math.abs(d).toFixed(2);
    });

    const withM = D.families.filter((f) => f.paper_n_milestones > 0);
    const onlyF = D.families.filter((f) => f.paper && !f.paper_n_milestones);
    const bits = [];
    bits.push("A positive \u0394 means the re-trained run is <b>ahead</b> of the published run "
      + "after the same fraction of the same step budget; negative means it is behind.");
    if (withM.length)
      bits.push("Paper-run milestone evals exist for " + withM.map((f) => "<b>" + f.label + "</b> ("
        + f.paper_n_milestones + " intermediate checkpoint" + (f.paper_n_milestones === 1 ? "" : "s") + ")")
        .join(", ") + " — those are true curve-vs-curve comparisons.");
    if (onlyF.length)
      bits.push(onlyF.map((f) => "<b>" + f.label + "</b>").join(", ")
        + (onlyF.length === 1 ? " has" : " have") + " only a final paper checkpoint, so the comparison "
        + "there exists at 100% or not at all.");
    $("#delta-fine").innerHTML = bits.join(" ");
  }

  /* ---------------- coverage matrix ---------------- */
  (function coverage() {
    const t = $("#coverage");
    const thead = t.createTHead().insertRow();
    thead.appendChild(el("th", null, "Family"));
    D.milestones.forEach((p) => thead.appendChild(el("th", null, p + "%")));
    thead.appendChild(el("th", null, "Have"));
    const tb = t.createTBody();
    D.families.forEach((f) => {
      const tr = tb.insertRow();
      const c0 = tr.insertCell();
      const fam = el("span", "fam");
      fam.style.setProperty("--sc", SLOT[f.slot]);
      fam.appendChild(el("span", "dot"));
      fam.appendChild(el("span", null, f.label));
      c0.appendChild(fam);
      f.expected.forEach((e) => {
        const c = tr.insertCell();
        c.className = "num";
        const s = el("span", "mark " + (e.present ? "yes" : "no"), e.present ? "●" : "–");
        s.title = e.present
          ? "step " + int(e.step) + " — present"
          : "step " + int(e.step) + " — " + e.reason;
        c.appendChild(s);
        c.appendChild(el("div", null, ""));
        const sub = el("div");
        sub.style.cssText = "font-family:var(--mono);font-size:10px;color:var(--ink-mute)";
        sub.textContent = int(e.step);
        c.appendChild(sub);
      });
      const last = tr.insertCell();
      last.className = "num";
      last.appendChild(el("span", "pill " + f.status, f.status_text));
    });
  })();

  /* ---------------- big table ---------------- */
  (function big() {
    const t = $("#bigtable");
    const head = t.createTHead().insertRow();
    ["Family / point", "Progress", "Step"].forEach((h) => head.appendChild(el("th", null, h)));
    D.tasks.forEach((x) => head.appendChild(el("th", null, x.label)));
    ["Avg5", "Avg7", "Source dir(s)"].forEach((h) => head.appendChild(el("th", null, h)));
    const tb = t.createTBody();

    const numCell = (tr, v) => {
      const c = tr.insertCell();
      c.className = "num" + (v == null ? " na" : "");
      c.textContent = v == null ? "—" : fmt(v);
    };

    D.families.forEach((f) => {
      const hr = tb.insertRow();
      hr.className = "famhead";
      const c = hr.insertCell();
      c.colSpan = 3 + D.tasks.length + 3;
      const fam = el("span", "fam");
      fam.style.setProperty("--sc", SLOT[f.slot]);
      fam.appendChild(el("span", "dot"));
      fam.appendChild(el("span", null, f.label + " — " + f.run_dir + " · " + int(f.total_steps) + " steps"
        + (f.paper ? "   ·   paper run " + f.paper.ckpt_dir.split("/").pop()
                      + " · " + int(f.paper_total_steps) + " steps" : "")));
      c.appendChild(fam);

      if (!f.milestones.length) {
        const r = tb.insertRow();
        const cc = r.insertCell();
        cc.colSpan = 3 + D.tasks.length + 3;
        cc.className = "na";
        cc.textContent = f.run_dir_exists
          ? "No stage_* directory with a usable results.json."
          : "Run directory does not exist — nothing has been trained or evaluated under this name.";
      }

      f.milestones.forEach((m) => {
        const tr = tb.insertRow();
        tr.insertCell().textContent = m.nominal != null ? m.nominal + "% milestone" : "off-schedule point";
        const pc = tr.insertCell(); pc.className = "num"; pc.textContent = m.pct.toFixed(1) + "%";
        const sc = tr.insertCell(); sc.className = "num"; sc.textContent = int(m.step);
        D.tasks.forEach((x) => numCell(tr, m.scores[x.key]));
        numCell(tr, m.avg5);
        numCell(tr, m.avg7);
        const src = tr.insertCell();
        src.style.cssText = "font-family:var(--mono);font-size:10.5px;color:var(--ink-mute);text-align:left";
        src.textContent = m.dirs.join(" + ");
      });

      (f.paper_curve || []).forEach((c) => {
        const tr = tb.insertRow();
        tr.className = "paper";
        tr.insertCell().textContent = "paper run† — " + (c.kind === "final"
          ? "final checkpoint"
          : (c.nominal != null ? c.nominal + "% milestone" : "off-schedule point"));
        const pc = tr.insertCell(); pc.className = "num"; pc.textContent = c.pct.toFixed(1) + "%";
        const sc = tr.insertCell(); sc.className = "num"; sc.textContent = int(c.step);
        D.tasks.forEach((x) => numCell(tr, c.scores[x.key]));
        numCell(tr, c.avg5);
        numCell(tr, c.avg7);
        const src = tr.insertCell();
        src.style.cssText = "font-family:var(--mono);font-size:10.5px;color:var(--ink-mute);text-align:left";
        src.textContent = c.dir;
      });
    });
  })();

  /* ---------------- notes ---------------- */
  (function notes() {
    const host = $("#notes");
    const add = (html) => { const p = el("p"); p.innerHTML = html; host.appendChild(p); };

    add("<b>Metrics.</b> Defined in full beside the metric selector at the top of the page. In short: "
      + D.tasks.map((t) => t.label + " <code>" + t.metric.replace(/,none$/, "") + "</code>").join(", ")
      + ", every one read at n-shot 0 — any task whose recorded <code>n-shot</code> is not 0 is discarded by the scanner.");

    add("<b>† Paper rows are a different training run</b> of the same model, from the previous submission. "
      + "Where that run kept intermediate checkpoints and evaluated them, its milestones are drawn as a second curve in the "
      + "same hue, dashed with hollow markers, so the re-trained curve can be read against it at the same fraction of the "
      + "step budget. Where only a final checkpoint survives, it stays a dashed horizontal reference.");

    add("<b>Which eval directories are read.</b> Only " + D.paper_sources.map((x) => "<code>" + x.dir + "</code>").join(", ")
      + " for a final checkpoint, and " + D.paper_milestone_sources.map((x) => "<code>" + x.dir + "/&lt;task&gt;</code>").join(", ")
      + " for a paper milestone. The checkpoint directories also hold few-shot and variant reruns "
      + "(<code>evals_5shot</code>, <code>evals_hellaswag_3shot</code>, <code>evals_arc_challenge_5shot</code>, "
      + "<code>evals_oldeval_*</code>, <code>eval_gr90_fg</code>, <code>evals_fg_162k</code>, …) whose numbers look plausible "
      + "but do not reproduce the published table — globbing <code>evals_*</code> puts the BPEByte final at Avg5 61.10 instead "
      + "of the correct 0-shot 60.45. The scanner never globs, and filters on <code>n-shot == 0</code> on top of that.");

    const ul = el("ul");
    D.families.forEach((f) => {
      if (!f.paper) return;
      const chk = f.paper.headline_check;
      const verdict = !chk ? "" :
        (chk.ok ? " — reproduces the published Avg5 of " + chk.expected.toFixed(1)
                : " — <b>does not match</b> the published Avg5 of " + chk.expected.toFixed(1) + " (extracted " + chk.got.toFixed(2) + ")")
        + (chk.note ? " <span style=\"color:var(--warn)\">(" + chk.note + ")</span>" : "");
      const li = el("li");
      li.innerHTML = "<b>" + f.label + " paper checkpoint</b> — Avg5 " + (f.paper.avg5 == null ? "n/a" : f.paper.avg5.toFixed(2))
        + ", Avg7 " + (f.paper.avg7 == null ? "n/a" : f.paper.avg7.toFixed(2)) + verdict
        + "<br>" + f.paper.sources.map((s) => "<code>" + s + "</code>").join(" + ")
        + "<br>" + (f.paper_n_milestones
            ? f.paper_n_milestones + " intermediate paper milestone" + (f.paper_n_milestones === 1 ? "" : "s") + ": "
              + f.paper_curve.filter((c) => c.kind === "milestone")
                  .map((c) => "<code>" + c.dir + "</code> (" + c.pct.toFixed(0) + "%)").join(", ")
            : "no <code>stage_*</code> milestone directory under this checkpoint — final checkpoint only.");
      ul.appendChild(li);
    });
    if (ul.children.length) { add("<b>Where the paper numbers come from.</b>"); host.appendChild(ul); }

    const gaps = el("ul");
    D.families.forEach((f) => {
      const miss = f.expected.filter((e) => !e.present);
      if (!miss.length) return;
      const li = el("li");
      li.innerHTML = "<b>" + f.label + "</b>: missing " + miss.map((m) => m.pct + "%").join(", ")
        + " — " + Array.from(new Set(miss.map((m) => m.reason))).join("; ");
      gaps.appendChild(li);
    });
    if (gaps.children.length) { add("<b>Gaps.</b>"); host.appendChild(gaps); }

    const skipped = [];
    D.families.forEach((f) => f.skipped.forEach((s) => skipped.push(f.label + ": <code>" + s.dir + "</code> — " + s.why)));
    if (skipped.length) {
      const d = el("div", "flag");
      d.innerHTML = "<b>Stage directories skipped (not counted as data points):</b><br>" + skipped.join("<br>");
      host.appendChild(d);
    }
    if (D.warnings.length) {
      const d = el("div", "flag");
      d.innerHTML = "<b>Scanner warnings:</b><br>" + D.warnings.map((w) => w.replace(/</g, "&lt;")).join("<br>");
      host.appendChild(d);
    }

    add("<b>Palette.</b> Four categorical series colours, validated with the data-viz six checks "
      + "(<code>--pairs all</code>) against this page's own chart surface in both themes: all checks pass. "
      + "No hue was added for the paper run: it re-uses its family's hue and is separated only by a dashed "
      + "stroke and hollow markers, because it is the same model, not a fifth series. "
      + "Series identity is carried by a legend, direct end-of-line labels and this table as well as by hue.");
    add("<b>Regenerate</b> with <code>python3 scripts/probes/build_stage_viewer.py</code>. Data is embedded at build time.");
  })();

  draw();
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output HTML path")
    args = ap.parse_args()

    data = collect()
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    # keep the JSON safe inside a <script> element
    blob = blob.replace("</", "<\\/")
    html = HTML.replace("__DATA__", blob)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)

    # console summary
    print("wrote", args.out, "(%.1f KB)" % (len(html) / 1024.0))
    for fam in data["families"]:
        print("\n== %-12s %s  [%s]" % (fam["label"], fam["run_dir"], fam["status_text"]))
        for m in fam["milestones"]:
            print("   %5.1f%%  step %-8d Avg5=%-7s Avg7=%-7s  %s" % (
                m["pct"], m["step"],
                "n/a" if m["avg5"] is None else "%.2f" % m["avg5"],
                "n/a" if m["avg7"] is None else "%.2f" % m["avg7"],
                " + ".join(m["dirs"])))
        for e in fam["expected"]:
            if not e["present"]:
                print("   ---- %3d%% (step %d): %s" % (e["pct"], e["step"], e["reason"]))
        for c in fam.get("paper_curve", []):
            print("   paper %5.1f%% step %-8d Avg5=%-7s Avg7=%-7s  %s%s" % (
                c["pct"], c["step"],
                "n/a" if c["avg5"] is None else "%.2f" % c["avg5"],
                "n/a" if c["avg7"] is None else "%.2f" % c["avg7"],
                c["dir"], "  (final)" if c["kind"] == "final" else ""))
        if fam["paper"]:
            p = fam["paper"]
            c = p["headline_check"]
            print("   paper final: Avg5=%s Avg7=%s%s" % (
                "n/a" if p["avg5"] is None else "%.2f" % p["avg5"],
                "n/a" if p["avg7"] is None else "%.2f" % p["avg7"],
                "" if not c else ("  [%s vs published %.1f%s]" % (
                    "MATCH" if c["ok"] else "MISMATCH", c["expected"],
                    "; " + c["note"] if c.get("note") else ""))))
        for r in fam.get("paper_shared", []):
            print("   delta %3d%% (re-trained %d vs paper %d): Avg5 %s  Avg7 %s" % (
                r["pct"], r["run_step"], r["paper_step"],
                "n/a" if r["delta"]["avg5"] is None else "%+.2f" % r["delta"]["avg5"],
                "n/a" if r["delta"]["avg7"] is None else "%+.2f" % r["delta"]["avg7"]))
    if data["warnings"]:
        print("\nwarnings:")
        for w in data["warnings"]:
            print("  -", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

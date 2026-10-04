#!/usr/bin/env python3
"""논문 robustness 표 생성 — 통일 5태스크 프로토콜 (HellaSwag/ARC-E/ARC-C/PIQA/BoolQ, limit 2000).

두 표를 TeX 로 직접 쓴다 (손 전사 금지):
  table_appendix/robustness_detail.tex : 태스크별 clean→perturbed (Δ), 축 평균 = tab:main_13b 값
  tables/robustness_category.tex       : 축 안의 카테고리별 |Δ| (5태스크 macro 평균)
      Noise   : 5 전략 (antspeak/drop/randomcase/repeat/uppercase), prompt/completion/both 평균
      Typo    : 4 편집 (delete/swap/key/insert), char/word 평균
      Despace : 공백 100% 제거 영역 (context / answer / both)

지표 규약 (robustness_paper_vs_retrain.py 와 동일):
  noise/typo  : acc_norm (BoolQ 는 acc), despace/pbp : acc
  BoolQ noise 는 prompt 변형만 존재(보기 "yes"/"no" 는 오염시키면 정답이 바뀜).

소스:
  matched  : runs/robustness_paper1p3b{,_ext}/<arm>/results.json (noise/typo/pbp)
             runs/robustness_despace_bits/<arm>/results.json     (despace, per-item bits 재실행)
  BLT/H-Net: reports/ext_ci/{blt,hnet}/<axis>.json (per-item 재실행, seed 1234, BLT 512 window) 우선,
             없으면 reports/robustness_ext/{blt,hnet}_<axis>_{hs_arce,ext}.json 의 raw
             BLT despace HS/ARC-E 는 reports/blt_threshold_ab/rob_despace_thr1.3354.json (threshold 1.335)
             BLT PBP 는 HS/ARC-E 미측정.

  python paper_robustness_tables.py [--overleaf /mnt/ssd2/hyun2/AUNet/paper_overleaf]
"""
import argparse, glob, json, os, statistics as st

L = "/mnt/ssd2/hyun2/AUNet"
# Leet (NL-Augmenter leet_letters) comes from the format_mc runs committed next to this script.
FMT = os.environ.get("FORMAT_ROBUSTNESS_DIR",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "format_robustness"))
FMT_SRC = {"llama": ["leet_extra/llama/results.json"], "aunet": ["raw/aunet/results.json"],
           "bpebyte": ["raw/bpebyte/results.json"], "blt": ["raw/blt_*_B.json"], "hnet": ["raw/hnet.json"]}
TASKS = ("hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq")
TNAME = {"hellaswag": "HellaSwag", "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge",
         "piqa": "PIQA", "boolq": "BoolQ"}
MODELS = ("llama", "aunet", "bpebyte", "blt", "hnet")
# Axes still measured WITHOUT BLT's 512-byte local attention window get NOWIN_MARK. Empty since the
# official-bytelatent rerun (reports/ext_ci/blt_official: entropy + local windows native, batch 1);
# PBP was re-run with the window in reports/robustness_ext/blt_pbp_{hs_arce,ext}.json.
NOWIN = {}
NOWIN_MARK = r"$^{\circ}$"
NOWIN_NOTE = ""
MATCHED = ("llama", "aunet", "bpebyte")
NOISE = ("antspeak", "drop", "randomcase", "repeat", "uppercase")
NOISE_POS = ("prompt", "completion", "both")
TYPO = ("delete", "swap", "key", "insert")
TYPO_LVL = ("char", "word")
DESPACE = (("despace", "Context"), ("despaceans", "Answer"), ("despaceall", "Both"))


def _get(r, metric):
    """lm-eval 행({'acc_norm,none':..}) 과 despace/pbp 행({'acc':..}) 모두에서 metric 을 꺼낸다."""
    if not isinstance(r, dict):
        return None
    for k in (f"{metric},none", metric):
        if r.get(k) is not None:
            return 100 * r[k]
    return None


def _norm(r, task):
    """noise/typo 기준: acc_norm, BoolQ 는 acc."""
    return _get(r, "acc" if task == "boolq" else "acc_norm")


AXES = ("pbp", "noise", "typo", "despace")   # + "leet", loaded separately (format_mc)


def load_results():
    """model -> axis -> flat results 딕셔너리 (키 이름은 lm-eval/despace_mc 규약).

    축별로 나누는 이유: BLT/H-Net 은 축마다 별도 실행이라 clean 기준선(`piqa` 등 같은 키)이 실행마다
    조금씩 다르다. 한 딕셔너리로 합치면 뒤 실행의 clean 이 앞 축의 clean 을 덮어써 Δ 가 어긋난다."""
    R = {}
    for arm in MATCHED:
        res = {}
        for d in ("robustness_paper1p3b", "robustness_paper1p3b_ext"):
            res.update(json.load(open(f"{L}/runs/{d}/{arm}/results.json"))["results"])
        # despace 는 per-item 재실행본으로 교체 (CI 와 같은 실행)
        res = {k: v for k, v in res.items() if not k.startswith("despace_mc")}
        res.update(json.load(open(f"{L}/runs/robustness_despace_bits/{arm}/results.json"))["results"])
        R[arm] = {ax: res for ax in AXES}
    E = f"{L}/reports/robustness_ext"
    for m in ("blt", "hnet"):
        R[m] = {}
        for ax in AXES:
            # Per-item reruns (scripts/probes/ext_ci/run_ext.py: perturbation seed 1234 = the trio's,
            # BLT with its 512-byte entropy window) supersede the Aug-29 runs (seed 0) when present.
            # BLT: official bytelatent (xformers windows, batch 1); noise/typo are per-task shards.
            d = f"{L}/reports/ext_ci/{'blt_official' if m == 'blt' else m}"
            files = [f"{d}/{ax}.json"] if os.path.exists(f"{d}/{ax}.json") else sorted(
                glob.glob(f"{d}/{ax}_*.json"))
            if ax != "pbp" and files:
                R[m][ax] = {}
                for f in files:
                    R[m][ax].update(json.load(open(f))["results"])
                continue
            res = {}
            for part in ("hs_arce", "ext"):
                f = f"{E}/{m}_{ax}_{part}.json"
                if not os.path.exists(f):
                    continue
                j = json.load(open(f))
                if ax == "pbp":   # pbp 는 raw 대신 태스크별 canonical/space 요약만 있다
                    for t in j["tasks"]:
                        res[f"pbp_mc_{t}_canonical"] = {"acc": j[t]["canonical"]}
                        res[f"pbp_mc_{t}_space"] = {"acc": j[t]["space"]}
                else:
                    res.update(j["raw"])
            if m == "blt" and ax == "despace":
                res.update(json.load(open(f"{L}/reports/blt_threshold_ab/rob_despace_thr1.3354.json"))["raw"])
            R[m][ax] = res
    for m in MODELS:          # Leet: clean / nla_leet per-item bits of the format_mc runs
        rows = {}
        for pat in FMT_SRC[m]:
            for f in glob.glob(os.path.join(FMT, pat)):
                rows.update(json.load(open(f))["results"])
        R[m]["leet"] = rows
    return R


def noise_keys(t, strat=None):
    ss = (strat,) if strat else NOISE
    return [f"{t}_noise_{s}_{p}" for s in ss for p in NOISE_POS]


def typo_keys(t, op=None):
    ops = (op,) if op else TYPO
    return [f"{t}_typo_{o}_{l}" for o in ops for l in TYPO_LVL]


def cell(model_res, t, axis, sub=None):
    """(clean, perturbed) in %, 또는 측정이 없으면 None. model_res: load_results()[model]."""
    res = model_res[axis]
    if axis in ("noise", "typo"):
        keys = noise_keys(t, sub) if axis == "noise" else typo_keys(t, sub)
        vals = [_norm(res.get(k), t) for k in keys if k in res]
        c = _norm(res.get(t), t)
        if c is None or not vals:
            return None
        return c, st.mean(vals)
    if axis == "despace":
        c = _get(res.get(f"despace_mc_{t}_clean"), "acc")
        p = _get(res.get(f"despace_mc_{t}_{sub or 'despaceall'}"), "acc")
        return None if c is None or p is None else (c, p)
    if axis == "leet":       # acc_norm, BoolQ acc (as Noise/Typo); from per-item bits
        c, p = res.get(f"fmt_{t}_clean"), res.get(f"fmt_{t}_nla_leet")
        if not c or not p:
            return None
        k = "acc" if t == "boolq" else "acc_norm"
        return 100 * st.mean(c["bits"][k]), 100 * st.mean(p["bits"][k])
    if axis == "pbp":
        c = _get(res.get(f"pbp_mc_{t}_canonical"), "acc")
        p = _get(res.get(f"pbp_mc_{t}_space"), "acc")
        return None if c is None or p is None else (c, p)


def axis_delta(res, axis, sub=None):
    """5태스크 macro 평균 Δ (signed). 한 태스크라도 없으면 None."""
    cs = [cell(res, t, axis, sub) for t in TASKS]
    return None if any(c is None for c in cs) else st.mean(p - c for c, p in cs)


def _bold_set(vals, key, nd):
    """matched 3모델 중 best 인덱스 집합 (key 가 작을수록 좋음). 표시 자릿수(nd)로 반올림한 값이
    같으면 동률로 모두 굵게 한다 — 보이는 숫자가 같은데 하나만 굵으면 독자가 오해한다."""
    cand = [(round(key(v), nd), i) for i, v in enumerate(vals[:3]) if v is not None]
    if not cand:
        return set()
    best = min(c for c, _ in cand)
    return {i for c, i in cand if c == best}


def _signed(x, nd):
    """부호를 $-$/$+$ 로, 숫자는 텍스트로 — \\textbf 가 숫자까지 굵게 하도록. -0.0 은 0.0 으로."""
    v = round(x, nd)
    if v == 0:
        return f"{0:.{nd}f}"
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.{nd}f}"


def detail_tex(R):
    head = " & " + " & ".join([r"\textbf{Transformer}", r"\textbf{AUNet}", r"\textbf{BPEByte}",
                               r"\textbf{BLT}$^{\dagger}$", r"\textbf{H-Net}$^{\ddagger}$"]) + r" \\"
    blocks = [("pbp", r"\emph{Prompt-boundary shift} (acc, canonical $\rightarrow$ shifted, $\Delta$)"),
              ("noise", r"\emph{Character noise} (acc\_norm; BoolQ acc, clean $\rightarrow$ perturbed, $\Delta$)"),
              ("typo", r"\emph{Typos} (acc\_norm; BoolQ acc, clean $\rightarrow$ perturbed, $\Delta$)"),
              ("despace", r"\emph{Despace} (acc, clean $\rightarrow$ all spaces removed, $\Delta$)"),
              ("leet", r"\emph{Leet} (acc\_norm; BoolQ acc, clean $\rightarrow$ leetspeak, $\Delta$)")]
    out = [r"% GENERATED by AUNet/scripts/probes/paper_robustness_tables.py — edit the script, not this file.",
           r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.5pt}",
           r"\fitcolumn[\textwidth]{%", r"\begin{tabular}{lccccc}", r"\toprule", head, r"\midrule"]
    for bi, (ax, title) in enumerate(blocks):
        if bi:
            out.append(r"\midrule")
        out.append(r"\multicolumn{6}{l}{" + title + r"} \\")
        for t in TASKS:
            cs = [cell(R[m], t, ax) for m in MODELS]
            b = _bold_set(cs, lambda c: abs(c[1] - c[0]), 2 if ax == "pbp" else 1)
            row = []
            for i, c in enumerate(cs):
                if c is None:
                    row.append("--"); continue
                nd = 2 if ax == "pbp" else 1
                s = f"{c[1]:.1f} ({_signed(c[1] - c[0], nd)})"
                s = r"\textbf{" + s + "}" if i in b else s
                row.append(s + NOWIN_MARK if ax in NOWIN.get(MODELS[i], ()) else s)
            out.append(r"\quad " + TNAME[t] + " & " + " & ".join(row) + r" \\")
        ds = [axis_delta(R[m], ax) for m in MODELS]
        b = _bold_set(ds, abs, 2 if ax == "pbp" else 1)
        row = []
        for i, d in enumerate(ds):
            s = "--" if d is None else _signed(d, 2 if ax == "pbp" else 1)
            s = r"\textbf{" + s + "}" if i in b and d is not None else s
            row.append(s + NOWIN_MARK if d is not None and ax in NOWIN.get(MODELS[i], ()) else s)
        out.append(r"\quad \emph{Average $\Delta$} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "}",
            r"\caption{Per-task robustness at 1B under the unified five-task protocol (up to $2{,}000$ items per task, "
            r"shared perturbation seed). Each cell gives the perturbed accuracy and, in parentheses, its change from the "
            r"model's own clean baseline; the \emph{Average $\Delta$} rows are the five-task means whose absolute values "
            r"appear in Table~\ref{tab:main_13b}. \textit{Noise} averages $15$ variants ($5$ strategies $\times$ "
            r"prompt/completion/both; BoolQ: $5$ prompt-only variants, since its fixed yes/no options cannot be perturbed "
            r"without changing the label); \textit{Typo} averages $8$ variants ($4$ edits $\times$ character/word); "
            r"\textit{Leet} applies the Leet Transformation of NL-Augmenter~\citep{dhole2023nl} at its default setting to the question. "
            r"Bold marks the smallest degradation among the three matched models; $^{\dagger}$BLT and $^{\ddagger}$H-Net "
            r"are external references (``--'': not measured). " + NOWIN_NOTE + "}",
            r"\label{tab:robustness_detail}", r"\end{table*}", ""]
    return "\n".join(out)


def category_tex(R):
    cols = ([("noise", s) for s in NOISE] + [("typo", o) for o in TYPO] + [("despace", d) for d, _ in DESPACE])
    label = {"antspeak": "Ant", "drop": "Drop", "randomcase": "Case", "repeat": "Rep.", "uppercase": "Upper",
             "delete": "Del.", "swap": "Swap", "key": "Key", "insert": "Ins.",
             "despace": "Ctx", "despaceans": "Ans", "despaceall": "Both"}
    name = {"llama": "Transformer", "aunet": "AUNet", "bpebyte": "BPEByte (ours)",
            "blt": r"BLT$^{\dagger}$", "hnet": r"H-Net$^{\ddagger}$"}
    vals = {m: [axis_delta(R[m], ax, sub) for ax, sub in cols] for m in MODELS}
    best = [_bold_set([vals[m][j] for m in MODELS], abs, 1) for j in range(len(cols))]
    out = [r"% GENERATED by AUNet/scripts/probes/paper_robustness_tables.py — edit the script, not this file.",
           r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l|ccccc|cccc|ccc}", r"\toprule",
           r" & \multicolumn{5}{c|}{\textbf{Noise}} & \multicolumn{4}{c|}{\textbf{Typo}} & \multicolumn{3}{c}{\textbf{Despace}} \\",
           r"\textbf{Model} & " + " & ".join(label[s] for _, s in cols) + r" \\", r"\midrule"]
    for m in MODELS:
        if m == "blt":
            out.append(r"\specialrule{0.01em}{0.3ex}{0.35ex}")
        row = []
        for j, v in enumerate(vals[m]):
            s = "--" if v is None else f"{abs(v):.1f}"
            row.append(r"\textbf{" + s + "}" if MODELS.index(m) in best[j] and v is not None else s)
        out.append(name[m] + " & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Robustness by perturbation category at 1B: $|\Delta\mathrm{Acc}|$ in percentage points, averaged "
            r"over the five-task suite of Table~\ref{tab:main_13b}; lower is better. \textit{Noise}: AntSpeak, character "
            r"drop, random case, character repetition, and uppercasing, each averaged over prompt/completion/both "
            r"targets. \textit{Typo}: character- and word-level deletion, swap, keyboard substitution, and insertion. "
            r"\textit{Despace}: all spaces removed from the context only, the answer options only, or both. "
            r"\textbf{Bold} marks the best matched model; $^{\dagger}$BLT and $^{\ddagger}$H-Net are external references. " + NOWIN_NOTE + "}",
            r"\label{tab:robustness_category}", r"\end{table*}", ""]
    return "\n".join(out), vals, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overleaf", default=f"{L}/paper_overleaf")
    a = ap.parse_args()
    R = load_results()
    open(f"{a.overleaf}/table_appendix/robustness_detail.tex", "w").write(detail_tex(R))
    tex, vals, cols = category_tex(R)
    open(f"{a.overleaf}/tables/robustness_category.tex", "w").write(tex)
    print("axis means |Δ| (main table):")
    for m in MODELS:
        ax = {x: axis_delta(R[m], x) for x in ("pbp", "noise", "typo", "despace", "leet")}
        ok = [abs(v) for v in ax.values() if v is not None]
        print(f"  {m:8s} " + "  ".join(f"{k}={'--' if v is None else f'{abs(v):.2f}'}" for k, v in ax.items())
              + (f"  avg={st.mean(ok):.2f}" if len(ok) == 5 else ""))
    print("category Δ (signed):")
    for m in MODELS:
        print(f"  {m:8s} " + " ".join("  --  " if v is None else f"{v:+6.2f}" for v in vals[m]))


if __name__ == "__main__":
    main()

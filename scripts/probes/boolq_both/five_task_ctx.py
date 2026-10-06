"""Main-table robustness with BoolQ as a fifth task, perturbed on the CONTEXT only (passage + question; the yes/no labels
are left untouched, see label_bias.py for why the both-region BoolQ numbers measure a label preference instead).
BoolQ cells per axis (all limit 2000, perturbation seed 1234, within-run clean baseline):
  Noise   : boolq_noise_<s>_prompt (passage field 0 + question field 1), runs/robustness_paper1p3b_ext, ext_ci/*/noise*
  Typo    : boolq_typoctx_<s> (passage + question), runs/robustness_boolq_ctx (new; the old boolq_typo was passage only)
  Despace : despace_mc_boolq_despace (context; = despaceall since "yes"/"no" have no spaces)
  Leet    : fmt_boolq_nla_leet (format_mc, leet seed 0, the context block; options untouched)
  PBP     : pbp_mc_boolq (as before)
The other four tasks keep the both-region cells of paper_robustness_tables.py. Prints BoolQ deltas and |mean delta|
over 4 tasks (current paper) vs 5 tasks; writes five_task_ctx_summary.json.
"""
import glob, json, os, statistics as st, sys

L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, f"{L}/scripts/probes")
import paper_robustness_tables as P  # noqa: E402

AX = ("pbp", "noise", "typo", "despace", "leet")
TC = f"{L}/runs/robustness_boolq_ctx"


def acc(r):
    return P._get(r, "acc")


def ld(f):
    return json.load(open(f))["results"] if os.path.exists(f) else {}


def typo_ctx(m):
    r = ld(f"{TC}/trio_tc_{m}/results.json") if m in P.MATCHED else ld(
        f"{TC}/snu20/{'blt1335' if m == 'blt' else m}_typoctx_boolq.json")
    ks = [f"boolq_typoctx_{o}_{l}" for o in P.TYPO for l in P.TYPO_LVL]
    if "boolq" not in r or not all(k in r for k in ks):
        return None
    return acc(r["boolq"]), st.mean(acc(r[k]) for k in ks)


def leet_ctx(m):
    r = {}
    for p in P.FMT_SRC[m]:
        for f in glob.glob(os.path.join(P.FMT, p)):
            r.update(ld(f))
    c, p_ = r["fmt_boolq_clean"], r["fmt_boolq_nla_leet"]
    return 100 * st.mean(c["bits"]["acc"]), 100 * st.mean(p_["bits"]["acc"])


def boolq_cells(R, m):
    nr, dr, pr = R[m]["noise"], R[m]["despace"], R[m]["pbp"]
    return {"noise": (acc(nr["boolq"]), st.mean(acc(nr[f"boolq_noise_{s}_prompt"]) for s in P.NOISE)),
            "typo": typo_ctx(m),
            "despace": (acc(dr["despace_mc_boolq_clean"]), acc(dr["despace_mc_boolq_despace"])),
            "leet": leet_ctx(m),
            "pbp": (acc(pr["pbp_mc_boolq_canonical"]), acc(pr["pbp_mc_boolq_space"]))}


def main():
    R = P.load_results()
    rows = {}
    for m in P.MODELS:
        c = boolq_cells(R, m)
        d4 = {x: P.axis_delta(R[m], x) for x in AX}
        bq = {x: None if c[x] is None else c[x][1] - c[x][0] for x in AX}
        d5 = {x: None if bq[x] is None else (4 * d4[x] + bq[x]) / 5 for x in AX}
        rows[m] = dict(clean={x: c[x][0] for x in AX if c[x]}, bq=bq, d4=d4, d5=d5)
    f = lambda v, s="": "  --  " if v is None else f"{v:{s}6.2f}"
    print("BoolQ (context only) delta, signed pts")
    for m, r in rows.items():
        print(f"  {m:8s}", "  ".join(f"{x}={f(r['bq'][x], '+')}" for x in AX))
    print("\n|mean delta|: 4 tasks (paper) -> 5 tasks")
    for m, r in rows.items():
        av4 = st.mean(abs(r["d4"][x]) for x in AX)
        ok = all(r["d5"][x] is not None for x in AX)
        av5 = st.mean(abs(r["d5"][x]) for x in AX) if ok else None
        print(f"  {m:8s}", "  ".join(f"{x}={abs(r['d4'][x]):5.2f}->{f(None if r['d5'][x] is None else abs(r['d5'][x]))}"
                                     for x in AX), f"  Avg={av4:5.2f}->{f(av5)}")
    # BLT theta=1.61: not in paper_robustness_tables.MODELS; four-task |delta| from tab:main_robust (all degradations,
    # PBP 0), BoolQ cells from the t1609 runs.
    E = f"{L}/reports/ext_ci/blt_official"
    nr, dr = ld(f"{E}/t1609_noise_boolq.json"), ld(f"{E}/t1609_despace.json")
    lt = ld(f"{P.FMT}/leet_extra/blt1609_boolq.json")
    tr = ld(f"{TC}/snu20/blt1609_typoctx_boolq.json")
    ks = [f"boolq_typoctx_{o}_{l}" for o in P.TYPO for l in P.TYPO_LVL]
    bq = {"pbp": 0.0,
          "noise": st.mean(acc(nr[f"boolq_noise_{s}_prompt"]) for s in P.NOISE) - acc(nr["boolq"]),
          "typo": st.mean(acc(tr[k]) for k in ks) - acc(tr["boolq"]) if all(k in tr for k in ks) else None,
          "despace": acc(dr["despace_mc_boolq_despace"]) - acc(dr["despace_mc_boolq_clean"]),
          "leet": 100 * (st.mean(lt["fmt_boolq_nla_leet"]["bits"]["acc"]) - st.mean(lt["fmt_boolq_clean"]["bits"]["acc"]))}
    d4 = {"pbp": 0.0, "noise": -16.9, "typo": -15.9, "despace": -8.1, "leet": -23.3}
    d5 = {x: None if bq[x] is None else (4 * d4[x] + bq[x]) / 5 for x in AX}
    rows["blt1609"] = dict(bq=bq, d4=d4, d5=d5)
    print(f"  {'blt1609':8s}", "  ".join(f"{x}={abs(d4[x]):5.2f}->{f(None if d5[x] is None else abs(d5[x]))}" for x in AX),
          "  BoolQ:", " ".join(f"{x}={f(bq[x], '+')}" for x in AX))
    json.dump(rows, open(f"{TC}/five_task_ctx_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

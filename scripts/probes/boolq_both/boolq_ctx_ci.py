"""BoolQ robustness with the perturbation on the CONTEXT only (passage + question; the yes/no labels untouched), for the
appendix table: clean accuracy and per-axis delta (perturbed - clean, variants averaged per item) with a paired item
bootstrap 95% CI (B=2000, seed 0, items resampled with the same indices for every model and for clean/perturbed, as
ci_main_table.py). PBP is point-only, as in the main table. limit 2000, perturbation seed 1234.
  Noise   boolq_noise_<s>_prompt   runs/robustness_paper1p3b_ext/<arm>, reports/ext_ci/{blt_official,hnet}
  Typo    boolq_typoctx_<s>        runs/robustness_boolq_ctx (trio_tc_<arm>, snu20/<blt1335|blt1609|hnet>_typoctx_boolq.json);
                                   BPEByte from the 2026-10-07 rerun, runs/robustness_boolq_rerun/trio_tc_bpebyte
  Despace despace_mc_boolq_despace runs/robustness_despace_bits/<arm>, reports/ext_ci/*/despace.json (bits)
  Leet    fmt_boolq_nla_leet       reports/format_robustness (bits; leet seed 0)
  python boolq_ctx_ci.py -> runs/robustness_boolq_rerun/boolq_ctx_ci.json
(supersedes runs/robustness_boolq_ctx/boolq_ctx_ci.json)"""
import glob, json, os, sys

import numpy as np

L = "/mnt/ssd2/hyun2/AUNet"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/probes
import ci_main_table as C  # noqa: E402
import paper_robustness_tables as P  # noqa: E402

TC = f"{L}/runs/robustness_boolq_ctx"
RR = f"{L}/runs/robustness_boolq_rerun"
E = f"{L}/reports/ext_ci"
MODELS = ("llama", "aunet", "bpebyte", "blt", "blt1609", "hnet")
NOISE = [f"boolq_noise_{s}_prompt" for s in P.NOISE]
TYPO = [f"boolq_typoctx_{o}_{l}" for o in P.TYPO for l in P.TYPO_LVL]


def J(p):
    return json.load(open(p))


def samples_noise(m):
    if m in P.MATCHED:
        return J(f"{L}/runs/robustness_paper1p3b_ext/{m}/results.json")["samples"]
    return J({"blt": f"{E}/blt_official/noise_boolq.json", "blt1609": f"{E}/blt_official/t1609_noise_boolq.json",
              "hnet": f"{E}/hnet/noise.json"}[m])["samples"]


def samples_typo(m):
    if m == "bpebyte":   # 2026-10-07 rerun with lingua main
        return J(f"{RR}/trio_tc_bpebyte/results.json")["samples"]
    if m in P.MATCHED:
        return J(f"{TC}/trio_tc_{m}/results.json")["samples"]
    return J(f"{TC}/snu20/{ {'blt': 'blt1335'}.get(m, m) }_typoctx_boolq.json")["samples"]


def despace_res(m):
    if m in P.MATCHED:
        return J(f"{L}/runs/robustness_despace_bits/{m}/results.json")["results"]
    return J({"blt": f"{E}/blt_official/despace.json", "blt1609": f"{E}/blt_official/t1609_despace.json",
              "hnet": f"{E}/hnet/despace.json"}[m])["results"]


def leet_res(m):
    src = dict(P.FMT_SRC, blt1609=["leet_extra/blt1609_boolq.json"])[m]
    r = {}
    for p in src:
        for f in glob.glob(os.path.join(P.FMT, p)):
            r.update(J(f)["results"])
    return r


def per_item(m):
    """{axis: (clean {id: v}, perturbed-mean {id: v})}"""
    out = {}
    for ax, S, keys in (("noise", samples_noise(m), NOISE), ("typo", samples_typo(m), TYPO)):
        c = C._vec(S["boolq"], "acc")
        vs = [C._vec(S[k], "acc") for k in keys]
        out[ax] = (c, {i: np.mean([v[i] for v in vs]) for i in c if all(i in v for v in vs)})
    D = despace_res(m)
    out["despace"] = (dict(enumerate(D["despace_mc_boolq_clean"]["bits"]["acc"])),
                      dict(enumerate(D["despace_mc_boolq_despace"]["bits"]["acc"])))
    R = leet_res(m)
    out["leet"] = (dict(enumerate(R["fmt_boolq_clean"]["bits"]["acc"])), dict(enumerate(R["fmt_boolq_nla_leet"]["bits"]["acc"])))
    return out


def main():
    data = {m: per_item(m) for m in MODELS}
    rng = np.random.default_rng(0)
    res = {m: {} for m in MODELS}
    for ax in ("noise", "typo", "despace", "leet"):
        ids = sorted(set.intersection(*[set(data[m][ax][0]) & set(data[m][ax][1]) for m in MODELS]))
        arr = {m: (np.array([data[m][ax][0][i] for i in ids], float), np.array([data[m][ax][1][i] for i in ids], float))
               for m in MODELS}

        def stat(idx):
            return {m: 100 * (arr[m][1][idx["b"]].mean() - arr[m][0][idx["b"]].mean()) for m in MODELS}

        point, boots = C.bootstrap(stat, {"b": len(ids)}, 2000, rng)
        for m in MODELS:
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            res[m][ax] = dict(n=len(ids), clean=100 * arr[m][0].mean(), delta=point[m], lo=lo, hi=hi)
    R = P.load_results()
    for m in MODELS:
        if m == "blt1609":
            res[m]["pbp"] = dict(delta=0.0)
            continue
        pr = R[m]["pbp"]
        if "pbp_mc_boolq_space" not in pr:   # the 4-task BLT PBP re-run (pbp4.json) has no BoolQ: use the windowed BoolQ run
            pr = J(f"{L}/reports/robustness_ext/{m}_pbp_ext.json")["raw"]
        res[m]["pbp"] = dict(delta=P._get(pr["pbp_mc_boolq_space"], "acc") - P._get(pr["pbp_mc_boolq_canonical"], "acc"))
    for m in MODELS:
        print(f"{m:8s} clean {res[m]['noise']['clean']:5.1f}  PBP {res[m]['pbp']['delta']:+6.2f}  " + "  ".join(
            f"{ax} {res[m][ax]['delta']:+6.2f} [{res[m][ax]['lo']:+.1f},{res[m][ax]['hi']:+.1f}] (n={res[m][ax]['n']})"
            for ax in ("noise", "typo", "despace", "leet")))
    json.dump(res, open(f"{RR}/boolq_ctx_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()

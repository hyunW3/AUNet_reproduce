#!/usr/bin/env python3
"""Compression stability of computation-boundary parsers under distribution shift.

plans/plan_compression_stability.md. Measures, for each parser and each held-out domain,
how many bytes land in one pooled unit. The claim under test: OnlineBPE is the only parser
whose patch length stays bounded everywhere -- heuristic parsers blow UP when their
delimiter disappears, learned parsers COLLAPSE to ~1 byte/patch off-distribution.

Patch convention follows tab:zh_stats ("shared boundary byte counted once"): boundaries are
patch START indices and patch k spans [start_k, start_{k+1}), so the windows partition
exactly and CR = total_bytes / total_patches.

Arms:
    llama     subword BPE token boundaries (Llama-3 tiktoken)
    aunet     whitespace/word regex pooling      (RegexPool word1)
    bpebyte   OnlineBPE greedy/root trie walk    (RegexPool bpe_br, bpe_online)
    blt       entropy patcher (--entropy_model)  -- see --theta
    (hnet     learned router: separate GPU harness, merged in via --extra)

Usage:
    python scripts/probes/compression_stability.py --gate-reproduce
    python scripts/probes/compression_stability.py --all
    python scripts/probes/compression_stability.py --domain te --parser bpebyte
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/compression_stability"
TOK = str(ROOT / "tokenizer/llama3/tokenizer.model")

sys.path.insert(0, str(ROOT / "lingua"))

# Domain -> window file. "en" is the reference the instability metric is measured against.
DOMAINS = {
    "en":           ROOT / "reports/bpb_windows.jsonl",
    "zh":           ROOT / "reports/bpb_windows_zh.jsonl",
    "te":           ROOT / "reports/windows/te.jsonl",
    "code":         ROOT / "reports/windows/code.jsonl",
    "despace_p025": ROOT / "reports/windows/despace_p025.jsonl",
    "despace_p050": ROOT / "reports/windows/despace_p050.jsonl",
    "despace_p075": ROOT / "reports/windows/despace_p075.jsonl",
    "despace_p100": ROOT / "reports/windows/despace_p100.jsonl",
    "random":       ROOT / "reports/windows/random.jsonl",
}
REFERENCE_DOMAIN = "en"

# tab:zh_stats, bytes/patch. Gate 1: a new measurement must reproduce these before any
# new-domain number is trusted (plan section 9, falsifier F2).
GATE = {
    ("llama",   "en"): 4.6, ("llama",   "zh"): 3.4,
    ("aunet",   "en"): 4.9, ("aunet",   "zh"): 20.3,
    ("bpebyte", "en"): 4.6, ("bpebyte", "zh"): 2.8,
    ("blt",     "en"): 1.2, ("blt",     "zh"): 1.1,
}
GATE_TOL = 0.2


# --------------------------------------------------------------------------- parsers
def build_parser(name: str, **kw):
    """Return f(text) -> sorted list of patch START byte-indices (always starts at 0)."""
    from apps.aunet.data.regex_cutting import RegexPool, RegexArgs

    def mask_starts(pool, text):
        b = list(text.encode())
        m = pool.get_levels_mask(b)
        s = sorted({i for i, v in enumerate(m[:len(b)]) if v >= 1})
        return s if s and s[0] == 0 else [0] + s

    if name == "llama":
        from lingua.tokenizer import TikTokenTokenizer
        lt = TikTokenTokenizer(TOK)

        def llama_starts(text):
            toks = lt.tkt_model.decode_tokens_bytes(lt.tkt_model.encode_ordinary(text))
            starts, off = [], 0
            for t in toks:
                starts.append(off)
                off += len(t)
            return starts if starts and starts[0] == 0 else [0] + starts
        return llama_starts

    if name == "aunet":
        pool = RegexPool(RegexArgs(strategy={"word1": "1@1"}, bpe_tokenizer_path=TOK))
        return lambda t: mask_starts(pool, t)

    if name == "bpebyte":
        pool = RegexPool(RegexArgs(
            strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=TOK,
            bpe_online=True, bpe_online_mode="greedy", bpe_online_placement="root",
            bpe_augment_tokenizer_path=kw.get("augment_tokenizer"),
            bpe_augment_unicode_range=kw.get("augment_range")))
        return lambda t: mask_starts(pool, t)

    if name == "blt":
        if not kw.get("entropy_model"):
            raise SystemExit("blt arm needs --entropy_model <consolidated dir>")
        set_entropy_device(kw.get("device"))
        pool = RegexPool(RegexArgs(
            strategy={"bpe_br": "1@1"}, bpe_tokenizer_path=TOK,
            bpe_online=True, bpe_online_mode="entropy", bpe_online_placement="root",
            entropy_model_path=kw["entropy_model"], entropy_rule=kw.get("rule", "monotonic"),
            entropy_theta=float(kw.get("theta", 0.0))))
        return lambda t: mask_starts(pool, t)

    raise SystemExit(f"unknown parser: {name}")


def set_entropy_device(device: str | None) -> str:
    """EntropyModel defaults to CPU (it is fork-safe for the dataloader). This probe is
    single-process, so opt into the GPU explicitly -- on CPU the sweep is ~100x slower."""
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    os.environ["ENTROPY_DEVICE"] = device
    print(f"  entropy model device: {device}")
    return device


def calibrate_on_english(entropy_model: str, rule: str, target_bps: float,
                         n_windows: int = 40) -> float:
    """Fit theta on the ENGLISH windows only, then hold it fixed across every domain.

    This is the whole point of the BLT arm: per-domain retuning is exactly the per-domain
    heuristic that learned/entropy patchers are supposed to avoid, so theta is calibrated
    once, in-distribution, and carried unchanged to zh/te/code/despace/random.
    """
    from apps.aunet.data.entropy_patch import EntropyPatcher, calibrate_theta
    patcher = EntropyPatcher(entropy_model, rule=rule, theta=0.0)
    wins = [json.loads(l) for l in open(DOMAINS[REFERENCE_DOMAIN])][:n_windows]
    ents = [patcher.entropies(list(w["text"].encode())) for w in wins]
    theta = calibrate_theta(ents, rule, target_bps)
    print(f"  calibrated theta={theta:.6f}  (rule={rule}, target={target_bps} B/patch, "
          f"{len(wins)} English windows)")
    return theta


# --------------------------------------------------------------------------- measure
def patch_lengths(starts: list[int], n_bytes: int) -> list[int]:
    """[start_k, start_{k+1}) partition; last patch runs to the window end."""
    ends = starts[1:] + [n_bytes]
    return [e - s for s, e in zip(starts, ends)]


def check_partition(lens: list[int], n_bytes: int, ctx: str) -> None:
    """Gate 3 (hygiene): patches must tile the window exactly, all lengths >= 1."""
    if sum(lens) != n_bytes:
        raise AssertionError(f"{ctx}: patch lengths sum to {sum(lens)} != {n_bytes} bytes")
    if any(x < 1 for x in lens):
        raise AssertionError(f"{ctx}: non-positive patch length ({min(lens)})")


def measure(parse, domain: str, windows_path: Path, limit: int | None = None) -> dict:
    wins = [json.loads(l) for l in open(windows_path)]
    if limit:
        wins = wins[:limit]
    all_lens: list[int] = []
    total_bytes = 0
    for w in wins:
        text, nb = w["text"], w["n_bytes"]
        lens = patch_lengths(parse(text), nb)
        check_partition(lens, nb, f"{domain}[{w['idx']}]")
        all_lens.extend(lens)
        total_bytes += nb
    n_patches = len(all_lens)
    return {
        "domain": domain, "n_windows": len(wins), "total_bytes": total_bytes,
        "n_patches": n_patches,
        "cr": total_bytes / n_patches,
        "mean": st.fmean(all_lens),
        "std": st.pstdev(all_lens) if n_patches > 1 else 0.0,
        "max": max(all_lens),
        "p99": sorted(all_lens)[min(n_patches - 1, int(0.99 * n_patches))],
        "frac_len1": sum(1 for x in all_lens if x == 1) / n_patches,
        "frac_bytes_in_long": sum(x for x in all_lens if x > 16) / total_bytes,
    }


# --------------------------------------------------------------------------- reporting
def instability(rows: list[dict]) -> None:
    """I_d = |log2(CR_d / CR_en)| -- penalises blow-up and collapse equally."""
    by_parser: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_parser.setdefault(r["parser"], {})[r["domain"]] = r
    for parser, doms in by_parser.items():
        ref = doms.get(REFERENCE_DOMAIN)
        for r in doms.values():
            r["instability"] = abs(math.log2(r["cr"] / ref["cr"])) if ref else None


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'parser':<9} {'domain':<13} {'B/patch':>8} {'std':>7} {'max':>5} "
           f"{'p99':>5} {'%len1':>7} {'%long':>7} {'I':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        inst = r.get("instability")
        print(f"{r['parser']:<9} {r['domain']:<13} {r['cr']:>8.2f} {r['std']:>7.2f} "
              f"{r['max']:>5} {r['p99']:>5} {100*r['frac_len1']:>6.1f}% "
              f"{100*r['frac_bytes_in_long']:>6.1f}% "
              f"{'--' if inst is None else f'{inst:>6.2f}'}")


def run_gate(rows: list[dict]) -> bool:
    print("\nGate 1 -- reproduce tab:zh_stats (tolerance +-%.1f B/patch)" % GATE_TOL)
    print(f"{'parser':<9} {'domain':<8} {'measured':>9} {'paper':>7} {'delta':>7}  verdict")
    print("-" * 52)
    ok = True
    for r in rows:
        key = (r["parser"], r["domain"])
        if key not in GATE:
            continue
        exp = GATE[key]
        d = r["cr"] - exp
        good = abs(d) <= GATE_TOL
        ok &= good
        print(f"{r['parser']:<9} {r['domain']:<8} {r['cr']:>9.2f} {exp:>7.1f} "
              f"{d:>+7.2f}  {'PASS' if good else 'FAIL'}")
    print("\n" + ("Gate 1 PASSED." if ok else
                  "Gate 1 FAILED -- do not trust new-domain numbers (falsifier F2)."))
    return ok


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", action="append", default=None,
                    choices=["llama", "aunet", "bpebyte", "blt"])
    ap.add_argument("--domain", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gate-reproduce", action="store_true",
                    help="measure en+zh only and compare against tab:zh_stats")
    ap.add_argument("--limit", type=int, default=None, help="windows per domain (debug)")
    ap.add_argument("--entropy_model", default=None, help="BLT arm: consolidated dir")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="BLT arm: explicit threshold (prefer --calibrate_bps)")
    ap.add_argument("--calibrate_bps", type=float, default=None,
                    help="BLT arm: fit theta on ENGLISH windows to this bytes/patch, then hold "
                         "it fixed across all domains (no per-domain retuning)")
    ap.add_argument("--rule", default="monotonic")
    ap.add_argument("--device", default=None, help="BLT arm: cuda|cpu (default: cuda if available)")
    ap.add_argument("--augment_tokenizer", default=None,
                    help="bpebyte arm: vocab to graft from (e.g. tokenizer/bloom)")
    ap.add_argument("--augment_range", default=None, help="e.g. 0C00-0C7F")
    ap.add_argument("--tag", default=None, help="label suffix, e.g. blt_theta2.85")
    ap.add_argument("--out", default=str(OUT / "measurements.jsonl"))
    args = ap.parse_args()

    if args.gate_reproduce:
        parsers = args.parser or ["llama", "aunet", "bpebyte"]
        domains = ["en", "zh"]
    else:
        parsers = args.parser or ["llama", "aunet", "bpebyte"]
        domains = args.domain or (list(DOMAINS) if args.all else ["en"])

    theta = args.theta
    if args.calibrate_bps is not None:
        set_entropy_device(args.device)
        theta = calibrate_on_english(args.entropy_model, args.rule, args.calibrate_bps)

    rows = []
    for pname in parsers:
        parse = build_parser(pname, entropy_model=args.entropy_model, theta=theta,
                             rule=args.rule, device=args.device,
                             augment_tokenizer=args.augment_tokenizer,
                             augment_range=args.augment_range)
        label = f"{pname}{'_' + args.tag if args.tag else ''}"
        for d in domains:
            path = DOMAINS[d]
            if not path.exists():
                print(f"  ! no windows for {d} ({path}) -- run build_windows.py", file=sys.stderr)
                continue
            r = measure(parse, d, path, args.limit)
            r["parser"] = label
            r["parser_base"] = pname
            if pname == "blt":
                r["theta"] = theta
                r["rule"] = args.rule
                r["calibrated_bps"] = args.calibrate_bps
            rows.append(r)
            print(f"  measured {label:<16} {d:<13} {r['cr']:.2f} B/patch", flush=True)

    instability(rows)
    print()
    print_table(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nappended {len(rows)} rows -> {args.out}")

    if args.gate_reproduce and not run_gate(rows):
        sys.exit(1)


if __name__ == "__main__":
    main()

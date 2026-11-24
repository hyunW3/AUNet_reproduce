#!/usr/bin/env python3
"""Build reports/patch_lag/results_table.md from the eval summaries and the offset profiles.

Generated rather than hand-written, because the hand-written version carried only the TOTAL BPB
column -- and reading the totals alone is exactly what led to the wrong conclusion that trunk
depth beats the scratchpad at patch lag.  The offset split is the point of the table: offset 0 is
the byte a patch-lag mechanism cannot reach, so a method's patch-lag effect is the offset>=1
column, never the total.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV, OFF = ROOT / "reports/patch_lag/sp_eval", ROOT / "reports/patch_lag/offset"
LN2 = math.log(2.0)

BLOCKS = [
    ("A. ece 블록", "`dclm_baseline_1.0_2shards_shuffled`, 3 GPU, 3.15 GB",
     "ece_base", "ece_base", [
        ("l3o_local — 경량 head (attn, 1층)",        "0.9 M head",  "ece_l3o_local",  "ece_l3o_local"),
        ("l3o_big — 경량 head (2층, hidden 4096)",   "~7 M head",   "ece_l3o_big",    "ece_l3o_big"),
        ("scratchpad1frac — scratch 위치 j+0.5",     "**0**",       "ece_scratchpad1frac", "ece_scratchpad1frac"),
        ("**scratchpad1** — scratch 위치 j+1",       "**0**",       "ece_scratchpad1", "ece_scratchpad1"),
        ("l3o_50m_start — 같은 head를 offset 0 에",  "50.7 M head", "ece_l3o_50m_start", "ece_l3o_50m_start"),
        ("**base_150m** — head 없이 trunk 10→17층",  "49.6 M trunk","ece_base_150m",  "ece_base_150m"),
        ("**scratchpad1_150m** — scratchpad + 깊이",  "49.6 M trunk","ece_scratchpad1_150m", "ece_scratchpad1_150m"),
        ("l3o_50m — 경량 head (4층, hidden 3840)",   "50.7 M head", "ece_l3o_50m",    "ece_l3o_50m"),
        ("llama (서브워드, 바이트매칭 3.07 GB)",       "—",           "ece_llama",      None),
     ]),
    ("B. snu20 블록", "`dclm_r10lr_2chunk`, 4 GPU, 3.15 GB (데이터가 달라 A와 합칠 수 없다)",
     "snu20_base", "s20_base", [
        ("l3o_local",                                 "0.9 M head",  "snu20_l3o_local", None),
        ("l3o_50m_gread — 현재 패치 바이트 없음",       "49.3 M head", "s20M_l3o_50m_gread", "s20_l3o_50m_gread"),
        ("l3o_50m_attn — bytes만",                    "50.7 M head", "s20M_l3o_50m_attn",  "s20_l3o_50m_attn"),
        ("l3o_50m_nooff — bytes + g_read",            "50.7 M head", "s20M_l3o_50m_nooff", "s20_l3o_50m_nooff"),
        ("l3o_50m — bytes + g_read + offset",         "50.7 M head", "s20M_l3o_50m",       "s20_l3o_50m"),
        ("**base_150m** — head 없이 trunk 10→17층",    "49.6 M trunk","s20M_base_150m",     "s20_base_150m"),
     ]),
]
HDR = ("| arm | 추가 params | BPB | Δ | Δ@offset 0 | Δ@offset≥1 | HellaSwag<br><sub>acc_norm</sub> | "
       "ARC-E<br><sub>acc_norm</sub> | BoolQ<br><sub>acc</sub> | avg3 |\n"
       "|---|---|---|---|---|---|---|---|---|---|")


def ds(tag):
    f = EV / f"{tag}_summary.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    return (d["hellaswag"]["acc_norm"] * 100, d["arc_easy"]["acc_norm"] * 100,
            d["boolq"]["acc"] * 100, d["avg3"] * 100, d["bpb"])


def split(tag, base_tag):
    """-> (delta at offset 0, delta at offset>=1) in BPB, or (None, None)."""
    if tag is None or not (OFF / f"{tag}.npz").exists():
        return None, None
    b, x = np.load(OFF / f"{base_tag}.npz"), np.load(OFF / f"{tag}.npz")
    m0 = b["offset"] == 0
    return tuple((x["nll"][m].mean() - b["nll"][m].mean()) / LN2 for m in (m0, ~m0))


out = ["# Scratchpad / Updater 프리트레이닝 — BPB + downstream + offset 분해", "",
       "BPEByte-rg 100M · 2,000 steps · held-out 320 windows / 1,228,796 bytes.",
       "`avg3` = HellaSwag(acc_norm) · ARC-Easy(acc_norm) · BoolQ(acc) 평균.", "",
       "**Δ@offset 0 / Δ@offset≥1** — offset 0은 패치의 첫 바이트로, 거기서 읽는 글로벌 벡터는",
       "이미 신선하다. 따라서 patch-lag 기법이 구조적으로 손댈 수 없는 구간이고, 어떤 방법의",
       "**patch-lag 효과는 offset≥1 열이지 총합이 아니다.** 총합만 보면 오독한다 (부록 I).", ""]
for title, note, ev_base, off_base, arms in BLOCKS:
    bd = ds(ev_base)
    out += [f"## {title} — {note}", "", HDR,
            f"| 원본 (base) |  | {bd[4]:.4f} | — | — | — | "
            f"{bd[0]:.2f} | {bd[1]:.2f} | {bd[2]:.2f} | {bd[3]:.2f} |"]
    rows = []
    for name, add, ev_tag, off_tag in arms:
        d = ds(ev_tag)
        if d is None:
            continue
        d0, d1 = split(off_tag, off_base)
        f = lambda v: "—" if v is None else f"{v:+.4f}"
        rows.append((d[4], f"| {name} | {add} | {d[4]:.4f} | {d[4] - bd[4]:+.4f} | {f(d0)} | "
                            f"{f(d1)} | {d[0]:.2f} | {d[1]:.2f} | {d[2]:.2f} | {d[3]:.2f} |"))
    out += [r for _, r in sorted(rows, key=lambda t: -t[0])] + [""]

out += ["## 읽는 법", "",
        "- `scratchpad1`은 offset 0에서 **+0.0048**(효과 없음), offset≥1에서 −0.0162 — 파라미터 0으로",
        "  patch lag만 정확히 고친다. `base_150m`은 offset≥1에서 −0.0165로 **사실상 동률**이고,",
        "  총합에서 앞선 몫은 전부 offset 0(−0.0363)에서 온다. 두 축은 경쟁이 아니라 직교다.",
        "- `l3o_50m_gread`(현재 패치 바이트를 못 보는 같은 크기 head)는 전 구간이 평평하다",
        "  (−0.0034 / −0.0052) — patch-lag 수정이 아니라 일반적 용량 증가의 서명이다.",
        "- **downstream 열은 신뢰할 수 없다.** paired McNemar 검정 결과 HellaSwag(p=0.63~1.00)과",
        "  ARC-Easy(p=0.17~0.78)는 모든 arm에서 null이다. 유의한 것은 BoolQ뿐인데, 그것도 능력이",
        "  아니라 답변 편향이다 — 다수 클래스가 62.2%인 2지선다에서 각 arm이 그것을 찍는 비율이",
        "  26.6%(l3o_50m) ~ 98.1%(l3o_50m_start)로 갈리고, 길이 정규화하면 네 arm 모두",
        "  60.8~62.2로 다수 클래스 베이스라인에 붙는다. avg3의 16 pt 차이는 전부 여기서 나온다.",
        "- 정확도 자체의 노이즈 하한도 크다. 같은 체크포인트를 하드웨어만 바꿔 재채점하면 BPB는",
        "  소수 4자리까지 같지만 정확도는 0.25 pt까지 움직인다 (A100 -> A5000, bf16).",
        "", "생성: `scripts/probes/mk_results_table.py`. offset 표 전체: `offset_table.md`."]
(ROOT / "reports/patch_lag/results_table.md").write_text("\n".join(out) + "\n")
print("\n".join(out))

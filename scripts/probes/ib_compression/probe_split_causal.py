#!/usr/bin/env python3
"""분절 규칙이 인과적인가 — 교란 기반 직접 검사.

왜 prefix 방식을 안 쓰는가: probe_root_causal.py 의 `mask_full` 은
`lm[min(e - off, n - 1)] = 1` 로 마지막 경계를 범위 안으로 클램프한다. 그래서 prefix
`b[:t+1]` 에서는 위치 t 가 거의 항상 경계로 표시되고, 위치 t 를 비교하는 어떤 지표든
프로덕션 root_greedy 에서도 ~78% 불일치가 난다(원본 프로브의 byte 열이 73.8% 인 이유).
원본이 유효 지표로 보고하는 것은 boundary-leak(0/124 = 0.0%)이다.

여기서는 클램프에 걸리지 않는 정의를 쓴다. **누출 = 나중 바이트가 앞쪽 경계 판정을 바꾼다.**
위치 j 의 바이트를 다른 값으로 바꾸고, j 보다 앞선 위치들의 경계 여부가 달라지는지 본다.
달라지면 그 판정은 j(자신의 미래)를 썼다는 뜻이다.

  base   root_greedy 경계만 (프로덕션)      -> 0 이어야 한다 (알려진 성질)
  mid    각 패치 중점에 경계 추가 (P2)       -> 중점은 (start_k + start_{k+1})//2 라
                                              start_{k+1} > mid 를 알아야 정해진다
  off<c> 패치 시작 + c 에 경계 추가          -> 위치가 직전 경계만으로 정해진다
"""
import os, sys, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lingua"))
os.chdir(ROOT / "lingua")

from lingua.tokenizer import TikTokenTokenizer                        # noqa: E402
from apps.aunet.data.byte_trie import build_byte_trie_from_tokenizer  # noqa: E402

tok = TikTokenTokenizer(str(ROOT / "tokenizer/llama3/tokenizer.model"))
trie = build_byte_trie_from_tokenizer(tok)


def mask_full(byte, off=0):
    """probe_root_causal.mask_full 그대로."""
    n = len(byte)
    if n == 0:
        return []
    lm = [0] * n
    for e in trie.boundaries(bytes(byte), mode=None):
        if 0 < e <= n:
            lm[min(e - off, n - 1)] = 1
    return lm


def add_mid(lm):
    n = len(lm)
    bs = [i for i, v in enumerate(lm) if v]
    out = list(lm)
    for k, s in enumerate(bs):
        e = bs[k + 1] if k + 1 < len(bs) else n
        out[(s + min(e, n - 1)) // 2] = 1
    return out


def add_off(lm, c):
    n = len(lm)
    out = list(lm)
    for i, v in enumerate(lm):
        if v and i + c < n:
            out[i + c] = 1
    return out


def probe(byte, tf, sites, rng):
    """-> (검사한 (위치,교란) 쌍 수, 앞쪽 경계가 바뀐 쌍 수, 바뀐 위치 총수)"""
    n = len(byte)
    base = tf(mask_full(byte))
    pairs = bad = moved = 0
    for j in sites:
        alt = list(byte)
        while True:
            v = rng.randrange(32, 127)
            if v != alt[j]:
                break
        alt[j] = v
        m = tf(mask_full(alt))
        # j 보다 **앞선** 위치(= j 의 과거)에서 경계 판정이 달라졌는가.
        # 마지막 위치 n-1 은 클램프 대상이라 제외한다.
        d = sum(1 for t in range(min(j, n - 1)) if m[t] != base[t])
        pairs += 1
        if d:
            bad += 1
            moved += d
    return pairs, bad, moved


def main():
    rng = random.Random(0)
    wins = [json.loads(l) for l in open(ROOT / "reports/bpb_windows.jsonl")]
    chunks = [w["text"].encode()[:1200] for w in wins[:60]]
    cases = [("base (프로덕션)", lambda m: m),
             ("mid — P2 로 돌린 것", add_mid),
             ("off2", lambda m: add_off(m, 2)),
             ("off3", lambda m: add_off(m, 3))]
    print(f"held-out DCLM {len(chunks)} 조각 x 1200B, 조각마다 교란 40회\n")
    print("| 분절 규칙 | 앞쪽 경계가 바뀐 교란 | 비율 | 바뀐 위치/교란 |")
    print("|---|---:|---:|---:|")
    for name, tf in cases:
        P = B = M = 0
        for c in chunks:
            sites = sorted(rng.sample(range(20, len(c) - 1), 40))
            p, b, m = probe(c, tf, sites, rng)
            P += p; B += b; M += m
        print(f"| {name} | {B:,} / {P:,} | **{100*B/P:.2f}%** | {M/P:.2f} |")


if __name__ == "__main__":
    main()

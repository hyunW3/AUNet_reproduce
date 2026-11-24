#!/usr/bin/env python3
"""Model-free ceiling for using the OnlineBPE trie as a speculative-decoding drafter.

Question (reports_afterAAAIsub/next_step.md §6.4): if, at generation time, we let the byte trie
propose the next bytes instead of running the local decoder, how many bytes per patch can it
propose and how often is it right?

This measures the DRAFTER alone -- no model, no GPU. Two policies:
  unique   : propose a byte only when the current trie node has exactly ONE child
             (the vocabulary forces it). Zero extra data.
  top1     : always propose the highest-scoring child, scored by
             `--score {minid,count}`:
               minid -> smallest token id in the child's subtree (earlier BPE merge = more
                        frequent). Pure tokenizer prior, no corpus needed.
               count -> corpus edge counts fitted on the FIT half of the windows.

Accuracy here is agreement with the REAL held-out text. The model's argmax agrees with real text
only ~1-BPB-worth of the time, so this is an upper bound on trie-vs-model agreement -- but the
acceptance test in speculative decoding compares the draft to the MODEL, and a well-trained model's
argmax is exactly what tracks real text. Treat this as the ceiling and the go/no-go screen.

Usage:
  python3 scripts/drafter/trie_draft_stats.py --windows reports/bpb_windows.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/mnt/ssd2/hyun2/AUNet/lingua")


def build_trie_with_ids(tokenizer_path):
    """ByteTrie plus, per node, the smallest token id whose byte string passes through it."""
    from apps.aunet.data.byte_trie import ByteTrie
    import base64

    trie = ByteTrie()
    min_id = {}  # id(node) -> smallest token id in its subtree

    def insert(bs, tid):
        node = trie.root
        cur = min_id.get(id(node))
        if cur is None or tid < cur:
            min_id[id(node)] = tid
        for b in bs:
            b = int(b)
            if b not in node.children:
                node.children[b] = ByteTrie.Node(depth=node.depth + 1)
            node = node.children[b]
            cur = min_id.get(id(node))
            if cur is None or tid < cur:
                min_id[id(node)] = tid
        node.is_token = True

    n = 0
    with open(tokenizer_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                bs = base64.b64decode(parts[0])
                tid = int(parts[1])
            except Exception:
                continue
            if bs:
                insert(bs, tid)
                n += 1
    for b in range(256):  # totality, as build_byte_trie_from_tokenizer does
        insert(bytes([b]), 10**9 + b)
    return trie, min_id, n


def fit_edge_counts(trie, texts):
    """Corpus counts for each (node, next-byte) edge, walked the way the parser walks."""
    counts = defaultdict(lambda: defaultdict(int))
    for t in texts:
        bs = t.encode()
        n = len(bs)
        pos = 0
        while pos < n:
            node = trie.root
            i = pos
            while i < n:
                b = bs[i]
                child = node.children.get(b)
                if child is None:
                    break
                counts[id(node)][b] += 1
                node = child
                i += 1
            pos = max(pos + 1, i)
    return counts


def propose(node, policy, min_id, counts):
    """The byte the drafter would propose at `node`, or None if it declines."""
    ch = node.children
    if not ch:
        return None
    if policy == "unique":
        return next(iter(ch)) if len(ch) == 1 else None
    if counts is not None:
        c = counts.get(id(node))
        if c:
            return max(ch, key=lambda k: (c.get(k, 0), -min_id.get(id(ch[k]), 10**9)))
    return min(ch, key=lambda k: min_id.get(id(ch[k]), 10**9))


def analyze(trie, min_id, counts, texts, policy, max_draft):
    """For every patch: the model emits byte 0, then the drafter proposes byte 1, 2, ... until it
    declines or is wrong. Records that accepted run length, plus per-position accuracy."""
    stats = {"bytes": 0, "patches": 0, "positions": 0, "proposed": 0, "correct": 0,
             "unique_child": 0, "runs": [],
             # per in-patch offset: [n_positions, n_correct, n_unique_child]
             "by_offset": defaultdict(lambda: [0, 0, 0]),
             # mean accepted run when the model is allowed to emit the first d bytes itself
             "runs_from": defaultdict(list)}
    for t in texts:
        bs = t.encode()
        n = len(bs)
        stats["bytes"] += n
        # patch spans from the SAME causal greedy parse the model was trained with
        ends = [e for e in trie.greedy_tokenize_boundaries(bs) if 0 < e <= n]
        start = 0
        for end in ends:
            stats["patches"] += 1
            node = trie.root
            run, broken = 0, False
            hits = []          # per in-patch offset (1-based): was the proposal correct?
            for j in range(start, end):
                if j > start:  # byte 0 of the patch always comes from the model
                    off = j - start
                    stats["positions"] += 1
                    uniq = len(node.children) == 1
                    stats["unique_child"] += uniq
                    pred = propose(node, policy, min_id, counts)
                    ok = pred is not None and pred == bs[j]
                    if pred is not None:
                        stats["proposed"] += 1
                        stats["correct"] += int(ok)
                    bo = stats["by_offset"][off]
                    bo[0] += 1; bo[1] += int(ok); bo[2] += int(uniq)
                    hits.append(ok)
                    if not broken:
                        if ok and run < max_draft:
                            run += 1
                        else:
                            broken = True
                nxt = node.children.get(bs[j])
                if nxt is None:
                    break
                node = nxt
            stats["runs"].append(run)
            # how far could we draft if the model emitted the first d bytes of the patch itself?
            for d in range(1, 5):
                r = 0
                for ok in hits[d - 1:]:
                    if not ok:
                        break
                    r += 1
                stats["runs_from"][d].append(r)
            start = end
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="/mnt/ssd2/hyun2/AUNet/reports/bpb_windows.jsonl")
    ap.add_argument("--tokenizer", default="/mnt/ssd2/hyun2/AUNet/tokenizer/llama3/tokenizer.model")
    ap.add_argument("--max-draft", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(args.windows, encoding="utf-8") if l.strip()]
    half = len(texts) // 2
    fit, ev = texts[:half], texts[half:]
    trie, min_id, ntok = build_trie_with_ids(args.tokenizer)
    print(f"trie: {ntok} vocab entries · fit windows {len(fit)} · eval windows {len(ev)}")

    counts = fit_edge_counts(trie, fit)
    rows = []
    for policy, cnt, tag in (("unique", None, "unique-child only"),
                             ("top1", None, "top-1 by merge rank (minid)"),
                             ("top1", counts, "top-1 by corpus count")):
        s = analyze(trie, min_id, cnt, ev, policy, args.max_draft)
        bpp = s["bytes"] / s["patches"]
        acc = s["correct"] / max(1, s["proposed"])
        cov = s["proposed"] / max(1, s["positions"])
        runs = s["runs"]
        mean_run = sum(runs) / max(1, len(runs))
        # one model step emits byte 0 of the patch; the drafter then covers `mean_run` more bytes
        # that a single parallel verification confirms -> bytes emitted per sequential model step.
        speedup = 1.0 + mean_run
        rows.append(dict(policy=tag, bytes_per_patch=round(bpp, 3),
                         draft_coverage=round(cov, 4), accuracy=round(acc, 4),
                         unique_child_frac=round(s["unique_child"] / max(1, s["positions"]), 4),
                         mean_accepted=round(mean_run, 3),
                         bytes_per_model_step=round(speedup, 3)))
        print(f"\n{tag}")
        print(f"  bytes/patch                 {bpp:.3f}")
        print(f"  in-patch positions w/ draft {cov*100:.1f}%   (unique-child {s['unique_child']/max(1,s['positions'])*100:.1f}%)")
        print(f"  proposal accuracy           {acc*100:.2f}%")
        print(f"  mean accepted run / patch   {mean_run:.3f} bytes")
        print(f"  bytes per model step        {speedup:.3f}x")
        offs = sorted(s["by_offset"])[:8]
        print("  accuracy by in-patch offset: " + "  ".join(
            f"+{o}:{s['by_offset'][o][1]/max(1,s['by_offset'][o][0])*100:4.1f}%"
            f"(u{s['by_offset'][o][2]/max(1,s['by_offset'][o][0])*100:.0f})" for o in offs))
        print("  mean run if model emits first d bytes: " + "  ".join(
            f"d={d}:{sum(v)/max(1,len(v)):.2f}" for d, v in sorted(s["runs_from"].items())))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

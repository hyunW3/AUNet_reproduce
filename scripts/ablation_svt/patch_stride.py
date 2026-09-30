import sys
p = sys.argv[1]
s = open(p).read()
init_anchor = '''            elif strategy.startswith("char"):
                # per-CHARACTER patches: one boundary at every UTF-8 codepoint (~3 B/patch on CJK).
                # Boundaries are emitted directly in str_offset; no regex needed.
                self.patterns.append((None, None))
'''
init_add = '''            elif strategy.startswith("stride"):
                # FIXED-STRIDE patches (content-agnostic ablation): a boundary every k BYTES,
                # k = float before '@' (e.g. "4.5@1" -> alternating 5/4-byte patches). Emitted
                # directly in str_offset; no regex needed.
                self.patterns.append((None, None))
'''
assert s.count(init_anchor) == 1
s = s.replace(init_anchor, init_anchor + init_add)
off_anchor = '''            if strat.startswith("char"):
                # one patch per character: a boundary at every codepoint end (map_codepoint_to_byte
                # -> last byte of each char). On Chinese (3-byte chars) this is ~3 B/patch.
                offsets.append(list(range(len(text))))
                continue
'''
off_add = '''            if strat.startswith("stride"):
                # fixed byte stride k: patch j ends at byte ceil((j+1)*k)-1. str_offset speaks in
                # CHARACTER offsets, so each byte boundary snaps to the char containing that byte
                # (only moves on multi-byte chars; ASCII is byte-exact). The final char always
                # closes the last (partial) patch, like the bpe strategies.
                k = float(self.strategy[strat].split('@')[0])
                assert k >= 1.0, f"stride must be >= 1 byte, got {k}"
                if not text:
                    offsets.append([])
                    continue
                last_byte = map_codepoint_to_byte(text)       # char idx -> last byte idx
                n_bytes = int(last_byte[-1]) + 1
                ends = np.ceil(np.arange(1, int(n_bytes / k) + 2) * k - 1e-9).astype(np.int64) - 1
                ends = ends[ends < n_bytes]
                chars = np.searchsorted(last_byte, ends, side="left")
                chars = np.unique(np.append(chars, len(text) - 1))
                offsets.append(chars.tolist())
                continue
'''
assert s.count(off_anchor) == 1
s = s.replace(off_anchor, off_anchor + off_add)
assert "import numpy as np" in s
open(p, "w").write(s)
print("patched", p)

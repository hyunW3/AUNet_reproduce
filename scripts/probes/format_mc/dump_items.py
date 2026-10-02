#!/usr/bin/env python3
"""Freeze the 5-task x 2000 MC items to json (run once where lm-eval + lingua are available) and
print every variant of one item per task for eyeballing.

  PYTHONPATH=<lingua> python dump_items.py --items_dir reports/format_robustness/items [--show]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_mc as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--items_dir", required=True)
ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--show", action="store_true")
ap.add_argument("--idx", type=int, default=3)
a = ap.parse_args()
fs = F.formatspread_formats()
for t in F.TASKS:
    items = F.load_items(t, a.limit, a.items_dir)
    print(f"{t}: {len(items)} items, {sum(len(i['choices']) for i in items)} options")
    if a.show:
        ctx = items[a.idx]["context"].rstrip(" ")
        for v in F.all_variants():
            print(f"  [{v}] {F.perturb(t, ctx, v, a.idx, fs)!r}"[:400])

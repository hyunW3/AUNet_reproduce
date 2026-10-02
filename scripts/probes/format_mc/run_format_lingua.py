#!/usr/bin/env python3
"""format_mc on a matched lingua checkpoint (AU-Net / BPEByte) through the stock apps.aunet.eval
entry point, so model loading, boundary pool and batching are exactly those of the paper's
despace_mc robustness runs. The despace_mc sentinel's scorer is swapped for run_format_mc; nothing
in lingua is modified.

  cd <lingua> && PYTHONPATH=. FORMAT_ITEMS=<items_dir> FORMAT_CACHE=<cache.jsonl> \
      python <this> config=<eval yaml with harness.tasks=[despace_mc], limit 2000>
Env: FORMAT_TASKS (comma list, default all five), FORMAT_VARIANTS (comma list, default all).
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format_mc as F  # noqa: E402
import apps.aunet.eval as E  # noqa: E402

logger = logging.getLogger()


def _run(loglikelihood, limit=None, **_):
    tasks = os.environ.get("FORMAT_TASKS", ",".join(F.TASKS)).split(",")
    variants = os.environ.get("FORMAT_VARIANTS")
    return F.run_format_mc(loglikelihood, tasks=tasks, variants=variants.split(",") if variants else None,
                           limit=limit or 2000, items_dir=os.environ["FORMAT_ITEMS"],
                           cache=os.environ.get("FORMAT_CACHE"), chunk=int(os.environ.get("FORMAT_CHUNK", "8192")),
                           log=logger.info)


E.run_despace_mc = _run
if __name__ == "__main__":
    E.main()

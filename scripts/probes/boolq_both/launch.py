"""Run an eval entry point with the BoolQ both-region sentinels installed (boolq_both_tasks.install()).
  launch.py -m apps.aunet.eval config=...        (module, e.g. under torch.distributed.run)
  launch.py path/to/run_ext.py --family ...      (script)
lingua must be importable (PYTHONPATH=<lingua> for the trio; run_ext puts AUNET_LINGUA on sys.path itself)."""
import os, runpy, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
lingua = os.environ.get("AUNET_LINGUA")
if lingua and lingua not in sys.path:
    sys.path.insert(1, lingua)

import boolq_both_tasks  # noqa: E402

boolq_both_tasks.install()
if sys.argv[1] == "-m":
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
else:
    path = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    runpy.run_path(path, run_name="__main__")

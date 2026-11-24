"""run_redirect.py with OUT_DIR = reports/patch_stats/sniah_fit4k (texts.jsonl there = S-NIAH final set (fit-4k) + DCLM gate rows),
so the Leet histograms do not overwrite the condition histograms one directory up."""
import runpy, sys
from pathlib import Path
D = Path("/mnt/ssd2/hyun2/AUNet/scripts/probes/patch_stats")
sys.path.insert(0, str(D)); sys.path.insert(0, str(D.parent))
import patch_common as pc
pc.OUT_DIR = Path("/mnt/ssd2/hyun2/AUNet/reports/patch_stats/sniah_fit4k")
pc.TEXTS = pc.OUT_DIR / "texts.jsonl"
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
runpy.run_path(str(D / script), run_name="__main__")

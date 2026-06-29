#!/bin/bash
# Canonical, idempotent organizer for AUNet/runs/.
#   run dirs  -> 1.3B/ 760M/ small/   (scale buckets, with back-compat symlinks at old paths)
#   infra     -> _infra/              (seg_cache, .cache; back-compat symlinks — path-referenced)
#   loose     -> _logs/ _plots/ _reports/ _scripts/ _data/   (by extension; artifacts, no symlink)
# Safety: same-filesystem renames (instant; open FDs follow the inode). A run/infra dir an active
# job is touching is SKIPPED (re-run later to fold it in). Nothing is ever deleted. Re-runnable.
set -u
cd /NHNHOME/WORKSPACE/0226010285_F/MINDlab/hyunw3/AUNet/runs
inuse(){ pgrep -af 'torchrun|apps.aunet|apps.main' | grep -q "/$1[/ ]"; }
mkdir -p 1.3B 760M small _infra _logs _plots _reports _scripts _data

# ---- run dirs by scale (symlink back-compat) ----
declare -A GRP=(
 [bpebyte_br_greedy_root_1.3B]=1.3B [bpebyte_br_greedy_root_1.3B_oldseg_20260624]=1.3B
 [bpebyte_br_bt_1.3B]=1.3B [bpebyte_br_bt_online_1.3B]=1.3B [bpebyte_br_bt_committed_1.3B]=1.3B
 [aunet2_1.3B]=1.3B [llama_1.8B_paper]=1.3B [llama_1B_dm10_not_target]=1.3B
 [bpebyte_root_greedy_760M]=760M [bpebyte_root_greedy_760M_oldcode_4ed607e]=760M
 [bpebyte_root_greedy_760M_oldbatch12_20260622_151758]=760M [aunet2_760M]=760M [llama_760M]=760M
 [ablation_100M]=small [cmp_100M]=small [cmp_300M]=small [cmp_g10]=small [leakfree_100M]=small
 [aunet_100M_entropy_low]=small [entropy_model]=small [bpebyte_screen]=small [warm]=small
 [tokwarm]=small [poc]=small [pcut_rescore]=small [_smoke_online]=small
)
for name in "${!GRP[@]}"; do
  s=${GRP[$name]}
  [ -L "$name" ] && continue
  [ -d "$name" ] || continue
  inuse "$name" && { echo "skip (in use): $name"; continue; }
  mv "$name" "$s/$name" && ln -s "$s/$name" "$name" && echo "run-dir $name -> $s/"
done

# ---- infra dirs (symlink back-compat; path-referenced by REGEX_SEG_CACHE / caches) ----
for name in seg_cache .cache; do
  [ -L "$name" ] && continue
  [ -d "$name" ] || continue
  inuse "$name" && { echo "skip (in use): $name"; continue; }
  mv "$name" "_infra/$name" && ln -s "_infra/$name" "$name" && echo "infra $name -> _infra/"
done

# ---- loose files by extension (artifacts; same-fs mv keeps any open append FDs valid) ----
shopt -s nullglob
for f in *.log *.out;     do [ -f "$f" ] && mv "$f" _logs/    && echo "log    $f"; done
for f in *.png;           do [ -f "$f" ] && mv "$f" _plots/   && echo "plot   $f"; done
for f in *.md;            do [ -f "$f" ] && mv "$f" _reports/ && echo "report $f"; done
for f in *.py *.sh;       do [ -f "$f" ] && mv "$f" _scripts/ && echo "script $f"; done
for f in *.json *.jsonl;  do [ -f "$f" ] && mv "$f" _data/    && echo "data   $f"; done
shopt -u nullglob

echo "ORGANIZE DONE"

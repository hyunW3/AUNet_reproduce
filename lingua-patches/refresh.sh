#!/usr/bin/env bash
# Regenerate the lingua fork patch series from the current bpebyte branch.
#
# Run this whenever lingua/bpebyte gains commits (or after you commit WIP) so
# the snapshot in AUNet stays current. It is safe to run repeatedly.
#
#   ./lingua-patches/refresh.sh            # uses ../lingua and branch bpebyte
#   LINGUA=/path BRANCH=mybranch ./refresh.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGUA="${LINGUA:-$HERE/../lingua}"
BRANCH="${BRANCH:-bpebyte}"

cd "$LINGUA"

# Base = where bpebyte diverged from upstream. Self-corrects after a rebase
# onto newer upstream. Falls back to the recorded base if upstream isn't fetched.
if git rev-parse --verify -q upstream/main >/dev/null; then
  BASE="$(git merge-base upstream/main "$BRANCH")"
else
  BASE=437d680e521873bb5971067148a69587790da853
  echo "warn: no upstream/main ref; using recorded base $BASE" >&2
fi

COUNT="$(git rev-list --count "$BASE..$BRANCH")"
echo "Exporting $COUNT commits ($BASE..$BRANCH) -> $HERE"

# Clear stale patches first so squashes/drops don't leave orphans behind.
rm -f "$HERE"/[0-9][0-9][0-9][0-9]-*.patch

git format-patch --base="$BASE" "$BASE..$BRANCH" -o "$HERE" >/dev/null
echo "Done: $(ls "$HERE"/[0-9][0-9][0-9][0-9]-*.patch | wc -l) patch files."

if [ -n "$(git -C "$LINGUA" status --porcelain "$BRANCH" 2>/dev/null)" ]; then :; fi
WIP="$(git status --porcelain | grep -vE '^\?\? ' | wc -l)"
[ "$WIP" -gt 0 ] && echo "note: $WIP uncommitted change(s) in lingua are NOT captured (commit them first)." >&2
true

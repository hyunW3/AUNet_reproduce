# lingua submodule — network setup runbook

The local conversion is already committed in AUNet_reproduce: `lingua/` is now a
git **submodule** (`.gitmodules` + gitlink), with the inner `origin` repointed
to `git@github.com:hyunW3/lingua.git` and `upstream` still
`facebookresearch/lingua`.

These remaining steps need your GitHub credentials, so run them from your own
terminal (the assistant's sandbox shell has no push access). Run in order — the
outer repo must not be pushed before the submodule commit exists on the remote,
or fresh clones can't fetch it.

## 1. Create the empty fork repo on GitHub

Create `hyunW3/lingua` as an **empty** repo (no README/license/.gitignore), so
the first push isn't rejected for non-fast-forward. Public or private as you
prefer. Web UI, or with `gh` if installed:

```bash
gh repo create hyunW3/lingua --private --description "AU-Net reproduction fork of facebookresearch/lingua"
```

## 2. Push the lingua fork

```bash
cd lingua
# (optional) commit any work-in-progress first so it's included in the pin:
#   git add -A && git commit -m "wip: ..."
git push -u origin bpebyte
git push    origin main          # pristine upstream mirror, for clean pulls
git remote -v                    # origin -> hyunW3/lingua, upstream -> facebookresearch/lingua
```

Optionally set `bpebyte` as the default branch of `hyunW3/lingua` on GitHub.

## 3. Re-pin the gitlink if bpebyte moved, then push the outer repo

The committed gitlink points at `fabb2e2`. If `bpebyte` advanced (new commits /
you committed WIP in step 2), bump the pointer:

```bash
cd ..                            # AUNet_reproduce root
git add lingua                   # re-stage gitlink at current bpebyte HEAD
git status                       # confirm "modified: lingua (new commits)"
git commit -m "lingua: bump submodule pointer"
git push                         # push AUNet_reproduce (branch chore/lingua-patch-export, or merge to main first)
```

## 4. Verify a fresh clone resolves the submodule

```bash
git clone --recurse-submodules git@github.com:hyunW3/AUNet_reproduce.git /tmp/aunet-check
cd /tmp/aunet-check && git submodule status        # should show fabb2e2 (or your bumped sha)
```

---

## Ongoing workflow

**After doing work in lingua** (the submodule is just a normal checkout):

```bash
cd lingua
git add -A && git commit -m "..."
git push origin bpebyte                 # publish to hyunW3/lingua
cd ..
git add lingua && git commit -m "lingua: bump pointer"   # record new pin in AUNet
git push
```

**Pulling Meta upstream updates into the fork** (main stays pristine):

```bash
cd lingua
git fetch upstream
git checkout main && git pull           # fast-forward main = upstream/main
git checkout bpebyte && git rebase main # replay your work onto new upstream
git push --force-with-lease origin bpebyte
```

**Cloning the whole project elsewhere:**

```bash
git clone --recurse-submodules git@github.com:hyunW3/AUNet_reproduce.git
# or, after a plain clone:
git submodule update --init --recursive
```

## Note on `lingua-patches/`

These patch files are now **redundant** — the submodule carries full fork
history on `hyunW3/lingua`. They're kept as a frozen offline backup of the
22-commit series; safe to `git rm -r lingua-patches/` once the fork is pushed
and verified, if you don't want the duplicate.

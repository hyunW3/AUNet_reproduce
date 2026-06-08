# lingua-patches

The full series of local changes made on top of Meta's `lingua` (the
`facebookresearch/lingua` upstream), captured as a `git format-patch` series so
that the vendored `lingua/` checkout can be kept as a **pristine, cleanly
pull-able mirror of upstream** while our reproduction work lives here in the
AUNet repo.

## State this captures

- **Upstream base:** `437d680e521873bb5971067148a69587790da853`
  (`facebookresearch/lingua` `main`, commit "Update README.md")
- **Series tip:** `bpebyte` branch, commit "aunet/generate_bt: drive online bt
  generation through prefill (variable length)"
- **22 commits**, `0001-*.patch` … `0022-*.patch`, generated with
  `git format-patch --base=437d680… 437d680…..bpebyte`.

These are the B200/A100 enablement fixes, the AU-Net 1.3B reproduction configs,
the BPEByte variant + online `bt` generation, the noisy-eval suite, and the
launch/monitor scripts. (Uncommitted working-tree WIP at export time is **not**
included — only committed history.)

## What was done to the lingua checkout

`lingua/main` was reset to the pristine upstream tip and set to track
`upstream/main`, so future Meta updates are a clean fast-forward:

```bash
cd lingua
git fetch upstream
git checkout main && git pull        # clean, no local commits on main
```

`lingua/bpebyte` still carries all 22 commits (a live training job runs on it),
so the working code is unchanged. This patch series is the portable record /
backup of that work, owned by the AUNet repo.

## Reapply onto a fresh / updated upstream checkout

```bash
cd lingua
git checkout -b bpebyte-reapplied upstream/main      # or any base you want
git am /path/to/AUNet/lingua-patches/*.patch
```

To rebase the existing work onto newer upstream instead of reapplying:

```bash
cd lingua
git fetch upstream
git rebase upstream/main bpebyte                      # replay the 22 commits
```

Resolve conflicts as they arise (most changes are additive new files; the
inline edits touch `apps/aunet/eval.py`, `apps/aunet/data/regex_cutting.py`,
`apps/aunet/hierarchical.py`, `apps/{aunet,main}/train.py`,
`lingua/distributed.py`, `lingua/eval_milestones.py`,
`setup/download_prepare_hf_data.py`).

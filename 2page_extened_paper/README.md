# Extended abstract — Tokenization Workshop @ COLM 2026

2-page extended abstract on **`root_greedy`**: leak-free, causal patch boundaries for
hierarchical byte-level language models.

## Files
- `main.tex` — the paper (uses the official COLM 2026 style, `[submission]` = anonymized + line numbers).
- `references.bib` — bibliography.
- `colm2026_conference.sty`, `.bst`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex` — official COLM 2026
  template files (downloaded from `github.com/COLM-org/Template`, branch `2026`). Do **not** modify them.

## Build
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
No TeX engine is installed on the dev box, so the PDF was **not** compiled here — build on a
machine with TeX Live (or Overleaf) before submitting. Target length is **2 pages** excluding
references (workshop limit for extended abstracts); references/appendices are unlimited.

## Submission checklist (workshop)
- [ ] Compiles to ≤ 2 pages (body), references unlimited.
- [ ] Anonymous (the `[submission]` option handles this; verify no author leaks).
- [ ] Single PDF via OpenReview. **Deadline: 2026-06-23, 11:59pm UTC.**

## Source of the numbers
All results are from the 1.3B AU-Net/BPEByte reproduction in this repo:
`BPEByte_root_greedy_method.md`, `model_results_1.3B.md`, `bpb_compare_1.3B_275G.md`,
`evaluation_results.md`.

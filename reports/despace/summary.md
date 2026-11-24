# Graded despace-MC robustness

Each intra-line space removed independently with prob p. `ctxNN` = context only; `ansNN` = answer/choices only; `allNN` = context **and** answer despaced. Macro-avg acc over 5 tasks (500/task). BLT column merged from lzspacebyte/blt eval (legacy variants for the 100% rows, matching the other columns). Source: `reports/statetrack/despace_grid/` (single sweep, p ∈ {0.1, 0.2, 0.4, 0.7, 1.0}; the 100% rows are the legacy `despace`/`despaceans`/`despaceall` variants).

| variant | Llama | AUNet | BPEByte rg | BLT |
|---|---|---|---|---|
| clean | 0.558 | 0.569 | 0.553 | 0.583 |
| ctx10 | 0.547 | 0.549 | 0.543 | 0.561 |
| ctx20 | 0.542 | 0.538 | 0.538 | 0.573 |
| ctx40 | 0.526 | 0.500 | 0.530 | 0.554 |
| ctx70 | 0.522 | 0.462 | 0.508 | 0.548 |
| ctx100 | 0.501 | 0.418 | 0.497 | 0.555 |
| ans10 | 0.487 | 0.494 | 0.490 | 0.514 |
| ans20 | 0.470 | 0.463 | 0.480 | 0.498 |
| ans40 | 0.466 | 0.446 | 0.465 | 0.494 |
| ans70 | 0.460 | 0.431 | 0.456 | 0.512 |
| ans100 | 0.447 | 0.390 | 0.451 | 0.511 |
| all10 | 0.500 | 0.492 | 0.503 | 0.524 |
| all20 | 0.480 | 0.463 | 0.481 | 0.520 |
| all40 | 0.455 | 0.430 | 0.460 | 0.519 |
| all70 | 0.432 | 0.383 | 0.436 | 0.507 |
| all100 | 0.414 | 0.345 | 0.428 | 0.514 |

Answer-side despacing hurts more than context-side at every p (option strings being scored get corrupted); AU-Net collapses under context despace (ctx100 0.418) since whitespace defines its pooling boundaries; rg has the smallest drop in nearly every cell.

## Per-task acc drop at all70 (context+answer, p=70%)

| task | Llama | AUNet | BPEByte rg | BLT |
|---|---|---|---|---|
| hellaswag | 0.46→0.34 | 0.48→0.32 | 0.46→0.37 | 0.50→0.42 |
| arc_easy | 0.66→0.49 | 0.70→0.40 | 0.69→0.50 | 0.69→0.59 |
| arc_challenge | 0.28→0.26 | 0.32→0.17 | 0.30→0.24 | 0.34→0.30 |
| piqa | 0.76→0.55 | 0.73→0.51 | 0.73→0.57 | 0.77→0.66 |
| winogrande | 0.61→0.54 | 0.62→0.51 | 0.60→0.52 | 0.63→0.57 |

## ARC-Easy full-removal endpoint (from the same grid)

| variant | Llama | AUNet | BPEByte rg | BLT |
|---|---|---|---|---|
| clean | 0.658 | 0.696 | 0.686 | 0.686 |
| ctx100 | 0.568 | 0.362 | 0.584 | 0.640 |
| ans100 | 0.502 | 0.450 | 0.540 | 0.574 |
| all100 | 0.432 | 0.274 | 0.486 | 0.590 |

AU-Net's all100 ARC-Easy (0.274) is near 4-choice chance.

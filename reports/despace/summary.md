# Graded despace-MC robustness

Each intra-line space removed independently with prob p. `ctxNN` = context only; `allNN` = context **and** answer despaced. Macro-avg acc over 5 tasks (500/task).

| variant | Llama | AUNet | BPEByte rg |
|---|---|---|---|
| clean | 0.557 | 0.568 | 0.557 |
| ctx10 | 0.548 | 0.551 | 0.545 |
| ctx40 | 0.526 | 0.501 | 0.530 |
| ctx70 | 0.523 | 0.464 | 0.509 |
| all10 | 0.499 | 0.496 | 0.504 |
| all40 | 0.457 | 0.431 | 0.462 |
| all70 | 0.435 | 0.382 | 0.438 |

## Per-task acc drop at all70 (context+answer, p=70%)

| task | Llama | AUNet | BPEByte rg |
|---|---|---|---|
| hellaswag | 0.46→0.34 | 0.48→0.32 | 0.46→0.37 |
| arc_easy | 0.66→0.49 | 0.70→0.40 | 0.69→0.50 |
| arc_challenge | 0.28→0.26 | 0.32→0.17 | 0.30→0.24 |
| piqa | 0.76→0.55 | 0.73→0.51 | 0.73→0.57 |
| winogrande | 0.61→0.54 | 0.62→0.51 | 0.60→0.52 |

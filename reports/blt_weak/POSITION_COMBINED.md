
## Text OR letter (same prompt scored both ways)

`OR` = right if either the text-scored or the letter-scored prediction is right (two guesses: chance 0.4375
for 4 options, 0.75 for PIQA). `logsum` = per option log(P(text)+P(letter)), one prediction (chance 0.25 / 0.5).
Single-format rows repeated for reference.

### options BEFORE the question

| metric | subset | chance | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---|---:|---:|---:|---:|---:|---:|
| text | all 2,000 | ≈0.31 | 0.451 | 0.445 | 0.451 | 0.422 | 0.362 |
| text | 4-option | 0.25 | 0.408 | 0.397 | 0.402 | 0.376 | 0.309 |
| letter | all 2,000 | ≈0.31 | 0.320 | 0.322 | 0.321 | 0.316 | 0.322 |
| letter | 4-option | 0.25 | 0.252 | 0.264 | 0.248 | 0.254 | 0.256 |
| or | all 2,000 | ≈0.49 | 0.555 | 0.574 | 0.559 | 0.523 | 0.449 |
| or | 4-option | 0.4375 | 0.510 | 0.529 | 0.511 | 0.478 | 0.394 |
| logsum | all 2,000 | ≈0.31 | 0.365 | 0.378 | 0.355 | 0.343 | 0.349 |
| logsum | 4-option | 0.25 | 0.310 | 0.327 | 0.291 | 0.282 | 0.290 |

By gold position (4-option items):

| metric | gold | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---|---:|---:|---:|---:|---:|
| or | A | 0.922 | 0.861 | 0.882 | 0.922 | 0.928 |
| or | B | 0.433 | 0.358 | 0.376 | 0.349 | 0.215 |
| or | C | 0.333 | 0.435 | 0.421 | 0.347 | 0.221 |
| or | D | 0.352 | 0.464 | 0.365 | 0.296 | 0.213 |
| or | max − min | 0.589 | 0.503 | 0.517 | 0.626 | 0.714 |
| logsum | A | 0.753 | 0.584 | 0.633 | 0.727 | 0.810 |
| logsum | B | 0.194 | 0.194 | 0.159 | 0.126 | 0.121 |
| logsum | C | 0.136 | 0.221 | 0.184 | 0.168 | 0.104 |
| logsum | D | 0.160 | 0.309 | 0.189 | 0.107 | 0.125 |
| logsum | max − min | 0.617 | 0.391 | 0.474 | 0.620 | 0.706 |

### standard MCQ, options AFTER the question

| metric | subset | chance | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---|---:|---:|---:|---:|---:|---:|
| text | all 2,000 | ≈0.31 | 0.347 | 0.344 | 0.332 | 0.340 | 0.352 |
| text | 4-option | 0.25 | 0.276 | 0.272 | 0.250 | 0.275 | 0.289 |
| letter | all 2,000 | ≈0.31 | 0.317 | 0.318 | 0.309 | 0.317 | 0.326 |
| letter | 4-option | 0.25 | 0.250 | 0.260 | 0.247 | 0.255 | 0.258 |
| or | all 2,000 | ≈0.49 | 0.557 | 0.539 | 0.556 | 0.505 | 0.541 |
| or | 4-option | 0.4375 | 0.461 | 0.452 | 0.458 | 0.420 | 0.471 |
| logsum | all 2,000 | ≈0.31 | 0.322 | 0.318 | 0.309 | 0.319 | 0.326 |
| logsum | 4-option | 0.25 | 0.257 | 0.258 | 0.246 | 0.255 | 0.258 |

By gold position (4-option items):

| metric | gold | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---|---:|---:|---:|---:|---:|
| or | A | 0.429 | 0.375 | 0.552 | 0.898 | 0.694 |
| or | B | 0.548 | 0.309 | 0.546 | 0.188 | 0.320 |
| or | C | 0.221 | 0.232 | 0.352 | 0.259 | 0.397 |
| or | D | 0.645 | 0.888 | 0.381 | 0.336 | 0.472 |
| or | max − min | 0.424 | 0.656 | 0.200 | 0.710 | 0.374 |
| logsum | A | 0.000 | 0.008 | 0.011 | 0.831 | 0.330 |
| logsum | B | 0.454 | 0.164 | 0.446 | 0.065 | 0.196 |
| logsum | C | 0.061 | 0.013 | 0.288 | 0.093 | 0.221 |
| logsum | D | 0.512 | 0.845 | 0.240 | 0.032 | 0.285 |
| logsum | max − min | 0.512 | 0.837 | 0.436 | 0.799 | 0.134 |


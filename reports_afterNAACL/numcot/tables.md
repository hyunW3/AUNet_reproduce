# NUMCoT on the 1B models: tables

Accuracy %. `paper` = NUMCoT scoring (gold anywhere in the response); `strict` = first answer only. † external models (BLT: items whose prompt + budget exceed 4,096 bytes are left-truncated). CIs: item bootstrap B=10,000, seed 0; differences paired by item.


# A. Full protocol (every item, all conditions)

Models: Transformer, AU-Net, BPEByte; 18828 generations.


## Per split, metric = paper


### num2word

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot | 0.2 | 0.2 | 0.0 | 0.0 | 0.2 |
| BPEByte | 0-shot | 0.0 | 0.5 | 0.0 | 0.0 | 0.2 |
| Transformer | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot CoT | 1.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| BPEByte | 0-shot CoT | 0.5 | 0.0 | 0.0 | 0.0 | 0.2 |
| Transformer | few-shot | 4.0 | 2.8 | 0.0 | 0.0 | 2.2 |
| AU-Net | few-shot | 9.2 | 7.2 | 0.0 | 0.0 | 5.5 |
| BPEByte | few-shot | 17.0 | 9.8 | 0.5 | 0.5 | 9.1 |
| Transformer | few-shot CoT | 1.0 | 1.5 | 0.0 | 0.0 | 0.8 |
| AU-Net | few-shot CoT | 1.2 | 2.2 | 0.0 | 0.0 | 1.2 |
| BPEByte | few-shot CoT | 0.2 | 0.5 | 0.0 | 0.0 | 0.2 |

### word2num

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | 0-shot | 0.8 | 3.8 | 0.0 | 0.0 | 1.5 |
| AU-Net | 0-shot | 0.0 | 0.2 | 0.0 | 0.0 | 0.1 |
| BPEByte | 0-shot | 8.8 | 2.8 | 0.0 | 0.0 | 3.8 |
| Transformer | 0-shot CoT | 1.0 | 1.2 | 0.0 | 0.0 | 0.8 |
| AU-Net | 0-shot CoT | 0.2 | 0.8 | 0.0 | 0.0 | 0.3 |
| BPEByte | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | few-shot | 0.2 | 0.0 | 0.0 | 0.0 | 0.1 |
| AU-Net | few-shot | 1.0 | 1.8 | 0.5 | 0.5 | 1.1 |
| BPEByte | few-shot | 3.5 | 4.5 | 1.0 | 0.0 | 2.8 |
| Transformer | few-shot CoT | 2.8 | 2.5 | 0.0 | 0.0 | 1.8 |
| AU-Net | few-shot CoT | 14.5 | 4.0 | 0.5 | 0.0 | 6.2 |
| BPEByte | few-shot CoT | 4.2 | 0.2 | 0.0 | 0.0 | 1.5 |

### units

| model | cond | easy | medium | hard | all |
|---|---|---|---|---|---|
| Transformer | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot | 0.5 | 0.0 | 0.0 | 0.2 |
| BPEByte | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | 0-shot CoT | 0.5 | 0.0 | 0.0 | 0.2 |
| AU-Net | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 |
| BPEByte | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | few-shot | 3.0 | 0.5 | 0.5 | 1.3 |
| AU-Net | few-shot | 10.0 | 0.5 | 0.5 | 3.7 |
| BPEByte | few-shot | 2.5 | 0.5 | 0.5 | 1.2 |
| Transformer | few-shot CoT | 9.5 | 0.5 | 1.0 | 3.7 |
| AU-Net | few-shot CoT | 18.0 | 0.5 | 1.0 | 6.5 |
| BPEByte | few-shot CoT | 16.5 | 0.5 | 1.0 | 6.0 |
| Transformer | few-shot CoT + knowledge | 5.5 | 0.0 | 2.0 | 2.5 |
| AU-Net | few-shot CoT + knowledge | 13.0 | 0.0 | 1.5 | 4.8 |
| BPEByte | few-shot CoT + knowledge | 19.0 | 0.5 | 1.0 | 6.8 |

### mwp

| model | cond | AddSub | SingleOp | MultiArith | all |
|---|---|---|---|---|---|
| Transformer | 0-shot | 3.5 | 10.0 | 4.5 | 6.2 |
| AU-Net | 0-shot | 6.1 | 7.3 | 5.0 | 6.1 |
| BPEByte | 0-shot | 1.0 | 1.2 | 1.2 | 1.2 |
| Transformer | 0-shot CoT | 9.6 | 17.4 | 13.2 | 13.8 |
| AU-Net | 0-shot CoT | 5.3 | 10.1 | 8.5 | 8.3 |
| BPEByte | 0-shot CoT | 2.3 | 5.2 | 4.0 | 4.0 |
| Transformer | few-shot | 3.8 | 12.1 | 5.7 | 7.5 |
| AU-Net | few-shot | 2.3 | 4.8 | 1.8 | 3.0 |
| BPEByte | few-shot | 2.3 | 4.3 | 1.0 | 2.5 |
| Transformer | few-shot CoT | 11.6 | 18.9 | 11.8 | 14.3 |
| AU-Net | few-shot CoT | 8.4 | 11.6 | 10.8 | 10.5 |
| BPEByte | few-shot CoT | 6.1 | 12.8 | 9.8 | 10.0 |

## Per split, metric = strict


### num2word

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot | 0.0 | 0.2 | 0.0 | 0.0 | 0.1 |
| BPEByte | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BPEByte | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | few-shot | 0.8 | 0.2 | 0.0 | 0.0 | 0.3 |
| AU-Net | few-shot | 7.2 | 3.8 | 0.0 | 0.0 | 3.7 |
| BPEByte | few-shot | 16.5 | 7.0 | 0.0 | 0.0 | 7.8 |
| Transformer | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BPEByte | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### word2num

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | 0-shot | 0.8 | 2.2 | 0.0 | 0.0 | 1.0 |
| AU-Net | 0-shot | 0.0 | 0.2 | 0.0 | 0.0 | 0.1 |
| BPEByte | 0-shot | 8.8 | 3.0 | 0.0 | 0.0 | 3.9 |
| Transformer | 0-shot CoT | 1.0 | 0.8 | 0.0 | 0.0 | 0.6 |
| AU-Net | 0-shot CoT | 0.0 | 0.5 | 0.0 | 0.0 | 0.2 |
| BPEByte | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | few-shot | 0.2 | 0.0 | 0.0 | 0.0 | 0.1 |
| AU-Net | few-shot | 1.0 | 1.8 | 0.5 | 0.5 | 1.1 |
| BPEByte | few-shot | 3.5 | 4.5 | 1.0 | 0.0 | 2.8 |
| Transformer | few-shot CoT | 1.2 | 0.5 | 0.0 | 0.0 | 0.6 |
| AU-Net | few-shot CoT | 12.5 | 1.2 | 0.5 | 0.0 | 4.7 |
| BPEByte | few-shot CoT | 4.0 | 0.0 | 0.0 | 0.0 | 1.3 |

### units

| model | cond | easy | medium | hard | all |
|---|---|---|---|---|---|
| Transformer | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot | 0.5 | 0.0 | 0.0 | 0.2 |
| BPEByte | 0-shot | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 |
| BPEByte | 0-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 |
| Transformer | few-shot | 3.0 | 0.5 | 0.5 | 1.3 |
| AU-Net | few-shot | 10.0 | 0.5 | 0.5 | 3.7 |
| BPEByte | few-shot | 2.5 | 0.5 | 0.5 | 1.2 |
| Transformer | few-shot CoT | 2.5 | 0.5 | 0.0 | 1.0 |
| AU-Net | few-shot CoT | 6.5 | 0.5 | 0.0 | 2.3 |
| BPEByte | few-shot CoT | 5.5 | 0.5 | 0.0 | 2.0 |
| Transformer | few-shot CoT + knowledge | 0.5 | 0.0 | 0.0 | 0.2 |
| AU-Net | few-shot CoT + knowledge | 0.5 | 0.0 | 0.0 | 0.2 |
| BPEByte | few-shot CoT + knowledge | 10.5 | 0.5 | 0.0 | 3.7 |

### mwp

| model | cond | AddSub | SingleOp | MultiArith | all |
|---|---|---|---|---|---|
| Transformer | 0-shot | 0.5 | 0.9 | 3.0 | 1.6 |
| AU-Net | 0-shot | 0.3 | 1.6 | 2.7 | 1.7 |
| BPEByte | 0-shot | 0.0 | 0.4 | 0.8 | 0.4 |
| Transformer | 0-shot CoT | 0.8 | 0.9 | 4.0 | 2.1 |
| AU-Net | 0-shot CoT | 0.3 | 1.1 | 1.8 | 1.2 |
| BPEByte | 0-shot CoT | 0.0 | 0.4 | 0.8 | 0.4 |
| Transformer | few-shot | 3.8 | 11.7 | 5.7 | 7.4 |
| AU-Net | few-shot | 2.3 | 5.2 | 1.8 | 3.1 |
| BPEByte | few-shot | 2.3 | 4.4 | 1.0 | 2.6 |
| Transformer | few-shot CoT | 5.3 | 8.9 | 4.7 | 6.4 |
| AU-Net | few-shot CoT | 4.1 | 7.1 | 4.2 | 5.2 |
| BPEByte | few-shot CoT | 1.8 | 8.4 | 2.0 | 4.2 |

## Task totals with 95% CI, metric = paper

| task | cond | n | Transformer | AU-Net | BPEByte |
|---|---|---|---|---|---|
| num2word | 0-shot | 1200 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.4] | 0.2 [0.0, 0.4] |
| num2word | 0-shot CoT | 1200 | 0.0 [0.0, 0.0] | 0.3 [0.1, 0.7] | 0.2 [0.0, 0.4] |
| num2word | few-shot | 1200 | 2.2 [1.4, 3.1] | 5.5 [4.2, 6.8] | 9.1 [7.5, 10.8] |
| num2word | few-shot CoT | 1200 | 0.8 [0.3, 1.4] | 1.2 [0.6, 1.8] | 0.2 [0.0, 0.6] |
| word2num | 0-shot | 1200 | 1.5 [0.8, 2.2] | 0.1 [0.0, 0.2] | 3.8 [2.8, 4.9] |
| word2num | 0-shot CoT | 1200 | 0.8 [0.3, 1.3] | 0.3 [0.1, 0.7] | 0.0 [0.0, 0.0] |
| word2num | few-shot | 1200 | 0.1 [0.0, 0.2] | 1.1 [0.5, 1.7] | 2.8 [1.9, 3.8] |
| word2num | few-shot CoT | 1200 | 1.8 [1.1, 2.6] | 6.2 [4.9, 7.7] | 1.5 [0.8, 2.2] |
| units | 0-shot | 600 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| units | 0-shot CoT | 600 | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| units | few-shot | 600 | 1.3 [0.5, 2.3] | 3.7 [2.3, 5.3] | 1.2 [0.3, 2.2] |
| units | few-shot CoT | 600 | 3.7 [2.2, 5.3] | 6.5 [4.5, 8.5] | 6.0 [4.2, 8.0] |
| units | few-shot CoT + knowledge | 600 | 2.5 [1.3, 3.8] | 4.8 [3.2, 6.7] | 6.8 [4.8, 8.8] |
| mwp | 0-shot | 1557 | 6.2 [5.1, 7.5] | 6.1 [4.9, 7.3] | 1.2 [0.6, 1.7] |
| mwp | 0-shot CoT | 1557 | 13.8 [12.1, 15.5] | 8.3 [6.9, 9.7] | 4.0 [3.0, 4.9] |
| mwp | few-shot | 1557 | 7.5 [6.2, 8.9] | 3.0 [2.2, 3.9] | 2.5 [1.7, 3.3] |
| mwp | few-shot CoT | 1557 | 14.3 [12.6, 16.1] | 10.5 [8.9, 12.0] | 10.0 [8.5, 11.5] |

### Paired differences (pp, 95% CI, bootstrap p), metric = paper

| task | cond | BPEByte − Transformer | BPEByte − AU-Net |
|---|---|---|---|
| num2word | 0-shot | +0.2 [+0.0, +0.4] p=0.264 | +0.0 [-0.3, +0.3] p=1.000 |
| num2word | 0-shot CoT | +0.2 [+0.0, +0.4] p=0.262 | -0.2 [-0.6, +0.2] p=0.534 |
| num2word | few-shot | +6.8 [+5.2, +8.5] p=0.000 | +3.6 [+2.2, +4.9] p=0.000 |
| num2word | few-shot CoT | -0.6 [-1.1, -0.2] p=0.002 | -0.9 [-1.5, -0.4] p=0.000 |
| word2num | 0-shot | +2.3 [+1.2, +3.5] p=0.000 | +3.8 [+2.7, +4.8] p=0.000 |
| word2num | 0-shot CoT | -0.8 [-1.2, -0.3] p=0.000 | -0.3 [-0.7, -0.1] p=0.035 |
| word2num | few-shot | +2.8 [+1.8, +3.8] p=0.000 | +1.8 [+0.7, +2.8] p=0.001 |
| word2num | few-shot CoT | -0.2 [-1.2, +0.7] p=0.647 | -4.8 [-6.2, -3.2] p=0.000 |
| units | 0-shot | +0.0 [+0.0, +0.0] p=1.000 | -0.2 [-0.5, +0.0] p=0.723 |
| units | 0-shot CoT | -0.2 [-0.5, +0.0] p=0.726 | +0.0 [+0.0, +0.0] p=1.000 |
| units | few-shot | -0.2 [-1.0, +0.7] p=0.851 | -2.5 [-3.8, -1.3] p=0.000 |
| units | few-shot CoT | +2.3 [+0.7, +4.2] p=0.011 | -0.5 [-2.3, +1.3] p=0.655 |
| units | few-shot CoT + knowledge | +4.3 [+2.2, +6.5] p=0.000 | +2.0 [-0.2, +4.2] p=0.089 |
| mwp | 0-shot | -5.1 [-6.4, -3.8] p=0.000 | -4.9 [-6.2, -3.9] p=0.000 |
| mwp | 0-shot CoT | -9.8 [-11.6, -8.1] p=0.000 | -4.3 [-5.8, -2.9] p=0.000 |
| mwp | few-shot | -5.0 [-6.4, -3.7] p=0.000 | -0.5 [-1.2, +0.2] p=0.178 |
| mwp | few-shot CoT | -4.4 [-5.9, -2.8] p=0.000 | -0.5 [-1.9, +0.8] p=0.499 |

## Task totals with 95% CI, metric = strict

| task | cond | n | Transformer | AU-Net | BPEByte |
|---|---|---|---|---|---|
| num2word | 0-shot | 1200 | 0.0 [0.0, 0.0] | 0.1 [0.0, 0.2] | 0.0 [0.0, 0.0] |
| num2word | 0-shot CoT | 1200 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| num2word | few-shot | 1200 | 0.3 [0.1, 0.7] | 3.7 [2.7, 4.8] | 7.8 [6.3, 9.4] |
| num2word | few-shot CoT | 1200 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| word2num | 0-shot | 1200 | 1.0 [0.5, 1.6] | 0.1 [0.0, 0.2] | 3.9 [2.9, 5.0] |
| word2num | 0-shot CoT | 1200 | 0.6 [0.2, 1.1] | 0.2 [0.0, 0.4] | 0.0 [0.0, 0.0] |
| word2num | few-shot | 1200 | 0.1 [0.0, 0.2] | 1.1 [0.6, 1.7] | 2.8 [1.9, 3.8] |
| word2num | few-shot CoT | 1200 | 0.6 [0.2, 1.1] | 4.7 [3.5, 5.8] | 1.3 [0.8, 2.0] |
| units | 0-shot | 600 | 0.0 [0.0, 0.0] | 0.2 [0.0, 0.5] | 0.0 [0.0, 0.0] |
| units | 0-shot CoT | 600 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| units | few-shot | 600 | 1.3 [0.5, 2.3] | 3.7 [2.2, 5.2] | 1.2 [0.3, 2.2] |
| units | few-shot CoT | 600 | 1.0 [0.3, 1.8] | 2.3 [1.2, 3.7] | 2.0 [1.0, 3.2] |
| units | few-shot CoT + knowledge | 600 | 0.2 [0.0, 0.5] | 0.2 [0.0, 0.5] | 3.7 [2.2, 5.2] |
| mwp | 0-shot | 1557 | 1.6 [1.0, 2.2] | 1.7 [1.1, 2.3] | 0.4 [0.1, 0.8] |
| mwp | 0-shot CoT | 1557 | 2.1 [1.3, 2.8] | 1.2 [0.6, 1.7] | 0.4 [0.1, 0.8] |
| mwp | few-shot | 1557 | 7.4 [6.1, 8.7] | 3.1 [2.3, 4.0] | 2.6 [1.8, 3.3] |
| mwp | few-shot CoT | 1557 | 6.4 [5.2, 7.6] | 5.2 [4.1, 6.3] | 4.2 [3.3, 5.3] |

### Paired differences (pp, 95% CI, bootstrap p), metric = strict

| task | cond | BPEByte − Transformer | BPEByte − AU-Net |
|---|---|---|---|
| num2word | 0-shot | +0.0 [+0.0, +0.0] p=1.000 | -0.1 [-0.2, +0.0] p=0.719 |
| num2word | 0-shot CoT | +0.0 [+0.0, +0.0] p=1.000 | +0.0 [+0.0, +0.0] p=1.000 |
| num2word | few-shot | +7.5 [+6.0, +9.1] p=0.000 | +4.2 [+2.8, +5.5] p=0.000 |
| num2word | few-shot CoT | +0.0 [+0.0, +0.0] p=1.000 | +0.0 [+0.0, +0.0] p=1.000 |
| word2num | 0-shot | +2.9 [+1.8, +4.0] p=0.000 | +3.8 [+2.8, +5.0] p=0.000 |
| word2num | 0-shot CoT | -0.6 [-1.1, -0.2] p=0.001 | -0.2 [-0.4, +0.0] p=0.263 |
| word2num | few-shot | +2.8 [+1.8, +3.7] p=0.000 | +1.8 [+0.7, +2.8] p=0.002 |
| word2num | few-shot CoT | +0.8 [+0.0, +1.5] p=0.057 | -3.3 [-4.7, -1.9] p=0.000 |
| units | 0-shot | +0.0 [+0.0, +0.0] p=1.000 | -0.2 [-0.5, +0.0] p=0.743 |
| units | 0-shot CoT | +0.0 [+0.0, +0.0] p=1.000 | +0.0 [+0.0, +0.0] p=1.000 |
| units | few-shot | -0.2 [-1.0, +0.7] p=0.839 | -2.5 [-3.8, -1.3] p=0.000 |
| units | few-shot CoT | +1.0 [-0.3, +2.3] p=0.163 | -0.3 [-1.8, +1.2] p=0.738 |
| units | few-shot CoT + knowledge | +3.5 [+2.0, +5.2] p=0.000 | +3.5 [+2.0, +5.0] p=0.000 |
| mwp | 0-shot | -1.2 [-1.9, -0.5] p=0.001 | -1.2 [-1.9, -0.6] p=0.000 |
| mwp | 0-shot CoT | -1.6 [-2.2, -1.0] p=0.000 | -0.7 [-1.2, -0.2] p=0.007 |
| mwp | few-shot | -4.8 [-6.2, -3.5] p=0.000 | -0.6 [-1.3, +0.1] p=0.131 |
| mwp | few-shot CoT | -2.1 [-3.5, -0.8] p=0.003 | -1.0 [-2.2, +0.4] p=0.178 |

# B. Shared subset (all models): few-shot on every item, few-shot CoT (+ knowledge) on 50 questions per split

Models: Transformer, AU-Net, BPEByte, BLT†, H-Net†; 5407 generations.


## Per split, metric = paper


### num2word

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | few-shot | 4.0 | 2.8 | 0.0 | 0.0 | 2.2 |
| AU-Net | few-shot | 9.2 | 7.2 | 0.0 | 0.0 | 5.5 |
| BPEByte | few-shot | 17.0 | 9.8 | 0.5 | 0.5 | 9.1 |
| BLT† | few-shot | 27.5 | 22.5 | 0.0 | 15.0 | 19.2 |
| H-Net† | few-shot | 9.2 | 6.8 | 0.5 | 1.0 | 5.6 |
| Transformer | few-shot CoT | 0.0 | 2.0 | 0.0 | 0.0 | 0.5 |
| AU-Net | few-shot CoT | 2.0 | 0.0 | 0.0 | 0.0 | 0.5 |
| BPEByte | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BLT† | few-shot CoT | 4.0 | 6.0 | 8.0 | 2.0 | 5.0 |
| H-Net† | few-shot CoT | 2.0 | 2.0 | 0.0 | 0.0 | 1.0 |

### word2num

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | few-shot | 0.2 | 0.0 | 0.0 | 0.0 | 0.1 |
| AU-Net | few-shot | 1.0 | 1.8 | 0.5 | 0.5 | 1.1 |
| BPEByte | few-shot | 3.5 | 4.5 | 1.0 | 0.0 | 2.8 |
| BLT† | few-shot | 6.8 | 8.0 | 11.0 | 2.0 | 7.1 |
| H-Net† | few-shot | 1.2 | 1.2 | 0.0 | 0.0 | 0.8 |
| Transformer | few-shot CoT | 2.0 | 4.0 | 0.0 | 0.0 | 1.5 |
| AU-Net | few-shot CoT | 12.0 | 8.0 | 0.0 | 0.0 | 5.0 |
| BPEByte | few-shot CoT | 4.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| BLT† | few-shot CoT | 18.0 | 8.0 | 20.0 | 2.0 | 12.0 |
| H-Net† | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### units

| model | cond | easy | medium | hard | all |
|---|---|---|---|---|---|
| Transformer | few-shot | 3.0 | 0.5 | 0.5 | 1.3 |
| AU-Net | few-shot | 10.0 | 0.5 | 0.5 | 3.7 |
| BPEByte | few-shot | 2.5 | 0.5 | 0.5 | 1.2 |
| BLT† | few-shot | 5.0 | 0.5 | 5.0 | 3.5 |
| H-Net† | few-shot | 1.5 | 0.5 | 1.0 | 1.0 |
| Transformer | few-shot CoT | 14.0 | 0.0 | 2.0 | 5.3 |
| AU-Net | few-shot CoT | 16.0 | 0.0 | 2.0 | 6.0 |
| BPEByte | few-shot CoT | 18.0 | 0.0 | 2.0 | 6.7 |
| BLT† | few-shot CoT | 34.0 | 8.0 | 2.0 | 14.7 |
| H-Net† | few-shot CoT | 30.0 | 0.0 | 0.0 | 10.0 |
| Transformer | few-shot CoT + knowledge | 2.0 | 0.0 | 2.0 | 1.3 |
| AU-Net | few-shot CoT + knowledge | 16.0 | 0.0 | 2.0 | 6.0 |
| BPEByte | few-shot CoT + knowledge | 22.0 | 0.0 | 2.0 | 8.0 |
| BLT† | few-shot CoT + knowledge | 22.0 | 2.0 | 4.0 | 9.3 |
| H-Net† | few-shot CoT + knowledge | 18.0 | 0.0 | 2.0 | 6.7 |

### mwp

| model | cond | AddSub | SingleOp | MultiArith | all |
|---|---|---|---|---|---|
| Transformer | few-shot | 3.8 | 12.1 | 5.7 | 7.5 |
| AU-Net | few-shot | 2.3 | 4.8 | 1.8 | 3.0 |
| BPEByte | few-shot | 2.3 | 4.3 | 1.0 | 2.5 |
| BLT† | few-shot | 30.6 | 37.5 | 4.0 | 22.9 |
| H-Net† | few-shot | 3.0 | 4.1 | 1.2 | 2.7 |
| Transformer | few-shot CoT | 20.0 | 20.0 | 8.0 | 16.0 |
| AU-Net | few-shot CoT | 12.0 | 14.0 | 10.0 | 12.0 |
| BPEByte | few-shot CoT | 6.0 | 24.0 | 8.0 | 12.7 |
| BLT† | few-shot CoT | 40.0 | 54.0 | 20.0 | 38.0 |
| H-Net† | few-shot CoT | 6.0 | 18.0 | 8.0 | 10.7 |

## Per split, metric = strict


### num2word

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | few-shot | 0.8 | 0.2 | 0.0 | 0.0 | 0.3 |
| AU-Net | few-shot | 7.2 | 3.8 | 0.0 | 0.0 | 3.7 |
| BPEByte | few-shot | 16.5 | 7.0 | 0.0 | 0.0 | 7.8 |
| BLT† | few-shot | 24.0 | 19.2 | 0.0 | 14.0 | 16.8 |
| H-Net† | few-shot | 3.0 | 1.5 | 0.0 | 0.0 | 1.5 |
| Transformer | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BPEByte | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| BLT† | few-shot CoT | 2.0 | 0.0 | 8.0 | 2.0 | 3.0 |
| H-Net† | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### word2num

| model | cond | easy | medium | hard_fraction | hard_decimal | all |
|---|---|---|---|---|---|---|
| Transformer | few-shot | 0.2 | 0.0 | 0.0 | 0.0 | 0.1 |
| AU-Net | few-shot | 1.0 | 1.8 | 0.5 | 0.5 | 1.1 |
| BPEByte | few-shot | 3.5 | 4.5 | 1.0 | 0.0 | 2.8 |
| BLT† | few-shot | 6.8 | 8.0 | 11.0 | 3.0 | 7.2 |
| H-Net† | few-shot | 1.2 | 1.2 | 0.0 | 0.0 | 0.8 |
| Transformer | few-shot CoT | 2.0 | 4.0 | 0.0 | 0.0 | 1.5 |
| AU-Net | few-shot CoT | 12.0 | 4.0 | 0.0 | 0.0 | 4.0 |
| BPEByte | few-shot CoT | 4.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| BLT† | few-shot CoT | 18.0 | 4.0 | 20.0 | 4.0 | 11.5 |
| H-Net† | few-shot CoT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### units

| model | cond | easy | medium | hard | all |
|---|---|---|---|---|---|
| Transformer | few-shot | 3.0 | 0.5 | 0.5 | 1.3 |
| AU-Net | few-shot | 10.0 | 0.5 | 0.5 | 3.7 |
| BPEByte | few-shot | 2.5 | 0.5 | 0.5 | 1.2 |
| BLT† | few-shot | 5.0 | 0.5 | 5.0 | 3.5 |
| H-Net† | few-shot | 0.5 | 0.5 | 0.5 | 0.5 |
| Transformer | few-shot CoT | 2.0 | 0.0 | 0.0 | 0.7 |
| AU-Net | few-shot CoT | 6.0 | 0.0 | 0.0 | 2.0 |
| BPEByte | few-shot CoT | 8.0 | 0.0 | 0.0 | 2.7 |
| BLT† | few-shot CoT | 10.0 | 2.0 | 0.0 | 4.0 |
| H-Net† | few-shot CoT | 30.0 | 0.0 | 0.0 | 10.0 |
| Transformer | few-shot CoT + knowledge | 0.0 | 0.0 | 0.0 | 0.0 |
| AU-Net | few-shot CoT + knowledge | 2.0 | 0.0 | 0.0 | 0.7 |
| BPEByte | few-shot CoT + knowledge | 10.0 | 0.0 | 0.0 | 3.3 |
| BLT† | few-shot CoT + knowledge | 6.0 | 0.0 | 0.0 | 2.0 |
| H-Net† | few-shot CoT + knowledge | 12.0 | 0.0 | 0.0 | 4.0 |

### mwp

| model | cond | AddSub | SingleOp | MultiArith | all |
|---|---|---|---|---|---|
| Transformer | few-shot | 3.8 | 11.7 | 5.7 | 7.4 |
| AU-Net | few-shot | 2.3 | 5.2 | 1.8 | 3.1 |
| BPEByte | few-shot | 2.3 | 4.4 | 1.0 | 2.6 |
| BLT† | few-shot | 31.1 | 38.8 | 4.0 | 23.4 |
| H-Net† | few-shot | 2.8 | 3.4 | 1.2 | 2.4 |
| Transformer | few-shot CoT | 10.0 | 10.0 | 4.0 | 8.0 |
| AU-Net | few-shot CoT | 4.0 | 6.0 | 10.0 | 6.7 |
| BPEByte | few-shot CoT | 0.0 | 12.0 | 4.0 | 5.3 |
| BLT† | few-shot CoT | 12.0 | 24.0 | 4.0 | 13.3 |
| H-Net† | few-shot CoT | 4.0 | 8.0 | 0.0 | 4.0 |

## Task totals with 95% CI, metric = paper

| task | cond | n | Transformer | AU-Net | BPEByte | BLT† | H-Net† |
|---|---|---|---|---|---|---|---|
| num2word | few-shot | 1200 | 2.2 [1.5, 3.1] | 5.5 [4.2, 6.8] | 9.1 [7.5, 10.8] | 19.2 [17.0, 21.5] | 5.6 [4.3, 6.9] |
| num2word | few-shot CoT | 200 | 0.5 [0.0, 1.5] | 0.5 [0.0, 1.5] | 0.0 [0.0, 0.0] | 5.0 [2.5, 8.0] | 1.0 [0.0, 2.5] |
| word2num | few-shot | 1200 | 0.1 [0.0, 0.2] | 1.1 [0.6, 1.8] | 2.8 [1.9, 3.8] | 7.1 [5.7, 8.6] | 0.8 [0.3, 1.4] |
| word2num | few-shot CoT | 200 | 1.5 [0.0, 3.5] | 5.0 [2.0, 8.0] | 1.0 [0.0, 2.5] | 12.0 [7.5, 16.5] | 0.0 [0.0, 0.0] |
| units | few-shot | 600 | 1.3 [0.5, 2.3] | 3.7 [2.2, 5.2] | 1.2 [0.3, 2.2] | 3.5 [2.2, 5.0] | 1.0 [0.3, 1.8] |
| units | few-shot CoT | 150 | 5.3 [2.0, 9.3] | 6.0 [2.7, 10.0] | 6.7 [2.7, 10.7] | 14.7 [9.3, 20.7] | 10.0 [5.3, 14.7] |
| units | few-shot CoT + knowledge | 150 | 1.3 [0.0, 3.3] | 6.0 [2.7, 10.0] | 8.0 [4.0, 12.7] | 9.3 [5.3, 14.0] | 6.7 [2.7, 10.7] |
| mwp | few-shot | 1557 | 7.5 [6.2, 8.9] | 3.0 [2.2, 3.9] | 2.5 [1.7, 3.3] | 22.9 [20.8, 25.0] | 2.7 [1.9, 3.5] |
| mwp | few-shot CoT | 150 | 16.0 [10.7, 22.0] | 12.0 [7.3, 17.3] | 12.7 [7.3, 18.0] | 38.0 [30.0, 46.0] | 10.7 [6.0, 16.0] |

### Paired differences (pp, 95% CI, bootstrap p), metric = paper

| task | cond | BPEByte − Transformer | BPEByte − AU-Net | BPEByte − BLT† | BPEByte − H-Net† |
|---|---|---|---|---|---|
| num2word | few-shot | +6.8 [+5.2, +8.5] p=0.000 | +3.6 [+2.2, +4.9] p=0.000 | -10.1 [-12.2, -8.1] p=0.000 | +3.5 [+1.8, +5.2] p=0.000 |
| num2word | few-shot CoT | -0.5 [-1.5, +0.0] p=0.732 | -0.5 [-1.5, +0.0] p=0.751 | -5.0 [-8.5, -2.0] p=0.000 | -1.0 [-2.5, +0.0] p=0.271 |
| word2num | few-shot | +2.8 [+1.8, +3.8] p=0.000 | +1.8 [+0.8, +2.8] p=0.001 | -4.2 [-5.9, -2.7] p=0.000 | +2.0 [+1.0, +3.1] p=0.000 |
| word2num | few-shot CoT | -0.5 [-2.5, +1.5] p=0.826 | -4.0 [-7.5, -1.0] p=0.021 | -11.0 [-16.0, -6.5] p=0.000 | +1.0 [+0.0, +2.5] p=0.271 |
| units | few-shot | -0.2 [-1.0, +0.7] p=0.843 | -2.5 [-3.8, -1.3] p=0.000 | -2.3 [-3.8, -1.0] p=0.000 | +0.2 [-0.8, +1.2] p=0.863 |
| units | few-shot CoT | +1.3 [-2.0, +5.3] p=0.585 | +0.7 [-3.3, +4.7] p=0.870 | -8.0 [-14.0, -2.0] p=0.008 | -3.3 [-8.7, +2.0] p=0.273 |
| units | few-shot CoT + knowledge | +6.7 [+2.7, +11.3] p=0.001 | +2.0 [-2.7, +6.7] p=0.494 | -1.3 [-6.0, +3.3] p=0.690 | +1.3 [-3.3, +6.7] p=0.673 |
| mwp | few-shot | -5.0 [-6.4, -3.7] p=0.000 | -0.5 [-1.2, +0.2] p=0.177 | -20.4 [-22.5, -18.3] p=0.000 | -0.2 [-1.1, +0.7] p=0.721 |
| mwp | few-shot CoT | -3.3 [-8.7, +2.0] p=0.258 | +0.7 [-4.7, +6.0] p=0.907 | -25.3 [-33.3, -16.7] p=0.000 | +2.0 [-2.0, +6.0] p=0.432 |

## Task totals with 95% CI, metric = strict

| task | cond | n | Transformer | AU-Net | BPEByte | BLT† | H-Net† |
|---|---|---|---|---|---|---|---|
| num2word | few-shot | 1200 | 0.3 [0.1, 0.7] | 3.7 [2.7, 4.8] | 7.8 [6.3, 9.4] | 16.8 [14.7, 18.9] | 1.5 [0.8, 2.2] |
| num2word | few-shot CoT | 200 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 3.0 [1.0, 5.5] | 0.0 [0.0, 0.0] |
| word2num | few-shot | 1200 | 0.1 [0.0, 0.2] | 1.1 [0.5, 1.7] | 2.8 [1.9, 3.8] | 7.2 [5.8, 8.8] | 0.8 [0.3, 1.4] |
| word2num | few-shot CoT | 200 | 1.5 [0.0, 3.5] | 4.0 [1.5, 7.0] | 1.0 [0.0, 2.5] | 11.5 [7.5, 16.0] | 0.0 [0.0, 0.0] |
| units | few-shot | 600 | 1.3 [0.5, 2.3] | 3.7 [2.2, 5.2] | 1.2 [0.3, 2.2] | 3.5 [2.2, 5.0] | 0.5 [0.0, 1.2] |
| units | few-shot CoT | 150 | 0.7 [0.0, 2.0] | 2.0 [0.0, 4.7] | 2.7 [0.7, 5.3] | 4.0 [1.3, 7.3] | 10.0 [5.3, 15.3] |
| units | few-shot CoT + knowledge | 150 | 0.0 [0.0, 0.0] | 0.7 [0.0, 2.0] | 3.3 [0.7, 6.7] | 2.0 [0.0, 4.7] | 4.0 [1.3, 7.3] |
| mwp | few-shot | 1557 | 7.4 [6.1, 8.7] | 3.1 [2.3, 4.0] | 2.6 [1.8, 3.3] | 23.4 [21.3, 25.5] | 2.4 [1.6, 3.1] |
| mwp | few-shot CoT | 150 | 8.0 [4.0, 12.7] | 6.7 [2.7, 10.7] | 5.3 [2.0, 9.3] | 13.3 [8.0, 18.7] | 4.0 [1.3, 7.3] |

### Paired differences (pp, 95% CI, bootstrap p), metric = strict

| task | cond | BPEByte − Transformer | BPEByte − AU-Net | BPEByte − BLT† | BPEByte − H-Net† |
|---|---|---|---|---|---|
| num2word | few-shot | +7.5 [+6.0, +9.1] p=0.000 | +4.2 [+2.8, +5.5] p=0.000 | -8.9 [-10.9, -7.0] p=0.000 | +6.3 [+4.8, +7.9] p=0.000 |
| num2word | few-shot CoT | +0.0 [+0.0, +0.0] p=1.000 | +0.0 [+0.0, +0.0] p=1.000 | -3.0 [-5.5, -1.0] p=0.004 | +0.0 [+0.0, +0.0] p=1.000 |
| word2num | few-shot | +2.8 [+1.8, +3.8] p=0.000 | +1.8 [+0.7, +2.8] p=0.001 | -4.4 [-6.1, -2.8] p=0.000 | +2.0 [+1.0, +3.1] p=0.000 |
| word2num | few-shot CoT | -0.5 [-2.5, +1.5] p=0.825 | -3.0 [-6.0, +0.0] p=0.070 | -10.5 [-15.5, -6.0] p=0.000 | +1.0 [+0.0, +2.5] p=0.272 |
| units | few-shot | -0.2 [-1.0, +0.7] p=0.870 | -2.5 [-3.8, -1.3] p=0.000 | -2.3 [-3.8, -1.0] p=0.000 | +0.7 [+0.0, +1.5] p=0.129 |
| units | few-shot CoT | +2.0 [-0.7, +4.7] p=0.253 | +0.7 [-2.0, +3.3] p=0.829 | -1.3 [-5.3, +2.0] p=0.599 | -7.3 [-12.7, -2.7] p=0.002 |
| units | few-shot CoT + knowledge | +3.3 [+0.7, +6.7] p=0.011 | +2.7 [+0.0, +6.0] p=0.131 | +1.3 [-2.0, +4.7] p=0.540 | -0.7 [-5.3, +3.3] p=0.880 |
| mwp | few-shot | -4.8 [-6.2, -3.5] p=0.000 | -0.6 [-1.3, +0.1] p=0.141 | -20.9 [-23.0, -18.7] p=0.000 | +0.2 [-0.7, +1.1] p=0.731 |
| mwp | few-shot CoT | -2.7 [-7.3, +2.0] p=0.340 | -1.3 [-6.0, +3.3] p=0.665 | -8.0 [-14.0, -2.0] p=0.010 | +1.3 [-2.7, +5.3] p=0.631 |

# C. num2word few-shot: first word and demonstration copying

first word = first word of the response equals the gold's first word (case/punctuation ignored); 'One' = response starts with 'one'; copy = response contains 'twenty-three' (from the demonstrations).

| model | n | first word correct % | starts 'One' % | gold starts 'one' % | copies 'twenty-three' % |
|---|---|---|---|---|---|
| Transformer | 1200 | 10.8 | 99.8 | 10.8 | 40.7 |
| AU-Net | 1200 | 85.1 | 18.1 | 10.8 | 14.6 |
| BPEByte | 1200 | 88.2 | 13.0 | 10.8 | 6.2 |
| BLT† | 1200 | 37.7 | 62.4 | 10.8 | 21.5 |
| H-Net† | 1200 | 32.2 | 73.5 | 10.8 | 2.6 |

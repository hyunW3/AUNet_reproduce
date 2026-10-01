
### fixed
| task | metric | bs1 | bs4 | bs8 | bs16 | flips vs own bs1 (bs1, bs4, bs8, bs16) |
|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 63.2 | 63.0 | 64.2 | 63.6 | 0, 5, 7, 10 |
| arc_easy | acc_norm | 64.4 | 63.6 | 64.0 | 63.6 | 0, 4, 2, 4 |
| arc_challenge | acc_norm | 36.2 | 35.4 | 36.0 | 36.0 | 0, 6, 3, 3 |
| piqa | acc_norm | 81.0 | 80.2 | 79.6 | 80.0 | 0, 6, 7, 5 |
| winogrande | acc | 63.6 | 63.6 | 62.8 | 63.0 | 0, 16, 10, 13 |
| boolq | acc | 69.8 | 70.2 | 70.0 | 69.6 | 0, 10, 11, 11 |
| **avg** | | 63.03 | 62.67 | 62.77 | 62.63 | |
  bs1 vs own bs1: n_req=8995 mean|dlogp|=0.0000 max=0.000 frac>0.1=0.000 seconds=387
  bs4 vs own bs1: n_req=8995 mean|dlogp|=0.0921 max=4.146 frac>0.1=0.254 seconds=971
  bs8 vs own bs1: n_req=8995 mean|dlogp|=0.0826 max=3.604 frac>0.1=0.241 seconds=1119
  bs16 vs own bs1: n_req=8995 mean|dlogp|=0.0832 max=3.619 frac>0.1=0.237 seconds=1081

### fixed_entfp32
| task | metric | bs1 | bs4 | bs8 | bs16 | flips vs own bs1 (bs1, bs4, bs8, bs16) |
|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 63.2 | 63.2 | 63.6 | 63.4 | 0, 4, 4, 5 |
| arc_easy | acc_norm | 64.2 | 63.6 | 64.0 | 63.6 | 0, 3, 1, 3 |
| arc_challenge | acc_norm | 35.6 | 35.6 | 35.6 | 35.6 | 0, 4, 2, 2 |
| piqa | acc_norm | 80.4 | 80.4 | 80.0 | 80.2 | 0, 2, 2, 3 |
| winogrande | acc | 63.6 | 63.6 | 63.4 | 63.4 | 0, 10, 7, 7 |
| boolq | acc | 69.8 | 69.4 | 69.4 | 69.6 | 0, 6, 6, 5 |
| **avg** | | 62.80 | 62.63 | 62.67 | 62.63 | |
  bs1 vs own bs1: n_req=8995 mean|dlogp|=0.0000 max=0.000 frac>0.1=0.000 seconds=397
  bs4 vs own bs1: n_req=8995 mean|dlogp|=0.0614 max=1.852 frac>0.1=0.196 seconds=309
  bs8 vs own bs1: n_req=8995 mean|dlogp|=0.0593 max=0.969 frac>0.1=0.181 seconds=352
  bs16 vs own bs1: n_req=8995 mean|dlogp|=0.0589 max=0.969 frac>0.1=0.179 seconds=382

### unfixed
| task | metric | bs1 | bs4 | bs8 | bs16 | flips vs own bs1 (bs1, bs4, bs8, bs16) |
|---|---|---|---|---|---|---|
| hellaswag | acc_norm | 63.4 | 63.2 | 62.4 | 61.4 | 0, 31, 33, 32 |
| arc_easy | acc_norm | 64.2 | 62.6 | 64.2 | 64.2 | 0, 62, 60, 60 |
| arc_challenge | acc_norm | 36.2 | 36.8 | 37.6 | 37.2 | 0, 53, 53, 57 |
| piqa | acc_norm | 80.2 | 77.2 | 76.6 | 77.8 | 0, 45, 46, 46 |
| winogrande | acc | 63.8 | 60.6 | 62.6 | 62.0 | 0, 106, 100, 97 |
| boolq | acc | 69.8 | 65.4 | 65.4 | 67.4 | 0, 124, 112, 108 |
| **avg** | | 62.93 | 60.97 | 61.47 | 61.67 | |
  bs1 vs own bs1: n_req=8995 mean|dlogp|=0.0000 max=0.000 frac>0.1=0.000 seconds=852
  bs4 vs own bs1: n_req=8995 mean|dlogp|=1.0845 max=155.206 frac>0.1=0.751 seconds=769
  bs8 vs own bs1: n_req=8995 mean|dlogp|=1.2331 max=140.250 frac>0.1=0.840 seconds=860
  bs16 vs own bs1: n_req=8995 mean|dlogp|=1.3078 max=67.517 frac>0.1=0.891 seconds=717

# Position-balanced listing probe — five models (all scored on snu55)

Options are permuted per item so the gold sits at position idx mod k (A/B/C/D equally often). Each listing
format is scored three ways, separately (never OR-ed). `ref_*` = the original-order prompts of the 2nd batch,
re-scored on the same machine. ARC-E / ARC-C / PIQA / HellaSwag x 500. Cells: accuracy ±95% half-width.

## 1. Overall accuracy (all 2,000 items)

| condition | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| clean (no listing; option text) | 0.541 ±0.02 | 0.557 ±0.02 | 0.546 ±0.02 | 0.563 ±0.02 | 0.573 ±0.02 |
| original order, options before (echo_all) | 0.433 ±0.02 | 0.447 ±0.02 | 0.445 ±0.02 | 0.415 ±0.02 | 0.366 ±0.02 |
| balanced, before / text | 0.451 ±0.02 | 0.445 ±0.02 | 0.451 ±0.02 | 0.422 ±0.02 | 0.362 ±0.02 |
| balanced, before / letter | 0.320 ±0.02 | 0.322 ±0.02 | 0.321 ±0.02 | 0.316 ±0.02 | 0.322 ±0.02 |
| balanced, before / full | 0.376 ±0.02 | 0.361 ±0.02 | 0.369 ±0.02 | 0.350 ±0.02 | 0.321 ±0.02 |
| balanced, after / text | 0.347 ±0.02 | 0.344 ±0.02 | 0.332 ±0.02 | 0.340 ±0.02 | 0.352 ±0.02 |
| balanced, after / letter | 0.317 ±0.02 | 0.318 ±0.02 | 0.309 ±0.02 | 0.317 ±0.02 | 0.326 ±0.02 |
| balanced, after / full | 0.308 ±0.02 | 0.322 ±0.02 | 0.333 ±0.02 | 0.316 ±0.02 | 0.328 ±0.02 |

Chance: ARC 4-way 0.25 (a few 3/5-way), PIQA 0.5, HellaSwag 0.25 → pooled ≈ 0.31.

## options listed BEFORE the question — scored by option text (" warm")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.625 ±0.05 | 0.584 ±0.05 | 0.622 ±0.05 | 0.713 ±0.05 | 0.761 ±0.04 |
| gold at B | 0.371 ±0.05 | 0.320 ±0.05 | 0.323 ±0.05 | 0.296 ±0.05 | 0.183 ±0.04 |
| gold at C | 0.307 ±0.05 | 0.325 ±0.05 | 0.317 ±0.05 | 0.232 ±0.04 | 0.115 ±0.03 |
| gold at D | 0.331 ±0.05 | 0.360 ±0.05 | 0.347 ±0.05 | 0.264 ±0.04 | 0.179 ±0.04 |
| all 4-option | 0.408 ±0.02 | 0.397 ±0.02 | 0.402 ±0.02 | 0.376 ±0.02 | 0.309 ±0.02 |
| max − min | 0.318 | 0.265 | 0.305 | 0.481 | 0.647 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.458 | 0.431 | 0.442 | 0.567 | 0.692 |
| predicts B | 0.193 | 0.175 | 0.186 | 0.177 | 0.106 |
| predicts C | 0.175 | 0.175 | 0.170 | 0.118 | 0.081 |
| predicts D | 0.175 | 0.218 | 0.202 | 0.138 | 0.120 |


PIQA (2 options) by gold position: Llama 0.848 / 0.304; AU-Net 0.856 / 0.316; BPEByte 0.852 / 0.332; H-Net 0.892 / 0.224; BLT-1B 0.888 / 0.156 (A / B)

## options listed BEFORE the question — scored by letter (" A")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.836 ±0.04 | 0.657 ±0.05 | 0.702 ±0.05 | 0.745 ±0.04 | 0.818 ±0.04 |
| gold at B | 0.086 ±0.03 | 0.065 ±0.02 | 0.070 ±0.03 | 0.070 ±0.03 | 0.043 ±0.02 |
| gold at C | 0.048 ±0.02 | 0.165 ±0.04 | 0.168 ±0.04 | 0.155 ±0.04 | 0.117 ±0.03 |
| gold at D | 0.037 ±0.02 | 0.168 ±0.04 | 0.053 ±0.02 | 0.048 ±0.02 | 0.048 ±0.02 |
| all 4-option | 0.252 ±0.02 | 0.264 ±0.02 | 0.248 ±0.02 | 0.254 ±0.02 | 0.256 ±0.02 |
| max − min | 0.799 | 0.592 | 0.649 | 0.697 | 0.775 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.819 | 0.641 | 0.702 | 0.749 | 0.804 |
| predicts B | 0.097 | 0.056 | 0.067 | 0.065 | 0.044 |
| predicts C | 0.038 | 0.155 | 0.170 | 0.134 | 0.114 |
| predicts D | 0.045 | 0.149 | 0.061 | 0.052 | 0.037 |


PIQA (2 options) by gold position: Llama 0.888 / 0.160; AU-Net 0.812 / 0.180; BPEByte 0.936 / 0.140; H-Net 0.864 / 0.140; BLT-1B 0.920 / 0.116 (A / B)

## options listed BEFORE the question — scored by label + text (" (A) warm" / " A. warm")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.705 ±0.05 | 0.836 ±0.04 | 0.472 ±0.05 | 0.898 ±0.03 | 0.885 ±0.03 |
| gold at B | 0.172 ±0.04 | 0.137 ±0.03 | 0.239 ±0.04 | 0.156 ±0.04 | 0.086 ±0.03 |
| gold at C | 0.155 ±0.04 | 0.064 ±0.02 | 0.323 ±0.05 | 0.051 ±0.02 | 0.016 ±0.01 |
| gold at D | 0.235 ±0.04 | 0.184 ±0.04 | 0.219 ±0.04 | 0.072 ±0.03 | 0.069 ±0.03 |
| all 4-option | 0.316 ±0.02 | 0.305 ±0.02 | 0.313 ±0.02 | 0.294 ±0.02 | 0.264 ±0.02 |
| max − min | 0.550 | 0.772 | 0.253 | 0.847 | 0.869 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.600 | 0.766 | 0.395 | 0.832 | 0.866 |
| predicts B | 0.126 | 0.090 | 0.197 | 0.098 | 0.072 |
| predicts C | 0.103 | 0.037 | 0.236 | 0.025 | 0.011 |
| predicts D | 0.171 | 0.107 | 0.171 | 0.045 | 0.051 |


PIQA (2 options) by gold position: Llama 0.896 / 0.208; AU-Net 0.948 / 0.112; BPEByte 0.860 / 0.220; H-Net 0.828 / 0.212; BLT-1B 0.916 / 0.068 (A / B)

## standard MCQ: options AFTER the question — scored by option text (" warm")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.426 ±0.05 | 0.375 ±0.05 | 0.547 ±0.05 | 0.466 ±0.05 | 0.504 ±0.05 |
| gold at B | 0.172 ±0.04 | 0.191 ±0.04 | 0.153 ±0.04 | 0.124 ±0.03 | 0.142 ±0.04 |
| gold at C | 0.179 ±0.04 | 0.219 ±0.04 | 0.115 ±0.03 | 0.187 ±0.04 | 0.237 ±0.04 |
| gold at D | 0.325 ±0.05 | 0.304 ±0.05 | 0.187 ±0.04 | 0.323 ±0.05 | 0.272 ±0.05 |
| all 4-option | 0.276 ±0.02 | 0.272 ±0.02 | 0.250 ±0.02 | 0.275 ±0.02 | 0.289 ±0.02 |
| max − min | 0.254 | 0.184 | 0.432 | 0.343 | 0.362 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.385 | 0.365 | 0.550 | 0.443 | 0.443 |
| predicts B | 0.171 | 0.182 | 0.161 | 0.126 | 0.120 |
| predicts C | 0.185 | 0.192 | 0.125 | 0.155 | 0.189 |
| predicts D | 0.260 | 0.261 | 0.164 | 0.277 | 0.247 |


PIQA (2 options) by gold position: Llama 0.700 / 0.424; AU-Net 0.700 / 0.416; BPEByte 0.672 / 0.484; H-Net 0.660 / 0.404; BLT-1B 0.636 / 0.444 (A / B)

## standard MCQ: options AFTER the question — scored by letter (" A")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.003 ±0.01 | 0.008 ±0.01 | 0.011 ±0.01 | 0.834 ±0.04 | 0.346 ±0.05 |
| gold at B | 0.473 ±0.05 | 0.172 ±0.04 | 0.470 ±0.05 | 0.070 ±0.03 | 0.207 ±0.04 |
| gold at C | 0.053 ±0.02 | 0.016 ±0.01 | 0.285 ±0.05 | 0.088 ±0.03 | 0.219 ±0.04 |
| gold at D | 0.472 ±0.05 | 0.843 ±0.04 | 0.224 ±0.04 | 0.029 ±0.02 | 0.261 ±0.04 |
| all 4-option | 0.250 ±0.02 | 0.260 ±0.02 | 0.247 ±0.02 | 0.255 ±0.02 | 0.258 ±0.02 |
| max − min | 0.470 | 0.835 | 0.460 | 0.804 | 0.139 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.002 | 0.004 | 0.006 | 0.853 | 0.342 |
| predicts B | 0.478 | 0.165 | 0.468 | 0.048 | 0.180 |
| predicts C | 0.064 | 0.015 | 0.298 | 0.077 | 0.211 |
| predicts D | 0.456 | 0.816 | 0.228 | 0.022 | 0.266 |


PIQA (2 options) by gold position: Llama 0.092 / 0.940; AU-Net 0.312 / 0.660; BPEByte 0.316 / 0.668; H-Net 0.580 / 0.428; BLT-1B 0.604 / 0.452 (A / B)

## standard MCQ: options AFTER the question — scored by label + text (" (A) warm" / " A. warm")

Accuracy by gold position (4-option items: ARC-E / ARC-C / HellaSwag):

| gold position | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| gold at A | 0.064 ±0.02 | 0.032 ±0.02 | 0.075 ±0.03 | 0.410 ±0.05 | 0.528 ±0.05 |
| gold at B | 0.210 ±0.04 | 0.360 ±0.05 | 0.446 ±0.05 | 0.306 ±0.05 | 0.194 ±0.04 |
| gold at C | 0.168 ±0.04 | 0.091 ±0.03 | 0.032 ±0.02 | 0.187 ±0.04 | 0.163 ±0.04 |
| gold at D | 0.539 ±0.05 | 0.533 ±0.05 | 0.453 ±0.05 | 0.128 ±0.03 | 0.176 ±0.04 |
| all 4-option | 0.245 ±0.02 | 0.254 ±0.02 | 0.252 ±0.02 | 0.258 ±0.02 | 0.265 ±0.02 |
| max − min | 0.474 | 0.501 | 0.421 | 0.282 | 0.365 |


Prediction distribution (4-option items):

|  | Llama | AU-Net | BPEByte | H-Net | BLT-1B |
|---|---:|---:|---:|---:|---:|
| predicts A | 0.075 | 0.038 | 0.081 | 0.420 | 0.504 |
| predicts B | 0.213 | 0.349 | 0.434 | 0.300 | 0.181 |
| predicts C | 0.181 | 0.095 | 0.042 | 0.165 | 0.141 |
| predicts D | 0.531 | 0.518 | 0.443 | 0.116 | 0.175 |


PIQA (2 options) by gold position: Llama 0.496 / 0.500; AU-Net 0.456 / 0.592; BPEByte 0.576 / 0.564; H-Net 0.372 / 0.616; BLT-1B 0.388 / 0.648 (A / B)


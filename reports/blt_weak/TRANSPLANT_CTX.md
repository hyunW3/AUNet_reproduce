# Context-side segmentation transplant — BLT-1B (snu55)

Same bytes (echo_all prompt: options listed before the question); only the patch boundaries change.
Everything after the listing is byte-identical to the clean prompt, so `sfx_*` copy the clean prompt's
boundaries there exactly. Clean reference = BLT on the clean prompt, same machine.

| cond | what changes | n | acc | Δ vs nat [95% CI] | Δ vs clean [95% CI] | picks 1st listed (non-HS) |
|---|---|---:|---:|---:|---:|---:|
| clean | no listing (reference) | 2000 | 0.573 | | | |
| nat | natural entropy patching everywhere (echo_all as evaluated) | 2000 | 0.366 | +0.000 [+0.000, +0.000] | -0.207 [-0.235, -0.180] | 0.879 |
| sfx_clean | question + option boundaries copied from the clean prompt; listing natural | 2000 | 0.371 | +0.005 [-0.006, +0.018] | -0.202 [-0.228, -0.174] | 0.907 |
| sfx_list_byte | sfx_clean + listing as 1-byte patches | 2000 | 0.379 | +0.013 [-0.003, +0.027] | -0.195 [-0.221, -0.170] | 0.862 |
| sfx_list_word | sfx_clean + listing split at spaces (one patch per word) | 2000 | 0.423 | +0.058 [+0.041, +0.074] | -0.149 [-0.175, -0.123] | 0.772 |
| sfx_list_opt | sfx_clean + one patch per listed option | 2000 | 0.373 | +0.007 [-0.011, +0.027] | -0.200 [-0.227, -0.174] | 0.749 |

Segmentation diagnostics (mean over all scored options):

- question+option patches after the listing: natural 33.4 vs clean-prompt 46.7; boundaries changed by `sfx_clean`: 16.2 per option
- listing patches (natural): 50.0

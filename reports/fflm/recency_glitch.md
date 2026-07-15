# FFLM recency-glitch test

Accuracy split by whether the **nearest distractor bit disagrees** with the true state. A model that copies the nearest bit (attention glitch) collapses on the *disagree* subset; a true state-tracker is flat. `gap = acc(agree) − acc(disagree)` — bigger = more glitch-prone.

| family | acc (prev agrees) | acc (prev disagrees) | gap | n |
|---|---|---|---|---|
| subword (Llama) | 0.810 (8633) | 0.766 (4888) | +0.044 | 13521 |
| AUNet | 0.787 (8633) | 0.662 (4888) | +0.125 | 13521 |
| BPEByte rg | 0.844 (8633) | 0.751 (4888) | +0.093 | 13521 |

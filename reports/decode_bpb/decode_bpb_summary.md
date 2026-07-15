# Decode-part (Partial) BPB — hybrid vs rg, identical settings

Via `apps.aunet.eval_hybrid_bpb`, identical grid for every model: `prefill=offline_leaf`,
`fracs=0.25,0.5,0.75`, `max_docs=400`, `max_bytes=8192`. rg (C_online) scored with
`--force_hybrid` under the **same explicit boundary grid**. 100M on ece-agpu18, 1.3B on B200.

- **Partial/decode BPB** = NLL over the DECODE (root_greedy) region ÷ decode bytes — **leak-free,
  the actual generation regime → PRIMARY metric**.
- **Full BPB** = all bytes incl. the offline-BPE **prefill** region, which is **non-causal (sees
  future) → contaminated, SECONDARY only**.
- `b/N` = prefill fraction; decode region = the remaining (1−b/N) of the sequence.

## Partial/decode BPB (↓ better)

| scale | b/N | decode | hybrid | rg | Δ(hyb−rg) | winner |
|---|---|---|---|---|---|---|
| 100M | 0.25 | 75% | 4.097 | 3.958 | +0.139 | **rg** |
| 100M | 0.50 | 50% | 4.246 | 4.128 | +0.118 | **rg** |
| 100M | 0.75 | 25% | 4.348 | 4.256 | +0.092 | **rg** |
| 300M | 0.25 | 75% | 4.546 | 4.084 | +0.462 | **rg** |
| 300M | 0.50 | 50% | 4.638 | 4.166 | +0.472 | **rg** |
| 300M | 0.75 | 25% | 4.669 | 4.241 | +0.428 | **rg** |
| 1.3B | 0.25 | 75% | 4.271 | 4.282 | −0.011 | hybrid (tie) |
| 1.3B | 0.50 | 50% | 4.399 | 4.418 | −0.019 | hybrid (tie) |
| 1.3B | 0.75 | 25% | 4.511 | 4.464 | +0.048 | rg |

## Full BPB (↓, CONTAMINATED — secondary)

| scale | b/N | hybrid | rg |
|---|---|---|---|
| 100M | 0.25/0.50/0.75 | 3.581 / 3.540 / 3.464 | 3.771 / 3.893 / 3.960 |
| 1.3B | 0.25/0.50/0.75 | 3.642 / 3.484 / 3.325 | 4.025 / 4.085 / 4.092 |

## Reading

- On the **leak-free decode metric, hybrid does not beat rg** — clearly worse at 100M
  (+0.09…+0.14 BPB), essentially tied at 1.3B (±0.02, rg ahead only at the smallest decode
  region).
- Hybrid's large **Full-BPB** advantage and its strong **100M downstream** numbers come mostly
  from the **non-causal prefill leak**, not from better generation. Isolating the decode region
  removes the edge.
- Net: the leaf/B3 hybrid buys little-to-nothing on true generation quality vs online root_greedy
  at these scales; its benefit is in the (leaked) prefill-scoring regime.

## Pending
- **300M / 760M** decode-BPB pairs run as those hybrid trainings finish (300M is ~95% now).
- Data: `reports/decode_bpb/{hybrid,rg}_{100M,1.3B}.txt` (full per-fraction output + CIs).

#!/bin/bash
# Merge per-family scores of probe $1 from snu55 (<P>/out) and snu234 (<P>_out, <P>_out_hnet) into <P>/scored/.
P=$1; R=/mnt/ssd2/hyun2/AUNet/reports/rep_probes/$P; mkdir -p $R/scored
for f in llama aunet bpebyte blt; do [ -f $R/out/$f.jsonl ] && cp $R/out/$f.jsonl $R/scored/; done
[ -f $R/out_blt/blt.jsonl ] && cp $R/out_blt/blt.jsonl $R/scored/
scp -q snu234:aunet_ext/rep_probes/${P}_out/blt.jsonl $R/scored/ 2>/dev/null
scp -q snu234:aunet_ext/rep_probes/${P}_out_hnet/hnet.jsonl $R/scored/ 2>/dev/null
wc -l $R/scored/*.jsonl

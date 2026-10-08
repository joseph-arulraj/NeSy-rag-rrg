#!/bin/bash
# Lane A items 1-2 (user plan 2026-10-07): CheXmask-geometry ablation inside the Stage 3 head (with and
# without region features, CheXmask-derived region set B) and Stage 3 + retrieval for k = 5, 10, 25.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
CPU=37811782; E=NESY_EMB=letterbox; S4="NESY_S4_RUN=runs/20261006-224243_lb-stage4"; SB=NESY_S4_SETS=B
CON=data/concepts/concepts_final_k6_pruned.csv
S1=runs/20261006-224426_lb-stage1; S3=runs/20261006-231036_lb-stage3; S3nb10=runs/20261006-235600_lb-stage5-s3-nb10
log "lane A1 started"
step a1-s3-nocm $CPU $E -- scripts/stage3_factorised.py --name lb-s3-nocm --concepts $CON --extra "" --baseline $S3
step a1-s3-s4b $CPU $E $S4 $SB -- scripts/stage3_factorised.py --name lb-s3-s4b --concepts $CON --extra chexmask,s4 --baseline $S3
step a1-s3-s4b-nocm $CPU $E $S4 $SB -- scripts/stage3_factorised.py --name lb-s3-s4b-nocm --concepts $CON --extra s4 --baseline $S3
step a2-s3-nb5 $CPU $E -- scripts/stage3_factorised.py --name lb-s3-nb5 --concepts $CON --extra chexmask,nb5 --baseline $S3
step a2-s3-nb25 $CPU $E -- scripts/stage3_factorised.py --name lb-s3-nb25 --concepts $CON --extra chexmask,nb25 --baseline $S3
step a1-bootstrap $CPU $E -- scripts/bootstrap_compare.py --runs S1=$S1 S3=$S3 S3noCM=$(lastdir a1-s3-nocm) \
     S3s4B=$(lastdir a1-s3-s4b) S3s4BnoCM=$(lastdir a1-s3-s4b-nocm) --n-boot 2000
step a2-bootstrap $CPU $E -- scripts/bootstrap_compare.py --runs S3=$S3 S3nb5=$(lastdir a2-s3-nb5) S3nb10=$S3nb10 \
     S3nb25=$(lastdir a2-s3-nb25) --n-boot 2000
log "lane A1 FINISHED"; echo DONE > "$ORCH/STATUS"

#!/bin/bash
# Check that retrieval (k=25, chosen on the Stage 3 base without region features) still helps once region
# features are present: final head (C+A+R+nb25) vs C+A+R, patient bootstrap.
set -uo pipefail
ORCH=$1; A2=$2; A1=$3; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
until [ -f $A2/FINAL_HEAD ]; do sleep 60; done
step final-vs-car $(echo 37831634) NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs CAR=$(cat $A1/last_a1-s3-s4b) \
     FINAL=$(cat $A2/FINAL_HEAD) S1=runs/20261006-224426_lb-stage1 --n-boot 2000
log "lane A5 FINISHED"; echo DONE > "$ORCH/STATUS"

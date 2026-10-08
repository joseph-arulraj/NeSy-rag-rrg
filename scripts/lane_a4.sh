#!/bin/bash
# Input-group table (user, 2026-10-07): head on concepts (C), CheXmask anatomy (A), region scores (R, set B)
# alone and in every combination; patient bootstrap against C+A+R. C, C+A, C+R, C+A+R come from lane A1.
set -uo pipefail
ORCH=$1; A1=$2; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37811782; E="NESY_EMB=letterbox NESY_S4_RUN=runs/20261006-224243_lb-stage4 NESY_S4_SETS=B"
CON=data/concepts/concepts_final_k6_pruned.csv; S3=runs/20261006-231036_lb-stage3
log "lane A4 waiting for lane A1 ($A1)"
until [ -f "$A1/STATUS" ]; do sleep 60; done
a1() { cat "$A1/last_$1"; }
step grp-A $CPU $E -- scripts/stage3_factorised.py --name lb-grp-A --no-concepts --concepts $CON --extra chexmask --baseline $S3
step grp-R $CPU $E -- scripts/stage3_factorised.py --name lb-grp-R --no-concepts --concepts $CON --extra s4 --baseline $S3
step grp-AR $CPU $E -- scripts/stage3_factorised.py --name lb-grp-AR --no-concepts --concepts $CON --extra chexmask,s4 --baseline $S3
step grp-bootstrap $CPU NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs CAR=$(a1 a1-s3-s4b) C=$(a1 a1-s3-nocm) \
     A=$(lastdir grp-A) R=$(lastdir grp-R) CA=$S3 CR=$(a1 a1-s3-s4b-nocm) AR=$(lastdir grp-AR) --n-boot 2000
echo "CAR=$(a1 a1-s3-s4b) C=$(a1 a1-s3-nocm) A=$(lastdir grp-A) R=$(lastdir grp-R) CA=$S3 CR=$(a1 a1-s3-s4b-nocm) AR=$(lastdir grp-AR)" > $ORCH/RUNS
log "lane A4 FINISHED"; echo DONE > "$ORCH/STATUS"

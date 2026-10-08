#!/bin/bash
# Hypothesis-3 comparator (EVAL_PLAN.md): unrestricted linear probe on the CLEAR embedding + the same A (CheXmask
# anatomy) and R (Stage 4 region scores, set B) inputs as the C+A+R head. Fit split; Platt on calib; val metrics.
# Then a val patient bootstrap HEAD vs PROBE_EAR (development only; the H3 test comparison is on test, later).
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
JOB=37831634
HEAD=runs/20261007-065311_a1-s3-s4b; S4R=runs/20261006-224243_lb-stage4
echo RUNNING > $ORCH/STATUS
step h3-probe-ear $JOB NESY_EMB=letterbox NESY_S4_RUN=$S4R NESY_S4_SETS=B -- scripts/probe.py --name h3-probe-emb-a-r \
  --features emb,chexmask,s4 --baseline runs/20261006-224426_lb-stage1 || { echo FAILED > $ORCH/STATUS; exit 1; }
step h3-bootstrap $JOB NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs HEAD=$HEAD PROBE_EAR=$(lastdir h3-probe-ear) --n-boot 2000 || { echo FAILED > $ORCH/STATUS; exit 1; }
log "lane h3 DONE"; echo DONE > $ORCH/STATUS

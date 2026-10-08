#!/bin/bash
# Lane C post steps on the L40S node CPUs: per-view metrics of the final head and DenseNet bootstraps
# (main = inner-holdout epoch; optimistic = best-on-validate epoch, labelled as such).
set -uo pipefail
ORCH=$1; A2=$2; DN=$3; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
L40=37831634; S1=runs/20261006-224426_lb-stage1
log "lane C post (L40S CPUs) waiting for the final head ($A2/FINAL_HEAD)"
until [ -f $A2/FINAL_HEAD ]; do sleep 60; done
HEAD=$(cat $A2/FINAL_HEAD)
step c-view-metrics-head $L40 NESY_EMB=letterbox -- scripts/view_metrics.py --runs HEAD=$HEAD
step c-bootstrap $L40 NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNet=$DN --n-boot 2000
[ -f $DN/optimistic_best_on_val/predictions_non_test.parquet ] && step c-bootstrap-optimistic $L40 NESY_EMB=letterbox -- \
     scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNetOPTIMISTIC=$DN/optimistic_best_on_val --n-boot 2000
log "lane C post FINISHED"; echo DONE > "$ORCH/STATUS"

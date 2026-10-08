#!/bin/bash
# Lane C (user plan 2026-10-07): DenseNet-121 baseline on the L40S after the 256-px image cache is complete;
# then PA/AP metrics and patient bootstraps against letterbox Stage 1 and the final head (from lane A2).
set -uo pipefail
ORCH=$1; C0=$2; C1=$3; A2=$4; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
L40=37831634; CPU=37811782; S1=runs/20261006-224426_lb-stage1
log "lane C waiting for the image cache ($C0, $C1)"
until [ -f $C0/STATUS ] && [ "$(cat $C0/STATUS)" != RUNNING ] && [ -f $C1/STATUS ] && [ "$(cat $C1/STATUS)" != RUNNING ]; do sleep 60; done
if [ "$(cat $C0/STATUS)" != DONE ] || [ "$(cat $C1/STATUS)" != DONE ]; then log "image cache FAILED; stopping"; exit 1; fi
log "image cache done: $(du -sh data/image_cache_256 | cut -f1); free $(free_gb) GB"
until step densenet121 $L40 OMP_NUM_THREADS=8 -- scripts/densenet_train.py --epochs 8 --batch 96 --ckpt-dir $ORCH/densenet_checkpoints_innersel; do
  n=$(( ${n:-0} + 1 )); [ $n -ge 3 ] && { log "DenseNet failed 3 times; stopping"; exit 1; }
  log "DenseNet attempt $n failed; relaunching, resumes from $ORCH/densenet_checkpoints"
done
DN=$(lastdir densenet121)
step c-view-metrics $CPU NESY_EMB=letterbox -- scripts/view_metrics.py --runs S1=$S1 DenseNet=$DN DenseNetOPTIMISTIC=$DN/optimistic_best_on_val
log "waiting for the final head from lane A2 ($A2/FINAL_HEAD)"
until [ -f $A2/FINAL_HEAD ]; do sleep 60; done
HEAD=$(cat $A2/FINAL_HEAD)
step c-view-metrics-head $CPU NESY_EMB=letterbox -- scripts/view_metrics.py --runs HEAD=$HEAD
step c-bootstrap $CPU NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNet=$DN --n-boot 2000
[ -f $DN/optimistic_best_on_val/predictions_non_test.parquet ] && step c-bootstrap-optimistic $CPU NESY_EMB=letterbox -- \
     scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNetOPTIMISTIC=$DN/optimistic_best_on_val --n-boot 2000
log "lane C FINISHED"; echo DONE > "$ORCH/STATUS"

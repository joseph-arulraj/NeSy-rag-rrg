#!/bin/bash
# Close-out chain for the 2026-10-07 runner batch (validate only; no test or external data). CPU allocation.
# A failed step is logged and skipped; the results writer always runs at the end.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37811782
echo RUNNING > $ORCH/STATUS
step extpath-cache $CPU -- scripts/run_pipeline.py --split val --no-stage8 --feature-path external --s4-source cache \
  && step eq-extpath-cache $CPU -- scripts/equivalence_check.py --runner $(lastdir extpath-cache) || log "external path (cache) check FAILED"
log "waiting for the Stage 4 retrain (runs/20261007-1215_lane-extprep)"
until grep -qE "DONE|FAILED" runs/20261007-1215_lane-extprep/STATUS 2>/dev/null; do sleep 60; done
if [ "$(cat runs/20261007-1215_lane-extprep/STATUS)" = DONE ]; then
  S4N=$(cat runs/20261007-1215_lane-extprep/last_s4-saved)
  step s4-compare $CPU -- scripts/s4_compare.py --new $S4N || log "s4-compare FAILED"
  step extpath-models $CPU -- scripts/run_pipeline.py --split val --no-stage8 --feature-path external --s4-source models --s4-models-run $S4N \
    && step eq-extpath-models $CPU -- scripts/equivalence_check.py --runner $(lastdir extpath-models) || log "external path (models) check FAILED"
else
  log "Stage 4 retrain FAILED: skipping s4-compare and the models check"
fi
log "waiting for the Stage 8 rerun (runs/20261007-120046_pipeline-val-v2)"
until grep -qE "DONE|FAILED" runs/20261007-120046_pipeline-val-v2/STATUS 2>/dev/null; do sleep 60; done
step eval-val-v2 $CPU -- scripts/evaluate.py --pipeline-run runs/20261007-120046_pipeline-val-v2 --split val --labels chexpert \
  --comparator PROBE_EAR=runs/20261007-114130_h3-probe-ear/predictions_non_test.parquet --refs data/features/radgraph_val_refs || log "eval FAILED"
step results-runner $CPU -- scripts/results_runner.py --orch $ORCH || log "results writer FAILED"
log "lane close DONE"; echo DONE > $ORCH/STATUS

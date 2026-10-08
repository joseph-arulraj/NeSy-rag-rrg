#!/bin/bash
# After the 2026-10-08 repo restructure: runner on val (no LLM) + equivalence with the C+A+R reference, and the
# external feature path on val with cached scores + equivalence. CPU allocation.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37811782
echo RUNNING > $ORCH/STATUS
step rs-runner $CPU -- scripts/run_pipeline.py --split val --no-stage8 && step rs-eq $CPU -- scripts/equivalence_check.py --runner $(lastdir rs-runner) || log "runner check FAILED"
step rs-ext $CPU -- scripts/run_pipeline.py --split val --no-stage8 --feature-path external --s4-source cache && step rs-eq-ext $CPU -- scripts/equivalence_check.py --runner $(lastdir rs-ext) || log "external path check FAILED"
log "restructure check DONE"; echo DONE > $ORCH/STATUS

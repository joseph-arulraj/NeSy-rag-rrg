#!/bin/bash
# Writes report-draft points 7 and 8 to RESULTS.md after lane C post finishes (or fails); on L40S CPUs.
set -uo pipefail
ORCH=$1; A2=$2; CP=$3; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
log "lane R waiting for lane C post ($CP/STATUS) and the final head"
until [ -f $A2/FINAL_HEAD ] && { [ -f $CP/STATUS ] || grep -q "END   c-bootstrap " $CP/orchestrator.log 2>/dev/null; }; do sleep 60; done
BOOT=$(cat $CP/last_c-bootstrap 2>/dev/null || echo "")
step results-7-8 37831634 NESY_EMB=letterbox -- scripts/results_7_8.py --head $(cat $A2/FINAL_HEAD) --boot "${BOOT:-none}"
log "lane R FINISHED"; echo DONE > "$ORCH/STATUS"

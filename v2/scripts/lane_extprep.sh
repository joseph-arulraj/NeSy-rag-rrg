#!/bin/bash
# External-run prep (user request 2026-10-07): retrain Stage 4 region classifiers exactly as run
# 20261006-224243_lb-stage4 but saving the models (needed to score external images). Train/val labels only.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
echo RUNNING > $ORCH/STATUS
step s4-saved 37831634 NESY_EMB=letterbox -- scripts/stage4_regions.py --shards mimic_full_lb --crossfit 2 --skip-mscxr \
  --save-models --label-splits train,val || { echo FAILED > $ORCH/STATUS; exit 1; }
log "lane extprep (stage 4 retrain) DONE"; echo DONE > $ORCH/STATUS

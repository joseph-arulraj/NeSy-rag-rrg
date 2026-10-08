#!/bin/bash
# Lane A items 3-5 and 7 (user plan 2026-10-07), after lane A1: decide the final head (CheXmask keep/drop,
# retrieval k or drop), train it if needed, out-of-fold check on calib+thresh, knowledge rules R1/R2 on
# validate, association-links ablation against the final head (Stage 1 as accuracy reference).
set -uo pipefail
ORCH=$1; A1=$2; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
CPU=37811782; E=NESY_EMB=letterbox; S4R=runs/20261006-224243_lb-stage4; S4="NESY_S4_RUN=$S4R"; SB=NESY_S4_SETS=B
CON=data/concepts/concepts_final_k6_pruned.csv; S1=runs/20261006-224426_lb-stage1; S3=runs/20261006-231036_lb-stage3
log "lane A2 waiting for lane A1 ($A1)"
until [ -f "$A1/STATUS" ]; do sleep 60; done
a1() { cat "$A1/last_$1"; }
step decide-head $CPU -- scripts/decide_head.py --a1-boot $(a1 a1-bootstrap) --a2-boot $(a1 a2-bootstrap) \
     --s3s4b $(a1 a1-s3-s4b) --s3s4b-nocm $(a1 a1-s3-s4b-nocm) || { log "decision FAILED; stopping"; exit 1; }
BH=$(lastdir decide-head)/best_head.json
EXTRA=$($PY -c "import json;print(json.load(open('$BH'))['extra'])")
REUSE=$($PY -c "import json;print(json.load(open('$BH'))['reuse_run'] or '')")
if [ -n "$REUSE" ]; then HEAD=$REUSE; log "final head = $HEAD (extra $EXTRA)"; else
  step final-head $CPU $E $S4 $SB -- scripts/stage3_factorised.py --name lb-final-head --concepts $CON --extra $EXTRA --baseline $S3 \
       || { log "final head FAILED; stopping"; exit 1; }
  HEAD=$(lastdir final-head); log "final head = $HEAD (extra $EXTRA, new run)"; fi
echo $HEAD > $ORCH/FINAL_HEAD
step a4-oof-calib-thresh $CPU $E $S4 NESY_S4_SETS=A,B -- scripts/oof_check.py --groups s4 --compare calib,thresh
step a5-rules $CPU $E -- scripts/rules_eval.py --pred-run $HEAD --s4-run $S4R --n-boot 2000
echo $(lastdir a5-rules) > $ORCH/RULES_RUN
step a7-aw-head $CPU $E $S4 $SB -- scripts/stage3_factorised.py --name lb-final-head-aw --concepts $CON --extra $EXTRA,aw --baseline $HEAD
step a7-bootstrap $CPU $E -- scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD HEADaw=$(lastdir a7-aw-head) --n-boot 2000
log "lane A2 FINISHED"; echo DONE > "$ORCH/STATUS"

#!/bin/bash
# User items A and B (2026-10-07), on CPU after the lane A decisions: MLP head ablation against the final
# linear head (two hidden sizes; early stopping on val as requested, and on the inner holdout as the unbiased
# check), then case-based evidence evaluation on validate (needs the rules decision).
set -uo pipefail
ORCH=$1; A2=$2; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37814480; E="NESY_EMB=letterbox NESY_S4_RUN=runs/20261006-224243_lb-stage4 NESY_S4_SETS=B"
log "lane A3 (H100 node CPUs) waiting for the final head ($A2/FINAL_HEAD)"
until [ -f $A2/FINAL_HEAD ]; do sleep 60; done
HEAD=$(cat $A2/FINAL_HEAD); log "final head $HEAD"
for H in 64 256; do for ES in val inner; do
  step mlp-h$H-es$ES $CPU $E OMP_NUM_THREADS=6 -- scripts/stage3_mlp.py --linear-run $HEAD --hidden $H --es-split $ES --threads 6
done; done
step mlp-bootstrap $CPU NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs LIN=$HEAD MLP64val=$(lastdir mlp-h64-esval) \
     MLP256val=$(lastdir mlp-h256-esval) MLP64inner=$(lastdir mlp-h64-esinner) MLP256inner=$(lastdir mlp-h256-esinner) --n-boot 2000
log "waiting for the rules decision ($A2/RULES_RUN)"
until [ -f $A2/RULES_RUN ] && [ -f "$(cat $A2/RULES_RUN)/rules_eval.json" ]; do sleep 60; done
step case-eval $CPU NESY_EMB=letterbox -- scripts/case_eval.py --pred-run $HEAD --rules-run $(cat $A2/RULES_RUN) \
     --s4-run runs/20261006-224243_lb-stage4 --k 5
log "lane A3 FINISHED"; echo DONE > "$ORCH/STATUS"

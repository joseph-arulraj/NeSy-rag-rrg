#!/bin/bash
# Lane B (user plan 2026-10-07): Stage 8 on validate with the final head (from lane A2): LLM phrasing,
# RadGraph round trip, regeneration, template fallback. Uses the current H100 (37814480, expires 11:31)
# or the queued one (37831235) once it runs; results.jsonl in a fixed directory, so a relaunch resumes.
set -uo pipefail
ORCH=$1; A2=$2; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/v2/scripts/orch_lib.sh; cd $V2
S4R=runs/20261006-224243_lb-stage4
gpu() { for j in 37814480 37831235; do [ "$(squeue -h -j $j -o %T 2>/dev/null)" = RUNNING ] && { echo $j; return; }; done; }
log "lane B waiting for the final head and rules decision from lane A2 ($A2)"
until [ -f $A2/FINAL_HEAD ] && [ -f $A2/RULES_RUN ] && [ -f "$(cat $A2/RULES_RUN 2>/dev/null)/rules_eval.json" ]; do
  if [ -f $A2/RULES_RUN ] && [ "$(cat $(cat $A2/RULES_RUN)/STATUS 2>/dev/null)" = FAILED ]; then log "rules evaluation FAILED; stopping"; exit 1; fi
  sleep 60; done
HEAD=$(cat $A2/FINAL_HEAD); RULES=$(cat $A2/RULES_RUN)
log "final head $HEAD; rules $RULES"
for n in 1 2 3 4 5; do
  J=$(gpu); until [ -n "$J" ]; do sleep 60; J=$(gpu); done
  step stage8-val $J OMP_NUM_THREADS=8 NESY_EMB=letterbox -- scripts/stage8_phrase.py --pred-run $HEAD --rules-run $RULES --s4-run $S4R \
       --split val --chunk 32 --out $ORCH/stage8_results && break
  log "stage 8 attempt $n did not finish (GPU expiry or error); resuming"
  sleep 30
done
log "lane B FINISHED"; echo DONE > "$ORCH/STATUS"

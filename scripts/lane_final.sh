#!/bin/bash
# Final head = C+A+R (user decision 2026-10-07). Three chains in parallel, then the RESULTS.md writer.
#  CPU : rules R1/R2 measured on C+A+R -> rules-set (R2 keep, R1 drop) -> case-eval
#  L40S: view metrics -> hierarchy violations -> DenseNet bootstraps -> 4 MLP heads -> MLP bootstrap
#  GPU : waits for rules-set and for the running Stage 8 (head with retrieval, lane B) -> Stage 8 on C+A+R
#        (H100 37814480 until 11:31, then 37831235; resumes from $ORCH/stage8_results)
# A failed step skips only what depends on it. STATUS DONE at the end; results-final appends to RESULTS.md.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37811782; L40=37831634
HEAD=runs/20261007-065311_a1-s3-s4b; S1=runs/20261006-224426_lb-stage1; DN=runs/20261007-071846_densenet121
S4R=runs/20261006-224243_lb-stage4; LANEB=runs/20261007-0700_lane-b
E="NESY_EMB=letterbox NESY_S4_RUN=$S4R NESY_S4_SETS=B"
gpu() { for j in 37814480 37831235; do [ "$(squeue -h -j $j -o %T 2>/dev/null)" = RUNNING ] && { echo $j; return; }; done; }
echo $HEAD > $ORCH/FINAL_HEAD
log "lane final started; head $HEAD"
cpu_chain() {
  step rules $CPU NESY_EMB=letterbox -- scripts/rules_eval.py --pred-run $HEAD --s4-run $S4R --n-boot 2000 || { log "rules FAILED: skipping rules-set, case-eval, stage8-car"; return 1; }
  step rules-set $CPU -- scripts/rules_set.py --from-run $(lastdir rules) --r1 drop --r2 keep || { log "rules-set FAILED: skipping case-eval, stage8-car"; return 1; }
  echo $(lastdir rules-set) > $ORCH/RULES_RUN
  step case-eval $CPU NESY_EMB=letterbox -- scripts/case_eval.py --pred-run $HEAD --rules-run $(lastdir rules-set) --s4-run $S4R --k 5 || log "case-eval FAILED (nothing depends on it)"
  return 0
}
l40_chain() {
  step view-metrics $L40 NESY_EMB=letterbox -- scripts/view_metrics.py --runs HEAD=$HEAD S1=$S1 DenseNet=$DN || log "view-metrics FAILED"
  step violations $L40 -- scripts/hierarchy_violations.py --runs HEAD=$HEAD S1=$S1 DenseNet=$DN || log "violations FAILED"
  step c-bootstrap $L40 NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNet=$DN --n-boot 2000 || log "c-bootstrap FAILED"
  step c-bootstrap-optimistic $L40 NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs S1=$S1 HEAD=$HEAD DenseNetOPTIMISTIC=$DN/optimistic_best_on_val --n-boot 2000 || log "c-bootstrap-optimistic FAILED"
  local ok=1
  for H in 64 256; do for ES in inner val; do
    step mlp-h$H-es$ES $L40 $E OMP_NUM_THREADS=8 -- scripts/stage3_mlp.py --linear-run $HEAD --hidden $H --es-split $ES --threads 8 || { log "mlp-h$H-es$ES FAILED"; ok=0; }
  done; done
  if [ $ok = 1 ]; then
    step mlp-bootstrap $L40 NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs LIN=$HEAD MLP64inner=$(lastdir mlp-h64-esinner) \
         MLP256inner=$(lastdir mlp-h256-esinner) MLP64val=$(lastdir mlp-h64-esval) MLP256val=$(lastdir mlp-h256-esval) --n-boot 2000 || log "mlp-bootstrap FAILED"
  else log "an MLP run failed: mlp-bootstrap skipped"; fi
}
gpu_chain() {
  until [ -f $ORCH/RULES_RUN ] || grep -q "skipping .*stage8-car" $ORCH/orchestrator.log; do sleep 60; done
  [ -f $ORCH/RULES_RUN ] || { log "stage8-car skipped (rules missing)"; return 1; }
  log "stage8-car waiting for the running Stage 8 with retrieval to finish ($LANEB/STATUS)"
  until [ -f $LANEB/STATUS ]; do sleep 60; done
  for n in 1 2 3 4 5; do
    J=$(gpu); until [ -n "$J" ]; do sleep 60; J=$(gpu); done
    step stage8-car $J OMP_NUM_THREADS=8 NESY_EMB=letterbox -- scripts/stage8_phrase.py --pred-run $HEAD --rules-run $(cat $ORCH/RULES_RUN) \
         --s4-run $S4R --split val --chunk 32 --out $ORCH/stage8_results && return 0
    log "stage8-car attempt $n did not finish; resuming"; sleep 30
  done
  log "stage8-car FAILED after 5 attempts"; return 1
}
cpu_chain & P1=$!; l40_chain & P2=$!; gpu_chain & P3=$!
wait $P1; wait $P2; wait $P3
step results-final $L40 NESY_EMB=letterbox -- scripts/results_final.py --orch $ORCH --head $HEAD --stage8-retrieval "$(cat $LANEB/last_stage8-val 2>/dev/null)"
log "lane final FINISHED"; echo DONE > "$ORCH/STATUS"

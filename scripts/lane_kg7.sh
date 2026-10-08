#!/bin/bash
# KG task 7 (ablation only): glossary-derived concepts in place of the 77 in the C+A+R head, then a patient
# bootstrap against the C+A+R head. CPU allocation. STATUS DONE at the end.
set -uo pipefail
ORCH=$1; mkdir -p "$ORCH"; source /scratch/prj/bhi_zihe_imaging/NeSy-rag-rrg/scripts/orch_lib.sh; cd $V2
CPU=37811782
HEAD=runs/20261007-065311_a1-s3-s4b; S4R=runs/20261006-224243_lb-stage4
echo RUNNING > $ORCH/STATUS
log "lane kg7 started; reference head $HEAD"
step glossary-concepts $CPU NESY_EMB=letterbox -- scripts/glossary_concepts.py || { log "glossary-concepts FAILED"; echo FAILED > $ORCH/STATUS; exit 1; }
step kg7-s3-gloss $CPU NESY_EMB=letterbox NESY_S4_RUN=$S4R NESY_S4_SETS=B -- scripts/stage3_factorised.py --name kg7-gloss \
  --concepts data/concepts/concepts_glossary.csv --concept-emb data/concepts/glossary_embeddings.pt \
  --extra chexmask,s4 --baseline runs/20261006-231036_lb-stage3 || { log "head FAILED"; echo FAILED > $ORCH/STATUS; exit 1; }
step kg7-bootstrap $CPU NESY_EMB=letterbox -- scripts/bootstrap_compare.py --runs HEAD=$HEAD GLOSS=$(lastdir kg7-s3-gloss) --n-boot 2000 || { log "bootstrap FAILED"; echo FAILED > $ORCH/STATUS; exit 1; }
log "lane kg7 DONE"; echo DONE > $ORCH/STATUS

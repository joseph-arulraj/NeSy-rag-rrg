#!/bin/bash
# CTR-alone AUROC; Stage 3 on the pruned k=6 list (baseline Stage 2); normal-call table; sample reports; bootstrap S1/S2/S3.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
CHAIN=$1; S1=$2; S2=$3; mkdir -p "$CHAIN"; echo RUNNING > "$CHAIN/STATUS"; trap 'echo FAILED > "$CHAIN/STATUS"' ERR
step() { local name=$1; shift; local rd="runs/$(date +%Y%m%d-%H%M)_$name"
  echo "$(date '+%F %T') START $name -> $rd" >> "$CHAIN/chain.log"
  if "$PY" -u "$@" --run-dir "$rd" > "$rd.stdout" 2>&1; then echo "$(date '+%F %T') DONE  $name -> $rd" >> "$CHAIN/chain.log"; echo "$rd" > "$CHAIN/last_$name"
  else echo "$(date '+%F %T') FAIL  $name -> $rd" >> "$CHAIN/chain.log"; exit 1; fi; }
step ctr-auroc scripts/ctr_auroc.py
step stage3-k6-pruned scripts/stage3_factorised.py --name k6-pruned --concepts data/concepts/concepts_final_k6_pruned.csv --baseline "$S2"
S3=$(cat "$CHAIN/last_stage3-k6-pruned")
step normal-call-stage3-pruned scripts/abnormal_absent_check.py --pred-run "$S3" --label stage3-k6-pruned --rules "root 5%;root 2%"
step sample-reports-stage3-pruned scripts/sample_reports.py --pred-run "$S3" --n 80
step bootstrap-s123 scripts/bootstrap_compare.py --runs "S1=$S1" "S2=$S2" "S3=$S3" --n-boot 2000
echo DONE > "$CHAIN/STATUS"; echo "$(date '+%F %T') CHAIN FINISHED" >> "$CHAIN/chain.log"

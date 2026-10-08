#!/bin/bash
# v2 bands report + samples on Stage 3 pruned; Stage 3 with PA-only heart ratio; bootstrap; per-view metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/scratch/users/k23031260/.conda/envs/rrg/bin/python
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
CHAIN=$1; S1=$2; S2=$3; S3=$4; mkdir -p "$CHAIN"; echo RUNNING > "$CHAIN/STATUS"; trap 'echo FAILED > "$CHAIN/STATUS"' ERR
step() { local name=$1; shift; local rd="runs/$(date +%Y%m%d-%H%M)_$name"
  echo "$(date '+%F %T') START $name -> $rd" >> "$CHAIN/chain.log"
  if "$PY" -u "$@" --run-dir "$rd" > "$rd.stdout" 2>&1; then echo "$(date '+%F %T') DONE  $name -> $rd" >> "$CHAIN/chain.log"; echo "$rd" > "$CHAIN/last_$name"
  else echo "$(date '+%F %T') FAIL  $name -> $rd" >> "$CHAIN/chain.log"; exit 1; fi; }
step band-report-s3 scripts/band_report.py --pred-run "$S3"
step sample-reports-bands-v2 scripts/sample_reports.py --pred-run "$S3" --n 80
step stage3-pa-ctr scripts/stage3_factorised.py --name k6-pruned-pa-ctr --concepts data/concepts/concepts_final_k6_pruned.csv --extra chexmask_pa --baseline "$S3"
S3PA=$(cat "$CHAIN/last_stage3-pa-ctr")
step bootstrap-s3-pa scripts/bootstrap_compare.py --runs "S3=$S3" "S3pa=$S3PA" --n-boot 2000
step view-metrics scripts/view_metrics.py --runs "S1=$S1" "S2=$S2" "S3=$S3" "S3pa=$S3PA"
echo DONE > "$CHAIN/STATUS"; echo "$(date '+%F %T') CHAIN FINISHED" >> "$CHAIN/chain.log"
